import json
import logging
import os
import re
import tempfile

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


def _flipkart_scrape_stop_at() -> int:
    """Stop Selenium harvest early once we have enough unique reviews (faster UX)."""
    raw = (os.environ.get("FLIPKART_SCRAPE_STOP_AT") or "").strip()
    if raw:
        try:
            return max(3, min(500, int(raw)))
        except ValueError:
            pass
    try:
        lim = int((os.environ.get("PRODUCT_REVIEW_ANALYSIS_LIMIT") or "80").strip() or "80")
    except ValueError:
        lim = 80
    try:
        need = int((os.environ.get("PRODUCT_REVIEW_MIN_FOR_ANALYSIS") or "12").strip() or "12")
    except ValueError:
        need = 12
    need = max(1, min(need, 500))
    # Never use max(lim, need) — that forced scraping toward PRODUCT_REVIEW_ANALYSIS_LIMIT (e.g. 120)
    # even when only a few reviews were needed (very slow). Stop near min(need, lim) with a small floor.
    try:
        bump = int((os.environ.get("FLIPKART_SCRAPE_STOP_FLOOR") or "25").strip() or "25")
    except ValueError:
        bump = 25
    bump = max(1, min(bump, lim))
    target = min(lim, max(need, bump))
    return max(3, min(500, target))


def _flipkart_network_url_priority(url: str) -> tuple:
    """Prefer XHR URLs that are likely to carry review JSON (fewer wasted getResponseBody calls)."""
    u = (url or "").lower()
    score = 0
    if "review" in u:
        score += 100
    if "product-reviews" in u or "ratings" in u:
        score += 50
    if "/api/" in u or "/1/" in u:
        score += 25
    if "aggregat" in u or "social" in u:
        score += 10
    return (-score, len(u))


def _extract_amazon_reviews(soup):
    reviews = []
    for review_div in soup.select('div[data-hook="review"]'):
        body = review_div.select_one('span[data-hook="review-body"]')
        star_node = review_div.select_one('i[data-hook="review-star-rating"] span') or review_div.select_one('span[data-hook="review-star-rating"]')
        star = None
        if star_node:
            star_text = star_node.get_text(strip=True)
            # Example: '5.0 out of 5 stars'
            try:
                star = float(star_text.split(' ')[0])
            except (ValueError, IndexError):
                star = None

        if body:
            text = body.get_text(separator=" ", strip=True)
            if len(text) > 30:
                reviews.append({"text": text, "stars": star})
    return reviews


def _extract_flipkart_reviews(soup):
    reviews = []
    # Older grid layout
    for review_div in soup.select("div._16PBlm"):
        body = review_div.select_one("div.t-ZTKy") or review_div.select_one("div.qwjRop")
        rating_node = review_div.select_one("div._3LWZlK") or review_div.select_one("div._3Kxusg")
        star = None
        if rating_node:
            try:
                star = float(rating_node.get_text(strip=True))
            except (ValueError, TypeError):
                star = None

        if body:
            text = body.get_text(separator=" ", strip=True)
            if len(text) >= 12:
                reviews.append({"text": text, "stars": star})
    # Newer layouts — keep in sync with app.py scrape_flipkart_selectors_from_soup
    selectors = (
        "div.ZmyHeo",
        "div._6K-7Co",
        "div.t-ZTKy",
        "div[class*='ZmyHeo']",
        "div[class*='t-ZTKy']",
        "div[class*='_27MvfV']",
        "p.z9E0IQ",
    )
    for sel in selectors:
        for div in soup.select(sel):
            text = re.sub(r"\s+", " ", div.get_text(" ", strip=True)).strip()
            if len(text) >= 12:
                reviews.append({"text": text, "stars": None})
    return reviews


def _flipkart_reviews_tab_url(product_url: str):
    """.../p/itmXXX -> .../product-reviews/itmXXX"""
    p = urlparse(product_url)
    path = p.path or ""
    if re.search(r"/product-reviews/itm", path, re.I):
        return None
    if not re.search(r"/p/itm", path, re.I):
        return None
    new_path = re.sub(r"/p/(itm[a-z0-9]+)", r"/product-reviews/\1", path, count=1, flags=re.I)
    if new_path == path:
        return None
    return urlunparse((p.scheme or "https", p.netloc, new_path, "", p.query, ""))


def _flipkart_star_from_obj(obj: dict):
    """Match app.py _flipkart_rating_from_obj (nested reviewRating, string numbers)."""
    for k in ("rating", "ratingValue", "value", "stars"):
        v = obj.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    rr = obj.get("reviewRating")
    if isinstance(rr, dict):
        for k in ("ratingValue", "value", "rating"):
            v = rr.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return None


def _walk_flipkart_json_for_reviews(obj, out: list, depth: int = 0) -> None:
    if depth > 26:
        return
    if isinstance(obj, dict):
        rt = obj.get("reviewText") or obj.get("reviewTextOriginal") or obj.get("reviewBody")
        if rt is None and (
            obj.get("reviewId")
            or obj.get("reviewID")
            or obj.get("entityId")
            or obj.get("reviewerName")
            or obj.get("author")
        ):
            for k in ("text", "value", "description", "comment"):
                v = obj.get(k)
                if isinstance(v, str) and len(v.strip()) > 12:
                    rt = v
                    break
        if isinstance(rt, str) and len(rt.strip()) > 12:
            star = _flipkart_star_from_obj(obj)
            out.append({"text": re.sub(r"\s+", " ", rt).strip(), "stars": star})
            return
        for v in obj.values():
            _walk_flipkart_json_for_reviews(v, out, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _walk_flipkart_json_for_reviews(item, out, depth + 1)


def _extract_flipkart_embedded_json(html: str) -> list:
    out = []
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", id="__NEXT_DATA__"):
        raw = (script.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        _walk_flipkart_json_for_reviews(data, out)
    for script in soup.find_all("script", attrs={"type": "application/json"}):
        raw = (script.string or "").strip()
        if not raw or "reviewText" not in raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        _walk_flipkart_json_for_reviews(data, out)
    # Bundled chunks: Flipkart often inlines large JSON in anonymous <script> (not __NEXT_DATA__).
    for script in soup.find_all("script"):
        if script.get("id") == "__NEXT_DATA__":
            continue
        raw = (script.string or "").strip()
        if not raw or len(raw) < 80 or "reviewText" not in raw:
            continue
        if not (raw.startswith("{") or raw.startswith("[")):
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        _walk_flipkart_json_for_reviews(data, out)
    return out


def _dedupe_review_texts(rows: list) -> list:
    seen = set()
    out = []
    for r in rows:
        t = (r.get("text") or "").strip()
        if len(t) < 12:
            continue
        k = t[:140].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _extract_flipkart_regex_reviewtext(html: str) -> list:
    """
    Fallback when __NEXT_DATA__ / script JSON is escaped or fragmented:
    pull string values for reviewText* keys from raw HTML.
    """
    out: list = []
    if not html or "review" not in html.lower():
        return out
    for key in ("reviewText", "reviewTextOriginal", "reviewBody"):
        pat = re.compile(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % re.escape(key))
        for m in pat.finditer(html):
            raw = m.group(1)
            try:
                t = json.loads('"' + raw + '"')
            except Exception:
                t = raw.replace("\\n", " ").replace('\\"', '"').replace("\\\\", "\\")
            t = re.sub(r"\s+", " ", str(t)).strip()
            if len(t) >= 12 and not t.startswith("http"):
                snippet = html[m.end() : m.end() + 600]
                star = None
                for rpat in (
                    r'"rating"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                    r'"ratingValue"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                ):
                    rm = re.search(rpat, snippet)
                    if rm:
                        try:
                            star = float(rm.group(1))
                        except ValueError:
                            star = None
                        break
                out.append({"text": t, "stars": star})
    return out


def _flipkart_mobile_mirror_url(product_url: str) -> str | None:
    """Some regions serve lighter markup on m.flipkart.com."""
    p = urlparse(product_url)
    host = (p.netloc or "").lower()
    if host == "www.flipkart.com":
        return urlunparse((p.scheme or "https", "m.flipkart.com", p.path or "", "", p.query or "", ""))
    return None


def _flipkart_try_open_reviews(driver) -> None:
    """Click Reviews / All reviews so client-rendered review lists load."""
    try:
        cur = (driver.current_url or "").lower()
        if "product-reviews" in cur:
            return
        time.sleep(1.0)
        xpaths = (
            "//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'product-reviews')]",
            "//a[contains(normalize-space(.),'All reviews')]",
            "//a[contains(normalize-space(.),'Customer Reviews')]",
        )
        for xp in xpaths:
            for el in driver.find_elements(By.XPATH, xp)[:6]:
                try:
                    if el.is_displayed():
                        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                        time.sleep(0.4)
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(4.0)
                        return
                except Exception:
                    continue
    except Exception:
        pass


def _flipkart_scroll_inner_containers(driver) -> None:
    """Scroll overflow containers (Flipkart often puts reviews in a scrollable column)."""
    try:
        driver.execute_script(
            """
            var nodes = document.querySelectorAll('div,section,main,article');
            for (var i = 0; i < nodes.length; i++) {
                var el = nodes[i];
                var st = window.getComputedStyle(el);
                if ((st.overflowY === 'auto' || st.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 80) {
                    el.scrollTop = el.scrollHeight;
                }
            }
            """
        )
        time.sleep(0.9)
    except Exception:
        pass


def _flipkart_send_end_key_scroll(driver, times: int = 28) -> None:
    """Lazy-loaded reviews often respond to keyboard End more reliably than window.scrollTo."""
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        for _ in range(times):
            body.send_keys(Keys.END)
            time.sleep(0.22)
    except Exception:
        pass


def _flipkart_click_next_reviews_page(driver) -> bool:
    """Best-effort pagination on the reviews listing."""
    xpaths = (
        "//a[@aria-label='Next Page']",
        "//a[contains(translate(@aria-label,'NEXT','next'),'next')]",
        "//a[contains(.,'Next Page')]",
        "//nav//a[contains(.,'Next')]",
    )
    for xp in xpaths:
        try:
            for el in driver.find_elements(By.XPATH, xp)[:6]:
                try:
                    if not el.is_displayed():
                        continue
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.3)
                    driver.execute_script("arguments[0].click();", el)
                    time.sleep(3.0)
                    return True
                except Exception:
                    continue
        except Exception:
            continue
    return False


def _flipkart_reviews_from_network_logs(driver):
    """
    Flipkart loads many reviews via XHR; they never appear in document.documentElement.innerHTML.
    Pull JSON bodies from Chrome DevTools Protocol (requires performance logging enabled).
    """
    out: list = []
    if os.environ.get("FLIPKART_NETWORK_LOGS", "1").strip().lower() in ("0", "false", "no"):
        return out
    try:
        logs = driver.get_log("performance")
    except Exception as e:
        logger.debug("[Flipkart] performance log unavailable: %s", e)
        return out

    candidates = []
    for entry in logs:
        try:
            msg = json.loads(entry.get("message", "{}"))
            m = msg.get("message", {})
            if m.get("method") != "Network.responseReceived":
                continue
            params = m.get("params") or {}
            response = params.get("response") or {}
            url = (response.get("url") or "").lower()
            mime = (response.get("mimeType") or "").lower()
            if "flipkart" not in url:
                continue
            if "json" not in mime and "javascript" not in mime and "text" not in mime:
                continue
            if not any(
                k in url
                for k in (
                    "review",
                    "rating",
                    "product",
                    "aggregat",
                    "social",
                    "list",
                    "/api/",
                    "/1/",
                    "browse",
                    "swidget",
                    "retail",
                    "rukmini",
                    "4.se",
                    "discover",
                )
            ):
                continue
            rid = params.get("requestId")
            if rid:
                candidates.append((rid, url))
        except Exception:
            continue

    try:
        max_bodies = int((os.environ.get("FLIPKART_NETWORK_BODY_MAX") or "45").strip() or "45")
    except ValueError:
        max_bodies = 45
    max_bodies = max(5, min(max_bodies, 120))

    candidates.sort(key=lambda pair: _flipkart_network_url_priority(pair[1]))
    seen: set[str] = set()
    fetched = 0
    for rid, url in candidates:
        if fetched >= max_bodies:
            break
        if rid in seen:
            continue
        seen.add(rid)
        try:
            body = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": rid})
            fetched += 1
            text = body.get("body") or ""
            if body.get("base64Encoded"):
                import base64

                text = base64.b64decode(text).decode("utf-8", errors="replace")
            if not text or len(text) < 40:
                continue
            tl = text.lower()
            if "reviewtext" not in tl and not any(
                x in tl for x in ('"reviews"', "reviewid", "reviewername", '"text":"', "reviewrating")
            ):
                continue
            try:
                data = json.loads(text)
            except Exception:
                continue
            chunk: list = []
            _walk_flipkart_json_for_reviews(data, chunk)
            if chunk:
                logger.info("[Flipkart] network JSON %s -> %s review snippets", url[:100], len(chunk))
            out.extend(chunk)
        except Exception as e:
            logger.debug("[Flipkart] skip response %s: %s", url[:90], e)
            continue

    return out


def _flipkart_enable_network_capture(driver) -> None:
    try:
        driver.execute_cdp_cmd("Network.enable", {})
    except Exception as e:
        logger.debug("[Flipkart] Network.enable: %s", e)


def _flipkart_collect_from_page(driver, include_network: bool = True):
    page_source = driver.page_source or ""
    embedded = _extract_flipkart_embedded_json(page_source)
    regex_hits = _extract_flipkart_regex_reviewtext(page_source)
    soup = BeautifulSoup(page_source, "html.parser")
    dom = _extract_flipkart_reviews(soup)
    net = _flipkart_reviews_from_network_logs(driver) if include_network else []
    return _dedupe_review_texts(embedded + dom + regex_hits + net)


def _flipkart_expand_reviews_in_page(driver) -> None:
    """
    Flipkart often loads a handful of reviews first; more appear after scroll / 'View more' / 'See all'.
    """
    try:
        max_rounds = int((os.environ.get("FLIPKART_LOAD_MORE_ROUNDS") or "6").strip() or "6")
    except ValueError:
        max_rounds = 6
    for round_i in range(max_rounds):
        prev_snapshot = len(driver.page_source or "")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.0)
        clicked = False
        for xp in (
            "//button[contains(., 'View more')]",
            "//button[contains(., 'VIEW MORE')]",
            "//span[contains(., 'View more')]",
            "//button[contains(., 'See all')]",
            "//button[contains(., 'SEE ALL')]",
            "//*[contains(., 'View all reviews')]",
            "//*[contains(., 'Read all')]",
        ):
            try:
                for el in driver.find_elements(By.XPATH, xp)[:8]:
                    try:
                        if el.is_displayed() and el.is_enabled():
                            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                            time.sleep(0.35)
                            driver.execute_script("arguments[0].click();", el)
                            clicked = True
                            time.sleep(2.2)
                            break
                    except Exception:
                        continue
                if clicked:
                    break
            except Exception:
                continue
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.0)
        new_snapshot = len(driver.page_source or "")
        if not clicked and new_snapshot < prev_snapshot + 120 and round_i >= 3:
            break


def _extract_generic_reviews(soup):
    reviews = []
    for tag in soup.find_all('span'):
        text = tag.get_text(separator=' ', strip=True)
        if len(text) > 40:
            reviews.append({"text": text, "stars": None})
    return reviews


def _create_flipkart_webdriver():
    """Prefer undetected-chromedriver (harder for Flipkart to flag than stock Selenium)."""
    headless = os.environ.get("FLIPKART_SELENIUM_HEADLESS", "1").strip().lower() not in ("0", "false", "no")
    use_uc = os.environ.get("FLIPKART_USE_UNDETECTED_CHROME", "1").strip().lower() not in ("0", "false", "no")
    if use_uc:
        try:
            import undetected_chromedriver as uc

            opts = uc.ChromeOptions()
            opts.add_argument("--window-size=1920,1080")
            opts.add_argument("--lang=en-US,en")
            opts.add_argument("--disable-blink-features=AutomationControlled")
            if headless:
                opts.add_argument("--headless=new")
                opts.add_argument("--disable-gpu")
                opts.add_argument("--no-sandbox")
                opts.add_argument("--disable-dev-shm-usage")
            opts.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "OFF"})
            driver = uc.Chrome(options=opts, use_subprocess=True)
            driver.set_page_load_timeout(60)
            _flipkart_enable_network_capture(driver)
            logger.info("[Flipkart] Using undetected-chromedriver (headless=%s)", headless)
            return driver
        except Exception as e:
            logger.warning("[Flipkart] undetected-chromedriver failed (%s); using stock Chrome + webdriver-manager", e)

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--lang=en-US,en")
    _chrome_profile = os.path.join(tempfile.gettempdir(), "sentiment_flipkart_chrome")
    chrome_options.add_argument(f"--user-data-dir={_chrome_profile}")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "OFF"})
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(60)
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"},
        )
    except Exception as cdp_e:
        logger.debug("[Selenium] CDP patch skipped: %s", cdp_e)
    _flipkart_enable_network_capture(driver)
    logger.info("[Flipkart] Using stock Chrome + webdriver-manager (headless=%s)", headless)
    return driver


def _scrape_flipkart_selenium(url):
    """Scrape Flipkart reviews using Selenium to bypass anti-bot measures"""
    driver = None

    try:
        wait_s = float((os.environ.get("FLIPKART_SELENIUM_PAGE_WAIT") or "8").strip() or "8")
        driver = _create_flipkart_webdriver()

        skip_mobile = (os.environ.get("FLIPKART_SKIP_MOBILE_URL") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        # Product page first (often more SSR / embedded snippets in headless than /product-reviews/ alone).
        # Then reviews tab; optional m.flipkart mirror (slow — off by default).
        urls: list = []
        alt = _flipkart_reviews_tab_url(url)
        urls.append(url)
        if alt and alt.rstrip("/") != url.rstrip("/"):
            urls.append(alt)
        if not skip_mobile:
            mob = _flipkart_mobile_mirror_url(url)
            if mob:
                m_alt = _flipkart_reviews_tab_url(mob)
                if m_alt and m_alt.rstrip("/") != mob.rstrip("/"):
                    urls.append(m_alt)
                urls.append(mob)
        seen: set[str] = set()
        uniq_urls = []
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            uniq_urls.append(u)

        try:
            harvest_passes = int((os.environ.get("FLIPKART_REVIEW_HARVEST_PASSES") or "4").strip() or "4")
        except ValueError:
            harvest_passes = 4
        harvest_passes = max(1, min(harvest_passes, 15))
        try:
            end_key_times = int((os.environ.get("FLIPKART_END_KEY_SCROLLS") or "14").strip() or "14")
        except ValueError:
            end_key_times = 14
        end_key_times = max(6, min(end_key_times, 48))
        try:
            bottom_loops = int((os.environ.get("FLIPKART_BOTTOM_SCROLL_LOOPS") or "4").strip() or "4")
        except ValueError:
            bottom_loops = 4
        bottom_loops = max(2, min(bottom_loops, 20))

        net_every = int((os.environ.get("FLIPKART_NETWORK_LOGS_EVERY_NTH_PASS") or "1").strip() or "1")
        net_every = max(1, min(net_every, 5))

        stop_at = _flipkart_scrape_stop_at()
        all_reviews: list = []
        harvest_done = False
        for u in uniq_urls:
            if harvest_done:
                break
            logger.info("[Selenium] Navigating to %s", u)
            driver.get(u)
            WebDriverWait(driver, 40).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(wait_s)
            try:
                extra = float((os.environ.get("FLIPKART_PRODUCT_REVIEWS_EXTRA_WAIT") or "4").strip() or "4")
            except ValueError:
                extra = 4.0
            if "product-reviews" in (u or "").lower():
                time.sleep(max(0.0, min(extra, 25.0)))
            _flipkart_try_open_reviews(driver)

            prev_count = -1
            stagnant = 0
            for p in range(harvest_passes):
                _flipkart_send_end_key_scroll(driver, times=end_key_times)
                for _ in range(bottom_loops):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(0.45)
                _flipkart_scroll_inner_containers(driver)
                _flipkart_expand_reviews_in_page(driver)
                include_net = (p % net_every) == 0
                chunk = _flipkart_collect_from_page(driver, include_network=include_net)
                all_reviews = _dedupe_review_texts(all_reviews + chunk)
                n_after = len(all_reviews)
                logger.info(
                    "[Selenium] pass=%s url=%s chunk=%s total_unique=%s",
                    p + 1,
                    u[:90] + ("…" if len(u) > 90 else ""),
                    len(chunk),
                    n_after,
                )
                if n_after >= stop_at:
                    logger.info("[Selenium] Reached stop target (%s unique reviews); finishing early", stop_at)
                    harvest_done = True
                    break
                if n_after == prev_count:
                    stagnant += 1
                else:
                    stagnant = 0
                prev_count = n_after
                if stagnant >= 2 and p >= 2:
                    break
                # Multi-page review UIs: move to next page before the next pass (no-op if only infinite scroll).
                if p < harvest_passes - 1:
                    _flipkart_click_next_reviews_page(driver)

        logger.info("[Selenium] Found %s unique reviews (all URLs + passes)", len(all_reviews))
        return all_reviews

    except Exception as e:
        logger.warning("[Selenium] Error: %s", e)
        return []
    finally:
        if driver:
            driver.quit()


def _scrape_flipkart_requests(url):
    """Fallback method using requests for Flipkart"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
    ]
    
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate", 
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
        "Referer": "https://www.google.com/"
    }
    
    time.sleep(random.uniform(2, 5))
    
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return _extract_flipkart_reviews(soup)
    except:
        return []


def scrape_reviews(url, return_debug: bool = False):
    hostname = urlparse(url).hostname or ""
    debug: list[str] = []

    def _d(msg: str) -> None:
        debug.append(msg)
        logger.info(msg)

    _d(f"[scrape_reviews] Detected hostname: {hostname or '(unknown)'}")

    try:
        reviews = []
        if 'amazon.' in hostname:
            _d("[scrape_reviews] Using Amazon scraper (requests + BeautifulSoup)")
            reviews = _extract_amazon_reviews_soup(url)
        elif 'flipkart.' in hostname:
            _d("[scrape_reviews] Using Flipkart Selenium scraper")
            reviews = _scrape_flipkart_selenium(url)
            if not reviews:
                _d("[scrape_reviews] Selenium found no reviews, trying requests fallback...")
                reviews = _scrape_flipkart_requests(url)
        elif 'netlify.app' in hostname or 'solebazaar' in hostname:
            _d("[scrape_reviews] Using SoleBazaar (Netlify) Selenium scraper")
            reviews = _scrape_solebazaar_selenium(url, debug=debug)
            if not reviews:
                _d("[scrape_reviews] SoleBazaar Selenium found no reviews, trying generic fallback...")
                reviews = _extract_generic_reviews_soup(url)
        else:
            _d(f"Using generic scraper for hostname: {hostname}")
            reviews = _extract_generic_reviews_soup(url)

        if not reviews:
            raise Exception(f"No reviews found at URL: {url}")

        _d(f"[scrape_reviews] Successfully scraped {len(reviews)} reviews")
        if return_debug:
            return reviews, "\n".join(debug)
        return reviews
    except Exception as e:
        if return_debug:
            debug.append(f"[scrape_reviews] error: {e}")
            return [], "\n".join(debug)
        raise Exception(f"Failed to scrape URL: {str(e)}")


def _scrape_solebazaar_selenium(url: str, debug: list[str] | None = None):
    """
    Scraper for your SoleBazaar / Netlify demo shop.
    It mirrors the working logic in the separate scraper_api.py:
    - Open the page with Selenium so JS runs.
    - Ensure the review modal for the requested product is visible.
    - Harvest #reviewList .review blocks into plain {text, stars} rows.
    """
    if debug is None:
        debug = []

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1365,768")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    # Use webdriver-manager so ChromeDriver exists even on fresh machines.
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    wait = WebDriverWait(driver, 25)

    from urllib.parse import urlparse as _urlparse, parse_qs as _parse_qs

    product_id = None
    try:
        product_id = (_parse_qs(_urlparse(url).query).get("product") or [None])[0]
    except Exception:
        product_id = None
    debug.append(f"[SoleBazaar] Parsed product_id={product_id!r}")

    try:
        debug.append(f"[SoleBazaar] Opening URL: {url}")
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        def modal_is_open(d):
            el = d.find_element(By.ID, "reviewModal")
            cls = el.get_attribute("class") or ""
            return "hidden" not in cls

        try:
            # If the frontend already shows the modal for ?product=<id>, we can harvest immediately.
            debug.append("[SoleBazaar] Waiting for modal to be open...")
            wait.until(modal_is_open)
        except Exception:
            debug.append("[SoleBazaar] Modal not open yet; clicking product card...")
            # Fallback: click the matching product card (or first card) to open modal.
            wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".product-card")) > 0)
            cards = driver.find_elements(By.CSS_SELECTOR, ".product-card")
            target = None
            if product_id:
                for card in cards:
                    if (card.get_attribute("data-product-id") or "").strip() == str(product_id):
                        target = card
                        break
            if not target and cards:
                target = cards[0]
            if target:
                target.click()
            wait.until(modal_is_open)

        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#reviewList .review")))
        time.sleep(0.3)  # small settle so DOM is stable

        review_items = driver.find_elements(By.CSS_SELECTOR, "#reviewList .review")
        reviews = []
        for item in review_items:
            try:
                user_el = item.find_element(By.TAG_NAME, "strong")
            except Exception:
                user_el = None
            try:
                text_el = item.find_element(By.TAG_NAME, "p")
            except Exception:
                text_el = None
            stars = item.text.count("★")
            text_val = text_el.text.strip() if text_el else ""
            if text_val:
                reviews.append(
                    {"text": text_val, "stars": stars if stars else None, "source": "solebazaar"}
                )

        logger.info("[SoleBazaar] harvested %s reviews from modal", len(reviews))
        debug.append(f"[SoleBazaar] Harvested {len(reviews)} reviews")
        return reviews
    except Exception as e:
        logger.warning("[SoleBazaar] Selenium error: %s", e)
        debug.append(f"[SoleBazaar] Selenium error: {e}")
        return []
    finally:
        driver.quit()


def _extract_amazon_reviews_soup(url):
    """Extract Amazon reviews using requests + BeautifulSoup"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
    ]
    
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate", 
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
        "Referer": "https://www.google.com/"
    }
    
    time.sleep(random.uniform(2, 5))
    
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return _extract_amazon_reviews(soup)
    except Exception as e:
        raise Exception(f"Amazon scraping failed: {e}")


def _extract_generic_reviews_soup(url):
    """Extract generic reviews using requests + BeautifulSoup"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
    ]
    
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate", 
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
        "Referer": "https://www.google.com/"
    }
    
    time.sleep(random.uniform(2, 5))
    
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return _extract_generic_reviews(soup)
    except Exception as e:
        raise Exception(f"Generic scraping failed: {e}")