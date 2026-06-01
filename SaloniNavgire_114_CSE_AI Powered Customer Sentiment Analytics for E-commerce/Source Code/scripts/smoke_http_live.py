#!/usr/bin/env python3
"""
Live HTTP smoke test: prints each request method + URL + status (requires Flask already running).

  set BASE_URL=http://127.0.0.1:5001
  python scripts/smoke_http_live.py

Or from project root:
  .venv\\Scripts\\python scripts/smoke_http_live.py
"""
from __future__ import annotations

import os
import sys

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:5001").rstrip("/")


def log(method: str, path: str, **kwargs):
    url = f"{BASE}{path}"
    print(f"{method:7} {url}", end=" ")
    try:
        r = requests.request(method, url, timeout=30, allow_redirects=True, **kwargs)
    except requests.RequestException as e:
        print(f"-> ERROR: {e}")
        return None
    print(f"-> {r.status_code}")
    return r


def main():
    print(f"BASE_URL={BASE}\n")
    # GET
    for path in (
        "/",
        "/about",
        "/contact",
        "/auth/login",
        "/auth/signup",
        "/legacy/text",
        "/product",
        "/results",
        "/history",
        "/sklearn-analytics",
        "/api/trends",
        "/api/dashboard-stats",
    ):
        log("GET", path)

    # PUT / DELETE (expect 405 on typical pages)
    for path in ("/", "/product", "/about"):
        log("PUT", path)
        log("DELETE", path)

    # POST (no side effects where possible)
    log("POST", "/auth/logout")
    log("POST", "/product", data={"product_url": "not-a-url", "compare_url": ""})

    print(
        "\nDone. If GET /sklearn-analytics is 404 but pytest passes, restart `python app.py` "
        "so the live server loads the same app.py as your tests."
    )


if __name__ == "__main__":
    main()
