import importlib.util
import os
import sys
import types


def _stub_torchvision_for_transformers_if_broken():
    """
    Newer `transformers` imports `torchvision` when PIL is installed. If `torchvision`
    was built for a different `torch` (common on Windows), import fails with:
    RuntimeError: operator torchvision::nms does not exist.
    This app only uses text models; a tiny stub is enough for `transformers.image_utils`.
    """
    broken_torchvision = False
    if "torchvision" in sys.modules and getattr(sys.modules["torchvision"], "__spec__", None) is None:
        broken_torchvision = True
        del sys.modules["torchvision"]

    try:
        spec = importlib.util.find_spec("torchvision")
    except ValueError:
        broken_torchvision = True
        spec = None
    except Exception:
        spec = None

    if spec is None and not broken_torchvision:
        return

    try:
        import torchvision  # noqa: F401
        return
    except Exception:
        for key in list(sys.modules):
            if key == "torchvision" or key.startswith("torchvision."):
                del sys.modules[key]

        class InterpolationMode:
            NEAREST_EXACT = object()
            NEAREST = object()
            BOX = object()
            BILINEAR = object()
            HAMMING = object()
            BICUBIC = object()
            LANCZOS = object()

        transforms = types.ModuleType("torchvision.transforms")
        transforms.InterpolationMode = InterpolationMode
        transforms.__spec__ = importlib.util.spec_from_loader("torchvision.transforms", loader=None)

        tv = types.ModuleType("torchvision")
        tv.__spec__ = importlib.util.spec_from_loader("torchvision", loader=None, is_package=True)
        tv.__path__ = []
        tv.transforms = transforms

        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = transforms


_stub_torchvision_for_transformers_if_broken()

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, abort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re
import string
from datetime import datetime
import csv
import io
import json
import struct
import logging
import threading
import time
from functools import wraps
import smtplib
from email.message import EmailMessage
import html
# Add current directory to Python path so local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from bs4 import BeautifulSoup
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from dotenv import load_dotenv, dotenv_values
import certifi

# Load .env from this folder; override=True so a stale User/System RAINFOREST_API_KEY=your_rainforest_api_key_here does not win.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

_logger = logging.getLogger(__name__)
# Quiet HF Hub HTTP chatter before models load (configure_logging reinforces this).
for _quiet_hub in ("httpx", "httpcore"):
    logging.getLogger(_quiet_hub).setLevel(logging.WARNING)


def configure_logging(application):
    """
    Send all relevant logs to stderr so the dev terminal shows actions (HTTP + app INFO).
    Set LOG_LEVEL=DEBUG for more detail; LOG_REQUESTS=0 to skip per-request lines.
    """
    level_name = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=True, stream=sys.stderr)
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    for noisy in (
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "transformers",
        "huggingface_hub",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _logger.setLevel(level)
    application.logger.setLevel(level)


# PyMongo TLS: prefer stdlib ssl over PyOpenSSL; cap TLS 1.2 + lower OpenSSL SECLEVEL for Atlas on Windows.
import pymongo.ssl_support as _pymongo_ssl

if os.getenv("MONGO_USE_PYOPENSSL", "0").strip().lower() not in ("1", "true", "yes"):
    _pymongo_ssl.HAVE_PYSSL = False

_orig_get_ssl_context = _pymongo_ssl.get_ssl_context


def _get_ssl_context_tls12(*args, **kwargs):
    ctx = _orig_get_ssl_context(*args, **kwargs)
    try:
        import ssl as _stdlib_ssl

        if hasattr(_stdlib_ssl, "TLSVersion") and hasattr(ctx, "minimum_version"):
            ctx.minimum_version = _stdlib_ssl.TLSVersion.TLSv1_2
            ctx.maximum_version = _stdlib_ssl.TLSVersion.TLSv1_2
        try:
            ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        except Exception:
            pass
    except Exception:
        pass
    return ctx


if os.getenv("MONGO_TLS_NO_CAP", "0").strip().lower() not in ("1", "true", "yes"):
    _pymongo_ssl.get_ssl_context = _get_ssl_context_tls12

from pymongo import MongoClient
from bson.objectid import ObjectId

from prediction.predict import predict_sentiment
from prediction.smart_reply import generate_smart_reply, enrich_replies_with_contact_info
from scraper.review_scraper import scrape_reviews

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

configure_logging(app)
app.logger.info(
    "SentimentPulse logging to stderr | LOG_LEVEL=%s | LOG_REQUESTS=%s",
    os.environ.get("LOG_LEVEL", "INFO"),
    os.environ.get("LOG_REQUESTS", "1"),
)


def _should_log_request() -> bool:
    return os.environ.get("LOG_REQUESTS", "1").strip().lower() not in ("0", "false", "no")


def _should_run_startup_request_probe() -> bool:
    """One local GET after bind so >>> / <<< lines appear without opening a browser first."""
    if not _should_log_request():
        return False
    return os.environ.get("STARTUP_REQUEST_PROBE", "1").strip().lower() not in ("0", "false", "no")


def _schedule_startup_request_probe(port: int) -> None:
    if not _should_run_startup_request_probe():
        return

    def _run():
        time.sleep(0.75)
        try:
            requests.get(f"http://127.0.0.1:{port}/", timeout=15)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


@app.before_request
def _log_incoming_request():
    if request.path.startswith("/static"):
        return
    if _should_log_request():
        line = f">>> {request.method} {request.path}"
        # Use Flask's logger (not __main__._logger) so lines always show after requests hit the server.
        app.logger.info(line)
        print(line, file=sys.stderr, flush=True)


@app.after_request
def _log_outgoing_response(response):
    if request.path.startswith("/static"):
        return response
    if _should_log_request():
        line = f"<<< {request.method} {request.path} -> {response.status_code}"
        app.logger.info(line)
        print(line, file=sys.stderr, flush=True)
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANON_PRODUCT_RUN_LIMIT = int(os.environ.get("ANON_PRODUCT_RUN_LIMIT", "2"))

# MongoDB — PyMongo builds TLS from tls* kwargs only; stdlib ssl.SSLContext is not a valid MongoClient option.
MONGO_URI = os.getenv("MONGO_URI")
_MONGO_SEL_MS = int(os.environ.get("MONGO_SERVER_SELECTION_TIMEOUT_MS", "30000"))


def _mongo_client_options():
    """
    Atlas + Windows + Python 3.13: use tlsInsecure=True by default (set MONGO_TLS_STRICT=1 for full verification).
    """
    opts = {
        "serverSelectionTimeoutMS": _MONGO_SEL_MS,
        "connectTimeoutMS": 20_000,
        "socketTimeoutMS": 20_000,
        "tls": True,
        "tlsCAFile": certifi.where(),
    }
    strict = os.getenv("MONGO_TLS_STRICT", "0").strip().lower() in ("1", "true", "yes")
    if strict:
        ocsp = os.getenv("MONGO_TLS_DISABLE_OCSP", "1" if sys.platform.startswith("win") else "0").strip().lower()
        if ocsp in ("1", "true", "yes"):
            opts["tlsDisableOCSPEndpointCheck"] = True
    else:
        opts["tlsInsecure"] = True
    if os.getenv("MONGO_TLS_INSECURE", "").strip().lower() in ("1", "true", "yes"):
        opts["tlsInsecure"] = True
    return opts


client = MongoClient(MONGO_URI, **_mongo_client_options())

db = client["sentiment_db"]
users_collection = db.users
analyses_collection = db.analyses
contact_collection = db.contact_messages
product_reviews_collection = db.product_reviews
sentiment_runs_collection = db.sentiment_runs

# -----------------------------
# Load GO-Emotions Model (kept for "text analyzer" feature)
model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sentiment model for product reviews (implemented without transformers.pipeline to avoid torchvision dependency)
sent_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)

# List of emotions
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise',
    'neutral'
]

# -----------------------------
# Validation Helpers
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def validate_name(name, field_label="Name"):
    name = (name or "").strip()
    if not name:
        return False, f"{field_label} is required."
    if any(ch.isdigit() for ch in name):
        return False, f"{field_label} cannot contain numbers."
    if any(ch in string.punctuation for ch in name):
        return False, f"{field_label} cannot contain special characters."
    if len(name) < 2:
        return False, f"{field_label} must be at least 2 characters."
    return True, ""


def validate_email(email):
    email = (email or "").strip().lower()
    if not email or not EMAIL_REGEX.match(email):
        return False, "Please enter a valid email address."
    return True, email


def validate_password(password, confirm_password=None):
    password = password or ""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(ch.isupper() for ch in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(ch in string.punctuation for ch in password):
        return False, "Password must contain at least one special character."
    if confirm_password is not None and password != confirm_password:
        return False, "Password and Confirm Password do not match."
    return True, ""


def send_otp_email(to_email: str, otp: str):
    """
    Send OTP via SMTP.
    Configure via environment:
      SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM
      SMTP_USE_TLS=true/false (default true)
    """
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USERNAME", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr = os.environ.get("SMTP_FROM", username).strip()
    use_tls = (os.environ.get("SMTP_USE_TLS", "true").strip().lower() != "false")

    if not host or not username or not password or not from_addr:
        raise RuntimeError(
            "SMTP is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM in your .env"
        )

    msg = EmailMessage()
    msg["Subject"] = "Your SentimentPulse password reset OTP"
    msg["From"] = from_addr
    msg["To"] = to_email
    msg.set_content(
        f"Your OTP for password reset is: {otp}\n\n"
        "If you did not request this, you can ignore this email."
    )

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.ehlo()
        if use_tls:
            server.starttls()
            server.ehlo()
        server.login(username, password)
        server.send_message(msg)


def send_contact_email(name: str, email: str, message: str):
    """
    Send contact message to navgiresaloni10@gmail.com.

    Uses SendGrid API when SENDGRID_API_KEY is provided.
    Falls back to SMTP using SMTP_* vars.
    """
    to_address = "navgiresaloni10@gmail.com"

    # Prefer SendGrid API key if provided
    sendgrid_api_key = os.environ.get("SENDGRID_API_KEY", "").strip()
    if sendgrid_api_key:
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("requests library is required for SendGrid API use") from e

        sg_data = {
            "personalizations": [{
                "to": [{"email": to_address}],
                "subject": f"New Contact Message from {name}"
            }],
            "from": {"email": os.environ.get("SMTP_FROM", username if (username := os.environ.get('SMTP_USERNAME', '')).strip() else to_address)},
            "content": [{
                "type": "text/plain",
                "value": (
                    f"New contact message received:\n\n"
                    f"Name: {name}\n"
                    f"Email: {email}\n"
                    f"Message:\n{message}\n\n"
                    f"Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            }]
        }
        r = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {sendgrid_api_key}",
                "Content-Type": "application/json"
            },
            json=sg_data,
            timeout=20
        )
        if r.status_code not in (200, 202):
            raise RuntimeError(f"SendGrid request failed: {r.status_code} {r.text}")
        return

    # Fallback to SMTP
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USERNAME", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr = os.environ.get("SMTP_FROM", username).strip()
    use_tls = (os.environ.get("SMTP_USE_TLS", "true").strip().lower() != "false")

    if not host or not username or not password or not from_addr:
        raise RuntimeError(
            "SMTP is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM in your .env"
        )

    msg = EmailMessage()
    msg["Subject"] = f"New Contact Message from {name}"
    msg["From"] = from_addr
    msg["To"] = "navgiresaloni10@gmail.com"  # Send to admin's email
    msg.set_content(
        f"New contact message received:\n\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Message:\n{message}\n\n"
        f"Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.ehlo()
        if use_tls:
            server.starttls()
            server.ehlo()
        server.login(username, password)
        server.send_message(msg)


# -----------------------------
# Text Validation Functions
def validate_text(text):
    """
    Comprehensive text validation for emotion/intent recognition.
    Returns (is_valid, error_message)
    """
    # 1. Empty field check
    if not text or not text.strip():
        return False, "⚠️ Please enter some text. Empty field is not allowed."
    
    text = text.strip()
    
    # 2. Minimum length check (at least 10 characters)
    if len(text) < 10:
        return False, "⚠️ Text is too short. Please enter at least 10 characters."
    
    # 3. Minimum word count check (at least 3 words)
    words = text.split()
    if len(words) < 3:
        return False, "⚠️ Please enter at least 3 words for meaningful analysis."
    
    # 4. Check for numbers only
    if re.match(r'^[\d\s]+$', text):
        return False, "⚠️ Input cannot contain only numbers."
    
    # 5. Check for special characters only
    # Remove spaces and check if remaining are only special chars
    text_no_spaces = re.sub(r'\s+', '', text)
    if text_no_spaces and not re.search(r'[a-zA-Z0-9]', text_no_spaces):
        return False, "⚠️ Input cannot contain only special characters."
    
    # 6. Check if text contains any alphabetic characters
    if not re.search(r'[a-zA-Z]', text):
        return False, "⚠️ Please enter text containing English letters."
    
    # 7. Check for excessive special characters (more than 30% special chars)
    special_char_count = sum(1 for c in text if c in string.punctuation)
    if len(text) > 0 and (special_char_count / len(text)) > 0.3:
        return False, "⚠️ Too many special characters. Please enter meaningful text."
    
    # 8. Gibberish detection - check vowel/consonant ratio
    # Extract only alphabetic characters
    alpha_text = re.sub(r'[^a-zA-Z]', '', text.lower())
    if len(alpha_text) > 0:
        vowels = sum(1 for c in alpha_text if c in 'aeiou')
        consonants = len(alpha_text) - vowels
        if consonants > 0:
            vowel_ratio = vowels / len(alpha_text)
            # Normal English text has roughly 40% vowels
            if vowel_ratio < 0.15:  # Too few vowels suggests gibberish
                return False, "⚠️ Text appears to be gibberish. Please enter meaningful English text."
    
    # 9. Check for repeated characters/patterns (like "aaaa", "abcabc")
    if len(text) >= 4:
        # Check for 4+ consecutive identical characters
        if re.search(r'(.)\1{3,}', text):
            return False, "⚠️ Text contains excessive repeated characters. Please enter meaningful text."
        
        # Check for repeated word patterns
        words_lower = [w.lower() for w in words if len(w) > 2]
        if len(words_lower) >= 2:
            # Check if same word repeats too many times
            word_counts = {}
            for word in words_lower:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repeat = max(word_counts.values()) if word_counts else 0
            if max_repeat > len(words_lower) * 0.6:  # More than 60% same word
                return False, "⚠️ Text contains too many repeated words. Please enter meaningful text."
    
    # 10. Check for meaningful word length (most words should be 2+ characters)
    short_words = sum(1 for w in words if len(w) <= 1)
    if len(words) > 0 and (short_words / len(words)) > 0.5:
        return False, "⚠️ Text contains too many single-character words. Please enter meaningful sentences."
    
    # 11. Basic grammar/structure check
    # Check if text starts with capital letter (for sentences) or at least has proper structure
    # Allow lowercase for casual text, but check for basic sentence structure
    # Check for at least one word that's 4+ characters (common words)
    long_words = sum(1 for w in words if len(re.sub(r'[^a-zA-Z]', '', w)) >= 4)
    if len(words) >= 3 and long_words == 0:
        return False, "⚠️ Text appears meaningless. Please enter a proper sentence with meaningful words."
    
    # 12. Check for common English word patterns
    # At least some words should contain vowels
    words_with_vowels = sum(1 for w in words if re.search(r'[aeiouAEIOU]', w))
    if len(words) > 0 and (words_with_vowels / len(words)) < 0.3:
        return False, "⚠️ Text appears to be gibberish. Please enter meaningful English text."
    
    # 13. Check for excessive numbers mixed with text (like "abc123def456")
    # More than 50% numbers suggests invalid input
    digit_count = sum(1 for c in text if c.isdigit())
    if len(text) > 0 and (digit_count / len(text)) > 0.5:
        return False, "⚠️ Text contains too many numbers. Please enter meaningful text."
    
    # 14. Check for single word inputs that are too short (like "rakhi")
    if len(words) == 1 and len(words[0]) < 8:
        return False, "⚠️ Single word input is too short. Please enter a complete sentence."
    
    # All validations passed
    return True, ""


def generate_emotion_smart_reply(emotion: str, confidence: float) -> str:
    """
    Generate smart reply suggestions based on detected emotion.
    Returns a helpful response suggestion.
    """
    emotion = emotion.lower()
    
    # Convert confidence to float safely (handles both string and numeric values)
    try:
        confidence = float(confidence)
    except (ValueError, TypeError):
        confidence = 0.0
    
    confidence_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"

    replies = {
        "joy": {
            "high": [
                "I'm thrilled to hear that! What made your experience so positive?",
                "That's wonderful! I'd love to hear more about what you enjoyed.",
                "Great to hear you're happy! How can we make it even better?"
            ],
            "moderate": [
                "Glad you're feeling positive! What aspects did you enjoy most?",
                "That's good to hear! Is there anything we could improve?",
                "Nice to see you're satisfied! What stood out to you?"
            ],
            "low": [
                "Seems like you're somewhat pleased. What worked well for you?",
                "Good to hear you're content. Any suggestions for improvement?",
                "Thanks for the positive feedback! What could we enhance?"
            ]
        },
        "sadness": {
            "high": [
                "I'm sorry to hear you're feeling down. How can I help make this better?",
                "I understand this is disappointing. What specifically concerns you?",
                "I'm truly sorry for your negative experience. Let's work on improving this."
            ],
            "moderate": [
                "I see you're not entirely satisfied. What can we do to help?",
                "Sorry to hear you're feeling this way. How can we make it right?",
                "I understand your disappointment. What would help improve your experience?"
            ],
            "low": [
                "Seems like you're a bit disappointed. What could we do better?",
                "I hear your concern. How can we address this for you?",
                "Thanks for letting us know. What would make this more positive?"
            ]
        },
        "anger": {
            "high": [
                "I apologize for upsetting you. This is not the experience we want for our customers.",
                "I'm very sorry this has frustrated you. Let's resolve this immediately.",
                "I understand your anger and I want to make this right. How can I help?"
            ],
            "moderate": [
                "I apologize for any frustration this caused. What can we do to fix this?",
                "Sorry this has upset you. How can we improve your experience?",
                "I understand you're frustrated. Let's work together to resolve this."
            ],
            "low": [
                "Sorry if this has annoyed you. What would help make this better?",
                "I hear your frustration. How can we address your concerns?",
                "Apologies for any inconvenience. What can we do to help?"
            ]
        },
        "fear": {
            "high": [
                "I understand your concerns. Let me assure you we're here to help.",
                "I hear your worries. What information can I provide to ease your mind?",
                "Your concerns are important to us. How can we support you?"
            ],
            "moderate": [
                "I understand you have some concerns. What can we clarify for you?",
                "Thanks for sharing your worries. How can we help address them?",
                "I hear your apprehension. What would make you feel more comfortable?"
            ],
            "low": [
                "I see you have some concerns. What information would help?",
                "Thanks for letting us know. How can we assist you?",
                "I understand your hesitation. What would help reassure you?"
            ]
        },
        "surprise": {
            "high": [
                "Wow! What surprised you most about this experience?",
                "That's unexpected! I'd love to hear more about what caught you off guard.",
                "Interesting! What aspect surprised you the most?"
            ],
            "moderate": [
                "That caught you by surprise! What was unexpected?",
                "Interesting reaction! What surprised you about this?",
                "Unexpected! What stood out to you?"
            ],
            "low": [
                "Seems like this was a bit surprising. What caught your attention?",
                "Not what you expected? What surprised you?",
                "Interesting! What was unexpected for you?"
            ]
        },
        "neutral": {
            "high": [
                "Thanks for your feedback! How can we enhance your experience?",
                "Appreciate you taking the time to share. Any suggestions?",
                "Thank you for letting us know. What could we improve?"
            ],
            "moderate": [
                "Thanks for sharing your thoughts. How can we serve you better?",
                "I appreciate your feedback. What would make this more valuable?",
                "Thank you for your input. Any suggestions for improvement?"
            ],
            "low": [
                "Thanks for your feedback. How can we improve?",
                "Appreciate your thoughts. What would help?",
                "Thank you for sharing. Any suggestions?"
            ]
        }
    }

    # Get replies for the emotion, fallback to neutral if not found
    emotion_replies = replies.get(emotion, replies["neutral"])
    confidence_replies = emotion_replies.get(confidence_level, emotion_replies["moderate"])

    # Return a random reply from the appropriate category
    import random
    return random.choice(confidence_replies)


# -----------------------------
# Prediction function
def predict_emotion(text, threshold=0.3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    max_prob = probs.max()
    if max_prob < threshold:
        predicted_emotion = "neutral"
    else:
        predicted_emotion = EMOTIONS[probs.argmax()]
    
    return predicted_emotion, probs

# -----------------------------
@app.route("/")
def home():
    # If logged in, route to dashboard
    if session.get("user_id"):
        return redirect(url_for("user_dashboard"))
    return render_template("index.html")


@app.route("/about")
def about():
    # Public "About us" page (also accessible when logged in)
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        # Validation
        if not all([name, email, message]):
            return render_template("contact.html", error="All fields are required"), 400

        # Store contact message in MongoDB
        try:
            contact_doc = {
                "name": name,
                "email": email,
                "message": message,
                "created_at": datetime.utcnow().isoformat()
            }
            contact_collection.insert_one(contact_doc)

            # Send email notification
            try:
                send_contact_email(name, email, message)
            except Exception as e:
                # Log the error but don't fail the request
                print(f"Failed to send contact email: {e}")

            return render_template("contact.html", success="Message sent successfully! We'll get back to you soon.")
        except Exception as e:
            return render_template("contact.html", error=f"Error sending message: {str(e)}"), 500

    return render_template("contact.html")


@app.errorhandler(403)
def forbidden(_e):
    return render_template("forbidden.html"), 403


def login_required(role=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not session.get("user_id"):
                return redirect(url_for("login", next=request.path))
            if role and session.get("role") != role:
                abort(403)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@app.route("/legacy/text")
def legacy_text_tool():
    return render_template("legacy_text.html")

@app.route("/text")
def text_tool_alias():
    return redirect(url_for("legacy_text_tool"))


@app.route("/auth/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", info=request.args.get("info"), next=request.args.get("next"))

    raw_email = request.form.get("email") or ""
    password = request.form.get("password") or ""
    role = "user"
    next_url = request.form.get("next") or request.args.get("next")

    is_valid_email, email_or_error = validate_email(raw_email)
    if not is_valid_email:
        return render_template("login.html", error=email_or_error, next=next_url)
    email = email_or_error

    user = users_collection.find_one({"email": email, "role": role})

    if not user or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid credentials.", next=next_url)

    session["user_id"] = str(user["_id"])
    session["email"] = user["email"]
    session["role"] = user["role"]
    # Store first name for personalized greetings when available
    session["first_name"] = user.get("first_name", "").strip() or None

    if next_url:
        return redirect(next_url)
    return redirect(url_for("user_dashboard"))


@app.route("/auth/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    first_name_raw = request.form.get("first_name") or ""
    last_name_raw = request.form.get("last_name") or ""
    raw_email = request.form.get("email") or ""
    password = request.form.get("password") or ""
    confirm_password = request.form.get("confirm_password") or ""
    role = "user"

    valid_first, first_err = validate_name(first_name_raw, "First name")
    if not valid_first:
        return render_template("signup.html", error=first_err)

    valid_last, last_err = validate_name(last_name_raw, "Last name")
    if not valid_last:
        return render_template("signup.html", error=last_err)

    is_valid_email, email_or_error = validate_email(raw_email)
    if not is_valid_email:
        return render_template("signup.html", error=email_or_error)
    email = email_or_error

    valid_password, pwd_err = validate_password(password, confirm_password)
    if not valid_password:
        return render_template("signup.html", error=pwd_err)

    # Check if user already exists
    existing_user = users_collection.find_one({"email": email, "role": role})
    if existing_user:
        return render_template("signup.html", error="Account already exists for that email/role.")

    # Insert new user
    user_doc = {
        "email": email,
        "password_hash": generate_password_hash(password),
        "role": role,
        "first_name": first_name_raw.strip(),
        "last_name": last_name_raw.strip(),
        "created_at": datetime.utcnow().isoformat(),
    }
    result = users_collection.insert_one(user_doc)

    return redirect(url_for("login"))


@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/auth/forgot", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html", step="1")

    step = request.form.get("step") or "1"
    raw_email = request.form.get("email") or ""

    # Validate email first
    is_valid_email, email_or_error = validate_email(raw_email)
    if not is_valid_email:
        return render_template("forgot_password.html", error=email_or_error, step="1")
    email = email_or_error

    user = users_collection.find_one({"email": email})

    if not user:
        return render_template(
            "forgot_password.html",
            step="1",
            error="No account found for that email.",
        )

    if step == "1":
        # Generate and store OTP
        import random

        otp = f"{random.randint(100000, 999999)}"
        now_str = datetime.utcnow().isoformat()
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"reset_otp": otp, "reset_otp_created_at": now_str}}
        )

        # Send OTP by email
        try:
            send_otp_email(email, otp)
        except Exception as e:
            # Keep OTP visible in logs for debugging when SMTP isn't configured
            print(f"[DEBUG] Password reset OTP for {email}: {otp}")
            return render_template(
                "forgot_password.html",
                step="1",
                error=f"Could not send OTP email. Configure SMTP in your .env. ({e})",
            )

        return render_template(
            "forgot_password.html",
            step="2",
            email=email,
            info="We have sent a 6-digit OTP to your email address. Please enter it below.",
        )

    if step == "2":
        entered_otp = (request.form.get("otp") or "").strip()
        stored_otp = user.get("reset_otp", "")

        if not entered_otp:
            return render_template(
                "forgot_password.html",
                step="2",
                email=email,
                error="Please enter the OTP sent to your email.",
            )

        if not stored_otp or entered_otp != stored_otp:
            return render_template(
                "forgot_password.html",
                step="2",
                email=email,
                error="Invalid OTP. Please check and try again.",
            )

        return render_template(
            "forgot_password.html",
            step="3",
            email=email,
            info="OTP verified. Please choose a new password.",
        )

    if step == "3":
        new_password = request.form.get("new_password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        valid_password, pwd_err = validate_password(new_password, confirm_password)
        if not valid_password:
            return render_template(
                "forgot_password.html",
                step="3",
                email=email,
                error=pwd_err,
            )

        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"password_hash": generate_password_hash(new_password)}, "$unset": {"reset_otp": "", "reset_otp_created_at": ""}}
        )

        success_message = (
            "Password changed successfully. You can now log in with your new password."
        )
        return render_template(
            "forgot_password.html",
            step="3",
            email=email,
            success=success_message,
        )

    return render_template("forgot_password.html", step="1")


@app.route("/user")
@login_required(role="user")
def user_dashboard():
    first_name = session.get("first_name")
    if not first_name and session.get("email"):
        # Fallback to part before @ if first name is missing
        first_name = session.get("email").split("@")[0]
    return render_template("user_dashboard.html", first_name=first_name)

@app.route("/dashboard/analytics")
@login_required(role="user")
def analytics_dashboard():
    return render_template("analytics_dashboard.html")


def get_all_sentiment_words(reviews_list):
    """Convert review text into word frequency list for word cloud.

    Returns a simple list of [word, frequency] pairs.
    If filtering removes all words, returns unfiltered words instead.
    """
    if not reviews_list:
        return []

    stopwords = {
        "the", "and", "for", "with", "that", "this", "was", "from",
        "have", "has", "had", "are", "were", "but", "not", "you",
        "your", "their", "they", "them", "then", "when", "while",
        "what", "which", "can", "will", "would", "could", "should",
        "our", "out", "too", "very", "just", "about", "into", "its",
        "it's", "i", "me", "my", "we", "us", "he", "she",
        "is", "on", "in", "at", "of", "a", "an", "to",
        "if", "or", "as", "by", "so", "be", "do", "did", "does",
        "done", "been", "also", "more", "most", "over", "under", "per",
        "really", "still", "only", "even", "after", "before",
        "than", "such",
    }

    word_freq = {}

    def normalize_text(text):
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"[^\w\s']+", " ", text)
        return text.strip()

    # Count all words
    for review in reviews_list:
        normalized = normalize_text(review.get("text", ""))
        for token in normalized.split():
            token = token.strip("'\"`_")
            if len(token) >= 3 and not token.isnumeric():
                word_freq[token] = word_freq.get(token, 0) + 1

    # Try filtering
    filtered_freq = {w: f for w, f in word_freq.items() if w not in stopwords}

    # If filtering removes everything, use unfiltered words
    final_freq = filtered_freq if filtered_freq else word_freq

    if not final_freq:
        return []

    # Sort by frequency and return as list of [word, frequency] pairs
    sorted_words = sorted(final_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:100]


@app.route("/sklearn-analytics", methods=["GET", "POST"])
def sklearn_analytics():
    """Classic scikit-learn sentiment pipeline (merged from sentiment_analytics_project)."""
    used = int(session.get("anon_product_runs") or 0)
    remaining = max(0, ANON_PRODUCT_RUN_LIMIT - used) if not session.get("user_id") else None

    if request.method == "POST":
        review = request.form.get("review")
        product_url = request.form.get("product_url")

        if product_url:
            if not session.get("user_id"):
                if used >= ANON_PRODUCT_RUN_LIMIT:
                    return redirect(
                        url_for(
                            "login",
                            info=f"You’ve used your {ANON_PRODUCT_RUN_LIMIT} free analyses. Please log in to continue.",
                            next=url_for("sklearn_analytics"),
                        )
                    )
                session["anon_product_runs"] = used + 1
                used += 1
                remaining = max(0, ANON_PRODUCT_RUN_LIMIT - used)

            try:
                scraped_reviews = scrape_reviews(product_url)
            except Exception as e:
                return render_template(
                    "sklearn_analytics/index.html",
                    error=f"Failed to scrape URL: {e}",
                    free_remaining=remaining,
                )

            if not scraped_reviews:
                return render_template(
                    "sklearn_analytics/index.html",
                    error="No reviews found at this URL",
                    free_remaining=remaining,
                )

            min_rev = _product_review_min_for_analysis()
            if len(scraped_reviews) < min_rev:
                return render_template(
                    "sklearn_analytics/index.html",
                    error=(
                        f"Only {len(scraped_reviews)} review(s) found. "
                        f"Analysis requires at least {min_rev} reviews (set PRODUCT_REVIEW_MIN_FOR_ANALYSIS in .env)."
                    ),
                    free_remaining=remaining,
                )

            results = []
            total_score = 0.0
            sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
            score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
            star_sum = 0.0
            star_count = 0

            for item in scraped_reviews:
                text = item.get("text", "")
                stars = item.get("stars")
                
                # Ensure stars is a float if not None
                if stars is not None and not isinstance(stars, (int, float)):
                    try:
                        stars = float(stars)
                    except (ValueError, TypeError):
                        stars = None
                
                sentiment = str(predict_sentiment(text))
                if sentiment not in score_map:
                    sentiment = "Neutral"
                # Generate smart reply for this review
                smart_reply = generate_smart_reply(text, sentiment)
                results.append({"text": text, "sentiment": sentiment, "stars": stars, "smart_reply": smart_reply})
                total_score += score_map.get(sentiment, 0)
                sentiment_counts[sentiment] += 1

                if stars is not None:
                    star_sum += stars
                    star_count += 1

            avg_score = total_score / len(results)
            if avg_score > 0.3:
                overall = "Positive"
            elif avg_score < -0.3:
                overall = "Negative"
            else:
                overall = "Neutral"

            total_reviews = len(results)
            percentages = {
                "Positive": round((sentiment_counts["Positive"] / total_reviews) * 100, 1),
                "Neutral": round((sentiment_counts["Neutral"] / total_reviews) * 100, 1),
                "Negative": round((sentiment_counts["Negative"] / total_reviews) * 100, 1),
            }

            avg_star_rating = round(star_sum / star_count, 2) if star_count > 0 else None
            star_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for item in results:
                stars = item.get("stars")
                if isinstance(stars, (int, float)) and 1 <= stars <= 5:
                    key = int(round(stars))
                    star_distribution[key] += 1

            return render_template(
                "sklearn_analytics/result.html",
                results=results,
                overall=overall,
                avg_score=round(avg_score, 2),
                percentages=percentages,
                avg_star_rating=avg_star_rating,
                star_distribution=star_distribution,
                total_reviews=total_reviews,
            )

        if not review:
            return render_template("sklearn_analytics/index.html", error="Please enter a review or URL", free_remaining=remaining)

        prediction = predict_sentiment(review)
        return render_template("sklearn_analytics/index.html", prediction=str(prediction), free_remaining=remaining)

    return render_template("sklearn_analytics/index.html", free_remaining=remaining)


@app.route("/sklearn-analytics/rating")
def sklearn_rating():
    avg_star_rating = request.args.get("avg_star_rating")
    positive = request.args.get("positive")
    neutral = request.args.get("neutral")
    negative = request.args.get("negative")
    total_reviews = request.args.get("total_reviews")
    star_1 = int(request.args.get("star_1", 0))
    star_2 = int(request.args.get("star_2", 0))
    star_3 = int(request.args.get("star_3", 0))
    star_4 = int(request.args.get("star_4", 0))
    star_5 = int(request.args.get("star_5", 0))

    return render_template(
        "sklearn_analytics/rating.html",
        avg_star_rating=avg_star_rating,
        positive=positive,
        neutral=neutral,
        negative=negative,
        total_reviews=total_reviews,
        star_1=star_1,
        star_2=star_2,
        star_3=star_3,
        star_4=star_4,
        star_5=star_5,
    )


@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    """
    Classic ML PDF download (ReportLab).
    The Classic ML results page posts a pre-formatted title/summary/keywords.
    """
    title = (request.form.get("title") or "SentimentPulse - Classic ML Result").strip()
    summary = request.form.get("summary") or ""
    keywords = request.form.get("keywords") or ""

    # Keep ReportLab markup safe/stable (allow <br/> only).
    def _safe_para(s: str) -> str:
        s = "" if s is None else str(s)
        # Normalize newlines -> <br/>
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = s.replace("\n", "<br/>")
        # Escape everything, then unescape our <br/> tags.
        esc = html.escape(s, quote=False)
        return esc.replace("&lt;br/&gt;", "<br/>")

    buffer = io.BytesIO()

    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
    except Exception:
        return (
            jsonify(
                {
                    "error": (
                        "PDF generation dependency is missing. "
                        "Run: pip install -r requirements.txt"
                    )
                }
            ),
            500,
        )

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )

    styles = getSampleStyleSheet()
    # Slightly larger body for clarity
    body = ParagraphStyle(
        "BodyClear",
        parent=styles["BodyText"],
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )

    content = []
    content.append(Paragraph(f"<b>{html.escape(title)}</b>", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("<b>Summary:</b>", styles["Heading2"]))
    content.append(Paragraph(_safe_para(summary) or "-", body))
    content.append(Spacer(1, 10))

    content.append(Paragraph("<b>Keywords:</b>", styles["Heading2"]))
    content.append(Paragraph(_safe_para(keywords) or "-", body))

    doc.build(content)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="classic_ml_result.pdf",
        mimetype="application/pdf",
    )


@app.route("/api/generate-smart-reply", methods=["POST"])
def api_generate_smart_reply():
    """Generate a smart reply based on review text and sentiment."""
    data = request.get_json() or {}
    review_text = data.get("review_text", "").strip()
    sentiment = data.get("sentiment", "Neutral").strip()
    
    if not review_text:
        return jsonify({"error": "Review text is required"}), 400
    
    if sentiment not in ["Positive", "Negative", "Neutral"]:
        sentiment = "Neutral"
    
    try:
        smart_reply = generate_smart_reply(review_text, sentiment)
        return jsonify({"success": True, "reply": smart_reply})
    except Exception as e:
        app.logger.error(f"Error generating smart reply: {e}")
        return jsonify({"error": str(e)}), 500


def _url_path_looks_like_direct_product(path: str, host: str) -> bool:
    """True when the path is clearly a product page (avoids false positives from query tracking)."""
    p = (path or "").lower()
    hl = (host or "").lower()
    if "flipkart." in hl:
        return bool(re.search(r"/p/itm|/product-reviews/itm", p, re.I))
    if "amazon." in hl:
        return "/dp/" in p or "/gp/product/" in p or "/gp/aw/d/" in p
    return False


def _query_suggests_search_results(parsed) -> bool:
    """
    Detect search/browse URLs using query keys and path — not raw substring 'search',
    which false-matches Flipkart iid values like '....SEARCH' in tracking params.
    """
    path = (parsed.path or "").lower()
    if path.startswith("/search"):
        return True
    qs = parse_qs(parsed.query)
    for key in qs.keys():
        kl = key.lower()
        if kl in ("q", "query", "search", "keywords", "keyword"):
            return True
    return False


def _search_url_error_message(host: str) -> str:
    hl = (host or "").lower()
    if "flipkart." in hl:
        return (
            "Please provide a direct Flipkart product URL (path contains /p/itm… or /product-reviews/itm…), "
            "not a search or category listing page."
        )
    if "amazon." in hl:
        return (
            "Please provide a direct product page URL, not a search results page. "
            "Example: https://www.amazon.in/dp/B0XXXXXXX or https://www.amazon.in/gp/product/B0XXXXXXX"
        )
    return "Please provide a direct product page URL, not a search or category listing page."


def get_all_sentiment_words(reviews_list):
    """Convert review text + sentiment labels into word cloud payload.

    Returns either a list of word objects with text/size/color or a message dict
    when no reviews are available.
    """
    print(f"[DEBUG] get_all_sentiment_words called with {len(reviews_list) if reviews_list else 0} reviews")
    if not reviews_list:
        print("[DEBUG] No reviews - returning message")
        return {"message": "No Data Available"}

    # Minimal stopword filter to keep the cloud focused on meaningful terms.
    stopwords = {
        "the", "and", "for", "with", "that", "this", "was", "from",
        "have", "has", "had", "are", "were", "but", "not", "you",
        "your", "their", "they", "them", "then", "when", "while",
        "what", "which", "can", "will", "would", "could", "should",
        "our", "out", "too", "very", "just", "about", "into", "its",
        "it's", "it's", "it's", "i", "me", "my", "we", "us", "he",
        "she", "it", "is", "on", "in", "at", "of", "a", "an", "to",
        "if", "or", "as", "by", "so", "be", "do", "did", "does",
        "done", "been", "also", "more", "most", "over", "under", "per",
        "really", "still", "only", "even", "after", "before", "then",
        "than", "such",
    }

    sentiment_map = {"Positive": "green", "Negative": "red", "Neutral": "grey"}
    word_stats = {}

    def normalize_text(text):
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"[^\w\s']+", " ", text)
        return text.strip()

    for review in reviews_list:
        sentiment = review.get("sentiment", "Neutral")
        if sentiment not in sentiment_map:
            sentiment = "Neutral"
        normalized = normalize_text(review.get("text", ""))
        for token in normalized.split():
            token = token.strip("'\"`_")
            if len(token) < 3 or token in stopwords or token.isnumeric():
                continue
            stats = word_stats.setdefault(token, {"count": 0, "sentiments": {"Positive": 0, "Neutral": 0, "Negative": 0}})
            stats["count"] += 1
            stats["sentiments"][sentiment] += 1

    if not word_stats:
        print("[DEBUG] No word stats - returning message")
        return {"message": "No Data Available"}

    cloud = []
    for token, stats in word_stats.items():
        sentiment_counts = stats["sentiments"]
        net_score = sentiment_counts["Positive"] - sentiment_counts["Negative"]
        if net_score > 0:
            dominant = "Positive"
        elif net_score < 0:
            dominant = "Negative"
        else:
            dominant = "Neutral"

        size = min(max(stats["count"] * 4 + 12, 14), 60)
        cloud.append(
            {
                "text": token,
                "size": size,
                "color": sentiment_map[dominant],
                "sentiment": dominant,
            }
        )

    cloud.sort(key=lambda item: item["size"], reverse=True)
    result = cloud[:80]
    print(f"[DEBUG] Returning {len(result)} words for word cloud")
    return result


def _product_review_min_for_analysis() -> int:
    """Minimum reviews required before showing aggregate sentiment (default 12, range ~10–15 typical)."""
    raw = (os.environ.get("PRODUCT_REVIEW_MIN_FOR_ANALYSIS") or "12").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 12
    return max(1, min(n, 500))


def _run_product_sentiment(product_url: str, user_id):
    """
    Returns dict suitable for rendering product_result.html.
    """
    # Validate URL is a product page, not search results
    parsed = urlparse(product_url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    host = (parsed.netloc or "").lower()

    # Reject Amazon search URLs (/s?...)
    if "amazon." in host and (path.startswith("/s") or "s?" in query):
        raise RuntimeError(
            "Please provide a direct product page URL, not a search results page. "
            "Example: https://www.amazon.in/dp/B0XXXXXXX or https://www.amazon.in/gp/product/B0XXXXXXX"
        )

    # Reject search/browse URLs only when the path is not already a known product URL.
    # (Do not scan the raw query for the substring "search" — Flipkart iid often ends with ".SEARCH".)
    if not _url_path_looks_like_direct_product(path, host) and _query_suggests_search_results(parsed):
        raise RuntimeError(_search_url_error_message(host))

    if "flipkart." in host:
        if not re.search(r"/p/itm|/product-reviews/itm", path, re.I):
            raise RuntimeError(
                "Flipkart: use a product URL that contains /p/itm… (or a /product-reviews/itm… page). "
                "Search and category listing URLs (e.g. /search, /pr?sid=…) do not expose reviews to this tool."
            )

    reviews = []

    if "amazon." in host:
        _logger.info("📍 Detected hostname: %s", host)
        _logger.info("📦 Using Amazon reviews via Rainforest API")
        try:
            reviews = fetch_amazon_reviews_via_rainforest(product_url, max_pages=1)
        except Exception as e:
            # Network/lookup issues or API key issues: try fallback scraping then raise if no data
            reviews = scrape_reviews_from_jsonld(product_url)
            if not reviews:
                # Amazon typically blocks, so this is best-effort fallback.
                raise RuntimeError(
                    "Amazon review fetch failed via Rainforest and JSON-LD fallback.\n\n"
                    f"{e}\n\n"
                    "Tip: restart the Flask app after editing .env, or save .env (we reload it each request). "
                    "Remove any Windows User/System RAINFOREST_API_KEY if it is still the placeholder."
                ) from e

    elif "flipkart." in host:
        product_url = normalize_flipkart_product_url(product_url)
        _logger.info("Normalized Flipkart URL: %s", product_url[:160] + ("…" if len(product_url) > 160 else ""))
        try:
            reviews = fetch_flipkart_reviews_pipeline(product_url)
        except Exception as ex:
            # Never surface raw requests.HTTPError (403) to the UI — pipeline steps should catch these.
            _logger.warning("Flipkart fetch pipeline raised: %s", ex)
            reviews = []
    else:
        _logger.info("📍 Detected hostname: %s — trying JSON-LD review extraction", host or "(unknown)")
        try:
            reviews = scrape_reviews_from_jsonld(product_url)
        except Exception as ex:
            _logger.warning("JSON-LD scrape failed for %s: %s", product_url[:80], ex)
            reviews = []

    if not reviews:
        if "flipkart." in host:
            if os.environ.get("FLIPKART_USE_SELENIUM", "").strip().lower() in ("1", "true", "yes"):
                raise RuntimeError(
                    "Flipkart returned no extractable reviews even with Chrome (Selenium, usually headless). "
                    "Try FLIPKART_SELENIUM_PAGE_WAIT=18 in .env, or FLIPKART_SELENIUM_HEADLESS=0 temporarily "
                    "to open a visible browser if Flipkart shows a captcha. "
                    "SCRAPER_PROXY (residential) or a shorter …/p/itm…?pid=… URL can also help."
                )
            raise RuntimeError(
                "Flipkart returned no extractable reviews. Flipkart often blocks plain HTTP clients (403) and "
                "usually loads reviews via JavaScript/APIs, so headless automation may see an empty page. "
                "Try: (1) SCRAPER_PROXY (residential); (2) FLIPKART_USE_SELENIUM=1 with "
                "FLIPKART_SELENIUM_HEADLESS=1 (background Chrome, default); "
                "if you get a captcha, set FLIPKART_SELENIUM_HEADLESS=0 once to complete it in a visible window; "
                "(3) short product URL: …/p/itm…?pid=…"
            )
        raise RuntimeError(
            "No reviews were found for this product URL. "
            "For Amazon, set RAINFOREST_API_KEY and use a direct /dp/ product link."
        )

    min_reviews = _product_review_min_for_analysis()
    if len(reviews) < min_reviews:
        n = len(reviews)
        parts = [
            f"Only {n} review(s) were collected.",
            f"Sentiment analysis is shown only after at least {min_reviews} reviews are collected.",
        ]
        if "flipkart." in host:
            parts.append(
                "Try increasing FLIPKART_SELENIUM_PAGE_WAIT or FLIPKART_REVIEW_HARVEST_PASSES, "
                "or FLIPKART_SELENIUM_HEADLESS=0 if headless sees fewer reviews. "
                "Install undetected-chromedriver (`pip install undetected-chromedriver`) so FLIPKART_USE_UNDETECTED_CHROME can run. "
                "If Flipkart only shows a few reviews for this product, set PRODUCT_REVIEW_MIN_FOR_ANALYSIS lower in .env (e.g. 3)."
            )
        elif "amazon." in host:
            parts.append(
                "For Amazon, pick a product with more visible reviews, or verify RAINFOREST_API_KEY and your link."
            )
        else:
            parts.append("Try a product page that lists more reviews, or another store URL.")
        raise RuntimeError(" ".join(parts))

    reviews_fetched = len(reviews)
    analysis_limit = int(os.environ.get("PRODUCT_REVIEW_ANALYSIS_LIMIT", "80"))
    analysis_limit = max(1, min(analysis_limit, 500))
    store_limit = int(os.environ.get("PRODUCT_REVIEW_STORE_LIMIT", "200"))
    store_limit = max(1, min(store_limit, 500))

    # Store product reviews in MongoDB
    review_docs = []
    for r in reviews[:store_limit]:
        review_doc = {
            "product_url": product_url,
            "review_text": r["text"],
            "rating": r.get("rating"),
            "source": r.get("source", "jsonld"),
            "scraped_at": datetime.utcnow().isoformat()
        }
        review_docs.append(review_doc)
    if review_docs:
        product_reviews_collection.insert_many(review_docs)

    def _coerce_review_star(val):
        """Normalize rating/stars from scrapers (may be str, 1–5 or 1–10 scale)."""
        if val is None:
            return None
        try:
            if isinstance(val, str):
                x = float(val.strip())
            else:
                x = float(val)
        except (TypeError, ValueError):
            return None
        if 0 < x <= 5:
            return x
        if 5 < x <= 10:
            return x / 2.0
        return None

    scored = []
    pos = neg = neu = 0
    ratings = []
    rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in reviews[:analysis_limit]:
        sentiment, conf = classify_sentiment(r["text"])
        if sentiment == "POSITIVE":
            pos += 1
        elif sentiment == "NEGATIVE":
            neg += 1
        else:
            neu += 1

        raw_star = r.get("rating")
        if raw_star is None:
            raw_star = r.get("stars")
        rt = _coerce_review_star(raw_star)
        if rt is not None:
            rounded = int(round(rt))
            if 1 <= rounded <= 5:
                rating_dist[rounded] += 1
                ratings.append(rt)

        scored.append(
            {
                **r,
                "rating": rt,
                "sentiment": sentiment,
                "confidence": round(conf * 100, 2),
            }
        )

    total = len(scored)
    positive_pct = (pos / total) * 100 if total else 0
    negative_pct = (neg / total) * 100 if total else 0
    neutral_pct = (neu / total) * 100 if total else 0

    page_metadata = scrape_product_metadata_from_jsonld(product_url)
    page_rating = page_metadata.get("ratingValue")
    avg_rating = page_rating if page_rating is not None else ((sum(ratings) / len(ratings)) if ratings else None)

    # Store sentiment run in MongoDB
    sentiment_run_doc = {
        "user_id": user_id,
        "product_url": product_url,
        "total_reviews": total,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "neutral_pct": neutral_pct,
        "created_at": datetime.utcnow().isoformat()
    }
    sentiment_runs_collection.insert_one(sentiment_run_doc)

    return {
        "product_url": product_url,
        "total": total,
        "reviews_fetched": reviews_fetched,
        "analysis_limit": analysis_limit,
        "positive_pct": round(positive_pct, 1),
        "negative_pct": round(negative_pct, 1),
        "neutral_pct": round(neutral_pct, 1),
        "reviews": scored,
        "avg_rating": (round(avg_rating, 2) if avg_rating is not None else None),
        "rating_dist": rating_dist,
    }


def scrape_reviews_from_jsonld(product_url: str, timeout_s: int = 15):
    """
    Scrape reviews from many e-commerce product pages that publish JSON-LD.
    Returns list of dicts: {text, rating, source}
    Never raises — network/HTTP errors (e.g. Flipkart 403) return [] so callers can fall back.
    """
    try:
        resp = _http_get(product_url, timeout_s=timeout_s)

        soup = BeautifulSoup(resp.text, "html.parser")
        scripts = soup.find_all("script", attrs={"type": "application/ld+json"})

        reviews = []
        for s in scripts:
            raw = (s.string or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            # JSON-LD can be object or list
            nodes = data if isinstance(data, list) else [data]
            for node in nodes:
                if not isinstance(node, dict):
                    continue

                # Common patterns: Product -> review[], or graph
                candidates = []
                if "review" in node:
                    candidates.append(node)
                if "@graph" in node and isinstance(node["@graph"], list):
                    candidates.extend([x for x in node["@graph"] if isinstance(x, dict)])

                for cand in candidates:
                    revs = cand.get("review")
                    if not revs:
                        continue
                    if isinstance(revs, dict):
                        revs = [revs]
                    if not isinstance(revs, list):
                        continue

                    for r in revs:
                        if not isinstance(r, dict):
                            continue
                        text = r.get("reviewBody") or r.get("description") or ""
                        text = re.sub(r"\s+", " ", str(text)).strip()
                        if not text:
                            continue

                        rating = None
                        rr = r.get("reviewRating")
                        if isinstance(rr, dict):
                            rating_val = rr.get("ratingValue")
                            try:
                                rating = float(rating_val)
                            except Exception:
                                rating = None

                        reviews.append({"text": text, "rating": rating, "source": "jsonld"})

        return reviews
    except Exception as ex:
        _logger.warning("scrape_reviews_from_jsonld failed for %s: %s", (product_url or "")[:100], ex)
        return []


def scrape_product_metadata_from_jsonld(product_url: str, timeout_s: int = 15) -> dict:
    """
    Extract aggregate product metadata from JSON-LD on the product page.
    Returns dict with optional keys: ratingValue, ratingCount, reviewCount.
    """
    try:
        resp = _http_get(product_url, timeout_s=timeout_s)
        soup = BeautifulSoup(resp.text, "html.parser")
        scripts = soup.find_all("script", attrs={"type": "application/ld+json"})

        for s in scripts:
            raw = (s.string or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            nodes = data if isinstance(data, list) else [data]
            for node in nodes:
                if not isinstance(node, dict):
                    continue

                candidates = [node]
                if "@graph" in node and isinstance(node["@graph"], list):
                    candidates.extend([x for x in node["@graph"] if isinstance(x, dict)])

                for cand in candidates:
                    if not isinstance(cand, dict):
                        continue
                    agg = cand.get("aggregateRating")
                    if not isinstance(agg, dict):
                        continue

                    result: dict = {}
                    rating_value = agg.get("ratingValue")
                    if rating_value is not None:
                        try:
                            result["ratingValue"] = float(rating_value)
                        except Exception:
                            pass

                    for name in ("ratingCount", "reviewCount"):
                        count = agg.get(name)
                        if count is None or result.get("ratingCount") is not None:
                            continue
                        try:
                            result["ratingCount"] = int(float(count))
                        except Exception:
                            pass

                    if result.get("ratingValue") is not None:
                        return result

        return {}
    except Exception as ex:
        _logger.warning("scrape_product_metadata_from_jsonld failed for %s: %s", (product_url or "")[:100], ex)
        return {}


def _make_http_session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        }
    )
    return s


def _proxy_config():
    proxy = (os.environ.get("SCRAPER_PROXY") or "").strip()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _http_get(url: str, timeout_s: int = 20, referer: str | None = None, prime_cookie_url: str | None = None):
    """
    Best-effort GET that behaves more like a browser:
    - uses a session (cookies)
    - optional cookie priming (hit homepage first)
    - optional proxy
    - basic retries on 403/429/5xx
    """
    proxies = _proxy_config()
    s = _make_http_session()
    if prime_cookie_url:
        try:
            s.get(prime_cookie_url, timeout=timeout_s, proxies=proxies)
        except Exception:
            pass

    headers = {}
    if referer:
        headers["Referer"] = referer

    last_exc = None
    for attempt in range(3):
        try:
            resp = s.get(url, timeout=timeout_s, headers=headers, proxies=proxies)
            # Some sites block bots with 403 or rate-limit with 429
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                time.sleep(0.6 * (attempt + 1))
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            time.sleep(0.4 * (attempt + 1))

    raise last_exc


def extract_amazon_asin(url: str):
    """
    Tries to extract ASIN from common Amazon URL patterns.
    Returns ASIN string or None.
    """
    url = (url or "").strip()
    if not url:
        return None

    # Common Amazon product URL variants
    patterns = [
        r"/dp/([A-Za-z0-9]{10})",
        r"/gp/product/([A-Za-z0-9]{10})",
        r"/gp/aw/d/([A-Za-z0-9]{10})",
        r"/product/([A-Za-z0-9]{10})",
        r"/([A-Za-z0-9]{10})(?:[/?#]|$)",
    ]

    for pattern in patterns:
        m = re.search(pattern, url, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Sometimes ASIN is in query params: ?asin=...
    q = parse_qs(urlparse(url).query)
    asin = (q.get("asin") or [None])[0]
    if asin and re.fullmatch(r"[A-Za-z0-9]{10}", asin):
        return asin.upper()

    return None


def _extract_asin_from_page(product_url: str):
    """Fallback: fetch product page or search page and extract ASIN from HTML content."""
    try:
        resp = _http_get(product_url, timeout_s=20)
    except Exception:
        return None

    text = resp.text or ""

    # Amazon usually includes data-asin or asin variable in page
    m = re.search(r"[\"']asin[\"']\s*[:=]\s*[\"']([A-Za-z0-9]{10})[\"']", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"data-asin\s*=\s*[\"']([A-Za-z0-9]{10})[\"']", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"/dp/([A-Za-z0-9]{10})", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # If it's a search results page, try scraping the first product item
    try:
        soup = BeautifulSoup(text, "html.parser")

        # 1) Product links in search results
        for a in soup.select("a[href*='/dp/']"):
            href = a.get("href", "")
            m = re.search(r"/dp/([A-Za-z0-9]{10})", href, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        # 2) First element with data-asin attribute
        for el in soup.select("[data-asin]"):
            asin_value = (el.get("data-asin") or "").strip()
            if re.fullmatch(r"[A-Za-z0-9]{10}", asin_value):
                return asin_value.upper()

    except Exception:
        pass

    return None


def _mask_rainforest_key(key: str | None) -> str:
    if not key or not str(key).strip():
        return "(empty)"
    k = str(key).strip()
    if len(k) <= 8:
        return f"*** ({len(k)} chars)"
    return f"{k[:4]}…{k[-4:]} ({len(k)} chars)"


def _rainforest_key_problem(key: str | None) -> str | None:
    if not key or not str(key).strip():
        return "RAINFOREST_API_KEY is empty — set it in .env next to app.py and restart the server."
    k = str(key).strip().lower()
    if "your_rainforest" in k or k == "changeme":
        return (
            "RAINFOREST_API_KEY is still a placeholder. Replace it with your real key from Rainforest "
            "and restart. If it is set in Windows Environment Variables, remove the old value or "
            "use load_dotenv(..., override=True) (already in this app)."
        )
    return None


def _rainforest_response_debug(r: requests.Response) -> str:
    """Safe snippet for logs/UI (no secrets — response body should not echo api_key)."""
    text = (r.text or "")[:1500]
    try:
        data = r.json()
        if isinstance(data, dict):
            parts = []
            for key in ("error", "message", "detail", "status", "request_id"):
                if key in data and data[key] is not None:
                    parts.append(f"{key}={data[key]!r}")
            if parts:
                return " ".join(parts) + (" | " + text if text else "")
    except Exception:
        pass
    return text or "(empty body)"


def fetch_amazon_reviews_via_rainforest(product_url: str, max_pages: int = 1):
    """
    Reliable option for Amazon without direct scraping: Rainforest API (requires key).
    Env var: RAINFOREST_API_KEY
    Returns list of {text, rating, source}
    """
    # Reload .env each call. For Rainforest, read key/domain from the file first — Windows User/System
    # env often still has RAINFOREST_API_KEY=your_rainforest_api_key_here and would win over load_dotenv
    # unless we take values from dotenv_values (file-only) before falling back to os.environ.
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path, override=True)
    _file = dotenv_values(_env_path) or {}

    def _from_file(name: str) -> str:
        v = _file.get(name)
        return (v or "").strip() if v is not None else ""

    api_key = _from_file("RAINFOREST_API_KEY") or os.environ.get("RAINFOREST_API_KEY")
    if not api_key:
        raise RuntimeError("Amazon reviews require RAINFOREST_API_KEY (Rainforest API) to be set.")

    key_hint = _rainforest_key_problem(api_key)
    if key_hint:
        raise RuntimeError(f"{key_hint} (masked key: {_mask_rainforest_key(api_key)})")

    asin = extract_amazon_asin(product_url)
    if not asin:
        asin = _extract_asin_from_page(product_url)

    if not asin:
        raise RuntimeError(
            "Could not extract ASIN from the Amazon URL. "
            "Please provide a valid Amazon product page URL (contains /dp/ASIN or /gp/product/ASIN)."
        )

    host = _from_file("RAINFOREST_API_HOST") or os.environ.get("RAINFOREST_API_HOST", "api.rainforestapi.com")
    amazon_domain = _from_file("AMAZON_DOMAIN") or os.environ.get("AMAZON_DOMAIN") or "amazon.in"

    reviews = []
    for page in range(1, max_pages + 1):
        params = {
            "api_key": api_key,
            "type": "reviews",
            "amazon_domain": amazon_domain,
            "asin": asin,
            "page": page,
        }
        url = f"https://{host}/request"
        try:
            r = requests.get(url, params=params, timeout=25)
        except requests.exceptions.RequestException as e:
            _logger.exception("Rainforest request failed (network): asin=%s domain=%s", asin, amazon_domain)
            raise RuntimeError(
                f"Rainforest network error: {e!r}. "
                f"asin={asin} amazon_domain={amazon_domain} key={_mask_rainforest_key(api_key)}"
            ) from e

        _logger.info(
            "Rainforest response: status=%s asin=%s domain=%s page=%s key=%s",
            r.status_code,
            asin,
            amazon_domain,
            page,
            _mask_rainforest_key(api_key),
        )

        if r.status_code != 200:
            dbg = _rainforest_response_debug(r)
            _logger.warning(
                "Rainforest non-200: status=%s asin=%s body_snippet=%s",
                r.status_code,
                asin,
                dbg[:500],
            )
            raise RuntimeError(
                f"Rainforest HTTP {r.status_code} (Service Unavailable or error). "
                f"asin={asin} amazon_domain={amazon_domain} key={_mask_rainforest_key(api_key)}. "
                f"Details: {dbg}"
            )

        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(
                f"Rainforest returned non-JSON (status {r.status_code}): {(r.text or '')[:800]!r}"
            ) from e

        if isinstance(data, dict) and data.get("error"):
            err = data.get("error")
            _logger.warning("Rainforest API error field: %s", err)
            raise RuntimeError(
                f"Rainforest API error: {err!r}. asin={asin} domain={amazon_domain} key={_mask_rainforest_key(api_key)}"
            )

        # Expected shape: { "reviews": [ { "body": "...", "rating": 5, ... } ] }
        for item in (data.get("reviews") or []):
            body = (item.get("body") or item.get("review") or item.get("text") or "").strip()
            if not body:
                continue
            rating = item.get("rating")
            try:
                rating = float(rating) if rating is not None else None
            except Exception:
                rating = None
            reviews.append({"text": body, "rating": rating, "source": "amazon_rainforest"})

    return reviews


def normalize_flipkart_product_url(url: str) -> str:
    """Canonical host, strip tracking query params, keep product path."""
    url = (url or "").strip()
    if not url.startswith("http"):
        return url
    p = urlparse(url)
    host = (p.netloc or "").lower()
    if host == "flipkart.com":
        host = "www.flipkart.com"
    qs = parse_qs(p.query)
    drop_keys = {k for k in qs if k.lower().startswith("utm") or k.lower() in ("ref", "otracker", "otracker1")}
    for k in drop_keys:
        qs.pop(k, None)
    query = urlencode(qs, doseq=True) if qs else ""
    return urlunparse((p.scheme or "https", host, p.path or "", "", query, ""))


def flipkart_product_reviews_url(product_url: str) -> str | None:
    """Map .../slug/p/itmXXX -> .../slug/product-reviews/itmXXX (more review HTML/JSON)."""
    p = urlparse(product_url)
    path = p.path or ""
    if re.search(r"/product-reviews/itm", path, re.I):
        return None
    m = re.search(r"/p/(itm[a-z0-9]+)", path, re.I)
    if not m:
        return None
    new_path = re.sub(r"/p/(itm[a-z0-9]+)", r"/product-reviews/\1", path, count=1, flags=re.I)
    if new_path == path:
        return None
    return urlunparse((p.scheme or "https", (p.netloc or "").lower(), new_path, "", p.query, ""))


def _dedupe_flipkart_reviews(items: list) -> list:
    seen: set[str] = set()
    out = []
    for r in items:
        t = (r.get("text") or "").strip()
        if len(t) < 12:
            continue
        key = t[:160].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _flipkart_rating_from_obj(obj: dict) -> float | None:
    for key in ("rating", "ratingValue", "value", "stars"):
        v = obj.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    rr = obj.get("reviewRating")
    if isinstance(rr, dict):
        v = rr.get("ratingValue")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _flipkart_collect_review_dicts(obj, out: list, depth: int = 0) -> None:
    """Collect Flipkart-style review objects from nested JSON (__NEXT_DATA__, etc.)."""
    if depth > 24:
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
            rating = _flipkart_rating_from_obj(obj)
            out.append(
                {
                    "text": re.sub(r"\s+", " ", rt).strip(),
                    "rating": rating,
                    "source": "flipkart_embedded",
                }
            )
            return
        for v in obj.values():
            _flipkart_collect_review_dicts(v, out, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _flipkart_collect_review_dicts(item, out, depth + 1)


def scrape_flipkart_embedded_json_from_html(html: str) -> list:
    """Parse __NEXT_DATA__ / JSON blobs where Flipkart hides review text (CSR)."""
    soup = BeautifulSoup(html, "html.parser")
    found: list = []
    for script in soup.find_all("script", id="__NEXT_DATA__"):
        raw = (script.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            _flipkart_collect_review_dicts(data, found)
        except Exception:
            continue
    for script in soup.find_all("script", attrs={"type": "application/json"}):
        raw = (script.string or "").strip()
        if not raw or "reviewText" not in raw:
            continue
        try:
            data = json.loads(raw)
            _flipkart_collect_review_dicts(data, found)
        except Exception:
            continue
    return _dedupe_flipkart_reviews(found)


def scrape_flipkart_selectors_from_soup(soup: BeautifulSoup) -> list:
    """DOM selectors for review bodies (markup changes over time — keep several)."""
    reviews = []
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
            txt = re.sub(r"\s+", " ", div.get_text(" ", strip=True)).strip()
            if len(txt) >= 12:
                reviews.append({"text": txt, "rating": None, "source": "flipkart_html"})
    return _dedupe_flipkart_reviews(reviews)


def scrape_flipkart_reviews_html(product_url: str, timeout_s: int = 20) -> list:
    """
    Fetch Flipkart HTML and extract reviews from embedded JSON (preferred) and DOM.
    Never raises — callers rely on empty list when Flipkart returns 403.
    """
    try:
        resp = _http_get(
            product_url,
            timeout_s=timeout_s,
            referer="https://www.flipkart.com/",
            prime_cookie_url="https://www.flipkart.com/",
        )
        html = resp.text or ""
        embedded = scrape_flipkart_embedded_json_from_html(html)
        if embedded:
            return embedded

        soup = BeautifulSoup(html, "html.parser")
        return scrape_flipkart_selectors_from_soup(soup)
    except Exception as ex:
        _logger.warning("scrape_flipkart_reviews_html: %s", ex)
        return []


def _flipkart_reviews_via_selenium(url: str) -> list:
    """Optional real browser (set FLIPKART_USE_SELENIUM=1). Chrome must be installed."""
    try:
        from scraper.review_scraper import _scrape_flipkart_selenium

        raw = _scrape_flipkart_selenium(url)
        return [
            {"text": r["text"], "rating": r.get("stars"), "source": "flipkart_selenium"}
            for r in raw
            if len((r.get("text") or "").strip()) >= 12
        ]
    except Exception:
        return []


def _flipkart_selenium_reviews(product_url: str) -> list:
    """
    When FLIPKART_USE_SELENIUM=1: real Chrome session (often succeeds when requests get 403).
    Tries product URL, then /product-reviews/itm… if needed.
    """
    if os.environ.get("FLIPKART_USE_SELENIUM", "").strip().lower() not in ("1", "true", "yes"):
        return []
    _logger.info("🛒 Using Flipkart Selenium scraper (try before plain HTTP)")
    reviews = _flipkart_reviews_via_selenium(product_url)
    if reviews:
        _logger.info("✓ Flipkart: %s reviews from Selenium (product URL)", len(reviews))
        return reviews
    rev_url = flipkart_product_reviews_url(product_url)
    if rev_url:
        _logger.info("🛒 Flipkart Selenium: retrying reviews tab URL")
        reviews = _flipkart_reviews_via_selenium(rev_url)
        if reviews:
            _logger.info("✓ Flipkart: %s reviews from Selenium (reviews tab)", len(reviews))
    return reviews or []


def fetch_flipkart_reviews_pipeline(product_url: str) -> list:
    """
    If FLIPKART_USE_SELENIUM=1: try Chrome first (avoids wasted 403s from requests).
    Else: JSON-LD → product HTML → reviews-tab URL.
    Selenium can be retried implicitly via the same first step when enabled.
    """
    parsed = urlparse(product_url)
    host = (parsed.netloc or "").lower()
    _logger.info("📍 Detected hostname: %s", host or "(unknown)")
    _logger.info(
        "Flipkart pipeline: Selenium first (if enabled) → JSON-LD → product HTML → reviews-tab HTTP"
    )

    reviews: list = []

    def _try_jsonld(url: str) -> list:
        try:
            return scrape_reviews_from_jsonld(url)
        except Exception as e:
            _logger.warning("Flipkart JSON-LD fetch failed (%s): %s", url[:80], e)
            return []

    def _try_html(url: str) -> list:
        try:
            return scrape_flipkart_reviews_html(url)
        except Exception as e:
            _logger.warning("Flipkart HTML fetch failed (%s): %s", url[:80], e)
            return []

    merge_on = (os.environ.get("FLIPKART_MERGE_HTTP_SOURCES") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

    reviews = _flipkart_selenium_reviews(product_url)
    merged = _dedupe_flipkart_reviews(list(reviews or []))

    if merge_on:
        _logger.info(
            "Flipkart: merging JSON-LD / HTTP with Selenium (selenium=%s unique)",
            len(merged),
        )
        merged = _dedupe_flipkart_reviews(merged + _try_jsonld(product_url))
        merged = _dedupe_flipkart_reviews(merged + _try_html(product_url))
        rev_url_m = flipkart_product_reviews_url(product_url)
        if rev_url_m:
            merged = _dedupe_flipkart_reviews(merged + _try_jsonld(rev_url_m))
            merged = _dedupe_flipkart_reviews(merged + _try_html(rev_url_m))
        if merged:
            _logger.info("✓ Flipkart merged pipeline: %s unique reviews", len(merged))
            return merged
        _logger.warning("Flipkart: no reviews after Selenium + HTTP merge (set FLIPKART_MERGE_HTTP_SOURCES=0 to retry legacy path)")
        return []

    if merged:
        return merged

    if os.environ.get("FLIPKART_USE_SELENIUM", "").strip().lower() in ("1", "true", "yes"):
        _logger.info("Selenium returned no reviews; trying HTTP fetches…")
    else:
        _logger.info(
            "Flipkart Selenium not enabled (set FLIPKART_USE_SELENIUM=1 in .env to use Chrome)"
        )

    reviews = _try_jsonld(product_url)
    if reviews:
        _logger.info("✓ Flipkart: %s reviews from JSON-LD (product URL)", len(reviews))
        return reviews

    reviews = _try_html(product_url)
    if reviews:
        _logger.info("✓ Flipkart: %s reviews from embedded HTML / DOM (product URL)", len(reviews))
        return reviews

    rev_url = flipkart_product_reviews_url(product_url)
    if rev_url:
        _logger.info("Trying Flipkart reviews tab: %s", rev_url[:120] + ("…" if len(rev_url) > 120 else ""))
        reviews = _try_jsonld(rev_url)
        if reviews:
            _logger.info("✓ Flipkart: %s reviews from JSON-LD (reviews tab)", len(reviews))
            return reviews
        reviews = _try_html(rev_url)
        if reviews:
            _logger.info("✓ Flipkart: %s reviews from embedded HTML / DOM (reviews tab)", len(reviews))
            return reviews

    if not (reviews or []):
        _logger.warning("Flipkart pipeline returned no reviews after all steps")

    return reviews or []


def _site_hint_from_url(url: str) -> str:
    u = (url or "").lower()
    if "flipkart" in u:
        return "flipkart"
    if "amazon" in u:
        return "amazon"
    return "other"


def _format_product_analyze_error(failed_product_url: str, exc: BaseException) -> str:
    """
    User-visible error: never embed raw https URLs (requests errors can include huge links).
    """
    raw = str(exc)
    hint = _site_hint_from_url(failed_product_url)
    http_like = isinstance(exc, requests.exceptions.RequestException) or (
        "403 Client Error" in raw
        or "401 Client Error" in raw
        or "Forbidden for url" in raw
        or "Client Error:" in raw
    )
    cleaned = re.sub(r"https?://[^\s\)]+", "[url]", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if hint == "flipkart" and http_like:
        return (
            "Flipkart blocked automated access (HTTP error). "
            "Add SCRAPER_PROXY to .env, or FLIPKART_USE_SELENIUM=1 (Chrome runs headless in the background by default). "
            "If a captcha appears, set FLIPKART_SELENIUM_HEADLESS=0 temporarily. Restart the server after .env changes."
        )

    if hint == "flipkart":
        snippet = cleaned[:700] + ("…" if len(cleaned) > 700 else "")
        return f"Could not analyze this Flipkart product. {snippet}"

    if hint == "amazon" and http_like:
        return (
            "Could not fetch this Amazon page (network or access blocked). "
            "Set RAINFOREST_API_KEY in .env and use a direct /dp/ product URL."
        )

    snippet = cleaned[:1200] + ("…" if len(cleaned) > 1200 else "")
    return f"Could not fetch or analyze reviews. {snippet}"


def classify_sentiment(text: str):
    """
    SST-2 returns POSITIVE/NEGATIVE. We map low-confidence to "NEUTRAL".
    Returns: (label, confidence_float_0_to_1)
    """
    inputs = sent_tokenizer(text[:1000], return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = sent_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    # Label mapping for SST-2: 0=NEGATIVE, 1=POSITIVE
    idx = int(probs.argmax())
    score = float(probs[idx])
    if score < 0.60:
        return "NEUTRAL", score
    return ("POSITIVE" if idx == 1 else "NEGATIVE"), score

@app.route("/analyze", methods=["POST"])
@app.route("/analyze", methods=["POST"])

def analyze():
    text = request.form.get("text", "")
    
    # Validate text input
    is_valid, error_message = validate_text(text)
    if not is_valid:
        return render_template("result.html",
                               prediction=error_message,
                               confidence=None,
                               probs=None,
                               emotions=None,
                               text_input=text)
    
    # Predict emotion
    emotion, probs = predict_emotion(text)
    score = float(round(max(probs) * 100, 2))  # confidence of predicted emotion
    
    # Get top 8 emotions
    top_n = 8
    top_indices = probs.argsort()[-top_n:][::-1]
    top_probs = probs[top_indices].tolist()
    top_emotions = [EMOTIONS[i] for i in top_indices]

    # Save to MongoDB
    analysis_doc = {
        "text_input": text,
        "emotion": emotion,
        "confidence": score,
        "user_id": session.get("user_id"),  # Associate with logged-in user if available
        "created_at": datetime.utcnow().isoformat()
    }
    result = analyses_collection.insert_one(analysis_doc)
    analysis_id = str(result.inserted_id)  # Get the MongoDB ObjectId as string

    return render_template("result.html",
                           prediction=emotion,
                           confidence=score,
                           probs=top_probs,
                           emotions=top_emotions,
                           text_input=text,
                           analysis_id=analysis_id)

@app.route("/results")
def results():
    # Get analyses from MongoDB
    analyses_cursor = analyses_collection.find().sort("created_at", -1)
    analyses_raw = list(analyses_cursor)

    # Convert to format expected by template (tuple-like structure)
    analyses = []
    for a in analyses_raw:
        confidence = float(a.get("confidence", 0))
        # Create tuple-like structure: (id, text_input, emotion, confidence, created_at)
        analyses.append((
            str(a["_id"]),  # MongoDB ObjectId as string
            a.get("text_input", ""),
            a.get("emotion", ""),
            confidence,
            a.get("created_at", "")
        ))

    # Calculate statistics
    total_recordings = len(analyses)
    avg_confidence = sum(a[3] for a in analyses) / total_recordings if total_recordings > 0 else 0
    total_duration = sum(len(a[1].split()) for a in analyses)  # Word count as duration equivalent
    emotion_counts = {}
    for a in analyses:
        emotion_counts[a[2]] = emotion_counts.get(a[2], 0) + 1
    most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "Neutral"

    return render_template("results.html",
                          analyses=analyses,
                          total_recordings=total_recordings,
                          avg_confidence=round(avg_confidence, 1),
                          total_duration=total_duration,
                          most_common_emotion=most_common_emotion.capitalize())

@app.route("/history")
def history():
    # Get analyses from MongoDB
    analyses_cursor = analyses_collection.find().sort("created_at", -1)
    analyses_raw = list(analyses_cursor)

    # Convert to format expected by template
    analyses = []
    for a in analyses_raw:
        confidence = float(a.get("confidence", 0))
        # Create tuple-like structure: (id, text_input, emotion, confidence, created_at)
        analyses.append((
            str(a["_id"]),  # MongoDB ObjectId as string
            a.get("text_input", ""),
            a.get("emotion", ""),
            confidence,
            a.get("created_at", "")
        ))

    # Calculate statistics
    total_analyses = len(analyses)
    avg_confidence = sum(a[3] for a in analyses) / total_analyses if total_analyses > 0 else 0

    return render_template("history.html",
                          analyses=analyses,
                          total_analyses=total_analyses,
                          avg_confidence=round(avg_confidence, 1))

@app.route("/delete/<analysis_id>", methods=["POST"])
def delete_analysis(analysis_id):
    try:
        # Delete from MongoDB using ObjectId
        result = analyses_collection.delete_one({"_id": ObjectId(analysis_id)})
        if result.deleted_count > 0:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Analysis not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/export/json")
def export_json():
    # Get analyses from MongoDB
    analyses_cursor = analyses_collection.find().sort("created_at", -1)
    analyses_raw = list(analyses_cursor)

    data = []
    for a in analyses_raw:
        data.append({
            "id": str(a["_id"]),
            "text_input": a.get("text_input", ""),
            "emotion": a.get("emotion", ""),
            "confidence": float(a.get("confidence", 0)),
            "created_at": a.get("created_at", "")
        })

    output = io.BytesIO()
    output.write(json.dumps(data, indent=2).encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype='application/json', as_attachment=True, download_name='text_analyses.json')

@app.route("/export/csv")
def export_csv():
    # Get analyses from MongoDB
    analyses_cursor = analyses_collection.find().sort("created_at", -1)
    analyses_raw = list(analyses_cursor)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Text Input', 'Emotion', 'Confidence', 'Created At'])
    for a in analyses_raw:
        confidence = float(a.get("confidence", 0))
        writer.writerow([
            str(a["_id"]),
            a.get("text_input", ""),
            a.get("emotion", ""),
            confidence,
            a.get("created_at", "")
        ])

    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='text_analyses.csv')

@app.route("/api/trends")
def api_trends():
    # Get analyses from MongoDB
    analyses_cursor = analyses_collection.find().sort("created_at", -1)
    analyses_raw = list(analyses_cursor)

    # Convert to format expected by function
    analyses = []
    for a in analyses_raw:
        confidence = float(a.get("confidence", 0))
        analyses.append((
            str(a["_id"]),
            a.get("text_input", ""),
            a.get("emotion", ""),
            confidence,
            a.get("created_at", "")
        ))

    total_analyses = len(analyses)
    avg_confidence = sum(a[3] for a in analyses) / total_analyses if total_analyses > 0 else 0
    emotion_counts = {}
    for a in analyses:
        emotion_counts[a[2]] = emotion_counts.get(a[2], 0) + 1
    most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "Neutral"

    return jsonify({
        "total_analyses": total_analyses,
        "avg_confidence": round(avg_confidence, 1),
        "most_common_emotion": most_common_emotion.capitalize()
    })

@app.route("/api/smart-reply/<analysis_id>")
def get_smart_reply(analysis_id):
    """Generate smart reply for a specific analysis"""
    try:
        # Find analysis in MongoDB
        analysis = analyses_collection.find_one({"_id": ObjectId(analysis_id)})
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404

        emotion = analysis.get("emotion", "")
        confidence = float(analysis.get("confidence", 0))

        smart_reply = generate_emotion_smart_reply(emotion, confidence)
        return jsonify({"smart_reply": smart_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/result/<analysis_id>")
def download_single_result(analysis_id):
    """Download a single analysis result as JSON"""
    try:
        # Find analysis in MongoDB
        analysis = analyses_collection.find_one({"_id": ObjectId(analysis_id)})
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404

        confidence = float(analysis.get("confidence", 0))
        smart_reply = generate_emotion_smart_reply(analysis.get("emotion", ""), confidence)

        data = {
            "id": str(analysis["_id"]),
            "text_input": analysis.get("text_input", ""),
            "emotion": analysis.get("emotion", ""),
            "confidence": confidence,
            "created_at": analysis.get("created_at", ""),
            "smart_reply_suggestion": smart_reply
        }

        output = io.BytesIO()
        output.write(json.dumps(data, indent=2).encode('utf-8'))
        output.seek(0)
        return send_file(output, mimetype='application/json', as_attachment=True,
                        download_name=f'analysis_{analysis_id}.json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _load_pdf_font(pdf):
    """
    Register a Unicode font when available (so emojis / non-ASCII don't render as tofu).
    Falls back to Helvetica when fonts are missing.
    """
    try:
        from fpdf import FPDF  # noqa: F401

        font_path = os.path.join(BASE_DIR, "fonts", "DejaVuSans.ttf")
        if os.path.exists(font_path):
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", size=12)
            return "DejaVu"
    except Exception:
        pass
    pdf.set_font("Helvetica", size=12)
    return "Helvetica"


def _pdf_safe_text(value) -> str:
    """FPDF can't handle null bytes; keep output readable."""
    s = "" if value is None else str(value)
    s = s.replace("\x00", " ")
    return re.sub(r"\s+", " ", s).strip()


@app.route("/download/result/<analysis_id>.pdf")
def download_single_result_pdf(analysis_id):
    """Download a single analysis result as a PDF using ReportLab."""
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
    except ImportError:
        return (
            jsonify(
                {
                    "error": "PDF generation library not installed. Run: pip install -r requirements.txt"
                }
            ),
            500,
        )

    try:
        analysis = analyses_collection.find_one({"_id": ObjectId(analysis_id)})
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404

        # Extract data with safe defaults
        confidence = float(analysis.get("confidence", 0))
        emotion = str(analysis.get("emotion", "N/A")).strip()
        created_at = str(analysis.get("created_at", "")).strip()
        text_input = str(analysis.get("text_input", "")).strip()
        smart_reply = generate_emotion_smart_reply(emotion, confidence)

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm,
            title="SentimentPulse - Text Analysis Result",
        )

        styles = getSampleStyleSheet()
        
        # Define safe HTML content with proper escaping
        def _safe_para(s: str) -> str:
            if s is None:
                return ""
            text = str(s).replace("\r\n", "\n").replace("\r", "\n")
            text = text.replace("\n", "<br/>")
            escaped = html.escape(text, quote=False)
            return escaped.replace("&lt;br/&gt;", "<br/>")

        # Custom body style
        body_style = ParagraphStyle(
            "BodyClear",
            parent=styles["BodyText"],
            fontSize=11,
            leading=14,
            spaceAfter=6,
        )

        # Build content
        content = []
        
        # Title
        content.append(Paragraph("<b>SentimentPulse - Text Analysis Result</b>", styles["Title"]))
        content.append(Spacer(1, 10))
        
        # Metadata
        content.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", body_style))
        if created_at:
            content.append(Paragraph(f"<b>Analysis Date:</b> {html.escape(created_at)}", body_style))
        content.append(Spacer(1, 8))
        
        # Summary section
        content.append(Paragraph("<b>Analysis Summary</b>", styles["Heading2"]))
        content.append(Paragraph(f"<b>Detected Emotion:</b> {html.escape(emotion)}", body_style))
        content.append(Paragraph(f"<b>Confidence Score:</b> {confidence:.2f}%", body_style))
        content.append(Spacer(1, 10))
        
        # Input text section
        content.append(Paragraph("<b>Your Input Text</b>", styles["Heading2"]))
        if text_input:
            content.append(Paragraph(_safe_para(text_input), body_style))
        else:
            content.append(Paragraph("<i>(No text provided)</i>", body_style))
        content.append(Spacer(1, 10))
        
        # Smart reply section
        content.append(Paragraph("<b>AI-Generated Response</b>", styles["Heading2"]))
        if smart_reply:
            content.append(Paragraph(_safe_para(smart_reply), body_style))
        else:
            content.append(Paragraph("<i>Unable to generate response</i>", body_style))
        
        # Build PDF
        doc.build(content)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"emotion_analysis_{analysis_id}.pdf",
        )
    except Exception as e:
        _logger.error(f"PDF generation error: {str(e)}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

@app.route("/api/dashboard-stats")
def dashboard_stats():
    """Get comprehensive dashboard statistics for charts"""
    # Get analyses from MongoDB
    analyses_cursor = analyses_collection.find().sort("created_at", -1)
    analyses_raw = list(analyses_cursor)

    if not analyses_raw:
        return jsonify({
            "emotion_distribution": {},
            "confidence_trends": [],
            "daily_activity": [],
            "total_analyses": 0,
            "avg_confidence": 0,
            "most_common_emotion": "-"
        })

    # Convert data
    analyses = []
    for a in analyses_raw:
        confidence = float(a.get("confidence", 0))
        analyses.append({
            "id": str(a["_id"]),
            "text": a.get("text_input", ""),
            "emotion": a.get("emotion", ""),
            "confidence": confidence,
            "created_at": a.get("created_at", "")
        })

    # Emotion distribution
    emotion_counts = {}
    for a in analyses:
        emotion_counts[a["emotion"]] = emotion_counts.get(a["emotion"], 0) + 1

    # Confidence trends (last 10 analyses), shown oldest -> newest for readable chart flow.
    recent = list(reversed(analyses[:10]))
    confidence_trends = [{"id": str(i + 1), "confidence": a["confidence"]} for i, a in enumerate(recent)]
    most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "-"

    # Daily activity (group by date)
    daily_activity = {}
    for a in analyses:
        # Extract date part from ISO format (2026-04-03T...)
        date = a["created_at"].split("T")[0] if "T" in a["created_at"] else a["created_at"].split(" ")[0]
        daily_activity[date] = daily_activity.get(date, 0) + 1

    daily_activity_list = [{"date": date, "count": count} for date, count in daily_activity.items()]
    daily_activity_list.sort(key=lambda x: x["date"])

    return jsonify({
        "emotion_distribution": emotion_counts,
        "confidence_trends": confidence_trends,
        "daily_activity": daily_activity_list,
        "total_analyses": len(analyses),
        "avg_confidence": round(sum(a["confidence"] for a in analyses) / len(analyses), 2),
        "most_common_emotion": (most_common_emotion.capitalize() if most_common_emotion else "-")
    })

# -----------------------------
if __name__ == "__main__":
    # Disable the watchdog reloader to avoid duplicate Windows processes
    port = int(os.environ.get("PORT", "5001"))
    banner = (
        f"\n{'=' * 60}\n"
        f"SentimentPulse ready — http://127.0.0.1:{port}\n"
        f"Each browser/API action logs below as: >>> METHOD path  then  <<< METHOD path -> status\n"
        f"(Set LOG_REQUESTS=0 to hide these lines. STARTUP_REQUEST_PROBE=0 skips the initial sample request.)\n"
        f"{'=' * 60}\n"
    )
    print(banner, file=sys.stderr, flush=True)
    _logger.info("Server starting (stderr); request lines use >>> and <<<")
    _schedule_startup_request_probe(port)
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)


