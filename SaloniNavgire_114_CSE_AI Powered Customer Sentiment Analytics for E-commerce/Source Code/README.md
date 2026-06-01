# SentimentPulse (AI Customer Sentiment Analytics for E-commerce)

This project is a Flask website that:
- Scrapes **real-time product reviews** from product pages (via JSON-LD `application/ld+json` review markup)
- Runs **AI sentiment analysis** (Positive / Neutral / Negative)
- Includes a **Text Emotion Detection** tool (GO-Emotions)
- Allows **guest users** to analyze product links **2 times**, then requires login

## Run locally (Windows)

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Create a local `.env` file (recommended):

- Copy `env.example` to `.env` and fill in values:
  - `RAINFOREST_API_KEY` (needed for Amazon links)
  - `FLASK_SECRET_KEY` (recommended)

3. (Alternative) Set a secret key via PowerShell:

```powershell
$env:FLASK_SECRET_KEY="change-me"
```

4. Start the app:

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Guest limit + login

- Guests can use `/product` up to **2** times per browser session.
- After that, the app redirects to `/auth/login`.
- Logged-in users can analyze product links without this limit.

## Review scraping notes

The built-in scraper extracts reviews from JSON-LD (common on Shopify/WooCommerce sites that expose structured review data).
If a page doesn't expose JSON-LD reviews, the UI will tell you no reviews were found.

## Files

- `app.py`: Flask app, auth, role guards, scraping, sentiment pipeline
- `templates/`: UI pages (home, login/signup, dashboards, product analyzer, results)
- MongoDB Atlas: User authentication and data storage (all analyses, reviews, and user data)


