# src/config.py

import os
from datetime import date, timedelta

# --- Data root (change only here if you move your data folder)
DATA_ROOT = os.path.expanduser("~/mtp/data")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# --- Output files
NEWS_CSV = os.path.join(RAW_DIR, "news_NIFTY.csv")
SUMMARY_JSON = os.path.join(RAW_DIR, "ingestion_summary.json")

# --- RSS Config
QUERY = "Nifty"
RSS_URL = "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/1977021501.cms"
MAX_ENTRIES = 50

# --- Dates (for logging)
TODAY = date.today()
START_DATE = (TODAY - timedelta(days=2)).isoformat()
END_DATE = (TODAY - timedelta(days=1)).isoformat()
