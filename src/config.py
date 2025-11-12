# src/config.py
import os
from datetime import date, timedelta

# =========================================================
# --- Data root (read from env for container friendliness)
# =========================================================
# Use environment first; fall back to /app/data (container) then to ~/mtp/data (dev)
DATA_ROOT = os.getenv("DATA_ROOT", "/app/data")
if DATA_ROOT == "/app/data" and os.path.exists(os.path.expanduser("~/mtp/data")):
    # if developer already has local data, prefer it for local runs (optional)
    # comment out the following line if you always want /app/data default
    DATA_ROOT = os.path.expanduser("~/mtp/data")

RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# =========================================================
# --- Output files
# =========================================================
NEWS_CSV = os.path.join(RAW_DIR, "news_NIFTY.csv")
SUMMARY_JSON = os.path.join(RAW_DIR, "ingestion_summary.json")

# =========================================================
# --- RSS Config
# =========================================================
QUERY = os.getenv("RSS_QUERY", "Nifty")
RSS_URL = os.getenv(
    "RSS_URL",
    "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/1977021501.cms",
)
MAX_ENTRIES = int(os.getenv("MAX_ENTRIES", "50"))

# =========================================================
# --- Dates (for logging)
# =========================================================
TODAY = date.today()
START_DATE = (TODAY - timedelta(days=2)).isoformat()
END_DATE = (TODAY - timedelta(days=1)).isoformat()

# =========================================================
# --- MLflow Tracking Directory (centralized absolute path)
# =========================================================
MLFLOW_DIR = os.getenv("MLFLOW_DIR", os.path.join(DATA_ROOT, "mlruns"))
os.makedirs(MLFLOW_DIR, exist_ok=True)

# =========================================================
# --- Print summary (for debugging convenience)
# =========================================================
if __name__ == "__main__":
    print("✅ Configuration Summary:")
    print(f"DATA_ROOT     : {DATA_ROOT}")
    print(f"RAW_DIR       : {RAW_DIR}")
    print(f"PROCESSED_DIR : {PROCESSED_DIR}")
    print(f"MLFLOW_DIR    : {MLFLOW_DIR}")
    print(f"RSS_URL       : {RSS_URL}")
    print(f"DATE RANGE    : {START_DATE} → {END_DATE}")
