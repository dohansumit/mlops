# src/config.py
import os
from datetime import date, timedelta

# =========================================================
# --- Data root (read from env for container friendliness)
# =========================================================
# Use environment first; fall back to /app/data (container) then to ~/mtp/data (dev)
DATA_ROOT = os.getenv("DATA_ROOT", "/app/data")
if DATA_ROOT == "/app/data" and os.path.exists(os.path.expanduser("~/mtp/data")):
    # prefer local developer path when present (helps local dev)
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
# Resolution order (most authoritative first):
# 1. MLFLOW_TRACKING_URI env var (accepts file:/ or file:/// or plain path)
# 2. MLFLOW_DIR env var (plain path)
# 3. default: DATA_ROOT/mlruns
_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
if _mlflow_uri:
    # accept file:///abs/path, file:/abs/path, or plain filesystem path
    if _mlflow_uri.startswith("file://"):
        _mlflow_path = _mlflow_uri[len("file://") :]
    elif _mlflow_uri.startswith("file:"):
        _mlflow_path = _mlflow_uri[len("file:") :]
    else:
        _mlflow_path = _mlflow_uri
    MLFLOW_DIR = os.path.abspath(os.path.expanduser(_mlflow_path))
else:
    # fallback to explicit MLFLOW_DIR env or DATA_ROOT/mlruns
    MLFLOW_DIR = os.path.abspath(os.path.expanduser(os.getenv("MLFLOW_DIR", os.path.join(DATA_ROOT, "mlruns"))))

# create the mlflow dir if missing
os.makedirs(MLFLOW_DIR, exist_ok=True)

# =========================================================
# --- Print summary (for debugging convenience)
# =========================================================
# Prints only when run as a script, or when DEBUG_CONFIG=1 is set in env.
if __name__ == "__main__" or os.getenv("DEBUG_CONFIG", "0") == "1":
    print("✅ Configuration Summary:")
    print(f"DATA_ROOT     : {DATA_ROOT}")
    print(f"RAW_DIR       : {RAW_DIR}")
    print(f"PROCESSED_DIR : {PROCESSED_DIR}")
    print(f"MLFLOW_DIR    : {MLFLOW_DIR}   (derived from MLFLOW_TRACKING_URI={_mlflow_uri or '(not set)'})")
    print(f"RSS_URL       : {RSS_URL}")
    print(f"DATE RANGE    : {START_DATE} → {END_DATE}")
