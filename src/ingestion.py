# src/ingestion.py
import json
import pandas as pd
import feedparser
import subprocess
import os
import sys
from datetime import date
from config import (
    NEWS_CSV, SUMMARY_JSON, QUERY, RSS_URL, MAX_ENTRIES,
    START_DATE, END_DATE
)

# =========================================================
# Helper: Safe command execution
# =========================================================
def run_cmd(cmd):
    """Safely run shell commands (for DVC and Git)."""
    try:
        print("🔧", " ".join(cmd))
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"⚠️ Command failed: {e}")

# =========================================================
# Fetch and save news
# =========================================================
def fetch_and_save_news():
    """Fetch top N news articles and save as CSV."""
    print(f"📰 Fetching Economic Times RSS for '{QUERY}'")
    feed = feedparser.parse(RSS_URL)
    entries = getattr(feed, "entries", []) or []

    if not entries:
        print("⚠️ No articles found in RSS feed.")
        return 0

    articles = []
    for e in entries[:MAX_ENTRIES]:
        articles.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": getattr(e, "published", ""),
            "summary": getattr(e, "summary", "")
        })

    df = pd.DataFrame(articles)
    os.makedirs(os.path.dirname(NEWS_CSV), exist_ok=True)
    df.to_csv(NEWS_CSV, index=False)
    print(f"✅ Saved {len(df)} news articles to {NEWS_CSV}")
    return len(df)

# =========================================================
# Write summary file
# =========================================================
def write_summary(num_articles):
    """Write ingestion summary as JSON."""
    summary = {
        "query": QUERY,
        "rss_url": RSS_URL,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "fetched_on": date.today().isoformat(),
        "num_news_articles": num_articles,
        "news_csv_path": NEWS_CSV
    }

    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📝 Summary written to {SUMMARY_JSON}")
    return SUMMARY_JSON

# =========================================================
# DVC Tracking
# =========================================================
def dvc_track(file_path):
    """Track data files with DVC and commit .dvc files to Git."""
    run_cmd([sys.executable, "-m", "dvc", "add", file_path])
    dvc_file = f"{file_path}.dvc"

    if os.path.exists(dvc_file):
        run_cmd(["git", "add", dvc_file])
        run_cmd(["git", "commit", "-m", f"chore(dvc): track {file_path}"])
        print(f"✅ DVC tracked {file_path}")
    else:
        print(f"⚠️ No .dvc file found for {file_path}")

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    try:
        print("🚀 Starting data ingestion...")

        num_articles = fetch_and_save_news()
        if num_articles == 0:
            raise RuntimeError("No articles fetched — skipping tracking and summary.")

        summary_path = write_summary(num_articles)

        # DVC tracking
        dvc_track(NEWS_CSV)
        dvc_track(summary_path)

        print("🎉 Ingestion complete — data saved and tracked with DVC.")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
