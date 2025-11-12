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

def fetch_and_save_news():
    print(f"📰 Fetching Economic Times RSS for '{QUERY}'")
    feed = feedparser.parse(RSS_URL)
    entries = getattr(feed, "entries", []) or []

    articles = []
    for e in entries[:MAX_ENTRIES]:
        articles.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": getattr(e, "published", ""),
            "summary": getattr(e, "summary", "")
        })

    df = pd.DataFrame(articles)
    df.to_csv(NEWS_CSV, index=False)
    print(f"✅ Saved {len(df)} news articles to {NEWS_CSV}")
    return len(df)

def write_summary(num_articles):
    summary = {
        "query": QUERY,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "fetched_on": date.today().isoformat(),
        "num_news_articles": num_articles,
        "news_csv_path": NEWS_CSV
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📝 Summary written to {SUMMARY_JSON}")
    return SUMMARY_JSON

def run_cmd(cmd):
    try:
        print("🔧", " ".join(cmd))
        subprocess.run(cmd, check=False)
    except Exception as e:
        print("⚠️ Command failed:", e)

def dvc_track(file_path):
    run_cmd([sys.executable, "-m", "dvc", "add", file_path])
    dvc_file = f"{file_path}.dvc"
    if os.path.exists(dvc_file):
        run_cmd(["git", "add", dvc_file])
        run_cmd(["git", "commit", "-m", f"chore(dvc): track {file_path}"])
        print(f"✅ DVC tracked {file_path}")
    else:
        print(f"⚠️ No .dvc file found for {file_path}")

if __name__ == "__main__":
    try:
        n = fetch_and_save_news()
        summary = write_summary(n)
        dvc_track(NEWS_CSV)
        dvc_track(summary)
        print("🎉 Ingestion + DVC tracking done.")
    except Exception as e:
        print("❌ Ingestion failed:", e)
