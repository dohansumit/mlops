# src/preprocessing.py
import os
import json
import pandas as pd
import subprocess
import sys
from datetime import date
from config import (
    NEWS_CSV, SUMMARY_JSON, QUERY, START_DATE, END_DATE, DATA_ROOT
)

# =========================================================
# Helper: Run safe commands
# =========================================================
def run_cmd(cmd):
    """Safely execute shell commands (for DVC and Git)."""
    try:
        print("🔧", " ".join(cmd))
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"⚠️ Command failed: {e}")

# =========================================================
# DVC Tracking Helper
# =========================================================
def dvc_track(file_path):
    """Track data files with DVC and commit to Git."""
    run_cmd([sys.executable, "-m", "dvc", "add", file_path])
    dvc_file = f"{file_path}.dvc"

    if os.path.exists(dvc_file):
        run_cmd(["git", "add", dvc_file])
        run_cmd(["git", "commit", "-m", f"chore(dvc): track {file_path}"])
        print(f"✅ DVC tracked {file_path}")
    else:
        print(f"⚠️ No .dvc file found for {file_path}")

# =========================================================
# Preprocess News Data
# =========================================================
def preprocess_news():
    """Clean and preprocess raw news data."""
    if not os.path.exists(NEWS_CSV):
        raise FileNotFoundError(f"❌ Raw news file not found: {NEWS_CSV}")

    print(f"🧹 Reading and cleaning news data from {NEWS_CSV}")
    df = pd.read_csv(NEWS_CSV)

    if df.empty:
        raise ValueError("⚠️ Raw news file is empty — cannot preprocess.")

    # Basic cleaning
    df.drop_duplicates(subset=["title"], inplace=True)
    df = df[df["title"].notnull() & (df["title"].str.strip() != "")]
    df["title"] = df["title"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)

    # Add processing timestamp
    df["processed_on"] = date.today().isoformat()

    # Save processed file under data/processed
    processed_dir = os.path.join(DATA_ROOT, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, f"processed_news_{date.today().isoformat()}.csv")
    df.to_csv(processed_path, index=False)

    print(f"✅ Processed news saved to {processed_path}")
    return processed_path, len(df)

# =========================================================
# Write Summary
# =========================================================
def write_summary(num_articles, processed_path):
    """Write summary of preprocessing results."""
    summary = {
        "query": QUERY,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "processed_on": date.today().isoformat(),
        "num_news_articles": num_articles,
        "processed_csv_path": processed_path
    }

    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📝 Summary updated at {SUMMARY_JSON}")
    return SUMMARY_JSON

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    try:
        print("🚀 Starting preprocessing...")

        processed_file, n = preprocess_news()
        summary_path = write_summary(n, processed_file)

        # DVC tracking
        dvc_track(processed_file)
        dvc_track(summary_path)

        print("🎉 Preprocessing complete — data cleaned and tracked with DVC.")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
