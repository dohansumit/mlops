# src/app.py
from fastapi import FastAPI, BackgroundTasks
import subprocess
import os
import sys
import json
from pathlib import Path
from .config import DATA_ROOT
import pandas as pd
from typing import Dict, Any

app = FastAPI(title="Financial Sentiment API", version="1.0")

# =========================================================
# Utility: Run a script and capture output
# =========================================================
def run_script(script_name):
    script_path = Path(__file__).parent / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    return result

# =========================================================
# Lightweight summary/advice function (mirrors model.compute_summary)
# =========================================================
def compute_advice_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute avg sentiment, percentages, counts and heuristic advice from a dataframe."""
    if df is None or df.empty:
        return {
            "avg_sentiment": 0.0,
            "positive": 0.0,
            "neutral": 0.0,
            "negative": 0.0,
            "counts": {},
            "advice": "NO DATA"
        }

    # ensure we use the string labels as lowercase to be robust
    labels = df.get("sentiment_label", pd.Series([], dtype=str)).astype(str).str.lower()
    counts = labels.value_counts().to_dict()
    total = sum(counts.values()) or 0

    avg_sentiment = float(df["sentiment_score"].mean()) if "sentiment_score" in df.columns and len(df) > 0 else 0.0

    positive = (counts.get("positive", 0) / total) * 100 if total > 0 else 0.0
    neutral = (counts.get("neutral", 0) / total) * 100 if total > 0 else 0.0
    negative = (counts.get("negative", 0) / total) * 100 if total > 0 else 0.0

    # Heuristic-based advice (kept identical to your model.py logic)
    if positive >= 25 and neutral >= 25:
        advice = "GOOD TIME TO BUY STOCKS"
    elif 45 <= neutral <= 50:
        advice = "NORMAL DAY TO DO TRANSACTIONS"
    elif negative > 40:
        advice = "NOT GOOD TO INVEST"
    else:
        advice = "MIXED SIGNALS — USE CAUTION"

    return {
        "avg_sentiment": float(avg_sentiment),
        "positive": float(round(positive, 3)),
        "neutral": float(round(neutral, 3)),
        "negative": float(round(negative, 3)),
        "counts": {k: int(v) for k, v in counts.items()},
        "advice": advice
    }

# =========================================================
# Endpoints
# =========================================================

@app.get("/")
def home():
    return {
        "message": "📈 Financial Sentiment API is running!",
        "available_endpoints": ["/run_pipeline", "/latest_results", "/advice", "/health"]
    }

@app.get("/health")
def health_check():
    return {"status": "✅ OK"}

@app.post("/run_pipeline")
def run_pipeline(background_tasks: BackgroundTasks = None):
    """
    Run ingestion -> preprocess -> model sequentially.
    This runs synchronously by default.
    """
    steps = ["ingestion.py", "preprocess.py", "model.py"]
    logs = []

    for step in steps:
        result = run_script(step)
        logs.append({
            "step": step,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
        if result.returncode != 0:
            return {
                "status": "❌ Failed",
                "step": step,
                "error": result.stderr or result.stdout
            }

    return {
        "status": "✅ Pipeline completed successfully",
        "steps_executed": steps,
        "details": logs[-1]["stdout"][-1000:]
    }

@app.get("/latest_results")
def get_latest_results():
    """
    Return the latest sentiment results as JSON (a small sample) plus computed advice.
    """
    try:
        processed_dir = os.path.join(DATA_ROOT, "processed")
        if not os.path.isdir(processed_dir):
            return {"error": f"No processed directory found at {processed_dir}"}

        # Prefer CSV files
        csv_files = sorted(
            [
                f for f in os.listdir(processed_dir)
                if f.startswith("processed_news_sentiment_") and f.endswith(".csv")
            ],
            reverse=True
        )

        # If none, try to recover via dvc checkout
        if not csv_files:
            dvc_files = sorted(
                [f for f in os.listdir(processed_dir) if f.endswith(".dvc") and "processed_news_sentiment_" in f],
                reverse=True
            )
            if dvc_files:
                try:
                    subprocess.run([sys.executable, "-m", "dvc", "checkout", processed_dir], check=False)
                except Exception:
                    pass

                csv_files = sorted(
                    [
                        f for f in os.listdir(processed_dir)
                        if f.startswith("processed_news_sentiment_") and f.endswith(".csv")
                    ],
                    reverse=True
                )

        if not csv_files:
            return {
                "error": "No sentiment CSV results found in processed directory.",
                "hint": "If you recently pulled the repo, run `dvc checkout data/processed` to restore data."
            }

        latest_csv = os.path.join(processed_dir, csv_files[0])
        df = pd.read_csv(latest_csv)

        if df.empty:
            return {"latest_file": latest_csv, "num_articles": 0, "sample_results": [], "advice": "NO DATA"}

        # sample rows
        cols = [c for c in ["title", "sentiment_label", "sentiment_score"] if c in df.columns]
        sample = df[cols].tail(10).to_dict(orient="records")

        # compute advice & summary
        summary = compute_advice_from_df(df)

        return {
            "latest_file": latest_csv,
            "num_articles": len(df),
            "sample_results": sample,
            "summary": summary
        }

    except Exception as e:
        return {"error": "Internal error while reading latest results", "details": str(e)}

@app.get("/advice")
def get_advice():
    """
    Return only the computed summary & advice for the latest results.
    Useful for dashboards that poll a single endpoint.
    """
    try:
        processed_dir = os.path.join(DATA_ROOT, "processed")
        if not os.path.isdir(processed_dir):
            return {"error": f"No processed directory found at {processed_dir}"}

        csv_files = sorted(
            [
                f for f in os.listdir(processed_dir)
                if f.startswith("processed_news_sentiment_") and f.endswith(".csv")
            ],
            reverse=True
        )

        if not csv_files:
            return {"error": "No sentiment CSV results found in processed directory."}

        latest_csv = os.path.join(processed_dir, csv_files[0])
        df = pd.read_csv(latest_csv)
        summary = compute_advice_from_df(df)
        return {"latest_file": latest_csv, "summary": summary}

    except Exception as e:
        return {"error": "Internal error while computing advice", "details": str(e)}
