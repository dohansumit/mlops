# src/app.py
from fastapi import FastAPI, BackgroundTasks
import subprocess
import os
import sys
import json
from pathlib import Path
from .config import DATA_ROOT
import pandas as pd



app = FastAPI(title="Financial Sentiment API", version="1.0")

# =========================================================
# Utility: Run a script and capture output
# =========================================================
def run_script(script_name):
    """
    Run a Python script located in the same directory as this file
    and capture stdout/stderr. Returns a CompletedProcess.
    """
    script_path = Path(__file__).parent / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    return result

# =========================================================
# Endpoints
# =========================================================

@app.get("/")
def home():
    return {
        "message": "📈 Financial Sentiment API is running!",
        "available_endpoints": ["/run_pipeline", "/latest_results", "/health"]
    }

@app.get("/health")
def health_check():
    return {"status": "✅ OK"}

@app.post("/run_pipeline")
def run_pipeline(background_tasks: BackgroundTasks = None):
    """
    Run ingestion -> preprocess -> model sequentially.
    This runs synchronously by default. If you want non-blocking behavior,
    use BackgroundTasks and the commented example at the bottom.
    """
    # NOTE: your preprocessing script is named preprocess.py (not preprocessing.py)
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
            # return the error so you can debug from the client
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

# Example: how you'd run the pipeline in background instead of blocking:
# def background_run():
#     for step in ["ingestion.py", "preprocess.py", "model.py"]:
#         subprocess.run([sys.executable, str(Path(__file__).parent / step)])
#
# @app.post("/run_pipeline_bg")
# def run_pipeline_bg(background_tasks: BackgroundTasks):
#     background_tasks.add_task(background_run)
#     return {"status": "Pipeline started in background"}

@app.get("/latest_results")
def get_latest_results():
    """
    Return the latest sentiment results as JSON (a small sample).
    - Prefers actual CSV files (ignores .dvc files).
    - If only .dvc files exist, attempts `dvc checkout` for the processed dir.
    """
    try:
        processed_dir = os.path.join(DATA_ROOT, "processed")
        if not os.path.isdir(processed_dir):
            return {"error": f"No processed directory found at {processed_dir}"}

        # 1) Look for actual CSV sentiment files (ignore .dvc)
        csv_files = sorted(
            [
                f for f in os.listdir(processed_dir)
                if f.startswith("processed_news_sentiment_") and f.endswith(".csv")
            ],
            reverse=True
        )

        # 2) If no CSVs, but .dvc metadata exists, try to restore with dvc checkout
        if not csv_files:
            dvc_files = sorted(
                [
                    f for f in os.listdir(processed_dir)
                    if f.startswith("processed_news_sentiment_") and f.endswith(".dvc")
                ],
                reverse=True
            )
            if dvc_files:
                try:
                    # Attempt to restore processed files from DVC cache
                    subprocess.run([sys.executable, "-m", "dvc", "checkout", processed_dir], check=False)
                except Exception:
                    # don't crash the API if dvc isn't available or checkout fails
                    pass

                # recompute csv_files after attempted checkout
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

        # read csv safely
        df = pd.read_csv(latest_csv)
        if df.empty:
            return {"latest_file": latest_csv, "num_articles": 0, "sample_results": []}

        # only return a small set of useful columns to avoid huge payloads
        cols = [c for c in ["title", "sentiment_label", "sentiment_score"] if c in df.columns]
        sample = df[cols].tail(10).to_dict(orient="records")

        return {
            "latest_file": latest_csv,
            "num_articles": len(df),
            "sample_results": sample
        }

    except Exception as e:
        return {"error": "Internal error while reading latest results", "details": str(e)}
