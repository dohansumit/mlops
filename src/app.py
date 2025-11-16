# src/app.py
from fastapi import FastAPI, BackgroundTasks
import subprocess
import os
import sys
import json
import tempfile
from pathlib import Path
from .config import DATA_ROOT, MLFLOW_DIR
import pandas as pd
from typing import Dict, Any

# MLflow imports (used only by the two endpoints below)
import mlflow
from mlflow.tracking import MlflowClient

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
    Prefers MLflow latest run (experiment: Financial_Sentiment_Pipeline). Falls back to CSV in DATA_ROOT/processed.
    """
    # First attempt: read latest MLflow run if available
    try:
        # configure mlflow to local tracking dir
        mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
        client = MlflowClient(tracking_uri=f"file://{MLFLOW_DIR}")

        # find experiment by name
        exp = client.get_experiment_by_name("Financial_Sentiment_Pipeline")
        if exp is not None:
            exp_id = exp.experiment_id
            runs = client.search_runs(experiment_ids=[exp_id], order_by=["attribute.start_time DESC"], max_results=1)
            if runs:
                run = runs[0]
                run_id = run.info.run_id

                # collect params & metrics
                params = dict(run.data.params)
                metrics = dict(run.data.metrics)

                # try to fetch artifact(s) from 'results' artifact_path (model.py uses artifact_path="results")
                artifacts = client.list_artifacts(run_id, path="results")
                csv_path = None
                sample = []
                num_articles = int(params.get("num_articles") or metrics.get("num_articles") or 0)

                if artifacts:
                    # pick first CSV artifact in results
                    csv_art = next((a for a in artifacts if a.path.endswith(".csv")), None)
                    if csv_art:
                        # download artifact to a temp dir and read
                        tmpdir = tempfile.mkdtemp()
                        local_path = client.download_artifacts(run_id, path=csv_art.path, dst_path=tmpdir)
                        csv_path = local_path
                        try:
                            df = pd.read_csv(local_path)
                            cols = [c for c in ["title", "sentiment_label", "sentiment_score"] if c in df.columns]
                            sample = df[cols].tail(10).to_dict(orient="records")
                            num_articles = len(df)
                        except Exception:
                            # ignore reading errors and fall back to metadata only
                            sample = []

                # Advice: prefer the advice param if logged, else compute a lightweight summary from sample
                advice = params.get("advice") or params.get("advice".lower()) or None

                if not advice and sample:
                    # compute using same lightweight logic as compute_advice_from_df
                    try:
                        df_tmp = pd.DataFrame(sample)
                        advice = compute_advice_from_df(df_tmp)["advice"]
                    except Exception:
                        advice = "NO DATA"

                response = {
                    "source": "mlflow",
                    "run_id": run_id,
                    "params": params,
                    "metrics": metrics,
                    "latest_file": csv_path or "artifact_not_found",
                    "num_articles": num_articles,
                    "sample_results": sample,
                    "summary": {
                        "advice": advice
                    }
                }
                return response

    except Exception as e:
        # If anything fails when reading MLflow, we'll fall-through to CSV fallback
        print("⚠️ Warning: MLflow read failed in /latest_results:", e)

    # Fallback: previous CSV-based behavior (reads processed dir)
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
            "source": "csv",
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
    Prefer MLflow-run advice (advice param) when available, otherwise compute from the latest MLflow artifact or processed CSV.
    """
    # Try MLflow first
    try:
        mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
        client = MlflowClient(tracking_uri=f"file://{MLFLOW_DIR}")

        exp = client.get_experiment_by_name("Financial_Sentiment_Pipeline")
        if exp is not None:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["attribute.start_time DESC"], max_results=1)
            if runs:
                run = runs[0]
                run_id = run.info.run_id
                params = dict(run.data.params)
                metrics = dict(run.data.metrics)

                # advice may be logged as a param
                advice = params.get("advice") or params.get("advice".lower()) or None

                # if advice not present, try to read the results artifact and compute
                if not advice:
                    artifacts = client.list_artifacts(run_id, path="results")
                    if artifacts:
                        csv_art = next((a for a in artifacts if a.path.endswith(".csv")), None)
                        if csv_art:
                            tmpdir = tempfile.mkdtemp()
                            local_path = client.download_artifacts(run_id, path=csv_art.path, dst_path=tmpdir)
                            try:
                                df = pd.read_csv(local_path)
                                summary = compute_advice_from_df(df)
                                return {"source": "mlflow_artifact", "latest_file": local_path, "summary": summary}
                            except Exception:
                                advice = None

                # if we got advice from params or couldn't compute from artifact, return from mlflow metadata
                if advice:
                    summary = {
                        "avg_sentiment": float(metrics.get("avg_sentiment", 0.0)),
                        "positive": float(metrics.get("percent_positive", metrics.get("positive", 0.0))),
                        "neutral": float(metrics.get("percent_neutral", metrics.get("neutral", 0.0))),
                        "negative": float(metrics.get("percent_negative", metrics.get("negative", 0.0))),
                        "counts": {},  # counts aren't always logged as a dict; keep empty
                        "advice": advice
                    }
                    # attach run info
                    return {"source": "mlflow", "run_id": run_id, "summary": summary, "params": params, "metrics": metrics}

    except Exception as e:
        print("⚠️ Warning: MLflow read failed in /advice:", e)

    # CSV fallback
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
        return {"source": "csv", "latest_file": latest_csv, "summary": summary}

    except Exception as e:
        return {"error": "Internal error while computing advice", "details": str(e)}
