# src/model.py

import os
import json
import glob
import pandas as pd
import subprocess
import sys
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mlflow
from config import DATA_ROOT, QUERY, START_DATE, END_DATE, MLFLOW_DIR

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
# DVC Tracking Helper
# =========================================================
def dvc_track(file_path):
    """Track model outputs or data files with DVC and commit."""
    run_cmd([sys.executable, "-m", "dvc", "add", file_path])
    dvc_file = f"{file_path}.dvc"

    if os.path.exists(dvc_file):
        run_cmd(["git", "add", dvc_file])
        run_cmd(["git", "commit", "-m", f"chore(dvc): track {file_path}"])
        print(f"✅ DVC tracked {file_path}")
    else:
        print(f"⚠️ No .dvc file found for {file_path}")

# =========================================================
# Load Latest Processed News
# =========================================================
def load_latest_processed():
    """Find the latest processed news CSV."""
    processed_dir = os.path.join(DATA_ROOT, "processed")
    files = sorted(glob.glob(os.path.join(processed_dir, "processed_news_*.csv")))

    if not files:
        raise FileNotFoundError("❌ No processed CSV files found in data/processed/")

    latest_file = files[-1]
    print(f"📂 Using latest processed file: {latest_file}")
    return latest_file

# =========================================================
# FinBERT Sentiment Analysis (batched + robust mapping)
# =========================================================
def run_sentiment_analysis(input_file, batch_size=32):
    """Run FinBERT sentiment classification on news titles (batched)."""
    print("🔍 Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # derive id2label mapping robustly (keys -> label strings)
    raw_id2label = getattr(model.config, "id2label", None)
    if raw_id2label is None:
        # fallback to conventional ordering if missing
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
    else:
        # ensure keys are ints
        id2label = {int(k): v for k, v in raw_id2label.items()}

    print(f"✅ FinBERT loaded on {device.upper()}. Label map: {id2label}")

    df = pd.read_csv(input_file)
    if df.empty or "title" not in df.columns:
        raise ValueError("❌ Data missing 'title' column or empty file.")

    titles = df["title"].astype(str).tolist()

    def predict_batch(texts, bsize=32):
        all_labels = []
        all_scores = []
        all_probs = []
        for i in range(0, len(texts), bsize):
            batch = texts[i : i + bsize]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                out = model(**enc)
            probs = torch.nn.functional.softmax(out.logits, dim=1).cpu().numpy()  # shape (B, C)
            for p in probs:
                # map highest index to label via id2label
                idx = int(p.argmax())
                label = id2label.get(idx, str(idx))

                # try to find pos/neg indices by label string content
                pos_idx = next((k for k, v in id2label.items() if "pos" in v.lower()), None)
                neg_idx = next((k for k, v in id2label.items() if "neg" in v.lower()), None)

                if pos_idx is not None and neg_idx is not None:
                    score = float(p[pos_idx] - p[neg_idx])
                else:
                    # fallback: confidence-like score = max - second_max
                    sorted_p = sorted(p, reverse=True)
                    score = float(sorted_p[0] - sorted_p[1])

                all_labels.append(label)
                all_scores.append(score)
                all_probs.append(p.tolist())

        return all_labels, all_scores, all_probs

    print("🔬 Running batched sentiment predictions...")
    labels, scores, probs = predict_batch(titles, batch_size)

    # attach results to dataframe
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    # store probs as compact JSON strings (useful for debugging / analysis)
    df["sentiment_probs"] = [json.dumps(p) for p in probs]

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(DATA_ROOT, "processed", f"processed_news_sentiment_{timestamp}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Sentiment results saved to {output_file}")

    return df, output_file

# =========================================================
# Compute Metrics & Advice
# =========================================================
def compute_summary(df):
    """Compute sentiment statistics and investment advice."""
    label_counts = df["sentiment_label"].value_counts().to_dict()
    total = sum(label_counts.values()) if label_counts else 0
    avg_sentiment = df["sentiment_score"].mean() if len(df) > 0 else 0.0

    positive = (label_counts.get("positive", 0) / total) * 100 if total > 0 else 0.0
    neutral = (label_counts.get("neutral", 0) / total) * 100 if total > 0 else 0.0
    negative = (label_counts.get("negative", 0) / total) * 100 if total > 0 else 0.0

    # Heuristic-based advice
    if positive >= 25 and neutral >= 25:
        advice = "GOOD TIME TO BUY STOCKS"
    elif 45 <= neutral <= 50:
        advice = "NORMAL DAY TO DO TRANSACTIONS"
    elif negative > 40:
        advice = "NOT GOOD TO INVEST"
    else:
        advice = "MIXED SIGNALS — USE CAUTION"

    print(f"📈 Investment Advice: {advice}")

    return {
        "avg_sentiment": float(avg_sentiment),
        "positive": float(positive),
        "neutral": float(neutral),
        "negative": float(negative),
        "counts": {k: int(v) for k, v in label_counts.items()},
        "advice": advice
    }

# =========================================================
# MLflow Logging
# =========================================================
def log_to_mlflow(df, metrics, input_file, output_file):
    """Log model results and artifacts to MLflow."""
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("Financial_Sentiment_Pipeline")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=f"Sentiment_Analysis_{timestamp}"):
        mlflow.log_param("query", QUERY)
        mlflow.log_param("start_date", START_DATE)
        mlflow.log_param("end_date", END_DATE)
        mlflow.log_param("num_articles", len(df))
        mlflow.log_param("advice", metrics["advice"])

        mlflow.log_metric("avg_sentiment", metrics["avg_sentiment"])
        mlflow.log_metric("percent_positive", metrics["positive"])
        mlflow.log_metric("percent_neutral", metrics["neutral"])
        mlflow.log_metric("percent_negative", metrics["negative"])

        for label, count in metrics["counts"].items():
            # mlflow requires numeric metrics, ensure int
            mlflow.log_metric(f"count_{label}", int(count))

        mlflow.log_artifact(input_file, artifact_path="input_data")
        mlflow.log_artifact(output_file, artifact_path="results")

    print(f"✅ MLflow run logged at: {MLFLOW_DIR}")

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    try:
        print("🚀 Starting sentiment analysis pipeline...")

        latest_file = load_latest_processed()
        df, output_file = run_sentiment_analysis(latest_file)
        metrics = compute_summary(df)

        # Track and log
        dvc_track(output_file)
        log_to_mlflow(df, metrics, latest_file, output_file)

        print("🎉 Model pipeline completed — DVC tracked and MLflow logged.")
    except Exception as e:
        print(f"❌ Model pipeline failed: {e}")
