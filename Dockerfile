# ======================================================
# 1️⃣ Base image
# ======================================================
FROM python:3.10-slim

# Prevent Python from writing pyc files and using buffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ======================================================
# 2️⃣ Set working directory
# ======================================================
WORKDIR /app

# ======================================================
# 3️⃣ Copy dependencies and install them
# ======================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ======================================================
# 4️⃣ Copy project source code
# ======================================================
COPY src/ src/
COPY data/ data/

# Copy DVC configs only if they exist (use wildcard safely)
# These COPY commands will not fail if files are absent
COPY .dvc .dvc
COPY .dvcignore* ./
COPY dvc.yaml* ./

# ======================================================
# 5️⃣ Install extra dependencies (MLflow, FastAPI, DVC, etc.)
# ======================================================
RUN pip install --no-cache-dir \
    dvc[all] \
    mlflow \
    uvicorn \
    fastapi \
    torch \
    transformers \
    pandas \
    feedparser

# ======================================================
# 6️⃣ Environment variables
# ======================================================
ENV DATA_ROOT=/app/data
ENV MLFLOW_DIR=/app/data/mlruns
ENV MLFLOW_PORT=5050
ENV API_PORT=8080

# ======================================================
# 7️⃣ Expose both FastAPI and MLflow ports
# ======================================================
EXPOSE 8080
EXPOSE 5050

# ======================================================
# 8️⃣ Start both MLflow & FastAPI servers together
# ======================================================
# Runs MLflow in the background, then starts FastAPI
CMD ["bash", "-c", "\
    echo '🚀 Starting MLflow tracking server on port ${MLFLOW_PORT}...' && \
    mlflow server --backend-store-uri file://${MLFLOW_DIR} --host 0.0.0.0 --port ${MLFLOW_PORT} & \
    echo '🚀 Starting FastAPI app on port ${API_PORT}...' && \
    uvicorn src.app:app --host 0.0.0.0 --port ${API_PORT} --reload \
"]
