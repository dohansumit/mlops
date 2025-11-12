# Dockerfile: Full Pipeline + FastAPI + MLflow (root-run variant)
FROM python:3.10-slim

ARG USERNAME=mlflow
ARG USER_UID=1000
ARG USER_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Default runtime envs (can be overridden at runtime)
ENV DATA_ROOT=/app/data
ENV MLFLOW_DIR=/app/data/mlruns
ENV MLFLOW_PORT=5050
ENV API_PORT=8080
ENV SKIP_TRAIN=0

WORKDIR /app

# -------------------------
# System packages
# -------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git curl bash build-essential procps ca-certificates sudo \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Create unprivileged user (available if you want to drop privileges)
# -------------------------
RUN groupadd --gid ${USER_GID} ${USERNAME} || true \
 && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --home-dir /home/${USERNAME} --shell /bin/bash ${USERNAME} || true \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && chmod 0440 /etc/sudoers.d/${USERNAME}

# -------------------------
# Python deps (single install)
# -------------------------
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir mlflow "uvicorn[standard]" fastapi pandas feedparser yfinance

# -------------------------
# Project files
# -------------------------
RUN mkdir -p /app/data /app/src
COPY --chown=${USER_UID}:${USER_GID} src/ src/

# -------------------------
# Expose ports
# -------------------------
EXPOSE ${API_PORT}
EXPOSE ${MLFLOW_PORT}

# -------------------------
# Entrypoint Script (robust: starts MLflow, waits for HTTP 200 on /, then starts API)
# -------------------------
RUN mkdir -p /usr/local/bin
RUN cat > /usr/local/bin/fullpipeline-entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Configurable envs (can be overridden at runtime)
DATA_ROOT="${DATA_ROOT:-/app/data}"
MLFLOW_DIR="${MLFLOW_DIR:-/app/data/mlruns}"
MLFLOW_PORT="${MLFLOW_PORT:-5050}"
API_PORT="${API_PORT:-8080}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FLAG_FILE="${DATA_ROOT}/.model_trained"
LOG_FILE="${DATA_ROOT}/mlflow.log"

# Optional behaviour controlled by env args:
# CHOWN_ON_START=1 -> attempt to chown $MLFLOW_DIR to $USER_UID:$USER_GID
# DROP_TO_USER=1 -> after chown, drop to the unprivileged user and run services as that user
CHOWN_ON_START="${CHOWN_ON_START:-0}"
DROP_TO_USER="${DROP_TO_USER:-0}"
USER_UID="${USER_UID:-1000}"
USER_GID="${USER_GID:-1000}"
USERNAME="${USERNAME:-mlflow}"

# How long to wait for MLflow to become healthy (seconds)
MLFLOW_START_TIMEOUT="${MLFLOW_START_TIMEOUT:-60}"

cmd="${1:-start}"

function start_mlflow_and_wait() {
  mkdir -p "${MLFLOW_DIR}" "${DATA_ROOT}"
  echo "🚀 Starting MLflow tracking server on port ${MLFLOW_PORT} (backend: ${MLFLOW_DIR})..."
  nohup mlflow server \
    --backend-store-uri "file://${MLFLOW_DIR}" \
    --default-artifact-root "file://${MLFLOW_DIR}" \
    --host 0.0.0.0 \
    --port "${MLFLOW_PORT}" > "${LOG_FILE}" 2>&1 &

  MLFLOW_PID=$!

  echo "Waiting up to ${MLFLOW_START_TIMEOUT}s for MLflow to respond (HTTP 200 on /)..."
  local waited=0
  while [ "${waited}" -lt "${MLFLOW_START_TIMEOUT}" ]; do
    # prefer checking root for 200 OK; fall back to port listening
    if curl -sSf --max-time 2 "http://127.0.0.1:${MLFLOW_PORT}/" >/dev/null 2>&1; then
      echo "✅ MLflow root served HTTP 200 (PID ${MLFLOW_PID})."
      return 0
    fi
    # if root not yet 200, check if the port is listening (indicates process is up)
    if ss -ltn | grep -q ":${MLFLOW_PORT}"; then
      echo "ℹ️ Port ${MLFLOW_PORT} is listening (process up) but root not yet 200; continuing to wait..."
    fi
    sleep 1
    waited=$((waited + 1))
  done

  echo "⚠️ MLflow did not respond with HTTP 200 in ${MLFLOW_START_TIMEOUT}s. Last 200 lines of ${LOG_FILE}:"
  tail -n 200 "${LOG_FILE}" || true
  return 1
}

case "${cmd}" in
  shell)
    exec bash
    ;;

  run-pipeline)
    mkdir -p "${DATA_ROOT}"
    echo "🏗️ Running ingestion..."
    python -u src/ingestion.py || { echo "❌ Ingestion failed"; exit 1; }
    echo "🧹 Running preprocessing..."
    python -u src/preprocess.py || { echo "❌ Preprocessing failed"; exit 1; }
    echo "🤖 Training model..."
    python -u src/model.py || { echo "❌ Model training failed"; exit 1; }
    echo "✅ Pipeline completed."
    exit 0
    ;;

  start)
    mkdir -p "${DATA_ROOT}" "${MLFLOW_DIR}"

    if [ "${CHOWN_ON_START}" = "1" ]; then
      echo "🔧 CHOWN_ON_START=1 -> attempting to chown ${MLFLOW_DIR} to ${USER_UID}:${USER_GID}"
      chown -R ${USER_UID}:${USER_GID} "${MLFLOW_DIR}" || echo "⚠️ chown failed (continuing)"
    fi

    if [ "${SKIP_TRAIN}" = "1" ]; then
      echo "ℹ️ SKIP_TRAIN=1 -> Skipping ingestion/preprocess/model steps."
    else
      if [ -f "${FLAG_FILE}" ]; then
        echo "ℹ️ Found ${FLAG_FILE}, skipping model training."
      else
        echo "🏗️ Running ingestion..."
        python -u src/ingestion.py || { echo "❌ Ingestion failed"; exit 1; }
        echo "🧹 Running preprocessing..."
        python -u src/preprocess.py || { echo "❌ Preprocessing failed"; exit 1; }
        echo "🤖 Training model..."
        python -u src/model.py || { echo "❌ Model training failed"; exit 1; }
        touch "${FLAG_FILE}"
        echo "✅ Model trained and flagged at ${FLAG_FILE}"
      fi
    fi

    # Start MLflow and wait
    if ! start_mlflow_and_wait; then
      echo "⚠️ Aborting startup due to MLflow readiness failure." >&2
      exit 1
    fi

    echo "🚀 Starting FastAPI app on port ${API_PORT}..."

    # Run FastAPI (drop to unprivileged user optionally)
    if [ "${DROP_TO_USER}" = "1" ]; then
      echo "➡️ DROP_TO_USER=1 -> dropping to user ${USERNAME} (uid:${USER_UID}) to run uvicorn"
      exec su -s /bin/bash -c "exec uvicorn src.app:app --host 0.0.0.0 --port \"${API_PORT}\"" ${USERNAME}
    else
      exec uvicorn src.app:app --host 0.0.0.0 --port "${API_PORT}"
    fi
    ;;

  *)
    echo "❌ Unknown command: ${cmd}"
    exit 2
    ;;
esac
EOF

RUN chmod +x /usr/local/bin/fullpipeline-entrypoint.sh

WORKDIR /app

ENTRYPOINT ["/usr/local/bin/fullpipeline-entrypoint.sh"]
CMD ["start"]
