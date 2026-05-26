# ============================================================================
# Single-container LLM-Biopsy-Decision Evaluator
#
# Bundles the Ollama judge runtime AND the Python evaluator in one image so the
# whole pipeline runs from a single `docker run ...` invocation — as required
# by the Grand Challenge platform (https://grand-challenge.org/).
#
# Runtime contract (Grand-Challenge friendly):
#   /input/         (read-only)   target.json, evaluation_object.json,
#                                  section_variable_mapping.json
#   /output/        (writable)    per_case_results.csv, aggregate_metrics.json,
#                                  evaluation_results_summary.json
#   /models/        (read/write)  Ollama model store (weights for the judge LLM)
#
# GPU access requires the NVIDIA Container Toolkit on the host
# (`docker run --gpus all ...`).  CPU-only runs work too if
# USE_RATIONALE_JUDGE=0.
#
# Base: ollama/ollama:latest — which is Ubuntu 22.04 + full CUDA runtime.
# We install Python on top.  Using this as the base (rather than copying the
# binary into python:3.11-slim) ensures all CUDA shared libraries that Ollama
# needs at runtime are present.
# ============================================================================

FROM ollama/ollama:latest

# ── System dependencies ─────────────────────────────────────────────────────
#   python3 / pip : evaluator runtime
#   graphviz      : needed by the python `graphviz` package to render PNGs
#   curl          : used by the entrypoint Ollama health-check
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        graphviz \
        python3 \
        python3-venv \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────────
# Use a venv to avoid conflicts with Debian-managed pip/Python packages.
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Put the venv on PATH so `python` and `pip` resolve to it everywhere.
ENV PATH="/opt/venv/bin:$PATH"

# ── Application code ────────────────────────────────────────────────────────
WORKDIR /opt/app
COPY evaluate.py            /opt/app/evaluate.py
COPY docker/entrypoint.sh   /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

# Default copy of the section→variable mapping so the image is usable even if
# /input only contains the two case JSONs.  Anything mounted at /input
# overrides it via the env vars below.
COPY mimic_datasets/section_variable_mapping.json \
     /opt/app/defaults/section_variable_mapping.json

# ── Grand-Challenge style mount points ──────────────────────────────────────
RUN mkdir -p /input /output /models \
 && chmod 0777 /output /models

# ── Default environment ─────────────────────────────────────────────────────
# Ollama --------------------------------------------------------------------
ENV OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/models \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_BASE_URL=http://127.0.0.1:11434
# Judge --------------------------------------------------------------------
ENV JUDGE_MODEL=gemma4:e4b \
    USE_RATIONALE_JUDGE=1 \
    ALLOW_MODEL_PULL=1
# Evaluator I/O (point at Grand-Challenge mounts by default) ---------------
ENV TARGET_FILE=/input/target.json \
    EVAL_FILE=/input/evaluation_object.json \
    SECTION_MAPPING_FILE=/input/section_variable_mapping.json \
    EVAL_OUTPUT_DIR=/output
# DeepEval / misc ----------------------------------------------------------
ENV OPENAI_API_KEY=dummy-key-not-used \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

ENTRYPOINT ["/opt/app/entrypoint.sh"]
CMD ["python", "/opt/app/evaluate.py"]


# ── Python dependencies ─────────────────────────────────────────────────────
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# ── Application code ────────────────────────────────────────────────────────
WORKDIR /opt/app
COPY evaluate.py            /opt/app/evaluate.py
COPY docker/entrypoint.sh   /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

# Default copy of the section→variable mapping so the image is usable even if
# /input only contains the two case JSONs.  Anything mounted at /input
# overrides it via the env vars below.
COPY mimic_datasets/section_variable_mapping.json \
     /opt/app/defaults/section_variable_mapping.json

# ── Grand-Challenge style mount points ──────────────────────────────────────
RUN mkdir -p /input /output /models \
 && chmod 0777 /output /models

# ── Default environment ─────────────────────────────────────────────────────
# Ollama --------------------------------------------------------------------
ENV OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/models \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_BASE_URL=http://127.0.0.1:11434
# Judge --------------------------------------------------------------------
ENV JUDGE_MODEL=gemma4:e4b \
    USE_RATIONALE_JUDGE=1 \
    ALLOW_MODEL_PULL=1
# Evaluator I/O (point at Grand-Challenge mounts by default) ---------------
ENV TARGET_FILE=/input/target.json \
    EVAL_FILE=/input/evaluation_object.json \
    SECTION_MAPPING_FILE=/input/section_variable_mapping.json \
    EVAL_OUTPUT_DIR=/output
# DeepEval / misc ----------------------------------------------------------
ENV OPENAI_API_KEY=dummy-key-not-used \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

ENTRYPOINT ["/opt/app/entrypoint.sh"]
CMD ["python", "/opt/app/evaluate.py"]
