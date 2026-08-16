# ============================================================================
# Single-container LLM-Biopsy-Decision Evaluator
#
# Bundles the Ollama judge runtime AND the Python evaluator in one image so the
# whole pipeline runs from a single `docker run ...` invocation — as required
# by the Grand Challenge platform (https://grand-challenge.org/).
#
# Runtime contract (Grand-Challenge layout — see external/example_evaluation_method):
#   /input/                          (read-only)  predictions.json
#   /opt/ml/input/data/ground_truth/ (read-only)  taskN/case_id/<decision>.json
#                                                 taskN/case_id/<decision>-reasoning.json
#                                                 taskN/case_id/prostate-time-to-recurrence-or-last-follow-up.json
#                                                 section_variable_mapping.json
#   /output/                         (writable)   metrics.json, per_case_results.csv,
#                                                 aggregate_metrics.json,
#                                                 evaluation_results_summary.json
#
# The container runs as the unprivileged user `user`. Grand Challenge rejects
# images whose final USER is root.
#
# The judge LLM weights are baked into the image at /opt/ollama-models, so the
# container needs no network and no weights mount. Grand Challenge runs
# evaluation containers with `--network none`, which makes this mandatory.
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

# ── Unprivileged runtime user ───────────────────────────────────────────────
# Created early so the application COPYs below can set ownership directly.
# The switch to USER happens at the very end, after every step that needs root.
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

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
COPY --chown=user:user evaluate.py            /opt/app/evaluate.py
COPY --chown=user:user docker/entrypoint.sh   /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

# Default copy of the section→variable mapping so the image can score section
# grounding even when the mapping is not mounted separately.
COPY --chown=user:user ground_truth/section_variable_mapping.json \
     /opt/app/defaults/section_variable_mapping.json

# ── Runtime mount points (Grand-Challenge layout) ───────────────────────────
RUN mkdir -p /input /opt/ml/input/data/ground_truth /output \
 && chmod 0777 /output

# ── Judge model (baked into the image) ──────────────────────────────────────
# Pull the judge weights at build time into a directory that is NOT a runtime
# mount point, so no bind mount can shadow them. Override the tag with:
#   docker build --build-arg JUDGE_MODEL=<tag> ...
# The build fails loudly if the tag cannot be resolved.
ARG JUDGE_MODEL=gemma4:e4b
ENV OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/opt/ollama-models
RUN set -eu; \
    mkdir -p "${OLLAMA_MODELS}"; \
    ollama serve >/tmp/ollama-build.log 2>&1 & \
    OLLAMA_PID=$!; \
    for _ in $(seq 1 60); do \
        if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then break; fi; \
        sleep 1; \
    done; \
    if ! curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then \
        echo "ERROR: Ollama did not start during build" >&2; \
        cat /tmp/ollama-build.log >&2; \
        exit 1; \
    fi; \
    echo "Baking judge model ${JUDGE_MODEL} into the image..."; \
    ollama pull "${JUDGE_MODEL}"; \
    kill "${OLLAMA_PID}" 2>/dev/null || true; \
    wait "${OLLAMA_PID}" 2>/dev/null || true; \
    rm -f /tmp/ollama-build.log; \
    chmod -R a+rX "${OLLAMA_MODELS}"; \
    chown user:user "${OLLAMA_MODELS}"

# ── Default environment ─────────────────────────────────────────────────────
# Ollama --------------------------------------------------------------------
ENV OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_BASE_URL=http://127.0.0.1:11434
# Judge --------------------------------------------------------------------
# ALLOW_MODEL_PULL=0 by default: the model is already in the image, and Grand
# Challenge runs offline. Set it to 1 only when deliberately using another tag.
ENV JUDGE_MODEL=${JUDGE_MODEL} \
    USE_RATIONALE_JUDGE=1 \
    ALLOW_MODEL_PULL=0
# Evaluator I/O ------------------------------------------------------------
ENV GROUND_TRUTH_DIR=/opt/ml/input/data/ground_truth \
    INPUT_DIRECTORY=/input \
    PREDICTIONS_FILE=/input/predictions.json \
    SECTION_MAPPING_FILE=/opt/ml/input/data/ground_truth/section_variable_mapping.json \
    EVAL_OUTPUT_DIR=/output
# DeepEval / misc ----------------------------------------------------------
ENV OPENAI_API_KEY=dummy-key-not-used \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

# ── Drop root ───────────────────────────────────────────────────────────────
# Everything above needs root (apt, venv, baking the judge weights); nothing
# below does. The working directory moves to /tmp because DeepEval writes its
# .deepeval cache into the current directory, and on Grand Challenge /tmp and
# /output are the only writable paths. HOME is already /tmp, which is where
# Ollama keeps its runtime key.
WORKDIR /tmp
USER user

ENTRYPOINT ["/opt/app/entrypoint.sh"]
CMD ["python", "/opt/app/evaluate.py"]
