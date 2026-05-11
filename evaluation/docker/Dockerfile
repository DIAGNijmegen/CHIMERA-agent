# Lightweight evaluator image. No torch / no CUDA needed here — the GPU work
# happens in the separate `ollama` service. This keeps the image tiny and
# fast to build.
FROM python:3.11-slim

# graphviz binary is needed by the python `graphviz` package to render PNGs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Use uv for fast, reproducible installs.
RUN pip install --no-cache-dir --upgrade pip uv

WORKDIR /app

# Install python deps first so they cache between code edits.
COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

# DeepEval still pings telemetry / OpenAI on import; satisfy both with a stub.
ENV OPENAI_API_KEY="dummy-key-not-used"
ENV DEEPEVAL_TELEMETRY_OPT_OUT="YES"
ENV PYTHONUNBUFFERED=1

# Default command runs the evaluation; override via `docker compose run eval ...`.
CMD ["python", "/app/evaluate.py"]
