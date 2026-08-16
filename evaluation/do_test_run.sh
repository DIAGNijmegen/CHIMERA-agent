#!/usr/bin/env bash
# ============================================================================
# Build + run one complete phase evaluation locally, mirroring how Grand
# Challenge invokes the evaluation container.
#
#   /input                          <- ./test/input          (read-only)
#   /opt/ml/input/data/ground_truth <- ./ground_truth        (read-only)
#   /output                         <- ./results             (writable)
#   /tmp                            <- a throwaway volume    (GC does the same)
#
# The run is offline (`--network none`) exactly like Grand Challenge. The judge
# model is baked into the image at build time, so no weights mount is needed.
#
# Usage:
#   ./do_test_run.sh
#   GPU_DEVICE_ID=3 ./do_test_run.sh
#
# Config (env or ./.env):
#   GPU_DEVICE_ID        host GPU index to expose       (default: 0)
#   JUDGE_MODEL          Ollama judge model             (default: gemma4:e4b)
#   USE_RATIONALE_JUDGE  0 = deterministic, no GPU/LLM  (default: 1)
#   ALLOW_MODEL_PULL     1 = allow runtime pull         (default: 0, offline)
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Optionally load local overrides (GPU_DEVICE_ID, JUDGE_MODEL, ...).
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-chimera-evaluator:latest}"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG%%:*}-volume"

# ── Configuration ────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  echo "ERROR: this evaluator scores the complete phase in one run; no task arguments are accepted" >&2
  exit 1
fi
GPU_DEVICE_ID="${GPU_DEVICE_ID:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma4:e4b}"
USE_RATIONALE_JUDGE="${USE_RATIONALE_JUDGE:-1}"
ALLOW_MODEL_PULL="${ALLOW_MODEL_PULL:-0}"

INPUT_DIR="${SCRIPT_DIR}/test/input"
GROUND_TRUTH_DIR="${SCRIPT_DIR}/ground_truth"
RESULTS_DIR="${SCRIPT_DIR}/results"


# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -f "${INPUT_DIR}/predictions.json" ]]; then
  echo "ERROR: no predictions.json found at ${INPUT_DIR}/predictions.json" >&2
  exit 1
fi
if [[ ! -f "${GROUND_TRUTH_DIR}/debug_archive_pks.csv" ]]; then
  echo "ERROR: no case map found at ${GROUND_TRUTH_DIR}/debug_archive_pks.csv" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output.
    # This allows the host user (e.g. you) to access and handle these files.
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${RESULTS_DIR}":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# This allows for the Docker user to read
chmod -R -f o+rX "${INPUT_DIR}" "${GROUND_TRUTH_DIR}"
chmod -f o+rwX "${RESULTS_DIR}"

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

# GPU flag — only request a GPU when the judge is enabled.
GPU_ARGS=()
if [[ "${USE_RATIONALE_JUDGE}" == "1" ]]; then
  GPU_ARGS=(--gpus "device=${GPU_DEVICE_ID}")
fi

echo "=+= Cleaning up earlier phase output"
docker run --rm \
  --platform=linux/amd64 \
  --quiet \
  --volume "${RESULTS_DIR}":/output \
  --entrypoint /bin/sh \
  "$DOCKER_IMAGE_TAG" \
  -c "rm -rf /output/* || true"

echo "=+= Evaluating complete phase (judge=${USE_RATIONALE_JUDGE}, gpu=${GPU_DEVICE_ID})"
docker run --rm \
  "${GPU_ARGS[@]}" \
  --platform=linux/amd64 \
  --network none \
  --volume "${INPUT_DIR}":/input:ro \
  --volume "${GROUND_TRUTH_DIR}":/opt/ml/input/data/ground_truth:ro \
  --volume "${RESULTS_DIR}":/output \
  --volume "${DOCKER_NOOP_VOLUME}":/tmp \
  --env GROUND_TRUTH_DIR="/opt/ml/input/data/ground_truth" \
  --env JUDGE_MODEL="${JUDGE_MODEL}" \
  --env USE_RATIONALE_JUDGE="${USE_RATIONALE_JUDGE}" \
  --env ALLOW_MODEL_PULL="${ALLOW_MODEL_PULL}" \
  "$DOCKER_IMAGE_TAG"

echo "=+= Wrote Grand Challenge metrics to ${RESULTS_DIR}/metrics.json"

echo "=+= Save this image for uploading via ./do_save.sh"
