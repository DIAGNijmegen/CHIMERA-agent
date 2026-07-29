#!/usr/bin/env bash
# ============================================================================
# Build + run one or more evaluation tasks locally, mirroring how Grand
# Challenge invokes the evaluation container.
#
#   /input                          <- ./test/input          (read-only)
#   /opt/ml/input/data/ground_truth <- ./ground_truth        (read-only)
#   /output                         <- ./results/<TASK_ID>   (writable)
#   /tmp                            <- a throwaway volume    (GC does the same)
#
# The run is offline (`--network none`) exactly like Grand Challenge. The judge
# model is baked into the image at build time, so no weights mount is needed.
#
# Usage:
#   ./do_test_run.sh                 # TASK_ID=task1 on GPU_DEVICE_ID=0
#   TASK_ID=task2 GPU_DEVICE_ID=1 ./do_test_run.sh
#   ./do_test_run.sh task2           # positional TASK_ID override
#   ./do_test_run.sh task1 task2 task3   # run all three tasks sequentially
#   GPU_DEVICE_ID=3 ./do_test_run.sh task1 task2 task3   # all, on GPU 3
#
# Config (env or ./.env):
#   TASK_ID              task directory to evaluate     (default: task1)
#   GPU_DEVICE_ID        host GPU index to expose       (default: 0)
#   JUDGE_MODEL          Ollama judge model             (default: gemma4:e4b)
#   USE_RATIONALE_JUDGE  0 = deterministic, no GPU/LLM  (default: 1)
#   ALLOW_MODEL_PULL     1 = allow runtime pull         (default: 0, offline)
# ============================================================================

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Optionally load local overrides (GPU_DEVICE_ID, JUDGE_MODEL, ...).
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-chimera-evaluator:latest}"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG%%:*}-volume"

# ── Configuration ────────────────────────────────────────────────────────────
# Tasks: positional args win (one or more), else $TASK_ID, else task1.
#   ./do_test_run.sh                 -> task1
#   ./do_test_run.sh task2           -> task2
#   ./do_test_run.sh task1 task2     -> task1 then task2 (sequential)
if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=("${TASK_ID:-task1}")
fi
GPU_DEVICE_ID="${GPU_DEVICE_ID:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma4:e4b}"
USE_RATIONALE_JUDGE="${USE_RATIONALE_JUDGE:-1}"
ALLOW_MODEL_PULL="${ALLOW_MODEL_PULL:-0}"

INPUT_DIR="${SCRIPT_DIR}/test/input"
GROUND_TRUTH_DIR="${SCRIPT_DIR}/ground_truth"
RESULTS_DIR="${SCRIPT_DIR}/results"


# ── Sanity checks (validate every task up front) ─────────────────────────
if [[ ! -f "${INPUT_DIR}/predictions.json" ]]; then
  echo "ERROR: no predictions.json found at ${INPUT_DIR}/predictions.json" >&2
  exit 1
fi
for TASK_ID in "${TASKS[@]}"; do
  if [[ ! -d "${GROUND_TRUTH_DIR}/${TASK_ID}" ]]; then
    echo "ERROR: no ground truth found at ${GROUND_TRUTH_DIR}/${TASK_ID}" >&2
    exit 1
  fi
done

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

# ── Run each task sequentially (image is built once) ───────────────────────
for TASK_ID in "${TASKS[@]}"; do
  OUTPUT_DIR="${RESULTS_DIR}/${TASK_ID}"
  mkdir -p "${OUTPUT_DIR}"
  chmod -f o+rwX "${OUTPUT_DIR}"

  echo "=+= Cleaning up any earlier output for ${TASK_ID}"
  # Use the container itself to circumvent ownership problems
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "rm -rf /output/* || true"

  echo "=+= Evaluating ${TASK_ID} (judge=${USE_RATIONALE_JUDGE}, gpu=${GPU_DEVICE_ID})"
  ## Note the extra arguments that are passed here:
  # '--network none'
  #    entails there is no internet connection, exactly like Grand Challenge.
  #    The judge weights are baked into the image, so nothing needs to be pulled.
  # '--gpus device=N'
  #    enables access to the selected GPU (Grand Challenge uses '--gpus all')
  # '--volume <NAME>:/tmp'
  #    is added because on Grand Challenge this directory cannot be used to
  #    store permanent files
  # '--volume ./ground_truth:/opt/ml/input/data/ground_truth:ro'
  #    is added to provide access to the tarball-upload locally
  docker run --rm \
      "${GPU_ARGS[@]}" \
      --platform=linux/amd64 \
      --network none \
      --volume "${INPUT_DIR}":/input:ro \
      --volume "${GROUND_TRUTH_DIR}":/opt/ml/input/data/ground_truth:ro \
      --volume "${OUTPUT_DIR}":/output \
      --volume "${DOCKER_NOOP_VOLUME}":/tmp \
      --env TASK_ID="${TASK_ID}" \
      --env GROUND_TRUTH_DIR="/opt/ml/input/data/ground_truth/${TASK_ID}" \
      --env JUDGE_MODEL="${JUDGE_MODEL}" \
      --env USE_RATIONALE_JUDGE="${USE_RATIONALE_JUDGE}" \
      --env ALLOW_MODEL_PULL="${ALLOW_MODEL_PULL}" \
      "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}"
done

echo "=+= Save this image for uploading via ./do_save.sh"
