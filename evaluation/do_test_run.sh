#!/usr/bin/env bash
# ============================================================================
# Build + run one evaluation task locally, using the Grand-Challenge mount
# contract.
#
#   /input                          <- ./test/outputs        (read-only)
#   /opt/ml/input/data/ground_truth <- ./ground_truth        (read-only)
#   /output                         <- ./results/<TASK_ID>    (writable)
#   /models                         <- ./models              (Ollama weights)
#
# Usage:
#   ./do_test_run.sh                 # TASK_ID=task1 on GPU_DEVICE_ID=0
#   TASK_ID=task2 GPU_DEVICE_ID=1 ./do_test_run.sh
#   ./do_test_run.sh task2           # positional TASK_ID override
#   ./do_test_run.sh task1 task2     # run both tasks sequentially
#   GPU_DEVICE_ID=1 ./do_test_run.sh task1 task2   # both, on GPU 1
#
# Config (env or ./.env):
#   TASK_ID              task directory to evaluate     (default: task1)
#   GPU_DEVICE_ID        host GPU index to expose       (default: 0)
#   JUDGE_MODEL          Ollama judge model             (default: gemma4:e4b)
#   USE_RATIONALE_JUDGE  0 = deterministic, no GPU/LLM  (default: 1)
#   ALLOW_MODEL_PULL     0 = offline, fail if missing   (default: 1)
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
ALLOW_MODEL_PULL="${ALLOW_MODEL_PULL:-1}"

INPUT_DIR="${SCRIPT_DIR}/test/outputs"
GROUND_TRUTH_DIR="${SCRIPT_DIR}/ground_truth"
MODELS_DIR="${SCRIPT_DIR}/models"


# ── Sanity checks (validate every task up front) ─────────────────────────────
for TASK_ID in "${TASKS[@]}"; do
  if [[ ! -d "${INPUT_DIR}/${TASK_ID}" ]]; then
    echo "ERROR: no predictions found at ${INPUT_DIR}/${TASK_ID}" >&2
    exit 1
  fi
  if [[ ! -d "${GROUND_TRUTH_DIR}/${TASK_ID}" ]]; then
    echo "ERROR: no ground truth found at ${GROUND_TRUTH_DIR}/${TASK_ID}" >&2
    exit 1
  fi
done

mkdir -p "${MODELS_DIR}"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

# GPU flag — only request a GPU when the judge is enabled.
GPU_ARGS=()
if [[ "${USE_RATIONALE_JUDGE}" == "1" ]]; then
  GPU_ARGS=(--gpus "device=${GPU_DEVICE_ID}")
fi

# ── Run each task sequentially (image built once, model store reused) ────────
for TASK_ID in "${TASKS[@]}"; do
  OUTPUT_DIR="${SCRIPT_DIR}/results/${TASK_ID}"
  mkdir -p "${OUTPUT_DIR}"

  echo "=+= Evaluating ${TASK_ID} (judge=${USE_RATIONALE_JUDGE}, gpu=${GPU_DEVICE_ID})"
  # Notes:
  #  - Run as the host user so files under /output land with your ownership.
  #  - /etc/passwd + /etc/group are mounted read-only so the in-container UID
  #    resolves to a real username (deepeval / graphviz caches).
  #  - Grand Challenge runs offline (`--network none`). We keep the network on
  #    locally so the judge model can be pulled on first run; set
  #    ALLOW_MODEL_PULL=0 (with a pre-populated ./models) to mimic the GC run.
  docker run --rm \
      "${GPU_ARGS[@]}" \
      --platform=linux/amd64 \
      --user "$(id -u):$(id -g)" \
      --volume "${INPUT_DIR}":/input:ro \
      --volume "${GROUND_TRUTH_DIR}":/opt/ml/input/data/ground_truth:ro \
      --volume "${OUTPUT_DIR}":/output \
      --volume "${MODELS_DIR}":/models \
      --volume /etc/passwd:/etc/passwd:ro \
      --volume /etc/group:/etc/group:ro \
      --env HOME=/tmp \
      --env TASK_ID="${TASK_ID}" \
      --env GROUND_TRUTH_DIR="/opt/ml/input/data/ground_truth/${TASK_ID}" \
      --env TEST_OUTPUTS_DIR="/input/${TASK_ID}" \
      --env SECTION_MAPPING_FILE=/opt/ml/input/data/ground_truth/section_variable_mapping.json \
      --env EVAL_OUTPUT_DIR=/output \
      --env JUDGE_MODEL="${JUDGE_MODEL}" \
      --env USE_RATIONALE_JUDGE="${USE_RATIONALE_JUDGE}" \
      --env ALLOW_MODEL_PULL="${ALLOW_MODEL_PULL}" \
      "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}"
done
