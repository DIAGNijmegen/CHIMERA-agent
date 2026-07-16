#!/usr/bin/env bash
# ============================================================================
# Single-container entrypoint
#
# 1. Start the embedded Ollama server in the background (using OLLAMA_MODELS
#    which is mounted at /models — weights persist across runs).
# 2. Wait for the API to be ready.
# 3. If the rationale judge is enabled, ensure JUDGE_MODEL is present in the
#    model store; pull it if missing and ALLOW_MODEL_PULL=1, otherwise fail
#    with a clear message (this is what Grand-Challenge --network none runs
#    will hit when the weights mount is empty).
# 4. Run the evaluator.
# 5. Shut Ollama down cleanly and propagate the evaluator exit code.
#
# Runtime paths (Grand-Challenge layout):
#   /input/                          read-only predictions.json + pk_hash_to_case_map.json
#   /opt/ml/input/data/ground_truth/ read-only pathologist responses + section mapping
#   /output/                         writable results
#   /models/                         Ollama weight store (judge model)
# ============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Resolve mapping file: prefer /ground_truth/, fall back to the in-image default.
# --------------------------------------------------------------------------- #
if [[ -n "${SECTION_MAPPING_FILE:-}" && ! -f "${SECTION_MAPPING_FILE}" ]]; then
    if [[ -f /opt/app/defaults/section_variable_mapping.json ]]; then
        echo "[entrypoint] ${SECTION_MAPPING_FILE} not found — using bundled default."
        export SECTION_MAPPING_FILE=/opt/app/defaults/section_variable_mapping.json
    fi
fi

# --------------------------------------------------------------------------- #
# Decide whether we need Ollama at all.
# --------------------------------------------------------------------------- #
NEED_OLLAMA=1
if [[ "${USE_RATIONALE_JUDGE:-1}" == "0" ]]; then
    NEED_OLLAMA=0
    echo "[entrypoint] USE_RATIONALE_JUDGE=0 — skipping Ollama startup."
fi

OLLAMA_PID=""

cleanup() {
    if [[ -n "${OLLAMA_PID}" ]] && kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        echo "[entrypoint] Stopping Ollama (pid ${OLLAMA_PID})..."
        kill "${OLLAMA_PID}" 2>/dev/null || true
        # Wait up to 15 s for graceful shutdown, then force-kill.
        for _i in $(seq 1 15); do
            kill -0 "${OLLAMA_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${OLLAMA_PID}" 2>/dev/null || true
        wait "${OLLAMA_PID}" 2>/dev/null || true
        echo "[entrypoint] Ollama stopped."
    fi
}
trap cleanup EXIT INT TERM

if [[ "${NEED_OLLAMA}" == "1" ]]; then
    mkdir -p "${OLLAMA_MODELS:-/models}"

    echo "[entrypoint] Starting Ollama server"
    echo "             OLLAMA_HOST=${OLLAMA_HOST}"
    echo "             OLLAMA_MODELS=${OLLAMA_MODELS}"
    echo "             NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
    # Use process substitution so $! captures ollama's PID, not tee's.
    # (With a plain pipe `cmd | tee &`, $! is tee's PID and kill leaves
    # ollama running, causing the container to hang on exit.)
    ollama serve > >(tee /tmp/ollama.log) 2>&1 &
    OLLAMA_PID=$!

    echo "[entrypoint] Waiting for Ollama API on ${OLLAMA_BASE_URL}..."
    UP=0
    for _ in $(seq 1 60); do
        if curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
            UP=1
            break
        fi
        sleep 1
    done
    if [[ "${UP}" != "1" ]]; then
        echo "[entrypoint] ERROR: Ollama did not become ready in 60s." >&2
        echo "----- /tmp/ollama.log -----" >&2
        tail -n 50 /tmp/ollama.log >&2 || true
        exit 1
    fi
    echo "[entrypoint] Ollama is up."

    # ----------------------------------------------------------------------- #
    # Ensure the judge model is available locally.
    # ----------------------------------------------------------------------- #
    if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${JUDGE_MODEL}"; then
        echo "[entrypoint] Judge model '${JUDGE_MODEL}' is present in ${OLLAMA_MODELS}."
    else
        if [[ "${ALLOW_MODEL_PULL:-1}" == "1" ]]; then
            echo "[entrypoint] Judge model '${JUDGE_MODEL}' not found — pulling (requires network)..."
            ollama pull "${JUDGE_MODEL}"
        else
            echo "[entrypoint] ERROR: judge model '${JUDGE_MODEL}' is missing from ${OLLAMA_MODELS}" >&2
            echo "             and ALLOW_MODEL_PULL=0 (offline mode)." >&2
            echo "             Mount a pre-populated Ollama model store at /models" >&2
            echo "             or rerun with -e ALLOW_MODEL_PULL=1." >&2
            exit 1
        fi
    fi
fi

# --------------------------------------------------------------------------- #
# Run the evaluator (default CMD) or any user-supplied command.
# --------------------------------------------------------------------------- #
mkdir -p "${EVAL_OUTPUT_DIR:-/output}"

echo "[entrypoint] Launching: $*"
set +e
"$@"
RC=$?
set -e

echo "[entrypoint] Evaluator exited with code ${RC}."
exit ${RC}
