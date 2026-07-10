#!/usr/bin/env bash
# ============================================================================
# Build the unified evaluator image (Ollama judge + Python evaluator).
#
# The Dockerfile lives under docker/ but COPYs paths relative to this
# directory (evaluate.py, ground_truth/, docker/*), so the build context is
# the repository root (this script's directory), not docker/.
# ============================================================================

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-chimera-evaluator:latest}"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  --file "${SCRIPT_DIR}/docker/Dockerfile" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1
