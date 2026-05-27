# ============================================================================
# LLM-Biopsy-Decision Evaluator — top-level convenience wrapper
#
# The whole pipeline (Ollama judge + Python evaluator) lives in a SINGLE
# Docker image so it can be submitted to Grand Challenge
# (https://grand-challenge.org/) which runs:
#
#     docker run --rm --gpus all \
#         -v <input>:/input:ro -v <output>:/output [-v <models>:/models] \
#         biopsy-evaluator:latest
#
# Prerequisites:
#   - Docker engine (your user in the docker group)
#   - NVIDIA Container Toolkit, e.g.:
#       sudo nvidia-ctk runtime configure --runtime=docker
#       sudo systemctl restart docker
#
# Quick start:
#   cp .env.example .env        # edit GPU_DEVICE_ID / JUDGE_MODEL if needed
#   make run
#
# Targets:
#   make build       — build the unified evaluator image
#   make run         — build + run the evaluation (default)
#   make shell       — open a bash shell inside the container
#   make pull-model  — pre-populate ./models/ with the judge weights
#                      (handy before a Grand-Challenge submission so the
#                       offline run does not need network access)
#   make down        — stop / remove the container
#   make clean       — remove the built image
#   make help        — show this list
# ============================================================================

.PHONY: build run shell pull-model down clean help

# ── Identity ─────────────────────────────────────────────────────────────────
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

# ── Load .env if it exists ───────────────────────────────────────────────────
-include .env

# ── Resolve mount paths to absolute (compose handles relative too, but
#    keeping them absolute avoids surprises across shells / CI) ─────────────
INPUT_DIR         ?= $(CURDIR)/mimic_datasets
OUTPUT_DIR        ?= $(CURDIR)/results
OLLAMA_MODELS_DIR ?= $(CURDIR)/models
export INPUT_DIR         := $(abspath $(INPUT_DIR))
export OUTPUT_DIR        := $(abspath $(OUTPUT_DIR))
export OLLAMA_MODELS_DIR := $(abspath $(OLLAMA_MODELS_DIR))

JUDGE_MODEL       ?= gemma4:e4b
export JUDGE_MODEL

COMPOSE := docker-compose -f docker/docker-compose.yml
IMAGE   := biopsy-evaluator:latest

# ── Targets ──────────────────────────────────────────────────────────────────

help:  ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## Build the unified evaluator Docker image
	@mkdir -p "$(OUTPUT_DIR)" "$(OLLAMA_MODELS_DIR)"
	$(COMPOSE) build evaluator

run: build  ## Build + run the full evaluation  [default]
	@mkdir -p "$(OUTPUT_DIR)" "$(OLLAMA_MODELS_DIR)"
	$(COMPOSE) run --rm evaluator

shell: build  ## Open a bash shell in the container (Ollama is NOT auto-started)
	$(COMPOSE) run --rm --entrypoint bash evaluator

pull-model: build  ## Pre-pull JUDGE_MODEL into $(OLLAMA_MODELS_DIR) for offline runs
	@mkdir -p "$(OLLAMA_MODELS_DIR)"
	@echo "Pulling $(JUDGE_MODEL) into $(OLLAMA_MODELS_DIR) ..."
	$(COMPOSE) run --rm --entrypoint bash evaluator -c '\
	    ollama serve >/tmp/ollama.log 2>&1 & \
	    for i in $$(seq 1 30); do curl -fsS http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done; \
	    ollama pull $(JUDGE_MODEL)'

down:  ## Stop and remove the container
	$(COMPOSE) down

clean: down  ## Remove the built image
	-docker image rm $(IMAGE)
