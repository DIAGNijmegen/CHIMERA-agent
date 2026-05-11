# ============================================================================
# LLM-Evaluator — top-level convenience wrapper
#
# Prerequisites:
#   - Docker engine (your user in the docker group)
#   - NVIDIA Container Toolkit with the nvidia runtime registered, e.g.:
#       sudo nvidia-ctk runtime configure --runtime=docker
#       sudo systemctl restart docker
#
# Quick start:
#   cp .env.example .env        # edit GPU_DEVICE_ID and JUDGE_MODEL if needed
#   make run
#
# Targets:
#   make build   — (re)build the eval image
#   make run     — build + start ollama + run evaluation             [default]
#   make shell   — drop into the eval container for debugging
#   make ollama  — start only the ollama GPU service in the background
#   make down    — stop and remove all containers
#   make logs    — tail ollama logs
#   make help    — show this list
# ============================================================================

.PHONY: build run shell ollama down logs help

# ── Identity ─────────────────────────────────────────────────────────────────
# Pass host UID/GID into the compose file so result files are owned by you,
# not root.  These are always derived from the running user; never override them.
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

# ── Load .env if it exists ────────────────────────────────────────────────────
# Copy .env.example -> .env and customise.  Variables set here override the
# defaults below but are overridden by anything already in the shell environment.
-include .env

# ── Model cache path (always resolved to an absolute path) ───────────────────
# Default: ./models/ next to this Makefile (created on first `make run`).
# Override in .env:  OLLAMA_MODELS_DIR=/some/shared/path
OLLAMA_MODELS_DIR ?= $(CURDIR)/models
export OLLAMA_MODELS_DIR := $(abspath $(OLLAMA_MODELS_DIR))

# ── Compose shortcut ─────────────────────────────────────────────────────────
COMPOSE := docker compose -f docker/docker-compose.yml

# ── Targets ──────────────────────────────────────────────────────────────────

help:  ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## (Re)build the eval Docker image
	$(COMPOSE) build eval

ollama:  ## Start the Ollama GPU container in the background
	@mkdir -p "$(OLLAMA_MODELS_DIR)"
	$(COMPOSE) up -d ollama

run: build ollama  ## Build + start Ollama + run the full evaluation  [default]
	$(COMPOSE) run --rm eval

shell:  ## Open a bash shell in the eval container (for debugging)
	$(COMPOSE) run --rm eval bash

down:  ## Stop and remove all containers for this project
	$(COMPOSE) down

logs:  ## Tail the Ollama container logs
	$(COMPOSE) logs -f ollama
