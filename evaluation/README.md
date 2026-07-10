# CHIMERA Evaluation Pipeline

This directory contains the Grand-Challenge-style evaluator for CHIMERA agent
outputs. It compares per-case agent predictions against pathologist ground truth
for Task 1 biopsy decisions and Task 2 treatment decisions, then writes both the
Grand Challenge ranking file and local debugging reports.

The evaluator runs as a single Docker image that contains:

- the Python scoring pipeline in [evaluate.py](evaluate.py)
- an embedded Ollama server for the optional rationale judge
- all Python dependencies needed for deterministic metrics and LLM judging



## Repository Layout

```text
evaluation/
├── evaluate.py                       # main deterministic + optional LLM evaluator
├── do_build.sh                       # builds chimera-evaluator:latest
├── do_test_run.sh                    # builds once, then runs one or more tasks locally
├── do_save.sh                        # saves the image and packs ground_truth.tar.gz
├── .env.example                      # optional local configuration template
│
├── docker/
│   ├── Dockerfile                    # Ollama base image + Python evaluator runtime
│   ├── entrypoint.sh                 # starts Ollama, checks model, runs evaluate.py
│   └── requirements.txt              # Python dependencies
│
├── ground_truth/
│   ├── section_variable_mapping.json
│   ├── task1/<case_id>/pathologist_response.json
│   └── task2/<case_id>/pathologist_response.json
│
├── test/outputs/
│   ├── task1/<case_id>/prediction.json
│   └── task2/<case_id>/prediction.json
│
├── results/                          # generated local outputs, gitignored
├── models/                           # generated Ollama model cache, gitignored
├── pathologist_forms/                # HTML form templates

```

## Container Contract

The image mirrors the Grand Challenge evaluation-method layout while preserving
the CHIMERA per-case JSON schema.

| Mount | Mode | Contents |
| --- | --- | --- |
| `/input/` | read-only | `taskN/<case_id>/prediction.json` agent outputs |
| `/opt/ml/input/data/ground_truth/` | read-only | `taskN/<case_id>/pathologist_response.json` plus `section_variable_mapping.json` |
| `/output/` | writable | `metrics.json`, `aggregate_metrics.json`, `per_case_results.csv`, `evaluation_results_summary.json` |
| `/models/` | read/write | Ollama model store used by the rationale judge |

Local runs bind-mount [test/outputs](test/outputs) to `/input` and
[ground_truth](ground_truth) to `/opt/ml/input/data/ground_truth`. The directory
structure under those folders remains the original CHIMERA structure.

## Prerequisites

- Docker Engine with permission to run containers.
- NVIDIA driver and NVIDIA Container Toolkit for GPU judging.
- One GPU with enough VRAM for the judge model. The default `gemma4:e4b` needs
  roughly 10 GB of model storage and is intended for a 24 GB-class GPU.

For a deterministic smoke test with no GPU and no Ollama model, set
`USE_RATIONALE_JUDGE=0`.

## Quick Start

From this directory:

```bash
cd ~/CHIMERA-agent/evaluation

# Run Task 1 on the default GPU from .env or GPU_DEVICE_ID=0.
./do_test_run.sh task1

# Run Task 1 and Task 2 sequentially on GPU 1.
GPU_DEVICE_ID=1 ./do_test_run.sh task1 task2
```

`do_test_run.sh` builds `chimera-evaluator:latest` once, validates all requested
task directories up front, then runs each task in order. Results are written to
separate task directories:

```text
results/task1/
results/task2/
```

The first judge-enabled run may download `JUDGE_MODEL` into [models](models).
Subsequent runs reuse the same mounted model store.

## Configuration

You can either export variables inline or copy [.env.example](.env.example) to
`.env`. `do_test_run.sh` loads `.env` automatically if it exists.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOCKER_IMAGE_TAG` | `chimera-evaluator:latest` | Image tag used by all scripts |
| `TASK_ID` | `task1` | Task to run when no positional task args are provided |
| `GPU_DEVICE_ID` | `0` | Host GPU exposed to Docker when judging is enabled |
| `JUDGE_MODEL` | `gemma4:e4b` | Ollama model used by the rationale judge |
| `USE_RATIONALE_JUDGE` | `1` | `0` disables Ollama and runs deterministic metrics only |
| `ALLOW_MODEL_PULL` | `1` | `0` forbids runtime model download and fails if missing |

Examples:

```bash
# Deterministic, CPU-only smoke test.
USE_RATIONALE_JUDGE=0 ./do_test_run.sh task1

# Offline-style run after ./models has already been populated.
GPU_DEVICE_ID=1 ALLOW_MODEL_PULL=0 ./do_test_run.sh task1 task2

# Build under a custom image tag.
DOCKER_IMAGE_TAG=chimera-evaluator:dev ./do_build.sh
```

## Scripts

### Build Only

```bash
./do_build.sh
```

Builds `chimera-evaluator:latest` from [docker/Dockerfile](docker/Dockerfile)
using this directory as the build context.

### Local Test Run

```bash
./do_test_run.sh task1
./do_test_run.sh task1 task2
GPU_DEVICE_ID=1 ./do_test_run.sh task1 task2
```

For each requested task, the script mounts:

- [test/outputs](test/outputs) as `/input:ro`
- [ground_truth](ground_truth) as `/opt/ml/input/data/ground_truth:ro`
- `results/<task>` as `/output`
- [models](models) as `/models`

It also runs as the host user so generated files are not owned by root.

### Package for Grand Challenge

```bash
./do_save.sh
```

This rebuilds the image, writes an image tarball such as
`chimera-evaluator_<timestamp>.tar.gz`, and packs
`ground_truth.tar.gz` from [ground_truth](ground_truth).

Upload the image tarball as the evaluation method and upload
`ground_truth.tar.gz` separately as the phase ground truth. Grand Challenge
extracts that tarball to `/opt/ml/input/data/ground_truth/` at runtime.

## Manual Docker Run

The scripts are the preferred entry point, but this is the equivalent shape for
a single task on GPU 1:

```bash
mkdir -p results/task1 models

docker run --rm \
  --gpus "device=1" \
  --platform=linux/amd64 \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/test/outputs:/input:ro" \
  -v "$PWD/ground_truth:/opt/ml/input/data/ground_truth:ro" \
  -v "$PWD/results/task1:/output" \
  -v "$PWD/models:/models" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e HOME=/tmp \
  -e TASK_ID=task1 \
  -e GROUND_TRUTH_DIR=/opt/ml/input/data/ground_truth/task1 \
  -e TEST_OUTPUTS_DIR=/input/task1 \
  -e SECTION_MAPPING_FILE=/opt/ml/input/data/ground_truth/section_variable_mapping.json \
  -e EVAL_OUTPUT_DIR=/output \
  -e JUDGE_MODEL=gemma4:e4b \
  -e USE_RATIONALE_JUDGE=1 \
  -e ALLOW_MODEL_PULL=1 \
  chimera-evaluator:latest
```

For an offline check after the model exists in `/models`, add
`-e ALLOW_MODEL_PULL=0`. To disable the judge entirely, set
`-e USE_RATIONALE_JUDGE=0` and omit `--gpus`.

## Outputs

Each task writes these files under `results/<task>/` locally or `/output/` in
the container:

| File | Purpose |
| --- | --- |
| `metrics.json` | Grand-Challenge-facing ranking file with `aggregates` and per-case `results` |
| `aggregate_metrics.json` | Dataset-level metrics only |
| `per_case_results.csv` | Flat per-case table for quick inspection |
| `evaluation_results_summary.json` | Full run summary with aggregate metrics and per-case details |

## Evaluation Logic

### Stage 1: Decision Gate

The evaluator first matches predictions to ground truth by `case_id`.

For Task 1, the `biopsy_decision` must be valid (`yes` or `no`) and match the
pathologist response exactly. A mismatch receives `case_score = 0`.

For Task 2, `treatment_recommendation.primary` is scored across:

- `watchful_waiting`
- `active_surveillance`
- `continued_surveillance`
- `active_treatment`

Exact matches receive full decision credit. `active_surveillance` versus
`continued_surveillance` receives partial credit (`decision_score = 0.5`). Other
mismatches receive zero.

### Stage 2: Component Scores

Cases that pass the decision gate receive a weighted component score.

| Component | Method | Weight with judge | Weight without judge |
| --- | --- | ---: | ---: |
| Confidence | Ordinal distance over `uncertain`, `borderline`, `clear` | 0.20 | 0.225 |
| Variable weights | Mean ordinal error over `not_used`, `noted`, `important`, `decisive` | 0.25 | 0.275 |
| Important/decisive factors | Set F1 over variables marked `important` or `decisive` | 0.15 | 0.175 |
| Tool efficiency | Precision of agent revealed sections against pathologist revealed sections | 0.15 | 0.150 |
| Section grounding | Fraction of actively weighted variables grounded by revealed source sections | 0.15 | 0.175 |
| Rationale alignment | DeepEval GEval rubric via local Ollama | 0.10 | disabled |

If the rationale judge is disabled or unavailable, the non-rationale weights are
renormalized through the fixed no-judge weighting shown above.

### Tool-Use Policy

The tool score penalizes unnecessary reveals only:

```text
score = |agent_revealed ∩ pathologist_revealed| / |agent_revealed|
```

Missing pathologist reveals are not penalized. Extra agent reveals are
penalized uniformly.

### Section-Grounding Policy

Section grounding checks whether variables weighted above `not_used` are
supported by sections the agent actually revealed, using
[ground_truth/section_variable_mapping.json](ground_truth/section_variable_mapping.json).
Always-available variables such as `psa` and `age` are exempt.

## Key Aggregate Metrics

| Metric | Meaning |
| --- | --- |
| `final_DAG_score` | Mean case score across all cases, including gate failures |
| `decision_accuracy` | Mean decision score, including partial Task 2 credit |
| `exact_decision_accuracy` | Exact decision-match accuracy |
| `decision_f1_yes` | Task 1 positive-class F1 for biopsy `yes` |
| `decision_macro_f1` | Multiclass macro F1 for non-binary decision tasks |
| `confidence_weighted_kappa` | Quadratic weighted kappa over confidence labels |
| `variable_weight_weighted_kappa` | Quadratic weighted kappa over variable weights |
| `mean_tool_score` | Mean tool-efficiency precision |
| `mean_section_grounding_score` | Mean section-grounding score |
| `mean_rationale_score` | Mean LLM rationale score when enabled |
| `decision_gate_pass_rate` | Fraction of cases with nonzero decision score |
| `mean_case_score_among_gate_passed` | Mean case score excluding decision-gate failures |

## Adding or Replacing Data

Use the existing CHIMERA directory layout:

```text
ground_truth/<task_id>/<case_id>/pathologist_response.json
test/outputs/<task_id>/<case_id>/prediction.json
```

The evaluator accepts either a flat list, a task-keyed list such as
`{"biopsy_decision": [...]}`, or a single record object when loading each JSON
file. Records are matched by `case_id`, with `patient.id` as a fallback.

Keep [ground_truth/section_variable_mapping.json](ground_truth/section_variable_mapping.json)
updated when adding variables that should participate in section-grounding.

## Common Commands

```bash
# Build only.
./do_build.sh

# Run both supported tasks on GPU 1.
GPU_DEVICE_ID=1 ./do_test_run.sh task1 task2

# Run deterministic metrics only.
USE_RATIONALE_JUDGE=0 ./do_test_run.sh task1 task2

# Run offline after the model store has been populated.
GPU_DEVICE_ID=1 ALLOW_MODEL_PULL=0 ./do_test_run.sh task1 task2

# Package image + ground truth tarball for Grand Challenge.
./do_save.sh
```
