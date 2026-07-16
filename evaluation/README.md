# CHIMERA Evaluation Pipeline


This directory contains the Grand-Challenge-style evaluator for CHIMERA agent
outputs. It compares per-case agent predictions against pathologist ground truth
for Task 1 biopsy decisions, Task 2 treatment decisions, and Task 3
biochemical-recurrence prediction, then writes both the Grand Challenge ranking
file and local debugging reports.

Agent outputs are read from a single Grand-Challenge predictions dump
(`test/input/predictions.json`). That file carries no `case_id`, so
`test/input/pk_hash_to_case_map.json` maps each job `pk` to its `taskN/<case_id>`.

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
│   ├── task2/<case_id>/pathologist_response.json
│   └── task3/<case_id>/prostate-time-to-recurrence-or-last-follow-up.json
│
├── test/input/
│   ├── predictions.json              # Grand Challenge job dump (all tasks)
│   └── pk_hash_to_case_map.json      # job pk -> taskN/<case_id>
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
| `/input/` | read-only | `predictions.json` (all-task job dump) + `pk_hash_to_case_map.json` |
| `/opt/ml/input/data/ground_truth/` | read-only | `taskN/<case_id>/` pathologist responses (Task 1/2) or recurrence outcome (Task 3) plus `section_variable_mapping.json` |
| `/output/` | writable | `metrics.json`, `aggregate_metrics.json`, `per_case_results.csv`, `evaluation_results_summary.json` |
| `/models/` | read/write | Ollama model store used by the rationale judge |

Local runs bind-mount [test/input](test/input) to `/input` and
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

# Run all three tasks sequentially 
 ./do_test_run.sh task1 task2 task3
```

`do_test_run.sh` builds `chimera-evaluator:latest` once, validates all requested
task directories up front, then runs each task in order. Results are written to
separate task directories:

```text
results/task1/
results/task2/
results/task3/
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
GPU_DEVICE_ID=0 ./do_test_run.sh task1 task2
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

Agent outputs are read from `test/input/predictions.json`. Each job's task is
identified from its input sockets, and its `case_id` is recovered from
`test/input/pk_hash_to_case_map.json`. The evaluator iterates over ground-truth
cases; a ground-truth case with no matching prediction is reported as a missing
candidate with `case_score = 0`.

### Stage 1: Decision Gate (Task 1 / Task 2)

The evaluator matches predictions to ground truth by `case_id`.

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

## Task 3: Biochemical Recurrence

Task 3 predicts time to biochemical recurrence. The ground truth for each case
is a single object with `months_to_recurrence` and `event` (1 = recurrence
observed, 0 = censored at last follow-up). There are no variable weights,
confidence, or reveal sequences, so the decision gate and the Task 1/2 component
scores do not apply.

Each case is scored on three components:

| Component | Method | Weight with judge | Weight without judge |
| --- | --- | ---: | ---: |
| Event agreement | 1.0 if predicted `event` matches ground truth, else 0.0 | 0.35 | 0.50 |
| Time closeness | Censoring-aware closeness of `months_to_recurrence` | 0.35 | 0.50 |
| Reasoning | DeepEval GEval rubric via local Ollama (no reference rationale) | 0.30 | disabled |

Time closeness is censoring-aware: for observed recurrences (`event = 1`) the
predicted time should match the true time; for censored cases (`event = 0`) only
predictions that recur *earlier* than the last follow-up are penalized.

The reasoning judge has no reference rationale to compare against (the ground
truth carries only the outcome), so it scores clinical soundness and internal
consistency of the agent's free text against the clinical inputs and the
reference outcome.

At the dataset level, Task 3 also reports Harrell's concordance index over the
cohort, using predicted months as the risk ordering.

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

Task 3 (biochemical recurrence) reports a different set:

| Metric | Meaning |
| --- | --- |
| `final_DAG_score` | Mean case score across all Task 3 cases |
| `recurrence_event_accuracy` | Fraction of cases with correct `event` |
| `mean_event_score` | Mean event-agreement score |
| `mean_time_score` | Mean censoring-aware time score |
| `event1_time_mae_months` | Mean absolute months error over observed recurrences |
| `concordance_index` | Harrell's C-index over the cohort (None if no comparable pairs) |
| `mean_rationale_score` | Mean Task 3 reasoning-judge score when enabled |

## Adding or Replacing Data

Ground truth uses the CHIMERA per-case directory layout:

```text
ground_truth/task1/<case_id>/pathologist_response.json
ground_truth/task2/<case_id>/pathologist_response.json
ground_truth/task3/<case_id>/prostate-time-to-recurrence-or-last-follow-up.json
```

The case directory name is the authoritative `case_id`. For Task 1 and Task 2
the evaluator accepts a flat list, a task-keyed list such as
`{"biopsy_decision": [...]}`, or a single record object. Task 3 ground truth is a
bare `{"months_to_recurrence": ..., "event": ...}` object.

Agent outputs come from `test/input/predictions.json` (a Grand Challenge job
dump). Each job's task is identified from its input sockets, and its case is
recovered from `test/input/pk_hash_to_case_map.json`, which maps each job `pk`
to `taskN/<case_id>`. Ground-truth cases with no matching prediction are
reported as missing candidates.

Keep [ground_truth/section_variable_mapping.json](ground_truth/section_variable_mapping.json)
updated when adding variables that should participate in section-grounding
(Task 1 / Task 2 only).

## Common Commands

```bash
# Build only.
./do_build.sh

# Run all three tasks on GPU 3.
GPU_DEVICE_ID=3 ./do_test_run.sh task1 task2 task3

# Run deterministic metrics only (no GPU / no judge).
USE_RATIONALE_JUDGE=0 ./do_test_run.sh task1 task2 task3

# Run offline after the model store has been populated.
GPU_DEVICE_ID=3 ALLOW_MODEL_PULL=0 ./do_test_run.sh task1 task2 task3

# Package image + ground truth tarball for Grand Challenge.
./do_save.sh
```
