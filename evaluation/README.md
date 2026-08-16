# CHIMERA Evaluation Pipeline

Grand-Challenge evaluation container for CHIMERA agent outputs. It scores agent
predictions against pathologist ground truth for three tasks and writes both the
Grand Challenge ranking file and local debugging reports.

| Task | Question | Decision label |
| --- | --- | --- |
| Task 1 | Should this patient be biopsied? | `yes` / `no` |
| Task 2 | Which treatment is recommended? | one of four labels |
| Task 3 | When will biochemical recurrence occur? | `months_to_recurrence` + `event` |

Everything ships in **one Docker image**: the Python scoring pipeline, an
embedded Ollama server, and the judge LLM weights themselves. The container
needs no network at runtime, which is what Grand Challenge requires.

## Leaderboard scoring

All leaderboard metrics range from 0 to 1, with higher scores indicating better performance.
The evaluator computes each task independently and writes one phase-level
`metrics.json`. The official score is `aggregates.overall_ranking_score`, the
equal-weighted mean of the `ranking_score` values for the tasks present in the
phase.

### Task 1: Biopsy decision

The Task 1 ranking score is the average of:

* **Mean case score:** A holistic score averaged across all patients. Per case - An incorrect biopsy decision receives a case score of zero. For a correct decision, the score evaluates agreement with the pathologist on confidence, clinical-variable importance and decisive factors, together with efficient information retrieval, evidence grounding and the quality of the explanation.
* **Biopsy F1 score:** The F1 score for the positive class—patients for whom biopsy is recommended.

[
\text{Task 1 score}=
\frac{\text{Mean case score}+\text{Biopsy F1}}{2}
]

### Task 2: Treatment recommendation

The Task 2 ranking score is the average of:

* **Mean case score:** A holistic score averaged across all patients. Per case- An incorrect primary treatment recommendation receives a case score of zero. For a correct recommendation, the score evaluates confidence, clinical-variable importance, identification of decisive factors, information-retrieval efficiency, evidence grounding and explanation quality.
* **Weighted F1 score:** F1 across the four treatment categories, weighted by the number of patients in each category. This accounts for imbalance between treatment classes.

[
\text{Task 2 score}=
\frac{\text{Mean case score}+\text{Weighted F1}}{2}
]

### Task 3: Time to biochemical recurrence

The Task 3 leaderboard is ranked using Harrell's Concordance Index (C-index).

The C-index measures how well a model ranks patients according to their risk of biochemical recurrence. Rather than requiring the exact recurrence time to be predicted, it evaluates whether patients predicted to recur earlier do, in fact, experience recurrence before patients predicted to recur later. The metric naturally accounts for censored patients by considering only comparable patient pairs.
The evaluator also reports Task 3 case scores and time-dependent AUC for
analysis. They do not affect the leaderboard: Task 3 `ranking_score` is exactly
the C-index.

The Grand Challenge score JSONPath is:

```text
aggregates.overall_ranking_score
```

Task-specific result columns can use `aggregates.task1.ranking_score`,
`aggregates.task2.ranking_score`, and `aggregates.task3.ranking_score`.

## How Input Handling Works

Input handling is a deliberate copy of the GC debug kit.
Only the *scoring* and the *emitted metrics* differ.

```text
/input/predictions.json          one entry per algorithm job
        │
        ├── job["inputs"]  ──► sorted tuple of socket slugs = "interface key"
        │                         └──► selects process_interf0 / 1 / 2
        │
        └── job["outputs"] ──► socket "relative_path"
                                  └──► /input/<job_pk>/<relative_path>
```

1. **Interface switch.** `get_interface_key(job)` builds the sorted tuple of the
   job's *input* socket slugs. That tuple identifies the task.
2. **Handler picking.** The key selects one of three handlers, exactly as in the
   template:

   | Interface key (sorted input slugs) | Handler | Task |
   | --- | --- | --- |
   | `prostate-biopsy-decision-clinical-data`, `prostate-modality-level-neural-representations`, `structured-prompt` | `process_interf0` | Task 1 |
   | `prostate-modality-level-neural-representations`, `prostate-treatment-decision-clinical-data`, `structured-prompt` | `process_interf1` | Task 2 |
   | `prostate-modality-level-neural-representations`, `prostate-time-to-recurrence-or-last-follow-up-clin`, `structured-prompt` | `process_interf2` | Task 3 |

3. **Reading outputs from the pk.** Each handler calls `get_file_location(job_pk=..., values=job["outputs"], slug=...)`,
   which resolves the socket's `relative_path` under the job pk directory and
   reads it with `load_json_file`. Agent outputs are always read from **files**,
   never from inline values.
4. **Case id from the input socket-value PK.** Production input files have
  `value: null`. The handler joins the structured-prompt input's `pk` against
  the read-only `debug_archive_pks.csv` supplied with the GC ground truth. It
  deliberately does not match the algorithm job PK or archive-item PK.

All recognized task interfaces in `predictions.json` are evaluated in one run.
Every recognized job must resolve to exactly one case, every mapped phase case
must have ground truth under `ground_truth/<task>/<case_id>/`, and duplicate
jobs for the same task/case are rejected.



## Repository Layout

```text
evaluation/
├── evaluate.py                       # deterministic + optional LLM evaluator
├── do_build.sh                       # builds chimera-evaluator:latest
├── do_test_run.sh                    # builds once, then runs one or more tasks
├── do_save.sh                        # saves the image and packs ground_truth.tar.gz
├── .env.example                      # optional local configuration template
│
├── docker/
│   ├── Dockerfile                    # Ollama base + Python runtime + baked judge model
│   ├── entrypoint.sh                 # starts Ollama, checks model, runs evaluate.py
│   └── requirements.txt              # Python dependencies
│
├── ground_truth/
│   ├── debug_archive_pks.csv
│   ├── section_variable_mapping.json
│   └── taskN/<case_id>/ decision, reasoning, and clinical-data JSON files
│
├── test/input/
│   ├── predictions.json              # Grand Challenge job dump (all tasks)
│   └── <job_pk>/<relative_path>.json # agent output files, one dir per job
│
├── results/                          # generated local outputs, gitignored
├── pathologist_forms/                # HTML form templates
```

## Container Contract

| Mount | Mode | Contents |
| --- | --- | --- |
| `/input/` | read-only | `predictions.json` and one directory per job pk holding that job's output files |
| `/opt/ml/input/data/ground_truth/` | read-only | `taskN/<case_id>/` decision + reasoning files (Task 1/2) or the recurrence outcome (Task 3) plus `section_variable_mapping.json` |
| `/output/` | writable | `metrics.json`, `aggregate_metrics.json`, `per_case_results.csv`, `evaluation_results_summary.json` |
| `/tmp/` | writable | scratch only; nothing here is persisted |

The judge weights live at `/opt/ollama-models` **inside the image**. That path is
not a mount point, so nothing can shadow it.

## Prerequisites

- Docker Engine with permission to run containers.
- NVIDIA driver and NVIDIA Container Toolkit for GPU judging.
- One GPU with enough VRAM for the judge model.
- Network access **at build time only**, to pull the judge weights into the image.

For a deterministic smoke test with no GPU and no judge, set
`USE_RATIONALE_JUDGE=0`.

## Quick Start

```bash
cd ~/CHIMERA-agent/evaluation

# Run the complete mixed-task phase on the default GPU.
./do_test_run.sh
```

`do_test_run.sh` builds `chimera-evaluator:latest`, then evaluates the complete
phase once **offline** (`--network none`), just like Grand Challenge. The
Grand Challenge metrics file is written locally as:

```text
results/metrics.json
```

## Configuration

Export variables inline or copy [.env.example](.env.example) to `.env`;
`do_test_run.sh` loads `.env` automatically if it exists.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOCKER_IMAGE_TAG` | `chimera-evaluator:latest` | Image tag used by all scripts |
| `GPU_DEVICE_ID` | `0` | Host GPU exposed to Docker when judging is enabled |
| `JUDGE_MODEL` | `gemma4:e4b` | Ollama model baked in at build time and used at runtime |
| `USE_RATIONALE_JUDGE` | `1` | `0` disables Ollama and runs deterministic metrics only |
| `ALLOW_MODEL_PULL` | `0` | `1` permits a runtime download; needs network, so not usable on Grand Challenge |

Examples:

```bash
# Deterministic, CPU-only smoke test.
USE_RATIONALE_JUDGE=0 ./do_test_run.sh

# Build with a different judge model baked in.
JUDGE_MODEL=gemma3:4b ./do_build.sh

# Build under a custom image tag.
DOCKER_IMAGE_TAG=chimera-evaluator:dev ./do_build.sh
```

> The build fails if `JUDGE_MODEL` cannot be resolved by `ollama pull`. Override
> it with `JUDGE_MODEL=<tag> ./do_build.sh` rather than editing the Dockerfile.

## Scripts

### Build Only

```bash
./do_build.sh
```

Builds `chimera-evaluator:latest` from [docker/Dockerfile](docker/Dockerfile)
using this directory as the build context, and pulls `JUDGE_MODEL` into the
image. This step needs network access; the resulting runs do not.

### Local Test Run

```bash
./do_test_run.sh
GPU_DEVICE_ID=0 ./do_test_run.sh
```

The script mounts:

- [test/input](test/input) as `/input:ro`
- [ground_truth](ground_truth) as `/opt/ml/input/data/ground_truth:ro`
- `results` as `/output`
- a throwaway Docker volume as `/tmp`

It runs with `--network none`, then fixes file ownership afterwards from inside
the container so the results stay readable on the host.

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
mkdir -p results

docker run --rm \
  --gpus "device=1" \
  --platform=linux/amd64 \
  --network none \
  -v "$PWD/test/input:/input:ro" \
  -v "$PWD/ground_truth:/opt/ml/input/data/ground_truth:ro" \
  -v "$PWD/results:/output" \
  -e GROUND_TRUTH_DIR=/opt/ml/input/data/ground_truth \
  chimera-evaluator:latest
```

All other paths (`INPUT_DIRECTORY`, `PREDICTIONS_FILE`, `SECTION_MAPPING_FILE`,
`EVAL_OUTPUT_DIR`) already have correct defaults baked
into the image. To disable the judge entirely, add `-e USE_RATIONALE_JUDGE=0`
and omit `--gpus`.

## Outputs

The phase evaluation writes these files under `results/` locally and `/output/`
in the container:

| File | Purpose |
| --- | --- |
| `metrics.json` | Grand-Challenge-facing ranking file with `aggregates` and per-case `results` |
| `aggregate_metrics.json` | Dataset-level metrics only |
| `per_case_results.csv` | Flat per-case table for quick inspection |
| `evaluation_results_summary.json` | Full run summary with aggregate metrics and per-case details |

## Evaluation Logic

Agent outputs are read from the per-job files described in
[How Input Handling Works](#how-input-handling-works). The evaluator groups all
recognized jobs by task, computes task-specific aggregates, and gives every
task present in the phase equal weight in `overall_ranking_score`.

### Stage 1: Decision Gate (Task 1 / Task 2)

Both gates are **exact match**. There is no partial credit.

For Task 1, the `biopsy_decision` must be valid (`yes` or `no`) and match the
pathologist response exactly.

For Task 2, `treatment_recommendation.primary` must exactly match one of:

- `watchful_waiting`
- `active_surveillance`
- `continued_surveillance`
- `active_treatment`

Any mismatch ends the case with `case_score = 0`. Only cases that pass the gate
proceed to Stage 2.

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

The rationale judge compares the agent's `free_text` against the pathologist's
`free_text`, expected decision, confidence, and clinical context. Since GC
input sockets are file-backed with `value: null`, clinical context is loaded
from the corresponding ground-truth case directory.

### Tool-Use Policy

`reveal_sequence` is a flat, ordered list naming the sections the agent actually
consulted. Exactly six names are recognised:

```text
radiology_report  laboratory_results  psa_trend
pathology_report  previous_notes      family_history
```

The tool score penalizes unnecessary reveals only:

```text
score = |agent_revealed ∩ pathologist_revealed| / |agent_revealed|
```

Missing pathologist reveals are not penalized. Extra agent reveals are
penalized uniformly, and a name outside the six above counts as an extra
reveal. Order is ignored, and repeated reveals of the same section count once.

### Section-Grounding Policy

Section grounding checks whether variables weighted above `not_used` are
supported by sections the agent actually revealed, using
[ground_truth/section_variable_mapping.json](ground_truth/section_variable_mapping.json).
Always-available variables such as `psa` and `age` are exempt.

A variable whose source sections lie outside the six-name reveal vocabulary
(currently only `comorbidity`, sourced from the comorbidities section) cannot be
grounded by any submission, so it is excluded from the score rather than counted
against the agent. Such variables are listed under `ungradable_variables` in
`evaluation_results_summary.json`.

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

At the dataset level, Task 3 reports two ranking metrics over the cohort, both
using predicted months as the risk ordering (a shorter predicted time means
higher risk):

- **Harrell's concordance index** — overall ranking quality across all
  comparable pairs.
- **Time-dependent AUC** — Uno's IPCW cumulative/dynamic AUC at 12, 24, 36 and
  60 months. At each horizon, cases are patients who recurred by then and
  controls are those still recurrence-free; inverse-probability-of-censoring
  weights correct for patients censored before the horizon. A horizon with no
  case/control pair reports `null`, and `mean_time_dependent_auc` averages only
  the horizons that are defined.

> Both metrics need a reasonable cohort size. On a two-case test set they will
> usually be `null`; that is expected, not a failure.

## Key Aggregate Metrics

Task 1 / Task 2:

| Metric | Meaning |
| --- | --- |
| `ranking_score` | **Leaderboard metric.** Mean of `mean_case_score` and the task's decision F1 |
| `mean_case_score` | Mean case score across all cases, including gate failures |
| `decision_accuracy` | Decision-match accuracy |
| `decision_f1_yes` | Task 1 positive-class F1 for biopsy `yes` |
| `decision_weighted_f1` | Support-weighted multiclass F1 for Task 2 |
| `confidence_weighted_kappa` | Quadratic weighted kappa over confidence labels |
| `variable_weight_weighted_kappa` | Quadratic weighted kappa over variable weights |
| `mean_tool_score` | Mean tool-efficiency precision |
| `mean_section_grounding_score` | Mean section-grounding score |
| `mean_rationale_score` | Mean LLM rationale score when enabled |
| `decision_gate_pass_rate` | Fraction of cases that passed the decision gate |
| `mean_case_score_among_gate_passed` | Mean case score excluding decision-gate failures |

Task 2 uses **weighted** F1 rather than macro F1 so that each class contributes
in proportion to how often it actually occurs, which is the more honest summary
on the imbalanced treatment-label distribution.

Both F1 metrics are computed over **every** case that has a ground-truth
decision. A case whose prediction is missing or fails schema validation is
scored against a sentinel label, so it costs the true class its recall instead
of quietly dropping out of the metric.

Task 3 (biochemical recurrence) reports a different set:

| Metric | Meaning |
| --- | --- |
| `ranking_score` | **Leaderboard metric.** Equal to `concordance_index` |
| `mean_case_score` | Mean case score across all Task 3 cases |
| `recurrence_event_accuracy` | Fraction of cases with correct `event` |
| `mean_event_score` | Mean event-agreement score |
| `mean_time_score` | Mean censoring-aware time score |
| `event1_time_mae_months` | Mean absolute months error over observed recurrences |
| `concordance_index` | Harrell's C-index over the cohort (`null` if no comparable pairs) |
| `time_dependent_auc` | IPCW cumulative/dynamic AUC per horizon (`12m`, `24m`, `36m`, `60m`) |
| `mean_time_dependent_auc` | Mean of the defined horizons |
| `mean_rationale_score` | Mean Task 3 reasoning-judge score when enabled |

## Adding or Replacing Data

Ground truth uses the CHIMERA per-case directory layout. For Task 1 and Task 2
it is split across the **same two files the agent submits**, so ground truth and
submission are directly comparable:

```text
ground_truth/task1/<case_id>/prostate-biopsy-decision.json
ground_truth/task1/<case_id>/prostate-biopsy-decision-reasoning.json
ground_truth/task1/<case_id>/prostate-biopsy-decision-clinical-data.json
ground_truth/task2/<case_id>/prostate-treatment-decision.json
ground_truth/task2/<case_id>/prostate-treatment-decision-reasoning.json
ground_truth/task2/<case_id>/prostate-treatment-decision-clinical-data.json
ground_truth/task3/<case_id>/prostate-time-to-recurrence-or-last-follow-up.json
ground_truth/task3/<case_id>/prostate-time-to-recurrence-or-last-follow-up-clinical-data.json
```

The case directory name is the authoritative `case_id`; the files carry none.
A case missing any of its required files is skipped with a warning.

The decision file is a bare JSON value — `"yes"` / `"no"` for Task 1, one of the
four treatment labels for Task 2. The reasoning file is an object with exactly
four keys:

```json
{
  "confidence": "clear",
  "variable_weights": {"pirads": "decisive", "psa": "noted"},
  "reveal_sequence": ["radiology_report", "laboratory_results"],
  "free_text": "PI-RADS 5 with high PSA density ..."
}
```

Task 3 ground truth is a bare `{"months_to_recurrence": ..., "event": ...}`
object and is unchanged.

Agent outputs come from `test/input/predictions.json` (a Grand Challenge job
dump) plus one directory per job pk holding that job's output files:

```text
test/input/predictions.json
test/input/<job_pk>/<relative_path>.json
```

The `<relative_path>` values come from each output socket in `predictions.json`,
so the filenames must match what that file declares. Case IDs are resolved by
joining each structured-prompt socket-value PK to the GC-provided
`debug_archive_pks.csv`. The evaluator reads that file but never modifies it.

Keep [ground_truth/section_variable_mapping.json](ground_truth/section_variable_mapping.json)
updated when adding variables that should participate in section-grounding
(Task 1 / Task 2 only).

## Common Commands

```bash
# Build only.
./do_build.sh

# Run the complete phase on GPU 3.
GPU_DEVICE_ID=3 ./do_test_run.sh

# Run deterministic metrics only (no GPU / no judge).
USE_RATIONALE_JUDGE=0 ./do_test_run.sh

# Rebuild with a different judge model baked into the image.
JUDGE_MODEL=gemma3:4b ./do_build.sh

# Package image + ground truth tarball for Grand Challenge.
./do_save.sh
```
