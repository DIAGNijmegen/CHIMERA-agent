# LLM Evaluator


## Repo layout

```
.
├── evaluate.py                       # main evaluation script
├── do_build.sh                       # build the evaluator image
├── do_test_run.sh                    # entry point: build + run one task
├── do_save.sh                        # package image + ground_truth.tar.gz for GC
├── .env.example                      # configuration template → copy to .env
│
├── docker/
│   ├── Dockerfile                    # single image: Python evaluator + Ollama judge
│   ├── entrypoint.sh                 # starts Ollama, ensures model, runs evaluate.py
│   └── requirements.txt              # Python dependencies
│
├── ground_truth/
│   ├── section_variable_mapping.json
│   └── task1/<case_id>/pathologist_response.json
├── test/outputs/
│   └── task1/<case_id>/prediction.json
│
├── external/                          # Grand-Challenge reference method (do not edit)
└── pathologist_forms/                 # ground-truth form templates (HTML)
```

Generated at runtime (gitignored, never committed):
- `results/` — evaluation output files
- `models/` — Ollama model-weight cache (the judge LLM)

---

## Container contract (Grand-Challenge layout)

The image follows the Grand-Challenge mount contract (mirrors
`external/example_evaluation_method`):

| Mount          | Mode | Contents                                                                                  |
|----------------|------|-------------------------------------------------------------------------------------------|
| `/input/`        | RO | `taskN/<case_id>/prediction.json` (algorithm outputs)                                    |
| `/opt/ml/input/data/ground_truth/` | RO | `taskN/<case_id>/pathologist_response.json`, `section_variable_mapping.json` (from the ground-truth tarball) |
| `/output/`     | RW   | `metrics.json`, `per_case_results.csv`, `aggregate_metrics.json`, `evaluation_results_summary.json` |
| `/models/`     | RW   | Ollama model store (`blobs/`, `manifests/`). Persist + reuse across runs.                 |

Nothing is baked into the image at runtime — both **data** and **model weights**
are pulled from these mounts, exactly as Grand Challenge expects.

---

## Prerequisites

- **Docker Engine** (your user in the `docker` group)
- **NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** for GPU access:
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- **One NVIDIA GPU with ≥ 10 GB VRAM** (RTX 3090, A10, A100, …) for the
  default `gemma4:e4b` judge.

> **No GPU / fast smoke test?** Set `USE_RATIONALE_JUDGE=0` in `.env`. The
> evaluator then runs fully deterministically with no GPU required.

---

## Quick start (local development)

```bash
# 1. Clone
git clone <repo-url>
cd <repo-name>/evaluation

# 2. Configure
cp .env.example .env
#    Edit GPU_DEVICE_ID if your target GPU is not device 0 (check: nvidia-smi).
#    Defaults are fine on a single-GPU workstation.

# 3. Build the image, start Ollama inside it, pull the judge model (first
#    run only, ~9.6 GB), and run the evaluator for one task.
./do_test_run.sh                       # TASK_ID=task1
TASK_ID=task2 GPU_DEVICE_ID=1 ./do_test_run.sh
```

Results land in `results/<TASK_ID>/` with host-user ownership (not root).

---

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable              | Default                  | Description                                                              |
|-----------------------|--------------------------|--------------------------------------------------------------------------|
| `GPU_DEVICE_ID`       | `0`                      | GPU index exposed to the container (`nvidia-smi` to check)               |
| `JUDGE_MODEL`         | `gemma4:e4b`             | Ollama model used as the rationale judge (~9.6 GB)                       |
| `USE_RATIONALE_JUDGE` | `1`                      | `0` = skip the LLM judge step, fully deterministic (no GPU)              |
| `ALLOW_MODEL_PULL`    | `1`                      | `0` = forbid runtime download; fail fast if weights missing (offline)    |
| `TASK_ID`             | `task1`                  | Task directory to evaluate (also positional arg 1 to `do_test_run.sh`)  |

---

## Scripts

| Command            | What it does                                                                |
|--------------------|-----------------------------------------------------------------------------|
| `./do_test_run.sh` | Build image + run one evaluation task *(default `TASK_ID=task1`)*           |
| `./do_build.sh`    | Build the unified evaluator image only                                      |
| `./do_save.sh`     | Build + `docker save` the image and pack `ground_truth.tar.gz` for GC       |

Run both tasks (task2 on GPU 1):

```bash
./do_test_run.sh task1
TASK_ID=task2 GPU_DEVICE_ID=1 ./do_test_run.sh
```

---

## Running on Grand Challenge (or any `docker run` host)

Grand Challenge invokes evaluation containers without Compose and (typically)
without network access. The image is built to handle that:

### 1. Prepare the model weights once

```bash
# Populate ./models/ with the judge weights by running one task with network
# access (the entrypoint pulls JUDGE_MODEL into the mounted /models store).
./do_test_run.sh task1
```

### 2. Test the offline run locally first

```bash
docker run --rm --gpus "device=1" \
  -v "$PWD/test/outputs:/input:ro" \
  -v "$PWD/ground_truth:/opt/ml/input/data/ground_truth:ro" \
  -v "$PWD/results/task1:/output" \
  -v "$PWD/models:/models" \
  -e TASK_ID=task1 \
  -e ALLOW_MODEL_PULL=0 \
  biopsy-evaluator:latest
```

If this succeeds with no network access, the same invocation will work on
Grand Challenge.

### 3. Package + submit

```bash
./do_save.sh    # writes biopsy-evaluator_<timestamp>.tar.gz + ground_truth.tar.gz
```

Upload the image tarball as the Evaluation Method and `ground_truth.tar.gz`
separately under **Phase settings > Ground Truths** (extracted to
`/opt/ml/input/data/ground_truth/` at runtime).

On submission, Grand Challenge mounts `/input/`,
`/opt/ml/input/data/ground_truth/`, `/output/`, and, if configured for the
phase, `/models/`.

> **If `/models` is NOT available on submission**, re-build the image with
> the weights baked in: populate `./models/` (run one task first), add a
> `COPY models/ /models/` line to the Dockerfile, and rebuild. (Not enabled
> by default to keep the image small.)

---

## How the evaluation works

### Stage 1 — Decision check (per case)

A Task 1 biopsy case must pass all three conditions to receive a component score:

1. A matching candidate record exists (matched by `case_id`)
2. The candidate passes schema validation (`biopsy_decision` ∈ {yes, no};
   `variable_weights` is a dict if present) — see
   [mimic_datasets/README.md](mimic_datasets/README.md) for the full schema
3. The `biopsy_decision` field matches the ground truth exactly

Task 1 cases that fail any gate receive `case_score = 0`.

For Task 2 treatment decisions, `treatment_recommendation.primary` is scored
across four canonical classes: `watchful_waiting`, `active_surveillance`,
`continued_surveillance`, and `active_treatment`. Mismatches score 0, except
`active_surveillance` vs `continued_surveillance`, which receives
`decision_score = 0.5`. The remaining component score is multiplied by this
decision score.

### Stage 2 — Component scores (gate-passed cases only)

All component scores are in `[0, 1]`:

| Component                       | Method                                                                                                                | Weight (w/ judge) | Weight (w/o) |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------|-------------------|--------------|
| **Confidence**                  | Ordinal distance: `1 − \|gt − pred\| / 2`                                                                             | 0.20              | 0.225        |
| **Variable weights**            | Mean ordinal MAE across all variables                                                                                 | 0.25              | 0.275        |
| **Important / decisive factors**| Set-F1 between `important + decisive` variable sets                                                                   | 0.15              | 0.175        |
| **Tool-efficiency precision**   | `\|agent ∩ pathologist\| / \|agent revealed\|`                                                                        | 0.15              | 0.150        |
| **Section grounding**           | `n_grounded / n_weighted` — fraction of weighted variables whose primary source section the agent actually revealed   | 0.15              | 0.175        |
| **Rationale alignment**         | GEval rubric judged by Ollama (`USE_RATIONALE_JUDGE=0` to disable)                                                    | 0.10              | —            |

**Tool-score policy:** the agent is penalised only for revealing sections the
pathologist did not reveal (unnecessary lookups). Missing reveals are not
penalised.

**Section-grounding policy:** penalises the agent for weighting a variable
above `not_used` without revealing the section that primarily provides that
variable's data, per
[ground_truth/section_variable_mapping.json](ground_truth/section_variable_mapping.json).
Always-available variables (`psa`, `age` from the patient card) are exempt.

### Aggregate metrics (dataset level)

| Metric                              | Description                                                       |
|-------------------------------------|-------------------------------------------------------------------|
| `final_DAG_score`                   | Mean case score (gate failures count as 0)                        |
| `decision_accuracy`                 | Fraction of cases with correct `biopsy_decision`                  |
| `decision_f1_yes`                   | F1 for the positive (`yes`) class                                 |
| `n_decision_correct/incorrect`      | Raw counts                                                        |
| `confidence_weighted_kappa`         | Quadratic Cohen's κ on confidence labels                          |
| `variable_weight_weighted_kappa`    | Quadratic Cohen's κ across all variable weights                   |
| `mean_tool_score`                   | Mean tool-efficiency precision                                    |
| `mean_section_grounding_score`      | Mean section grounding score across gate-passed cases             |
| `decision_gate_pass_rate`           | Fraction of cases that passed the hard gate                       |
| `mean_case_score_among_gate_passed` | Mean component score excluding gate failures                      |

---

## Plugging in your own data

Replace or augment the per-case JSON files under:

- `ground_truth/<task_id>/<case_id>/pathologist_response.json` — ground-truth expert response
- `test/outputs/<task_id>/<case_id>/prediction.json` — LLM-agent response to evaluate
- `ground_truth/section_variable_mapping.json` — form-section → variable map (optional;
  a bundled default is used as fallback)

See [mimic_datasets/README.md](mimic_datasets/README.md) for the full record
schema. Case IDs in both files are matched by the `case_id` field.

Point the evaluator at custom roots without editing `.env`:

```bash
TASK_ID=task1 \
GROUND_TRUTH_DIR=/path/to/ground_truth \
TEST_OUTPUTS_DIR=/path/to/test/outputs \
OUTPUT_DIR=/path/to/my-results \
make run
```

---

## Biopsy-decision guidelines (Task 1)

The pathologist ground-truth follows these rules:

**Biopsy YES**
- PI-RADS ≥ 4 → biopsy (targeted + perilesional)
- PI-RADS 3 + PSA density ≥ 0.10 ng/mL/cc → biopsy
- PI-RADS 3 + family history of PCa → biopsy
- PI-RADS ≤ 2 + PSA density ≥ 0.20 ng/mL/cc → biopsy
- PI-RADS ≤ 2 + family history → biopsy

**Biopsy NO (PSA monitoring instead)**
- PI-RADS 3 + PSA density < 0.10 + no family history → defer
- PI-RADS ≤ 2 + PSA density < 0.20 + no family history → defer

**Override (comorbidity / life expectancy)**
- Severe comorbidity with life expectancy < 10 years → watchful waiting,
  not biopsy
