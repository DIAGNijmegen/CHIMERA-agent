# LLM Biopsy-Decision Evaluator

Evaluates LLM-agent biopsy-decision form responses against pathologist
ground-truth using a hybrid deterministic + LLM scoring pipeline.

The whole pipeline (judge LLM + evaluator) lives in **one Docker image** so it
can be submitted directly to the [Grand Challenge](https://grand-challenge.org/)
platform.

---

## Repo layout

```
.
├── evaluate.py                       # main evaluation script
├── Makefile                          # entry point: `make run`
├── .env.example                      # configuration template → copy to .env
│
├── docker/
│   ├── Dockerfile                    # single image: Python evaluator + Ollama judge
│   ├── entrypoint.sh                 # starts Ollama, ensures model, runs evaluate.py
│   ├── docker-compose.yml            # single-service compose (dev convenience)
│   └── requirements.txt              # Python dependencies
│
├── mimic_datasets/
│   ├── target.json                   # ground-truth pathologist responses
│   ├── evaluation_object.json        # LLM-agent responses to evaluate
│   ├── section_variable_mapping.json # form-section → clinical-variable map
│   └── README.md                     # dataset schema documentation
│
├── gt_formats/                       # ground-truth form templates (HTML)
└── hpc/                              # optional Apptainer / SLURM wrappers
```

Generated at runtime (gitignored, never committed):
- `results/` — evaluation output files
- `models/` — Ollama model-weight cache (the judge LLM)

---

## Container contract (Grand-Challenge layout)

The image follows a single, platform-agnostic mount contract:

| Mount       | Mode | Contents                                                                                  |
|-------------|------|-------------------------------------------------------------------------------------------|
| `/input/`   | RO   | `target.json`, `evaluation_object.json`, `section_variable_mapping.json`                  |
| `/output/`  | RW   | `per_case_results.csv`, `aggregate_metrics.json`, `evaluation_results_summary.json`       |
| `/models/`  | RW   | Ollama model store (`blobs/`, `manifests/`). Persist + reuse across runs.                 |

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
cd <repo-name>

# 2. Configure
cp .env.example .env
#    Edit GPU_DEVICE_ID if your target GPU is not device 0 (check: nvidia-smi).
#    Defaults are fine on a single-GPU workstation.

# 3. Build the image, start Ollama inside it, pull the judge model (first
#    run only, ~9.6 GB), and run the evaluator.
make run
```

Results land in `results/` with host-user ownership (not root).

---

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable              | Default                  | Description                                                              |
|-----------------------|--------------------------|--------------------------------------------------------------------------|
| `GPU_DEVICE_ID`       | `0`                      | GPU index exposed to the container (`nvidia-smi` to check)               |
| `JUDGE_MODEL`         | `gemma4:e4b`             | Ollama model used as the rationale judge (~9.6 GB)                       |
| `USE_RATIONALE_JUDGE` | `1`                      | `0` = skip the LLM judge step, fully deterministic                       |
| `ALLOW_MODEL_PULL`    | `1`                      | `0` = forbid runtime download; fail fast if weights missing (offline)    |
| `INPUT_DIR`           | `./mimic_datasets`       | Host path mounted read-only at `/input`                                  |
| `OUTPUT_DIR`          | `./results`              | Host path mounted writable at `/output`                                  |
| `OLLAMA_MODELS_DIR`   | `./models`               | Host path mounted at `/models` (Ollama weight cache)                     |
| `COMPOSE_NAME_PREFIX` | `biopsy-eval`            | Container name prefix (useful on shared hosts)                           |

---

## Makefile targets

| Command            | What it does                                                                |
|--------------------|-----------------------------------------------------------------------------|
| `make run`         | Build image + run evaluation *(default)*                                    |
| `make build`       | Build the unified evaluator image only                                      |
| `make pull-model`  | Pre-populate `./models/` with `JUDGE_MODEL` for offline / Grand-Challenge   |
| `make shell`       | Open a bash shell in the container (Ollama not auto-started)                |
| `make down`        | Stop and remove the container                                               |
| `make clean`       | Remove the built image                                                      |
| `make help`        | List all targets                                                            |

---

## Running on Grand Challenge (or any `docker run` host)

Grand Challenge invokes evaluation containers without Compose and (typically)
without network access. The image is built to handle that:

### 1. Prepare the model weights once

```bash
# Downloads gemma4:e4b into ./models/ on the host
make pull-model
```

### 2. Test the offline run locally first

```bash
docker run --rm --gpus all \
    -v "$PWD/mimic_datasets:/input:ro" \
    -v "$PWD/results:/output" \
    -v "$PWD/models:/models" \
    -e ALLOW_MODEL_PULL=0 \
    biopsy-evaluator:latest
```

If this succeeds with no network access, the same invocation will work on
Grand Challenge.

### 3. Submit

Export and submit the tagged image as required by the challenge phase:

```bash
docker tag  biopsy-evaluator:latest  <your-gc-registry>/biopsy-evaluator:v1
docker push <your-gc-registry>/biopsy-evaluator:v1
# … or use `docker save` if the challenge expects a tarball.
```

On submission, Grand Challenge will mount its own `/input/`, `/output/`, and
(if configured for the phase) `/models/` directories and run the container.

> **If `/models` is NOT available on submission**, re-build the image with
> the weights baked in:
> ```bash
> make pull-model                              # populate ./models/
> docker build -t biopsy-evaluator:offline \
>     --build-arg BAKE_MODELS=1 -f docker/Dockerfile .
> ```
> and add a `COPY models/ /models/` line to the Dockerfile gated on the build
> arg. (Not enabled by default to keep the image small.)

---

## How the evaluation works

### Stage 1 — Gate check (per case)

A case must pass all three conditions to receive a component score:

1. A matching candidate record exists (matched by `case_id`)
2. The candidate passes schema validation (`biopsy_decision` ∈ {yes, no};
   `variable_weights` is a dict if present) — see
   [mimic_datasets/README.md](mimic_datasets/README.md) for the full schema
3. The `biopsy_decision` field matches the ground truth exactly

Cases that fail any gate receive `case_score = 0`.

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
[mimic_datasets/section_variable_mapping.json](mimic_datasets/section_variable_mapping.json).
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

Replace or augment the JSON files in `mimic_datasets/` (or point `INPUT_DIR`
at any directory laid out the same way):

- `target.json` — ground-truth expert responses
- `evaluation_object.json` — LLM-agent responses to evaluate
- `section_variable_mapping.json` — form-section → variable map (optional;
  a bundled default is used as fallback)

See [mimic_datasets/README.md](mimic_datasets/README.md) for the full record
schema. Case IDs in both files are matched by the `case_id` field.

Point the evaluator at custom files without editing `.env`:

```bash
INPUT_DIR=/path/to/my-inputs \
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
