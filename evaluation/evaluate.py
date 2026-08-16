"""Deterministic + optional LLM evaluation pipeline for clinical decision forms.

Compares LLM-agent outputs against pathologist ground-truth for three tasks:

    Task 1  biopsy decision      (yes/no)
    Task 2  treatment decision   (watchful_waiting / active_surveillance / ...)
    Task 3  biochemical recurrence prediction (months_to_recurrence + event)

Input handling mirrors the Grand-Challenge reference evaluation method
(external/example_evaluation_method/evaluate.py): /input/predictions.json is a
job dump, each job's task is identified by the sorted tuple of its *input*
socket slugs (the "interface key"), that key selects a per-interface handler,
and every agent artefact is read from a real file whose location is built from
the job pk plus the output socket's relative_path. Only the scoring and the
emitted metrics differ from the reference method.

A job's task comes from its interface key (the sorted tuple of its input socket
slugs). File-backed inputs have value=null, so the structured-prompt socket-value
pk is joined against debug_archive_pks.csv to recover the case id. Ground truth
is loaded per task and case from ground_truth/<task_id>/<case_id>/. Ground-truth
cases with no matching job are reported as missing candidates rather than
silently dropped.

Scoring:

        * A decision gate (Task 1 / Task 2 only). Both are exact-match gates: a
            wrong biopsy decision or a wrong primary treatment recommendation
            scores 0 for the case. Task 3 has no categorical gate.

        * Task 3 deterministic scores: event agreement, censoring-aware
            time-to-recurrence closeness, a cohort concordance index, and
            IPCW cumulative/dynamic time-dependent AUC at fixed horizons.

    * Deterministic ordinal scores:
        - confidence_score          ordinal distance on clear/borderline/uncertain
        - variable_weight_score     mean ordinal MAE across all variable weights
        - important_decisive_factor_score   set-F1 on important+decisive variables

    * Tool efficiency precision (deterministic):
        score = |agent revealed ∩ pathologist revealed| / |agent revealed|
        Extra agent reveals are penalised uniformly; missing reveals are NOT
        penalised. This encodes "don't look up unnecessary information."

    * Optional LLM rationale judge:
        GEval rubric evaluated by a local Ollama model (gemma4:e4b by default).
        Disabled when USE_RATIONALE_JUDGE=0 or Ollama is unreachable.

Outputs (written to /output — see README.md):

    metrics.json                      Grand-Challenge ranking file
                                      ({"aggregates": {...}, "results": [...]})
    evaluation_results_summary.json   full per-case + aggregate dump
    per_case_results.csv              one row per case, easy to scan
    aggregate_metrics.json            dataset-level summary

Grand-Challenge container contract (see external/example_evaluation_method):

    /input/                        (read-only)  predictions.json (job dump)
                                                → <job_pk>/<relative_path>
                                                  (or <job_pk>/output/<relative_path>)
    /opt/ml/input/data/ground_truth/ (read-only) ground-truth tarball
                                                → taskN/<case_id>/<decision>.json
                                                → taskN/<case_id>/<reasoning>.json
                                                → taskN/<case_id>/<clinical-data>.json
                                                → debug_archive_pks.csv
                                                → section_variable_mapping.json
    /output/                       (writable)   metrics.json + the reports above

Run via Docker (recommended — see README.md):

    ./do_test_run.sh                # builds + runs one task

Or directly (Ollama must be reachable at OLLAMA_BASE_URL):

    python evaluate.py

Environment variable overrides:

    GROUND_TRUTH_DIR   ground-truth root directory (default: ground_truth/)
    INPUT_DIRECTORY    job dump + job output dirs  (default: test/input)
    PREDICTIONS_FILE   agent predictions dump      (default: $INPUT_DIRECTORY/predictions.json)
    EVAL_OUTPUT_DIR    output directory            (default: results/)
    OLLAMA_BASE_URL    Ollama API base URL         (default: http://ollama:11434)
    JUDGE_MODEL        Ollama model name           (default: gemma4:e4b)
    USE_RATIONALE_JUDGE  "0" disables LLM judge    (default: "1")
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

# IMPORTANT: configure DeepEval timeouts BEFORE importing deepeval (settings
# are cached on first import). Local Ollama on a single 3090 is slower than
# cloud APIs; the default per-attempt timeout trips easily.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "3600")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "1800")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "2")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")

import requests

# --------------------------------------------------------------------------- #
# Paths and configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent

GROUND_TRUTH_DIR = Path(os.getenv(
    "GROUND_TRUTH_DIR",
    str(ROOT / "ground_truth"),
))
# Root of the Grand-Challenge input mount. Job outputs live under
# INPUT_DIRECTORY/<job_pk>/ and are located via the output socket relative_path.
INPUT_DIRECTORY = Path(os.getenv(
    "INPUT_DIRECTORY",
    str(ROOT / "test" / "input"),
))
PREDICTIONS_FILE = Path(os.getenv(
    "PREDICTIONS_FILE",
    str(INPUT_DIRECTORY / "predictions.json"),
))
OUTPUT_DIR = Path(os.getenv(
    "EVAL_OUTPUT_DIR",
    str(ROOT / "results"),
))
SECTION_MAPPING_FILE = Path(os.getenv(
    "SECTION_MAPPING_FILE",
    str(GROUND_TRUTH_DIR / "section_variable_mapping.json"),
))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma4:e4b")
USE_RATIONALE_JUDGE = bool(int(os.getenv("USE_RATIONALE_JUDGE", "1")))


# --------------------------------------------------------------------------- #
# Label conventions
# --------------------------------------------------------------------------- #

VALID_BIOPSY_DECISIONS = {"yes", "no"}
VALID_TREATMENT_DECISIONS = {
    "watchful_waiting",
    "active_surveillance",
    "continued_surveillance",
    "active_treatment",
}

CONF_MAP = {
    "uncertain":  0,
    "borderline": 1,
    "clear":      2,
}

WEIGHT_MAP = {
    "not_used":  0,
    "noted":     1,
    "important": 2,
    "decisive":  3,
}

# Returned by the form when the urologist never revealed a row. Treat as 0
# influence rather than crashing on label lookup.
WEIGHT_ALIAS = {
    "not_revealed": "not_used",
}

IMPORTANT_OR_DECISIVE = {"important", "decisive"}

# Label used in the dataset-level decision metrics when a case produced no
# usable prediction (job absent, or output that failed schema validation). It
# can never equal a ground-truth label, so such a case always counts as an
# error instead of silently dropping out of the F1.
MISSING_DECISION_LABEL = "__missing__"

# Reveal-sequence vocabulary. Ground truth and agent submissions name the
# sections they consulted with these six flat names, while
# section_variable_mapping.json still keys its primary_sections by the raw form
# section id, so that side is translated before anything is compared.
SECTION_KEY_TO_REVEAL_NAME = {
    "section_s3-prev": "previous_notes",
    "section_s3-labs": "laboratory_results",
    "section_s3-psa": "psa_trend",
    "section_s3-mri": "radiology_report",
    "section_s3-path": "pathology_report",
    "section_s3-fh": "family_history",
}

# Interface keys: the sorted tuple of a job's input socket slugs, exactly as the
# reference evaluation method identifies an interface. Used to pick the handler
# and group jobs into task-specific aggregates.
INTERF0_KEY = (
    "prostate-biopsy-decision-clinical-data",
    "prostate-modality-level-neural-representations",
    "structured-prompt",
)
INTERF1_KEY = (
    "prostate-modality-level-neural-representations",
    "prostate-treatment-decision-clinical-data",
    "structured-prompt",
)
INTERF2_KEY = (
    "prostate-modality-level-neural-representations",
    "prostate-time-to-recurrence-or-last-follow-up-clin",
    "structured-prompt",
)

INTERFACE_TASK_ID = {
    INTERF0_KEY: "task1",
    INTERF1_KEY: "task2",
    INTERF2_KEY: "task3",
}

# Output socket slugs per task, in preference order (predictions.json uses
# "biospy" and "-reas"; the per-case dumps use the corrected spellings —
# accept both so a fixed upstream slug keeps working).
TASK1_DECISION_SLUGS = ("prostate-biospy-decision", "prostate-biopsy-decision")
TASK1_REASONING_SLUGS = (
    "prostate-biospy-decision-reasoning",
    "prostate-biopsy-decision-reasoning",
)
TASK2_DECISION_SLUGS = ("prostate-treatment-decision",)
TASK2_REASONING_SLUGS = ("prostate-treatment-decision-reasoning",)
TASK3_OUTCOME_SLUGS = ("prostate-time-to-recurrence-or-last-follow-up",)
TASK3_REASONING_SLUGS = (
    "prostate-time-to-recurrence-or-last-follow-up-reas",
    "prostate-time-to-recurrence-or-last-follow-up-reasoning",
)
TASK3_CLIN_SLUGS = ("prostate-time-to-recurrence-or-last-follow-up-clin",)

# Input socket slugs used only when a job provides inline clinical context.
# File-backed production inputs have value=null, so clinical data normally
# comes from the corresponding ground-truth case directory.
STRUCTURED_PROMPT_SLUGS = ("structured-prompt",)
TASK1_CLIN_SLUGS = ("prostate-biopsy-decision-clinical-data",)
TASK2_CLIN_SLUGS = ("prostate-treatment-decision-clinical-data",)

# Ground-truth filenames per task. Task 1 and Task 2 ground truth mirrors the
# agent's own output sockets: a bare decision value plus a reasoning object.
# Task 3 ground truth stays a single {months_to_recurrence, event} object.
GT_FILENAMES = {
    "task1": {
        "decision": "prostate-biopsy-decision.json",
        "reasoning": "prostate-biopsy-decision-reasoning.json",
    },
    "task2": {
        "decision": "prostate-treatment-decision.json",
        "reasoning": "prostate-treatment-decision-reasoning.json",
    },
    "task3": {
        "outcome": "prostate-time-to-recurrence-or-last-follow-up.json",
    },
}

CLINICAL_FILENAMES = {
    "task1": "prostate-biopsy-decision-clinical-data.json",
    "task2": "prostate-treatment-decision-clinical-data.json",
    "task3": "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
}



# --------------------------------------------------------------------------- #
# JSON / record helpers
# --------------------------------------------------------------------------- #
CASE_MAP_FILE = Path(os.getenv(
    "CASE_MAP_FILE",
    str(GROUND_TRUTH_DIR / "debug_archive_pks.csv"),
))


def load_case_id_by_input_pk(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"Missing archive case map: {path}")

    mapping: dict[str, str] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            case_id = (row.get("case_id") or row.get("patient_title") or "").strip()
            if not case_id:
                continue

            raw_pk = row.get("structured-prompt_pk")
            if not raw_pk:
                continue
            pk = str(raw_pk).strip()
            previous = mapping.get(pk)
            if previous is not None and previous != case_id:
                raise RuntimeError(
                    f"Structured-prompt socket-value PK {pk} maps to both "
                    f"{previous!r} and {case_id!r}"
                )
            mapping[pk] = case_id

    return mapping


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth_records(root: Path, task_id: str) -> list[dict]:
    """Load one flattened ground-truth record per case from root/<case_id>/.

    Task 1 and Task 2 ground truth is split over the same two files the agent
    submits: a bare decision value and a reasoning object holding confidence,
    variable_weights, reveal_sequence and free_text. Task 3 ground truth is a
    single {months_to_recurrence, event} object. Records are flattened into the
    same shape the per-interface handlers build for predictions, so the scorers
    see identical keys on both sides.

    The case directory name is authoritative for case_id: the ground-truth
    files carry no case id of their own.
    """
    if not root.exists():
        sys.exit(f"Missing ground-truth directory: {root}")
    if not root.is_dir():
        sys.exit(f"ground-truth path is not a directory: {root}")
    names = GT_FILENAMES.get(task_id)
    if names is None:
        sys.exit(f"Unknown task id {task_id!r}; expected one of {sorted(GT_FILENAMES)}")

    records: list[dict] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        missing = [n for n in names.values() if not (case_dir / n).exists()]
        if missing:
            print(f"[warning] {case_dir.name}: missing ground-truth file(s) {missing}; skipping case")
            continue

        if task_id == "task3":
            outcome = load_json(case_dir / names["outcome"])
            record = dict(outcome) if isinstance(outcome, dict) else {}
        else:
            reasoning = load_json(case_dir / names["reasoning"])
            record = dict(reasoning) if isinstance(reasoning, dict) else {}
            decision = load_json(case_dir / names["decision"])
            if task_id == "task1":
                record["biopsy_decision"] = decision
            else:
                record["treatment_recommendation"] = {"primary": decision}

        clinical_path = case_dir / CLINICAL_FILENAMES[task_id]
        if clinical_path.exists():
            clinical_data = load_json(clinical_path)
            record["clinical_data"] = (
                clinical_data if isinstance(clinical_data, dict) else {}
            )
        else:
            record["clinical_data"] = {}
            if USE_RATIONALE_JUDGE:
                print(
                    f"[warning] {case_dir.name}: missing clinical data; "
                    "rationale judge will not receive the clinical context"
                )

        record["case_id"] = case_dir.name
        records.append(record)
    return records


def _socket_value(values: Any, slugs: tuple[str, ...]) -> Any:
    """Return an inline input value when a socket happens to provide one."""
    for slug in slugs:
        for sv in values or []:
            if isinstance(sv, dict) and sv.get("socket", {}).get("slug") == slug:
                return sv.get("value")
    return None


# --------------------------------------------------------------------------- #
# Grand-Challenge job input handling
#
# These helpers mirror the reference evaluation method
# (external/example_evaluation_method/evaluate.py) so that jobs are routed and
# their output files are located and read in exactly the same way. Only the
# scoring performed on the loaded results differs.
# --------------------------------------------------------------------------- #

def read_predictions() -> list[dict]:
    # The prediction file tells us all the tasks per job
    if not PREDICTIONS_FILE.exists():
        sys.exit(f"Missing predictions file: {PREDICTIONS_FILE}")
    jobs = load_json(PREDICTIONS_FILE)
    if not isinstance(jobs, list):
        sys.exit(f"predictions file is not a list of jobs: {PREDICTIONS_FILE}")
    return [job for job in jobs if isinstance(job, dict)]


def get_interface_key(job: dict) -> tuple[str, ...]:
    # Each interface is uniquely defined by the set of input sockets.
    socket_slugs = [sv["socket"]["slug"] for sv in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_interface_relative_path(*, values: list[dict], slug: str) -> str:
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["socket"]["relative_path"]
    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk: str, values: list[dict], slug: str) -> Path:
    # Where a job's output file will be located in the evaluation container.
    # Grand Challenge nests job outputs under <job_pk>/output/; the local test
    # fixtures keep them flat under <job_pk>/. Accept whichever is present.
    relative_path = get_interface_relative_path(values=values, slug=slug)
    flat = INPUT_DIRECTORY / job_pk / relative_path
    if flat.exists():
        return flat
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def get_file_location_any(
    *, job_pk: str, values: list[dict], slugs: tuple[str, ...]
) -> Path:
    """get_file_location over the accepted spellings of one output socket."""
    for slug in slugs:
        try:
            return get_file_location(job_pk=job_pk, values=values, slug=slug)
        except RuntimeError:
            continue
    raise RuntimeError(f"No value with any of the interfaces {slugs} found!")


def load_json_file(*, location: Path) -> Any:
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


# --------------------------------------------------------------------------- #
# Per-interface handlers
# --------------------------------------------------------------------------- #

def process(job: dict, ctx: dict) -> dict | None:
    """Processes a single algorithm job, looking at the outputs"""
    interface_key = get_interface_key(job)
    handler = {
        INTERF0_KEY: process_interf0,
        INTERF1_KEY: process_interf1,
        INTERF2_KEY: process_interf2,
    }[interface_key]
    return handler(job, ctx)


def process_interf0(job: dict, ctx: dict) -> dict | None:
    """Task 1 — prostate biopsy decision + reasoning."""
    case_id = _case_id_for_job(job, ctx)
    if not case_id:
        return None

    # Firstly, find the location of the results
    location_decision = get_file_location_any(
        job_pk=job["pk"], values=job["outputs"], slugs=TASK1_DECISION_SLUGS,
    )
    location_reasoning = get_file_location_any(
        job_pk=job["pk"], values=job["outputs"], slugs=TASK1_REASONING_SLUGS,
    )

    # Secondly, read the results
    result_decision = load_json_file(location=location_decision)
    result_reasoning = load_json_file(location=location_reasoning)

    # Thirdly, flatten them into the record shape the scorers expect
    pred = dict(result_reasoning) if isinstance(result_reasoning, dict) else {}
    pred["biopsy_decision"] = result_decision
    pred["case_id"] = case_id
    clin = _socket_value(job.get("inputs"), TASK1_CLIN_SLUGS)
    pred["clinical_data"] = clin if isinstance(clin, dict) else {}

    # Fourthly, score against the ground truth for this case
    return _score_job(job, ctx, pred)


def process_interf1(job: dict, ctx: dict) -> dict | None:
    """Task 2 — prostate treatment decision + reasoning."""
    case_id = _case_id_for_job(job, ctx)
    if not case_id:
        return None

    location_decision = get_file_location_any(
        job_pk=job["pk"], values=job["outputs"], slugs=TASK2_DECISION_SLUGS,
    )
    location_reasoning = get_file_location_any(
        job_pk=job["pk"], values=job["outputs"], slugs=TASK2_REASONING_SLUGS,
    )

    result_decision = load_json_file(location=location_decision)
    result_reasoning = load_json_file(location=location_reasoning)

    pred = dict(result_reasoning) if isinstance(result_reasoning, dict) else {}
    pred["treatment_recommendation"] = {"primary": result_decision}
    pred["case_id"] = case_id
    clin = _socket_value(job.get("inputs"), TASK2_CLIN_SLUGS)
    pred["clinical_data"] = clin if isinstance(clin, dict) else {}

    return _score_job(job, ctx, pred)


def process_interf2(job: dict, ctx: dict) -> dict | None:
    """Task 3 — time to biochemical recurrence + reasoning."""
    case_id = _case_id_for_job(job, ctx)
    if not case_id:
        return None

    location_outcome = get_file_location_any(
        job_pk=job["pk"], values=job["outputs"], slugs=TASK3_OUTCOME_SLUGS,
    )
    location_reasoning = get_file_location_any(
        job_pk=job["pk"], values=job["outputs"], slugs=TASK3_REASONING_SLUGS,
    )

    result_outcome = load_json_file(location=location_outcome)
    result_reasoning = load_json_file(location=location_reasoning)

    # Inline input values are accepted for compatibility; production inputs
    # are file-backed and fall back to the clinical copy in ground truth.
    clin = _socket_value(job.get("inputs"), TASK3_CLIN_SLUGS)

    outcome = result_outcome if isinstance(result_outcome, dict) else {}
    pred = {
        "case_id": case_id,
        "months_to_recurrence": outcome.get("months_to_recurrence"),
        "event": outcome.get("event"),
        "free_text": result_reasoning if isinstance(result_reasoning, str) else None,
        "clinical_data": clin if isinstance(clin, dict) else {},
    }

    return _score_job(job, ctx, pred)

def _case_id_for_job(job: dict, ctx: dict) -> str | None:
    # Keep compatibility with scalar/inline inputs if you introduce one later.
    prompt = _socket_value(job.get("inputs"), STRUCTURED_PROMPT_SLUGS)
    if isinstance(prompt, dict) and prompt.get("case_id"):
        return str(prompt["case_id"])

    # File-backed inputs have value=null. Join their ComponentInterfaceValue
    # PKs against the mapping exported after archive creation.
    matches = {
        ctx["case_id_by_input_pk"][str(sv.get("pk"))]
        for sv in job.get("inputs", [])
        if str(sv.get("pk")) in ctx["case_id_by_input_pk"]
    }

    if len(matches) == 1:
        return next(iter(matches))

    if len(matches) > 1:
        raise RuntimeError(
            f"Job {job.get('pk')} maps to multiple cases: {sorted(matches)}"
        )

    print(
        f"[warning] job {job.get('pk')} has no matching input socket-value "
        "PK in the archive case map; skipping"
    )
    return None

def _score_job(job: dict, ctx: dict, pred: dict) -> dict | None:
    """Score one loaded prediction against its ground-truth case."""
    case_id = pred["case_id"]
    task_id = INTERFACE_TASK_ID[get_interface_key(job)]
    gt = ctx["ground_truth"][task_id].get(case_id)
    if gt is None:
        print(f"[warning] no ground truth for case {case_id} (pk {job.get('pk')}); skipping")
        return None
    if not pred.get("clinical_data"):
        pred["clinical_data"] = gt.get("clinical_data", {})
    row = evaluate_case(gt, pred, ctx["tool_metric"], ctx["rationale_judge"])
    attach_kappa_fields(row, gt, pred)
    return row


def attach_kappa_fields(row: dict, gt: dict, pred: dict | None) -> None:
    """Attach raw label pairs needed for dataset-level kappas (kept out of CSV)."""
    row["gt_biopsy_decision_conf"] = _norm_conf(gt.get("confidence"))
    row["pred_biopsy_decision_conf"] = _norm_conf(pred.get("confidence")) if pred else None
    weight_pairs: list[tuple[int, int]] = []
    if pred and (row.get("decision_score") or 0.0) > 0.0:
        gt_w = gt.get("variable_weights") or {}
        pr_w = pred.get("variable_weights") or {}
        for var, gv in gt_w.items():
            g = _norm_weight(gv)
            if g is None:
                continue
            p = _norm_weight(pr_w.get(var, "not_used")) or "not_used"
            weight_pairs.append((WEIGHT_MAP[g], WEIGHT_MAP[p]))
    row["_weight_pairs"] = weight_pairs


def get_case_id(record: dict) -> str:
    cid = record.get("case_id")
    if cid:
        return str(cid)
    patient = record.get("patient") or {}
    if isinstance(patient, dict) and patient.get("id"):
        return str(patient["id"])
    return ""


def task_kind(record: dict) -> str:
    if "months_to_recurrence" in record:
        return "recurrence"
    if "treatment_recommendation" in record:
        return "treatment"
    return "biopsy"


def validate_record(record: dict, task: str) -> tuple[bool, str]:
    """Lightweight schema check on a candidate record."""
    if not isinstance(record, dict):
        return False, "candidate is not an object"
    if task == "recurrence":
        if _norm_event(record.get("event")) is None:
            return False, f"invalid event={record.get('event')!r}"
        if _norm_months(record.get("months_to_recurrence")) is None:
            return False, f"invalid months_to_recurrence={record.get('months_to_recurrence')!r}"
        return True, "ok"
    if task == "treatment":
        decision = _norm_treatment_decision(record)
        if decision not in VALID_TREATMENT_DECISIONS:
            raw = (record.get("treatment_recommendation") or {}).get("primary")
            return False, f"invalid treatment_recommendation.primary={raw!r}"
    else:
        decision = _norm_decision(record.get("biopsy_decision"))
        if decision not in VALID_BIOPSY_DECISIONS:
            return False, f"invalid biopsy_decision={record.get('biopsy_decision')!r}"
    weights = record.get("variable_weights")
    if weights is not None and not isinstance(weights, dict):
        return False, "variable_weights must be an object"
    return True, "ok"


def _norm_weight(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    v = WEIGHT_ALIAS.get(v, v)
    return v if v in WEIGHT_MAP else None


def _norm_conf(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in CONF_MAP else None


def _norm_decision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in VALID_BIOPSY_DECISIONS else None


def _norm_treatment_decision(record: dict | None) -> str | None:
    if not isinstance(record, dict):
        return None
    rec = record.get("treatment_recommendation") or {}
    if not isinstance(rec, dict):
        return None
    value = rec.get("primary")
    if not isinstance(value, str):
        return None
    v = value.strip().lower().replace("-", "_")
    v = "_".join(v.split())
    return v if v in VALID_TREATMENT_DECISIONS else None


def _norm_event(value: Any) -> int | None:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return None
    return iv if iv in (0, 1) else None


def _norm_months(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decision_score(task: str, gt: dict, pred: dict) -> tuple[float, str | None, str | None, str]:
    if task == "treatment":
        gt_decision = _norm_treatment_decision(gt)
        pred_decision = _norm_treatment_decision(pred)
        if gt_decision == pred_decision and gt_decision is not None:
            return 1.0, gt_decision, pred_decision, "treatment_decision matched"
        return 0.0, gt_decision, pred_decision, (
            f"treatment_decision mismatch: gt={gt_decision!r} pred={pred_decision!r}"
        )

    gt_decision = _norm_decision(gt.get("biopsy_decision"))
    pred_decision = _norm_decision(pred.get("biopsy_decision"))
    if gt_decision == pred_decision and gt_decision is not None:
        return 1.0, gt_decision, pred_decision, "biopsy_decision matched"
    return 0.0, gt_decision, pred_decision, (
        f"biopsy_decision mismatch: gt={gt_decision!r} pred={pred_decision!r}"
    )


# --------------------------------------------------------------------------- #
# Per-case deterministic component scores
# --------------------------------------------------------------------------- #

def confidence_score(gt: dict, pred: dict) -> float | None:
    g = _norm_conf(gt.get("confidence"))
    p = _norm_conf(pred.get("confidence"))
    if g is None or p is None:
        return None
    distance = abs(CONF_MAP[g] - CONF_MAP[p])
    max_dist = max(CONF_MAP.values()) - min(CONF_MAP.values())  # = 2
    return 1.0 - (distance / max_dist)


def variable_weight_score(gt: dict, pred: dict) -> float | None:
    gt_w = gt.get("variable_weights") or {}
    pr_w = pred.get("variable_weights") or {}
    if not isinstance(gt_w, dict) or not gt_w:
        return None
    max_w = max(WEIGHT_MAP.values()) - min(WEIGHT_MAP.values())  # = 3
    errors: list[float] = []
    for var, gv in gt_w.items():
        g = _norm_weight(gv)
        if g is None:
            continue
        # Missing prediction for a variable = treat as not_used.
        p = _norm_weight(pr_w.get(var, "not_used")) or "not_used"
        errors.append(abs(WEIGHT_MAP[g] - WEIGHT_MAP[p]) / max_w)
    if not errors:
        return None
    return 1.0 - mean(errors)


def _important_set(weights: dict) -> set[str]:
    if not isinstance(weights, dict):
        return set()
    out = set()
    for var, val in weights.items():
        v = _norm_weight(val)
        if v in IMPORTANT_OR_DECISIVE:
            out.add(var)
    return out


def _set_f1(gt_set: set, pred_set: set) -> float:
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def important_decisive_factor_score(gt: dict, pred: dict) -> float | None:
    gt_set = _important_set(gt.get("variable_weights") or {})
    pred_set = _important_set(pred.get("variable_weights") or {})
    if not gt_set and not pred_set:
        return 1.0
    return _set_f1(gt_set, pred_set)


# --------------------------------------------------------------------------- #
# Task 3: biochemical-recurrence (time-to-event) scores
# --------------------------------------------------------------------------- #

def recurrence_event_score(gt: dict, pred: dict) -> float | None:
    g = _norm_event(gt.get("event"))
    p = _norm_event(pred.get("event"))
    if g is None or p is None:
        return None
    return 1.0 if g == p else 0.0


def recurrence_time_score(gt: dict, pred: dict) -> float | None:
    """Censoring-aware closeness of predicted time-to-recurrence.

    For observed recurrences (event=1) the predicted time should match the true
    time. For censored cases (event=0) `months_to_recurrence` is the last
    follow-up time and the true event time is unknown but strictly later; we
    therefore only penalise predictions that recur *earlier* than that time.
    """
    g_event = _norm_event(gt.get("event"))
    g_t = _norm_months(gt.get("months_to_recurrence"))
    p_t = _norm_months(pred.get("months_to_recurrence"))
    if g_t is None or p_t is None:
        return None
    scale = max(g_t, 1.0)
    if g_event == 1:
        return max(0.0, 1.0 - abs(p_t - g_t) / scale)
    # Censored: predicting recurrence at or after the follow-up time is fine.
    if p_t >= g_t:
        return 1.0
    return max(0.0, 1.0 - (g_t - p_t) / scale)


def concordance_index(
    times: list[float], preds: list[float], events: list[int]
) -> float | None:
    """Harrell's concordance index.

    `preds` are predicted months-to-recurrence used as a risk ordering: a
    shorter predicted time means higher predicted risk. A pair is comparable
    when the earlier subject had an observed event (event=1).
    """
    num = 0.0
    den = 0.0
    n = len(times)
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j or not times[i] < times[j]:
                continue
            den += 1.0
            if preds[i] < preds[j]:
                num += 1.0
            elif preds[i] == preds[j]:
                num += 0.5
    return (num / den) if den > 0 else None


# Horizons (months) at which cumulative/dynamic AUC is reported.
TD_AUC_HORIZONS_MONTHS = (12.0, 24.0, 36.0, 60.0)


def _censoring_km(times: list[float], events: list[int]) -> list[tuple[float, float]]:
    """Kaplan-Meier estimate of the censoring survival G(t) = P(C > t).

    Returned as ascending (time, G) steps. Censoring is the "event" here, so a
    subject with event=0 counts as a censoring event.
    """
    order = sorted(range(len(times)), key=lambda i: times[i])
    steps: list[tuple[float, float]] = [(float("-inf"), 1.0)]
    at_risk = len(order)
    g = 1.0
    i = 0
    while i < len(order):
        t = times[order[i]]
        tied = 0
        censored = 0
        while i + tied < len(order) and times[order[i + tied]] == t:
            if events[order[i + tied]] == 0:
                censored += 1
            tied += 1
        if at_risk > 0 and censored > 0:
            g *= 1.0 - censored / at_risk
        steps.append((t, g))
        at_risk -= tied
        i += tied
    return steps


def _km_at(steps: list[tuple[float, float]], t: float) -> float:
    value = 1.0
    for step_t, step_g in steps:
        if step_t <= t:
            value = step_g
        else:
            break
    return value


def time_dependent_auc(
    times: list[float], preds: list[float], events: list[int], horizon: float
) -> float | None:
    """Uno's IPCW cumulative/dynamic AUC at one horizon.

    Cases are subjects with an observed recurrence at or before `horizon`;
    controls are subjects still recurrence-free after it. Censored subjects who
    drop out before the horizon are neither, and the remaining subjects are
    re-weighted by the inverse censoring probability to correct for that.
    Risk is `-predicted months`, so a shorter predicted time means higher risk.
    Returns None when the horizon has no case/control pair to compare.
    """
    steps = _censoring_km(times, events)
    cases: list[tuple[float, float]] = []
    controls: list[tuple[float, float]] = []
    for t, p, e in zip(times, preds, events):
        risk = -p
        if e == 1 and t <= horizon:
            g = _km_at(steps, t)
            if g > 0.0:
                cases.append((risk, 1.0 / g))
        elif t > horizon:
            g = _km_at(steps, horizon)
            if g > 0.0:
                controls.append((risk, 1.0 / g))
    if not cases or not controls:
        return None

    num = 0.0
    for case_risk, case_w in cases:
        for ctrl_risk, ctrl_w in controls:
            if case_risk > ctrl_risk:
                num += case_w * ctrl_w
            elif case_risk == ctrl_risk:
                num += 0.5 * case_w * ctrl_w
    den = sum(w for _, w in cases) * sum(w for _, w in controls)
    return (num / den) if den > 0 else None


# --------------------------------------------------------------------------- #
# Section-variable grounding
# --------------------------------------------------------------------------- #

_SECTION_VAR_MAPPING: dict = {}


def _get_section_var_mapping() -> dict:
    global _SECTION_VAR_MAPPING
    if not _SECTION_VAR_MAPPING:
        if SECTION_MAPPING_FILE.exists():
            _SECTION_VAR_MAPPING = load_json(SECTION_MAPPING_FILE)
        else:
            print(
                f"[warning] section_variable_mapping.json not found at "
                f"{SECTION_MAPPING_FILE}; section grounding check disabled"
            )
    return _SECTION_VAR_MAPPING


def section_grounding_score(pred: dict) -> tuple[float | None, dict]:
    """
    Penalise variables the agent weighted above 'not_used' whose primary
    source section was never revealed in the agent's own reveal_sequence.

    A variable is 'grounded' if:
      - It is an always-available variable (psa, age) readable from the
        patient card without any section reveal, OR
      - Its primary_sections list is empty, OR
      - At least one of its primary_sections appears in the agent's
        reveal_sequence.

    A variable whose primary sections are all outside the reveal vocabulary
    (e.g. comorbidities, which no submission can declare as revealed) is
    ungradable: it is excluded from the score entirely rather than counted as
    ungrounded, which would be an unavoidable penalty.

    Score = n_grounded / (n_grounded + n_ungrounded)

    Returns (None, details) when no variable is actively weighted (no
    penalisation possible).
    """
    mapping = _get_section_var_mapping()
    if not mapping:
        return None, {"grounded_variables": [], "ungrounded_variables": [],
                      "ungradable_variables": [],
                      "total_weighted": 0, "revealed_sections": []}

    var_to_sections = mapping.get("variable_to_sections", {})
    always_available = set(
        mapping.get("always_available_variables", {}).get("variables", [])
    )

    revealed = set(_reveal_keys(pred))
    weights = pred.get("variable_weights") or {}

    grounded: list[str] = []
    ungrounded: list[str] = []
    ungradable: list[str] = []

    for var, weight_val in weights.items():
        w = _norm_weight(weight_val)
        if w is None or w == "not_used":
            continue  # variable not actively used — skip

        # Always-available variables (psa, age) need no section reveal.
        if var in always_available:
            grounded.append(var)
            continue

        var_info = var_to_sections.get(var, {})
        primary_sections = var_info.get("primary_sections", [])
        always_avail_flag = var_info.get("always_available_baseline", False)

        if always_avail_flag or not primary_sections:
            grounded.append(var)
            continue

        primary_names = [
            SECTION_KEY_TO_REVEAL_NAME[s]
            for s in primary_sections
            if s in SECTION_KEY_TO_REVEAL_NAME
        ]
        if not primary_names:
            ungradable.append(var)
            continue

        # Grounded if at least one primary section was revealed.
        if any(s in revealed for s in primary_names):
            grounded.append(var)
        else:
            ungrounded.append(var)

    total = len(grounded) + len(ungrounded)
    if total == 0:
        return None, {
            "grounded_variables": [],
            "ungrounded_variables": [],
            "ungradable_variables": sorted(ungradable),
            "total_weighted": 0,
            "revealed_sections": sorted(revealed),
        }

    return len(grounded) / total, {
        "grounded_variables": sorted(grounded),
        "ungrounded_variables": sorted(ungrounded),
        "ungradable_variables": sorted(ungradable),
        "total_weighted": total,
        "revealed_sections": sorted(revealed),
    }


# --------------------------------------------------------------------------- #
# Reveal sequence / tool-use
# --------------------------------------------------------------------------- #

def _reveal_keys(record: dict) -> list[str]:
    """Return revealed section names in reveal order, deduplicated.

    ``reveal_sequence`` is a flat, already-ordered list of section names, e.g.
    ["radiology_report", "laboratory_results"]. Names outside the recognised
    vocabulary are kept verbatim, so inventing a section still counts as an
    extra (penalised) reveal.
    """
    seq = record.get("reveal_sequence") or []
    if not isinstance(seq, list):
        return []
    seen: set[str] = set()
    keys: list[str] = []
    for item in seq:
        if not isinstance(item, str) or not item:
            continue
        if item in seen:
            continue
        keys.append(item)
        seen.add(item)
    return keys


def cost_aware_tool_score(gt: dict, pred: dict) -> tuple[float, dict]:
    """
    Uniform-cost asymmetric tool score.

    Rule:
      - Penalize agent tools that the pathologist did NOT use (extra tools).
      - Do NOT penalize missing pathologist tools (under-use is fine).
      - Ignore reveal order.
      - Each extra tool incurs a uniform cost.

    Score = |agent_tools ∩ pathologist_tools| / |agent_tools|
          = precision of agent tool usage.

    If the agent uses no tools, return 1.0 (no unnecessary cost incurred).
    """
    expected = set(_reveal_keys(gt))
    actual = set(_reveal_keys(pred))

    if not actual:
        return 1.0, {
            "expected_tools": sorted(expected),
            "actual_tools": [],
            "extra_tools": [],
            "missing_tools_not_penalized": sorted(expected),
            "n_extra": 0,
            "n_actual": 0,
            "policy": "no_actual_tools_no_extra_cost",
        }

    extra = actual - expected
    approved = actual & expected
    missing = expected - actual
    score = len(approved) / len(actual)

    return score, {
        "expected_tools": sorted(expected),
        "actual_tools": sorted(actual),
        "approved_tools": sorted(approved),
        "extra_tools": sorted(extra),
        "missing_tools_not_penalized": sorted(missing),
        "n_extra": len(extra),
        "n_actual": len(actual),
        "policy": "penalize_extra_only_ignore_order_uniform_cost",
    }


def reveal_sequence_to_tool_calls(record: dict) -> list:
    """Convert a reveal_sequence into DeepEval ToolCall objects."""
    try:
        from deepeval.test_case import ToolCall  # type: ignore
    except Exception:
        return []
    seq = record.get("reveal_sequence") or []
    if not isinstance(seq, list):
        return []
    seq = sorted(seq, key=lambda x: x.get("order", 10**9) if isinstance(x, dict) else 10**9)
    calls = []
    for item in seq:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not key:
            continue
        # Construct a ToolCall in a defensive way; field names are stable
        # across recent deepeval versions but optional kwargs vary.
        tool_name = f"reveal_{key}"
        try:
            calls.append(ToolCall(
                name=tool_name,
                input_parameters={"key": str(key)},
                description=item.get("label"),
                reasoning=f"Reveal action from {item.get('via', 'unknown')}",
                output=item.get("value"),
            ))
        except TypeError:
            calls.append(ToolCall(
                name=tool_name,
                input_parameters={"key": str(key)},
            ))
    return calls


def build_tool_metric():
    """
    ToolCorrectnessMetric is NOT used as the primary score.

    Our policy is asymmetric: penalize extra agent tools only, do not penalize
    missing tools, ignore order. cost_aware_tool_score implements this exactly.
    DeepEval's ToolCorrectnessMetric asks "did the agent call the expected tools?"
    which would penalise under-use — the opposite of what we want.

    This stub keeps call-sites unchanged.
    """
    return None


def compute_tool_score(gt: dict, pred: dict, tool_metric) -> tuple[float, str]:
    """Return (score in [0,1], short reason) using cost_aware_tool_score.

    Metric: Tool Efficiency Precision
      score = |agent_tools ∩ pathologist_tools| / |agent_tools|
    Extra agent tools are penalised uniformly; missing tools are not penalised.
    """
    gt_keys = _reveal_keys(gt)
    pred_keys = _reveal_keys(pred)
    if not gt_keys and not pred_keys:
        return 1.0, "no reveals expected or produced"

    score, details = cost_aware_tool_score(gt, pred)
    n_extra = details["n_extra"]
    n_actual = details["n_actual"]
    extra = details.get("extra_tools", [])
    reason = (
        f"precision={score:.3f} extra={n_extra}/{n_actual}"
        + (f" extra_keys={extra}" if extra else "")
    )
    return score, reason


# --------------------------------------------------------------------------- #
# Rationale judge (LLM)
# --------------------------------------------------------------------------- #

def wait_for_ollama(base_url: str, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                print(f"[ollama] reachable at {base_url}")
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2)
    raise RuntimeError(f"Ollama not reachable at {base_url}: {last_err}")


def ensure_model_pulled(base_url: str, model: str) -> None:
    tags = requests.get(f"{base_url}/api/tags", timeout=5).json()
    have = {m["name"] for m in tags.get("models", [])}
    if model in have or any(name.startswith(model) for name in have):
        print(f"[ollama] model '{model}' already present")
        return
    print(f"[ollama] pulling '{model}' (first run takes a while) ...")
    with requests.post(
        f"{base_url}/api/pull",
        json={"name": model, "stream": True},
        stream=True,
        timeout=None,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in msg:
                raise RuntimeError(f"ollama pull error: {msg['error']}")
            if msg.get("status"):
                print(f"[ollama pull] {msg['status']}")
    print(f"[ollama] pull complete: {model}")


def build_rationale_judge():
    """Return a callable judge(gt, pred) -> (score in [0,1] or None, reason)."""
    if not USE_RATIONALE_JUDGE:
        return None
    try:
        wait_for_ollama(OLLAMA_BASE_URL)
        ensure_model_pulled(OLLAMA_BASE_URL, JUDGE_MODEL)
    except Exception as exc:  # noqa: BLE001
        print(f"[judge] ollama bootstrap failed ({exc}); rationale judging disabled")
        return None

    try:
        from deepeval.metrics import GEval  # type: ignore
        from deepeval.models import OllamaModel  # type: ignore
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[judge] DeepEval GEval unavailable ({exc}); rationale judging disabled")
        return None

    model = OllamaModel(model=JUDGE_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    rubric = (
        "Score the agent's free-text rationale (Actual Output) against the "
        "pathologist's rationale (Expected Output) and the case's expected clinical "
        "decision. Score HIGH if the rationale: (1) supports the same "
        "decision; (2) cites the same important/decisive clinical "
        "variables; (3) does not contradict the clinical_data; (4) does not "
        "invent unavailable information; (5) expresses uncertainty consistent "
        "with the stated confidence. Score LOW if it contradicts the decision, "
        "misses major decisive factors, hallucinates clinical facts, or gives "
        "generic case-agnostic reasoning."
    )

    geval = GEval(
        name="RationaleAlignment",
        model=model,
        criteria=rubric,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        async_mode=False,
        verbose_mode=False,
        strict_mode=False,
    )

    recurrence_rubric = (
        "Score the agent's free-text rationale (Actual Output) for a prostate "
        "biochemical-recurrence (time-to-event) prediction. Use the clinical "
        "inputs and reference outcome given in Input and the agent's own "
        "predicted event/time in Actual Output. Score HIGH if the rationale: "
        "(1) is consistent with its own predicted recurrence event and timing; "
        "(2) cites concrete post-operative prognostic features actually present "
        "in the clinical inputs (e.g. Gleason/ISUP grade, pathological stage, "
        "surgical margin status, seminal-vesicle invasion, extraprostatic "
        "extension, lymph-node status, PSA / PSA density); (3) does not "
        "contradict the clinical inputs; (4) does not invent unavailable "
        "information. Score LOW if it contradicts the inputs, ignores major "
        "prognostic factors, hallucinates facts, or gives generic case-agnostic "
        "reasoning. There is no reference free-text rationale, so judge clinical "
        "soundness and internal consistency, not verbatim agreement."
    )

    geval_recurrence = GEval(
        name="RecurrenceRationale",
        model=model,
        criteria=recurrence_rubric,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        async_mode=False,
        verbose_mode=False,
        strict_mode=False,
    )

    def judge(gt: dict, pred: dict) -> tuple[float | None, str]:
        if task_kind(gt) == "recurrence":
            pred_text = (pred.get("free_text") or "").strip()
            if not pred_text:
                return None, "missing free_text on pred"
            gt_event = _norm_event(gt.get("event"))
            gt_months = _norm_months(gt.get("months_to_recurrence"))
            input_ctx = {
                "case_id": get_case_id(gt),
                "task": "biochemical_recurrence",
                "clinical_inputs": pred.get("clinical_data", {}),
                "reference_event": gt_event,
                "reference_months_to_recurrence": gt_months,
            }
            actual = {
                "event": _norm_event(pred.get("event")),
                "months_to_recurrence": _norm_months(pred.get("months_to_recurrence")),
                "free_text": pred_text,
            }
            expected = {"event": gt_event, "months_to_recurrence": gt_months}
            tc = LLMTestCase(
                input=json.dumps(input_ctx, ensure_ascii=False),
                actual_output=json.dumps(actual, ensure_ascii=False),
                expected_output=json.dumps(expected, ensure_ascii=False),
            )
            try:
                geval_recurrence.measure(tc)
                score = getattr(geval_recurrence, "score", None)
                reason = getattr(geval_recurrence, "reason", "") or "GEval"
                if score is None:
                    return None, f"GEval returned no score: {reason}"
                return max(0.0, min(1.0, float(score))), str(reason)
            except Exception as exc:  # noqa: BLE001
                return None, f"GEval error: {exc}"

        gt_text = (gt.get("free_text") or "").strip()
        pred_text = (pred.get("free_text") or "").strip()
        if not gt_text or not pred_text:
            return None, "missing free_text on gt or pred"

        task = task_kind(gt)
        gt_decision = _norm_treatment_decision(gt) if task == "treatment" else _norm_decision(gt.get("biopsy_decision"))
        pred_decision = _norm_treatment_decision(pred) if task == "treatment" else _norm_decision(pred.get("biopsy_decision"))

        # The clinical context comes from the job's own inline input socket:
        # the ground truth now carries only the decision and reasoning fields.
        input_ctx = {
            "case_id": get_case_id(gt),
            "task": task,
            "clinical_data": pred.get("clinical_data", {}),
            "expected_decision": gt_decision,
            "expected_confidence": gt.get("confidence"),
        }
        actual = {
            "decision": pred_decision,
            "confidence": pred.get("confidence"),
            "free_text": pred_text,
        }
        expected = {
            "decision": gt_decision,
            "confidence": gt.get("confidence"),
            "free_text": gt_text,
        }

        tc = LLMTestCase(
            input=json.dumps(input_ctx, ensure_ascii=False),
            actual_output=json.dumps(actual, ensure_ascii=False),
            expected_output=json.dumps(expected, ensure_ascii=False),
        )
        try:
            geval.measure(tc)
            score = getattr(geval, "score", None)
            reason = getattr(geval, "reason", "") or "GEval"
            if score is None:
                return None, f"GEval returned no score: {reason}"
            return max(0.0, min(1.0, float(score))), str(reason)
        except Exception as exc:  # noqa: BLE001
            return None, f"GEval error: {exc}"

    return judge


# --------------------------------------------------------------------------- #
# Per-case + dataset-level evaluation
# --------------------------------------------------------------------------- #

def evaluate_recurrence_case(
    gt: dict,
    pred: dict | None,
    rationale_judge,
) -> dict:
    case_id = get_case_id(gt)
    gt_event = _norm_event(gt.get("event"))
    gt_months = _norm_months(gt.get("months_to_recurrence"))

    base = {
        "case_id": case_id,
        "task": "recurrence",
        "gate": "passed",
        "case_score": 0.0,
        "gt_event": gt_event,
        "pred_event": None,
        "gt_months": gt_months,
        "pred_months": None,
        "event_score": None,
        "time_score": None,
        "rationale_score": None,
        "reason": "",
    }

    if pred is None:
        base["gate"] = "missing_candidate"
        base["reason"] = "no candidate record for this case"
        return base

    ok, why = validate_record(pred, "recurrence")
    if not ok:
        base["gate"] = "schema_failed"
        base["reason"] = f"schema validation failed: {why}"
        base["pred_event"] = pred.get("event")
        base["pred_months"] = pred.get("months_to_recurrence")
        return base

    es = recurrence_event_score(gt, pred)
    tsc = recurrence_time_score(gt, pred)

    rs, r_reason = (None, "rationale judge disabled")
    if rationale_judge is not None:
        rs, r_reason = rationale_judge(gt, pred)

    base["pred_event"] = _norm_event(pred.get("event"))
    base["pred_months"] = _norm_months(pred.get("months_to_recurrence"))
    base["event_score"] = es
    base["time_score"] = tsc
    base["rationale_score"] = rs

    # Weighted composite. Drop the reasoning weight when the judge is
    # unavailable and renormalise onto the deterministic components.
    components = {
        "event":     (es,  0.35),
        "time":      (tsc, 0.35),
        "reasoning": (rs,  0.30),
    }
    if rs is None:
        components = {
            "event": (es, 0.50),
            "time":  (tsc, 0.50),
        }

    score = sum((v if v is not None else 0.0) * w for v, w in components.values())
    base["case_score"] = max(0.0, min(1.0, score))

    es_str = "n/a" if es is None else f"{es:.1f}"
    ts_str = "n/a" if tsc is None else f"{tsc:.3f}"
    parts = [f"event={es_str} time={ts_str}"]
    if rationale_judge is not None:
        parts.append(f"rationale: {r_reason}")
    base["reason"] = " | ".join(parts)
    return base


def evaluate_case(
    gt: dict,
    pred: dict | None,
    tool_metric,
    rationale_judge,
) -> dict:
    task = task_kind(gt)
    if task == "recurrence":
        return evaluate_recurrence_case(gt, pred, rationale_judge)
    case_id = get_case_id(gt)
    gt_decision = _norm_treatment_decision(gt) if task == "treatment" else _norm_decision(gt.get("biopsy_decision"))

    base = {
        "case_id": case_id,
        "task": task,
        "gate": "passed",
        "case_score": 0.0,
        "decision_score": 0.0,
        "decision_correct": False,
        "gt_decision": gt_decision,
        "pred_decision": None,
        "biopsy_decision_correct": False,
        "gt_biopsy_decision": gt_decision if task == "biopsy" else None,
        "pred_biopsy_decision": None,
        "treatment_decision_correct": False,
        "gt_treatment_decision": gt_decision if task == "treatment" else None,
        "pred_treatment_decision": None,
        "confidence_score": None,
        "variable_weight_score": None,
        "important_decisive_factor_score": None,
        "tool_score": None,
        "section_grounding_score": None,
        "rationale_score": None,
        "reason": "",
    }

    if pred is None:
        base["gate"] = "missing_candidate"
        base["reason"] = "no candidate record for this case"
        return base

    ok, why = validate_record(pred, task)
    if not ok:
        base["gate"] = "schema_failed"
        base["reason"] = f"schema validation failed: {why}"
        if task == "biopsy":
            base["pred_biopsy_decision"] = pred.get("biopsy_decision")
        else:
            rec = pred.get("treatment_recommendation") or {}
            base["pred_treatment_decision"] = rec.get("primary") if isinstance(rec, dict) else None
            base["pred_decision"] = base["pred_treatment_decision"]
        return base

    ds, gt_decision, pred_decision, d_reason = decision_score(task, gt, pred)
    base["decision_score"] = ds
    base["gt_decision"] = gt_decision
    base["pred_decision"] = pred_decision
    base["decision_correct"] = ds == 1.0
    if task == "biopsy":
        base["gt_biopsy_decision"] = gt_decision
        base["pred_biopsy_decision"] = pred_decision
        base["biopsy_decision_correct"] = ds == 1.0
    else:
        base["gt_treatment_decision"] = gt_decision
        base["pred_treatment_decision"] = pred_decision
        base["treatment_decision_correct"] = ds == 1.0

    if ds == 0.0:
        base["gate"] = f"{task}_decision_failed"
        base["reason"] = d_reason
        return base

    # Granular evaluation
    cs = confidence_score(gt, pred)
    vws = variable_weight_score(gt, pred)
    fs = important_decisive_factor_score(gt, pred)
    ts, t_reason = compute_tool_score(gt, pred, tool_metric)
    sgs, sg_details = section_grounding_score(pred)

    rs, r_reason = (None, "rationale judge disabled")
    if rationale_judge is not None:
        rs, r_reason = rationale_judge(gt, pred)

    base["confidence_score"] = cs
    base["variable_weight_score"] = vws
    base["important_decisive_factor_score"] = fs
    base["tool_score"] = ts
    base["section_grounding_score"] = sgs
    base["rationale_score"] = rs

    # Weighted composite. Drop rationale weight when unavailable and
    # renormalise the remaining weights.
    components = {
        "confidence":        (cs,  0.20),
        "var_weight":        (vws, 0.25),
        "factor_f1":         (fs,  0.15),
        "tool":              (ts,  0.15),
        "section_grounding": (sgs, 0.15),
        "rationale":         (rs,  0.10),
    }

    if rs is None:
        components = {
            "confidence":        (cs,  0.225),
            "var_weight":        (vws, 0.275),
            "factor_f1":         (fs,  0.175),
            "tool":              (ts,  0.150),
            "section_grounding": (sgs, 0.175),
        }

    # Replace any component that is None with 0 so the math is well-defined,
    # but record it in the reason.
    missing = [k for k, (v, _) in components.items() if v is None]
    score = sum((v if v is not None else 0.0) * w for v, w in components.values())
    base["case_score"] = max(0.0, min(1.0, score))

    parts = []
    parts.append(f"tool: {t_reason}")
    sg_ungrounded = sg_details.get("ungrounded_variables", [])
    if sg_ungrounded:
        parts.append(f"ungrounded_vars={sg_ungrounded}")
    if rationale_judge is not None:
        parts.append(f"rationale: {r_reason}")
    if missing:
        parts.append("missing components zeroed: " + ", ".join(missing))
    base["reason"] = " | ".join(parts)
    return base


def aggregate_recurrence_metrics(rows: list[dict]) -> dict:
    """Dataset-level aggregation for Task 3 (biochemical recurrence)."""
    n = len(rows)
    mean_case_score = mean(r["case_score"] for r in rows)

    evaluated = [
        r for r in rows
        if r.get("pred_event") is not None and r.get("pred_months") is not None
    ]
    event_pairs = [
        (r["gt_event"], r["pred_event"]) for r in rows
        if r.get("gt_event") is not None and r.get("pred_event") is not None
    ]
    event_scores = [r["event_score"] for r in rows if r.get("event_score") is not None]
    time_scores = [r["time_score"] for r in rows if r.get("time_score") is not None]
    rationale_scores = [r["rationale_score"] for r in rows if r.get("rationale_score") is not None]
    maes = [
        abs(r["pred_months"] - r["gt_months"]) for r in rows
        if r.get("gt_event") == 1
        and r.get("pred_months") is not None
        and r.get("gt_months") is not None
    ]

    times: list[float] = []
    preds: list[float] = []
    events: list[int] = []
    for r in rows:
        if (
            r.get("gt_months") is not None
            and r.get("pred_months") is not None
            and r.get("gt_event") is not None
        ):
            times.append(r["gt_months"])
            preds.append(r["pred_months"])
            events.append(r["gt_event"])

    td_auc = {
        f"{int(h)}m": time_dependent_auc(times, preds, events, h)
        for h in TD_AUC_HORIZONS_MONTHS
    }
    td_auc_values = [v for v in td_auc.values() if v is not None]
    c_index = concordance_index(times, preds, events)

    return {
        "n_cases": n,
        "n_evaluated": len(evaluated),
        "mean_case_score": mean_case_score,
        # Task 3 is ranked on the concordance index alone. Time-dependent AUC
        # and mean case score are diagnostic and do not feed the leaderboard.
        "ranking_score": c_index,
        "recurrence_event_accuracy": (mean(int(a == b) for a, b in event_pairs) if event_pairs else None),
        "mean_event_score": (mean(event_scores) if event_scores else None),
        "mean_time_score": (mean(time_scores) if time_scores else None),
        "event1_time_mae_months": (mean(maes) if maes else None),
        "concordance_index": c_index,
        "time_dependent_auc": td_auc,
        "mean_time_dependent_auc": (mean(td_auc_values) if td_auc_values else None),
        "mean_rationale_score": (mean(rationale_scores) if rationale_scores else None),
    }


def compute_aggregate_metrics(rows: list[dict]) -> dict:
    """Dataset-level aggregation. Pure-python fallback when sklearn missing."""
    n = len(rows)
    if n == 0:
        return {"n_cases": 0}

    if all(r.get("task") == "recurrence" for r in rows):
        return aggregate_recurrence_metrics(rows)

    mean_case_score = mean(r["case_score"] for r in rows)

    # The task decides which F1 goes on the leaderboard, so read it off the
    # rows rather than inferring it from which labels happen to be present.
    is_treatment = all(r.get("task") == "treatment" for r in rows)

    # Every case with a known ground-truth decision is graded. Cases whose
    # prediction is absent or schema-invalid get a sentinel label so they count
    # as errors: dropping them would make skipping a hard case free.
    graded = [r for r in rows if r.get("gt_decision") is not None]
    y_true = [r["gt_decision"] for r in graded]
    y_pred = [r.get("pred_decision") or MISSING_DECISION_LABEL for r in graded]
    n_evaluated = sum(1 for r in graded if r.get("pred_decision") is not None)
    n_correct = sum(int(r.get("decision_score") == 1.0) for r in graded)
    n_incorrect = len(graded) - n_correct

    conf_pairs = [
        (CONF_MAP[r["gt_biopsy_decision_conf"]], CONF_MAP[r["pred_biopsy_decision_conf"]])
        for r in rows
        if r.get("gt_biopsy_decision_conf") in CONF_MAP
        and r.get("pred_biopsy_decision_conf") in CONF_MAP
    ] if any("gt_biopsy_decision_conf" in r for r in rows) else []

    flat_gt_w = []
    flat_pred_w = []
    for r in rows:
        for gw, pw in r.get("_weight_pairs", []):
            flat_gt_w.append(gw)
            flat_pred_w.append(pw)

    gate_pass = [r for r in rows if r.get("decision_score", 0.0) > 0.0]
    gate_pass_rate = len(gate_pass) / n
    mean_among_pass = mean(r["case_score"] for r in gate_pass) if gate_pass else 0.0

    tool_scores = [r["tool_score"] for r in rows if r["tool_score"] is not None]
    section_grounding_scores = [r["section_grounding_score"] for r in rows if r["section_grounding_score"] is not None]
    rationale_scores = [r["rationale_score"] for r in rows if r["rationale_score"] is not None]

    out = {
        "n_cases": n,
        "n_evaluated": n_evaluated,
        "n_decision_correct": n_correct,
        "n_decision_incorrect": n_incorrect,
        "mean_case_score": mean_case_score,
        # Leaderboard ranking score, filled in below once the F1 is known.
        "ranking_score": None,
        "decision_accuracy": None,
        "decision_f1_yes": None,
        "decision_weighted_f1": None,
        "confidence_weighted_kappa": None,
        "variable_weight_weighted_kappa": None,
        "mean_tool_score": mean(tool_scores) if tool_scores else None,
        "mean_section_grounding_score": mean(section_grounding_scores) if section_grounding_scores else None,
        "mean_rationale_score": mean(rationale_scores) if rationale_scores else None,
        "decision_gate_pass_rate": gate_pass_rate,
        "mean_case_score_among_gate_passed": mean_among_pass,
    }

    # sklearn metrics (graceful fallback if missing).
    try:
        from sklearn.metrics import (
            f1_score,
            accuracy_score,
            cohen_kappa_score,
            classification_report,
        )
    except Exception as exc:  # noqa: BLE001
        out["sklearn_unavailable"] = str(exc)
        if y_true:
            out["decision_accuracy"] = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
        return out

    if y_true:
        out["decision_accuracy"] = float(accuracy_score(y_true, y_pred))
        if is_treatment:
            # Support-weighted F1 over the four treatment classes. Restricting
            # `labels` keeps the sentinel out of the class list while still
            # letting it cost the true class its recall.
            try:
                out["decision_weighted_f1"] = float(f1_score(
                    y_true, y_pred,
                    labels=sorted(VALID_TREATMENT_DECISIONS),
                    average="weighted", zero_division=0,
                ))
            except Exception:
                out["decision_weighted_f1"] = None
        else:
            # Positive-class F1 for biopsy 'yes'. average=None with an explicit
            # single label avoids the binary-average restriction, which the
            # sentinel label would otherwise trip.
            try:
                out["decision_f1_yes"] = float(f1_score(
                    y_true, y_pred, labels=["yes"], average=None, zero_division=0,
                )[0])
            except Exception:
                out["decision_f1_yes"] = None
        out["decision_classification_report"] = classification_report(
            y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)), zero_division=0,
        )

    if conf_pairs:
        ct, cp = zip(*conf_pairs)
        # Weighted kappa is undefined with fewer than 2 distinct labels (e.g. a
        # tiny cohort where every case shares one confidence level); skip the
        # sklearn call in that case to avoid an UndefinedMetricWarning + NaN.
        if len(set(ct) | set(cp)) >= 2:
            try:
                k = float(cohen_kappa_score(list(ct), list(cp), weights="quadratic"))
                out["confidence_weighted_kappa"] = k if k == k else None
            except Exception:
                out["confidence_weighted_kappa"] = None

    if flat_gt_w:
        if len(set(flat_gt_w) | set(flat_pred_w)) >= 2:
            try:
                k = float(cohen_kappa_score(flat_gt_w, flat_pred_w, weights="quadratic"))
                out["variable_weight_weighted_kappa"] = k if k == k else None
            except Exception:
                out["variable_weight_weighted_kappa"] = None

    # Leaderboard ranking score: the mean case score and the task's decision F1
    # contribute equally. F1 for the positive class on Task 1, support-weighted
    # F1 across the four treatment classes on Task 2.
    task_f1 = out["decision_weighted_f1"] if is_treatment else out["decision_f1_yes"]
    if task_f1 is not None:
        out["ranking_score"] = (mean_case_score + task_f1) / 2.0

    return out


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

CSV_COLUMNS = [
    "case_id",
    "task",
    "gate",
    "case_score",
    "decision_score",
    "decision_correct",
    "gt_decision",
    "pred_decision",
    "biopsy_decision_correct",
    "gt_biopsy_decision",
    "pred_biopsy_decision",
    "treatment_decision_correct",
    "gt_treatment_decision",
    "pred_treatment_decision",
    "confidence_score",
    "variable_weight_score",
    "important_decisive_factor_score",
    "tool_score",
    "section_grounding_score",
    "gt_event",
    "pred_event",
    "gt_months",
    "pred_months",
    "event_score",
    "time_score",
    "rationale_score",
    "reason",
]

# Task 3 has no decision gate, confidence, variable weights, reveal sequence or
# section grounding, so none of those columns apply. Emitting them would only
# produce a wall of empty cells and imply scores that were never computed.
RECURRENCE_CSV_COLUMNS = [
    "case_id",
    "task",
    "gate",
    "case_score",
    "gt_event",
    "pred_event",
    "gt_months",
    "pred_months",
    "event_score",
    "time_score",
    "rationale_score",
    "reason",
]


def write_csv(rows: list[dict], path: Path) -> None:
    columns = (
        RECURRENCE_CSV_COLUMNS
        if rows and all(r.get("task") == "recurrence" for r in rows)
        else CSV_COLUMNS
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions()
    case_id_by_input_pk = load_case_id_by_input_pk(CASE_MAP_FILE)
    task_ids = sorted({
        task_id
        for job in predictions
        if (task_id := INTERFACE_TASK_ID.get(get_interface_key(job))) is not None
    })
    if not task_ids:
        raise RuntimeError(
            "No jobs use a recognized CHIMERA task interface; check predictions.json"
        )

    mapping_ctx = {"case_id_by_input_pk": case_id_by_input_pk}
    phase_case_ids: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
    for job in predictions:
        task_id = INTERFACE_TASK_ID.get(get_interface_key(job))
        if task_id is None:
            continue
        case_id = _case_id_for_job(job, mapping_ctx)
        if case_id is None:
            raise RuntimeError(
                f"Job {job.get('pk')} could not be associated with ground truth"
            )
        phase_case_ids[task_id].add(case_id)

    targets_by_task: dict[str, list[dict]] = {}
    for task_id in task_ids:
        all_targets = load_ground_truth_records(GROUND_TRUTH_DIR / task_id, task_id)
        targets_by_task[task_id] = [
            gt for gt in all_targets if get_case_id(gt) in phase_case_ids[task_id]
        ]
        found_case_ids = {get_case_id(gt) for gt in targets_by_task[task_id]}
        missing_case_ids = sorted(phase_case_ids[task_id] - found_case_ids)
        if missing_case_ids:
            raise RuntimeError(
                f"Missing {task_id} ground truth for phase cases: {missing_case_ids}"
            )

    print(
        "Loaded target cases: "
        + ", ".join(
            f"{task_id}={len(targets_by_task[task_id])}" for task_id in task_ids
        )
    )
    print(f"Loaded {len(predictions)} jobs from {PREDICTIONS_FILE}")

    ctx = {
        "ground_truth": {
            task_id: {
                get_case_id(gt): gt
                for gt in targets
                if get_case_id(gt)
            }
            for task_id, targets in targets_by_task.items()
        },
        "case_id_by_input_pk": case_id_by_input_pk,
        "tool_metric": build_tool_metric(),
        "rationale_judge": build_rationale_judge(),
    }

    print("\nEvaluating cases...")
    rows_by_task: dict[str, list[dict]] = {task_id: [] for task_id in task_ids}
    scored_case_ids: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
    for job in predictions:
        task_id = INTERFACE_TASK_ID.get(get_interface_key(job))
        if task_id is None:
            print(f"[warning] job {job.get('pk')} has an unknown interface; skipping")
            continue
        row = process(job, ctx)
        if row is None:
            continue
        case_id = row["case_id"]
        if case_id in scored_case_ids[task_id]:
            raise RuntimeError(f"Multiple {task_id} jobs resolved to case {case_id}")
        rows_by_task[task_id].append(row)
        scored_case_ids[task_id].add(case_id)

    unscored_tasks = [
        task_id for task_id, case_ids in scored_case_ids.items() if not case_ids
    ]
    if unscored_tasks:
        raise RuntimeError(
            f"No jobs were successfully matched and scored for {unscored_tasks}. "
            "Check CASE_MAP_FILE, task interfaces, and the prediction output paths."
        )

    # Ground-truth cases with no matching job are reported, not skipped.
    for task_id, targets in targets_by_task.items():
        for gt in targets:
            case_id = get_case_id(gt)
            if case_id in scored_case_ids[task_id]:
                continue
            row = evaluate_case(gt, None, ctx["tool_metric"], ctx["rationale_judge"])
            attach_kappa_fields(row, gt, None)
            rows_by_task[task_id].append(row)

    task_aggregates = {
        task_id: compute_aggregate_metrics(rows_by_task[task_id])
        for task_id in task_ids
    }
    missing_ranking_scores = [
        task_id
        for task_id, aggregate in task_aggregates.items()
        if aggregate.get("ranking_score") is None
    ]
    if missing_ranking_scores:
        raise RuntimeError(
            "Ranking score is undefined for "
            f"{missing_ranking_scores}; Task 3 requires at least one comparable "
            "survival pair for its C-index"
        )
    aggregate = {
        **task_aggregates,
        "overall_ranking_score": mean(
            task_aggregates[task_id]["ranking_score"] for task_id in task_ids
        ),
    }
    rows = [row for task_id in task_ids for row in rows_by_task[task_id]]

    # Strip private-ish fields from the JSON dump for cleanliness, but keep
    # them in the in-memory `rows` for aggregate computation.
    public_rows = []
    for r in rows:
        pr = {k: v for k, v in r.items() if not k.startswith("_") and k not in {
            "gt_biopsy_decision_conf", "pred_biopsy_decision_conf",
        }}
        public_rows.append(pr)

    summary = {
        "task_ids": task_ids,
        "ground_truth_dir": str(GROUND_TRUTH_DIR),
        "predictions_file": str(PREDICTIONS_FILE),
        "judge_model": JUDGE_MODEL if USE_RATIONALE_JUDGE else None,
        # Tool use is a Task 1 / Task 2 concept only.
        "tool_metric": None if task_ids == ["task3"] else (
            "DeepEval ToolCorrectnessMetric" if ctx["tool_metric"] is not None else "fallback"
        ),
        "rationale_judge_enabled": ctx["rationale_judge"] is not None,
        "n_target": sum(len(targets) for targets in targets_by_task.values()),
        "n_scored": sum(len(case_ids) for case_ids in scored_case_ids.values()),
        "aggregates": aggregate,
        "per_case": public_rows,
    }

    summary_path = OUTPUT_DIR / "evaluation_results_summary.json"
    csv_path = OUTPUT_DIR / "per_case_results.csv"
    agg_path = OUTPUT_DIR / "aggregate_metrics.json"
    metrics_path = OUTPUT_DIR / "metrics.json"

    write_json(summary, summary_path)
    write_csv(public_rows, csv_path)
    write_json(aggregate, agg_path)

    # Grand-Challenge ranking file. Mirrors the shape of the reference
    # evaluation method (external/example_evaluation_method): an "aggregates"
    # block used for leaderboard ranking plus per-case "results".
    write_json({"aggregates": aggregate, "results": public_rows}, metrics_path)

    print()
    for task_id in task_ids:
        task_aggregate = task_aggregates[task_id]
        print(
            f"{task_id}: cases={task_aggregate.get('n_cases', 0)} "
            f"evaluated={task_aggregate.get('n_evaluated', 0)} "
            f"ranking_score={task_aggregate.get('ranking_score'):.3f}"
        )
    print(f"Overall ranking score: {aggregate['overall_ranking_score']:.3f}")
    print()
    print("Saved:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    print(f"  {agg_path}")


if __name__ == "__main__":
    run()
