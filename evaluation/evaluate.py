"""Deterministic + optional LLM evaluation pipeline for biopsy-decision forms.

Compares LLM-agent form responses (mimic_datasets/evaluation_object.json)
against pathologist ground-truth (mimic_datasets/target.json) using:

    * A hard biopsy-decision gate (per case) — cases that get the
      yes/no decision wrong score 0 regardless of other components.

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

Outputs (written to results/):

    evaluation_results_summary.json   full per-case + aggregate dump
    per_case_results.csv              one row per case, easy to scan
    aggregate_metrics.json            dataset-level summary

Run via Docker (recommended — see README.md):

    make run

Or directly (Ollama must be reachable at OLLAMA_BASE_URL):

    python evaluate.py

Environment variable overrides:

    TARGET_FILE        path to ground-truth JSON  (default: mimic_datasets/target.json)
    EVAL_FILE          path to candidate JSON      (default: mimic_datasets/evaluation_object.json)
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

TARGET_FILE = Path(os.getenv(
    "TARGET_FILE",
    str(ROOT / "mimic_datasets" / "target.json"),
))
EVAL_FILE = Path(os.getenv(
    "EVAL_FILE",
    str(ROOT / "mimic_datasets" / "evaluation_object.json"),
))
OUTPUT_DIR = Path(os.getenv(
    "EVAL_OUTPUT_DIR",
    str(ROOT / "results"),
))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma4:e4b")
USE_RATIONALE_JUDGE = bool(int(os.getenv("USE_RATIONALE_JUDGE", "1")))


# --------------------------------------------------------------------------- #
# Label conventions
# --------------------------------------------------------------------------- #

VALID_BIOPSY_DECISIONS = {"yes", "no"}

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


# --------------------------------------------------------------------------- #
# JSON / record helpers
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_task_records(data: Any, task_name: str = "biopsy_decision") -> list[dict]:
    """Accept either {task_name: [...]} or a flat [...] list and return a list."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        if task_name in data and isinstance(data[task_name], list):
            return [r for r in data[task_name] if isinstance(r, dict)]
        # Fallback: maybe the dict IS the single record.
        if "biopsy_decision" in data and isinstance(data["biopsy_decision"], str):
            return [data]
    return []


def get_case_id(record: dict) -> str:
    cid = record.get("case_id")
    if cid:
        return str(cid)
    patient = record.get("patient") or {}
    if isinstance(patient, dict) and patient.get("id"):
        return str(patient["id"])
    return ""


def validate_record(record: dict) -> tuple[bool, str]:
    """Lightweight schema check on a candidate record."""
    if not isinstance(record, dict):
        return False, "candidate is not an object"
    decision = str(record.get("biopsy_decision", "")).lower().strip()
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
# Reveal sequence / tool-use
# --------------------------------------------------------------------------- #

def _reveal_keys(record: dict) -> list[str]:
    """Return section keys in reveal order, deduplicated (first occurrence wins)."""
    seq = record.get("reveal_sequence") or []
    if not isinstance(seq, list):
        return []
    seq = sorted(seq, key=lambda x: x.get("order", 10**9) if isinstance(x, dict) else 10**9)
    seen: set[str] = set()
    keys: list[str] = []
    for item in seq:
        if not isinstance(item, dict):
            continue
        k = item.get("key")
        if not k:
            continue
        k = str(k)
        if k in seen:
            continue
        keys.append(k)
        seen.add(k)
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
        "pathologist's rationale (Expected Output) and the case's expected "
        "biopsy decision. Score HIGH if the rationale: (1) supports the same "
        "biopsy decision; (2) cites the same important/decisive clinical "
        "variables; (3) does not contradict the clinical_data; (4) does not "
        "invent unavailable information; (5) expresses uncertainty consistent "
        "with the stated confidence. Score LOW if it contradicts the decision, "
        "misses major decisive factors, hallucinates clinical facts, or gives "
        "generic case-agnostic reasoning. Repeat-test plans should also be "
        "judged for clinical reasonableness when present, but only as a minor "
        "modifier."
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

    def judge(gt: dict, pred: dict) -> tuple[float | None, str]:
        gt_text = (gt.get("free_text") or "").strip()
        pred_text = (pred.get("free_text") or "").strip()
        if not gt_text or not pred_text:
            return None, "missing free_text on gt or pred"

        input_ctx = {
            "case_id": get_case_id(gt),
            "patient": gt.get("patient", {}),
            "clinical_data": gt.get("clinical_data", {}),
            "expected_biopsy_decision": gt.get("biopsy_decision"),
            "expected_confidence": gt.get("confidence"),
            "expected_repeat_test": gt.get("repeat_test"),
        }
        actual = {
            "biopsy_decision": pred.get("biopsy_decision"),
            "confidence": pred.get("confidence"),
            "repeat_test": pred.get("repeat_test"),
            "free_text": pred_text,
        }
        expected = {
            "biopsy_decision": gt.get("biopsy_decision"),
            "confidence": gt.get("confidence"),
            "repeat_test": gt.get("repeat_test"),
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

def evaluate_case(
    gt: dict,
    pred: dict | None,
    tool_metric,
    rationale_judge,
) -> dict:
    case_id = get_case_id(gt)
    gt_decision = _norm_decision(gt.get("biopsy_decision"))

    base = {
        "case_id": case_id,
        "gate": "passed",
        "case_score": 0.0,
        "biopsy_decision_correct": False,
        "gt_biopsy_decision": gt_decision,
        "pred_biopsy_decision": None,
        "confidence_score": None,
        "variable_weight_score": None,
        "important_decisive_factor_score": None,
        "tool_score": None,
        "rationale_score": None,
        "reason": "",
    }

    if pred is None:
        base["gate"] = "missing_candidate"
        base["reason"] = "no candidate record for this case"
        return base

    ok, why = validate_record(pred)
    if not ok:
        base["gate"] = "schema_failed"
        base["reason"] = f"schema validation failed: {why}"
        base["pred_biopsy_decision"] = pred.get("biopsy_decision")
        return base

    pred_decision = _norm_decision(pred.get("biopsy_decision"))
    base["pred_biopsy_decision"] = pred_decision
    base["biopsy_decision_correct"] = (gt_decision == pred_decision and gt_decision is not None)

    if not base["biopsy_decision_correct"]:
        base["gate"] = "biopsy_decision_failed"
        base["reason"] = (
            f"biopsy_decision mismatch: gt={gt_decision!r} pred={pred_decision!r}"
        )
        return base

    # Granular evaluation
    cs = confidence_score(gt, pred)
    vws = variable_weight_score(gt, pred)
    fs = important_decisive_factor_score(gt, pred)
    ts, t_reason = compute_tool_score(gt, pred, tool_metric)

    rs, r_reason = (None, "rationale judge disabled")
    if rationale_judge is not None:
        rs, r_reason = rationale_judge(gt, pred)

    base["confidence_score"] = cs
    base["variable_weight_score"] = vws
    base["important_decisive_factor_score"] = fs
    base["tool_score"] = ts
    base["rationale_score"] = rs

    # Weighted composite. Drop rationale weight when unavailable and
    # renormalise the remaining weights.
    components = {
        "confidence": (cs, 0.25),
        "var_weight": (vws, 0.30),
        "factor_f1":  (fs, 0.20),
        "tool":       (ts, 0.15),
        "rationale":  (rs, 0.10),
    }

    if rs is None:
        components = {
            "confidence": (cs, 0.275),
            "var_weight": (vws, 0.350),
            "factor_f1":  (fs, 0.225),
            "tool":       (ts, 0.150),
        }

    # Replace any component that is None with 0 so the math is well-defined,
    # but record it in the reason.
    missing = [k for k, (v, _) in components.items() if v is None]
    score = sum((v if v is not None else 0.0) * w for v, w in components.values())
    base["case_score"] = max(0.0, min(1.0, score))

    parts = [f"tool: {t_reason}"]
    if rationale_judge is not None:
        parts.append(f"rationale: {r_reason}")
    if missing:
        parts.append("missing components zeroed: " + ", ".join(missing))
    base["reason"] = " | ".join(parts)
    return base


def compute_aggregate_metrics(rows: list[dict]) -> dict:
    """Dataset-level aggregation. Pure-python fallback when sklearn missing."""
    n = len(rows)
    if n == 0:
        return {"n_cases": 0}

    final_dag = mean(r["case_score"] for r in rows)

    decisions = [
        (r["gt_biopsy_decision"], r["pred_biopsy_decision"])
        for r in rows
        if r["gt_biopsy_decision"] in VALID_BIOPSY_DECISIONS
        and r["pred_biopsy_decision"] in VALID_BIOPSY_DECISIONS
    ]
    y_true = [g for g, _ in decisions]
    y_pred = [p for _, p in decisions]
    n_correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    n_incorrect = len(y_true) - n_correct

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

    gate_pass = [r for r in rows if r["gate"] == "passed"]
    gate_pass_rate = len(gate_pass) / n
    mean_among_pass = mean(r["case_score"] for r in gate_pass) if gate_pass else 0.0

    tool_scores = [r["tool_score"] for r in rows if r["tool_score"] is not None]
    rationale_scores = [r["rationale_score"] for r in rows if r["rationale_score"] is not None]

    out = {
        "n_cases": n,
        "n_evaluated": len(decisions),
        "n_decision_correct": n_correct,
        "n_decision_incorrect": n_incorrect,
        "final_DAG_score": final_dag,
        "decision_accuracy": None,
        "decision_f1_yes": None,
        "confidence_weighted_kappa": None,
        "variable_weight_weighted_kappa": None,
        "mean_tool_score": mean(tool_scores) if tool_scores else None,
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
        try:
            out["decision_f1_yes"] = float(f1_score(y_true, y_pred, pos_label="yes", zero_division=0))
        except Exception:
            out["decision_f1_yes"] = None
        out["decision_classification_report"] = classification_report(
            y_true, y_pred, labels=["yes", "no"], zero_division=0,
        )

    if conf_pairs:
        ct, cp = zip(*conf_pairs)
        try:
            out["confidence_weighted_kappa"] = float(
                cohen_kappa_score(list(ct), list(cp), weights="quadratic")
            )
        except Exception:
            out["confidence_weighted_kappa"] = None

    if flat_gt_w:
        try:
            out["variable_weight_weighted_kappa"] = float(
                cohen_kappa_score(flat_gt_w, flat_pred_w, weights="quadratic")
            )
        except Exception:
            out["variable_weight_weighted_kappa"] = None

    return out


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

CSV_COLUMNS = [
    "case_id",
    "gate",
    "case_score",
    "biopsy_decision_correct",
    "gt_biopsy_decision",
    "pred_biopsy_decision",
    "confidence_score",
    "variable_weight_score",
    "important_decisive_factor_score",
    "tool_score",
    "rationale_score",
    "reason",
]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run() -> None:
    if not TARGET_FILE.exists():
        sys.exit(f"Missing target file: {TARGET_FILE}")
    if not EVAL_FILE.exists():
        sys.exit(f"Missing candidate file: {EVAL_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = normalize_task_records(load_json(TARGET_FILE))
    candidates = normalize_task_records(load_json(EVAL_FILE))

    print(f"Loaded {len(targets)} target cases")
    print(f"Loaded {len(candidates)} candidate cases")

    cand_idx = {get_case_id(r): r for r in candidates if get_case_id(r)}

    tool_metric = build_tool_metric()
    rationale_judge = build_rationale_judge()

    print("\nEvaluating cases...")
    rows: list[dict] = []
    for gt in targets:
        cid = get_case_id(gt)
        pred = cand_idx.get(cid)
        row = evaluate_case(gt, pred, tool_metric, rationale_judge)

        # Attach raw values needed for dataset-level kappas (kept out of CSV).
        row["gt_biopsy_decision_conf"] = _norm_conf(gt.get("confidence"))
        row["pred_biopsy_decision_conf"] = _norm_conf(pred.get("confidence")) if pred else None
        weight_pairs: list[tuple[int, int]] = []
        if pred and row["gate"] == "passed":
            gt_w = gt.get("variable_weights") or {}
            pr_w = pred.get("variable_weights") or {}
            for var, gv in gt_w.items():
                g = _norm_weight(gv)
                if g is None:
                    continue
                p = _norm_weight(pr_w.get(var, "not_used")) or "not_used"
                weight_pairs.append((WEIGHT_MAP[g], WEIGHT_MAP[p]))
        row["_weight_pairs"] = weight_pairs
        rows.append(row)

    aggregate = compute_aggregate_metrics(rows)

    # Strip private-ish fields from the JSON dump for cleanliness, but keep
    # them in the in-memory `rows` for aggregate computation.
    public_rows = []
    for r in rows:
        pr = {k: v for k, v in r.items() if not k.startswith("_") and k not in {
            "gt_biopsy_decision_conf", "pred_biopsy_decision_conf",
        }}
        public_rows.append(pr)

    summary = {
        "judge_model": JUDGE_MODEL if USE_RATIONALE_JUDGE else None,
        "tool_metric": "DeepEval ToolCorrectnessMetric" if tool_metric is not None else "fallback",
        "rationale_judge_enabled": rationale_judge is not None,
        "n_target": len(targets),
        "n_candidate": len(candidates),
        "aggregate": aggregate,
        "per_case": public_rows,
    }

    summary_path = OUTPUT_DIR / "evaluation_results_summary.json"
    csv_path = OUTPUT_DIR / "per_case_results.csv"
    agg_path = OUTPUT_DIR / "aggregate_metrics.json"

    write_json(summary, summary_path)
    write_csv(public_rows, csv_path)
    write_json(aggregate, agg_path)

    # Console summary
    def fmt(v: Any) -> str:
        return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    print()
    print(f"Final case score (all cases): {fmt(aggregate.get('final_DAG_score'))}")
    print(f"Biopsy decision F1_yes: {fmt(aggregate.get('decision_f1_yes'))}")
    print(f"Biopsy decision accuracy: {fmt(aggregate.get('decision_accuracy'))}")
    n_eval = aggregate.get('n_evaluated', 0)
    n_corr = aggregate.get('n_decision_correct', 'n/a')
    n_incorr = aggregate.get('n_decision_incorrect', 'n/a')
    print(f"Biopsy decision correct/incorrect: {n_corr}/{n_incorr} (out of {n_eval} evaluated)")
    print(f"Confidence weighted kappa: {fmt(aggregate.get('confidence_weighted_kappa'))}")
    print(f"Variable-weight weighted kappa: {fmt(aggregate.get('variable_weight_weighted_kappa'))}")
    print(f"Mean tool score: {fmt(aggregate.get('mean_tool_score'))}")
    print(f"Mean rationale score: {fmt(aggregate.get('mean_rationale_score'))}")
    print(f"Mean case score (among gate passed cases): {fmt(aggregate.get('mean_case_score_among_gate_passed'))}")
    print()
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    print(f"  {agg_path}")


if __name__ == "__main__":
    run()
