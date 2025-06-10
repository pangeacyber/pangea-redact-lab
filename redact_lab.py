#!/usr/bin/env -S poetry run python
"""
redact_lab.py – Public CLI test tool for evaluating the efficacy of the Pangea
Redact service.

Features
--------
* Accepts one or more JSONL datasets with expected PII annotations.
* Calls the Pangea Redact API (requires PANGEA_BASE_URL & PANGEA_REDACT_TOKEN envs).
* Optional request‑per‑second throttling.
* Toggle to include/exclude partial / mismatch cases from metrics.
* Computes Accuracy, Precision, Recall, F1.
* Optionally writes metrics JSON plus false‑positive / false‑negative JSONL files.
* Verbose mode prints FP/FN examples to the console.

Usage Example
-------------
    python redact_lab.py -i dataset1.jsonl dataset2.jsonl \
        --output_metrics metrics.json \
        --fp_out fp.jsonl \
        --fn_out fn.jsonl \
        --include_partials \
        --rps 2 \
        --verbose \
        --redaction_type replacement
    # Optionally include: --config_id <ID>

Metrics & Scores
----------------
The tool reports three *tiers* of efficacy plus standard IR metrics:

* **Base** – Label‑level detection. A True‑Positive (TP) is counted when an
  entity with the *correct label* was redacted **somewhere** in the span
  (overlap is enough). This answers: *“Did we flag the right kind of thing?”*

* **Correct** – Exact match.  The redaction must have the right **label**
  **and** identical *start/end* offsets (unless the dataset omits coordinates,
  in which case any span counts).  This answers: *“Did we flag the *right
  characters*?”*

* **Factual** – Production realism.  Requires **exact label** plus a span
  that **covers** the expected text (overlap or containment is ok) and, when
  provided, the correct `redaction_type`.  It tolerates slightly larger
  redaction spans that are still factually correct.

For each tier we compute:

- **Accuracy** – Share of examples fully correct (no FP or FN).
- **Precision** – TP / (TP + FP).
- **Recall** – TP / (TP + FN).
- **F1** – Harmonic mean of Precision & Recall.
- **Specificity** – True‑Negative Rate (TN / (TN + FP)).
- **FP / FN rates** – Proportion of incorrect predictions.
- **Avg Duration** – Average API latency per example.

Combined score: we don’t collapse tiers into a single number; instead, read
*Base → Correct → Factual* left to right to understand precision–recall trade‑
offs as criteria tighten.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import datetime as _dt
import statistics as _stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

import requests
import re


def call_redact_api(
    base_url: str,
    token: str,
    text: str,
    rules: List[str] | None = None,
    redaction_type: str | None = None,
    config_id: str | None = None,
) -> Tuple[List[Dict], Dict]:
    """
    Invoke the Pangea Redact endpoint and return:
      • a list of entities (dicts) that were redacted
      • the full JSON response for reference
    Optional parameters:
      - rules: list of explicit redact rules to apply
      - redaction_type: redaction type override for the provided rules
      - config_id: optional config_id to include in the request
      - If redaction_type == "replacement", each rule gets a redaction_value placeholder of the form "<RULE>".
    """
    endpoint = base_url.rstrip("/") + "/v1/redact"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload: Dict = {"text": text, "debug": True}
    if rules:
        payload["rules"] = rules
        if redaction_type:
            overrides: Dict[str, Dict[str, str]] = {}
            for r in rules:
                ov = {"redaction_type": redaction_type}
                # When using replacement, also provide redaction_value like <EMAIL_ADDRESS>
                if redaction_type.lower() == "replacement":
                    ov["redaction_value"] = f"<{r}>"
                overrides[r] = ov
            payload["redaction_method_overrides"] = overrides
    if config_id:
        payload["config_id"] = config_id
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text}")
    data = resp.json()
    # print("\n")
    # print(data)
    # New response format: result.report.recognizer_results
    try:
        rec_results = (
            data.get("result", {})
                .get("report", {})
                .get("recognizer_results", [])
        )
    except AttributeError:
        rec_results = []

    # Collect detailed entities that were actually redacted
    redacted_entities = []
    for r in rec_results:
        if r.get("redacted") is True and r.get("field_type"):
            redacted_entities.append({
                "label": r.get("field_type"),
                "text": r.get("text"),
                "start": r.get("start"),
                "end": r.get("end"),
                "redaction_type": r.get("redaction_type"),
            })
    return redacted_entities, data


def spans_match(e: Dict, a: Dict, iou_thresh: float = 0.5) -> bool:
    """
    True if expected entity `e` matches actual entity `a`.
    · Same label, and either:
        (a) ≥1-char overlap with IoU ≥ iou_thresh (when both spans known); or
        (b) expected has no coords (wild-card); or
        (c) expected text is substring of actual text (case-insensitive).
    """
    if e["label"] != a["label"]:
        return False

    # Wild-card if coords missing
    if e["start"] is None or e["end"] is None:
        return True
    if a["start"] is None or a["end"] is None:
        return False

    left = max(e["start"], a["start"])
    right = min(e["end"], a["end"])
    overlap = right - left
    if overlap <= 0:
        return False

    span1 = e["end"] - e["start"]
    span2 = a["end"] - a["start"]
    union = span1 + span2 - overlap
    iou = overlap / union if union else 0.0
    if iou >= iou_thresh:
        return True

    # fallback: textual containment
    if e.get("text") and a.get("text"):
        return e["text"].lower() in a["text"].lower()
    return False


def match_entities(expected_list: List[Dict], actual_list: List[Dict]) -> Tuple[int, int, int, List[Dict], List[Dict]]:
    """
    Return TP, FP, FN plus lists of unmatched expected (for FN) and actual (for FP).
    Entity is considered matched if labels equal AND
    (a) both have start/end provided and spans overlap, or
    (b) start/end missing in expected (wild‑card).
    """
    tp = 0
    fn = 0
    fp = 0
    actual_unused = actual_list.copy()
    expected_unmatched = []

    for exp in expected_list:
        matched = False
        for act in actual_unused:
            if exp["label"] != act["label"]:
                continue
            span_ok = True
            if exp["start"] is not None and exp["end"] is not None:
                span_ok = (act["start"] is not None and act["end"] is not None and
                           exp["start"] < act["end"] and act["start"] < exp["end"])
            if span_ok:
                tp += 1
                matched = True
                actual_unused.remove(act)
                break
        if not matched:
            fn += 1
            expected_unmatched.append(exp)

    fp = len(actual_unused)
    return tp, fp, fn, expected_unmatched, actual_unused

# === Multi-tier efficacy matchers ===



def expected_within_actual(exp: Dict, act: Dict) -> bool:
    """True if the expected span lies fully inside the actual span."""
    if exp["start"] is None or exp["end"] is None:
        return False
    if act["start"] is None or act["end"] is None:
        return False
    return act["start"] <= exp["start"] and act["end"] >= exp["end"]


def match_effective(exp: List[Dict], act: List[Dict], redacted_text: str) -> Tuple[int, int, int]:
    """
    Base tier: counts a TP when an expected label (or its synonym) is found
    in actual entities and either:
      • spans overlap (any amount), OR
      • expected span is wholly contained within the actual span, OR
      • expected coords are missing (wild‑card).
    """
    # Early return: if all expected entities have no valid spans, skip Base evaluation
    if all(e["start"] is None or e["end"] is None for e in exp):
        return 0, 0, 0  # No span-based matching possible for Base tier
    tp = fp = fn = 0
    act_unused = act.copy()
    for e in exp:
        matched = None
        for a in act_unused:
            # Allow label mismatch for Base tier if redaction occurred over expected text span
            if e["start"] is None or e["end"] is None:
                matched = a
                break
            if expected_within_actual(e, a) or (e["start"] < a["end"] and a["start"] < e["end"]):
                matched = a
                break
        # Check if redaction occurred over expected text span and expected text is no longer present
        if matched and e.get("text") and e["text"] not in redacted_text:
            tp += 1
            act_unused.remove(matched)
            matched = "counted"
        # fallback: only count as FN if not already counted
        if matched != "counted":
            fn += 1
    fp = len(act_unused)
    return tp, fp, fn


def match_correct(exp: List[Dict], act: List[Dict]) -> Tuple[int, int, int]:
    """
    Correct tier:
      • labels must be exactly equal
      • if expected has coords, START & END must match exactly
      • if expected lacks coords (wild‑card), any span is accepted
    """
    tp = fp = fn = 0
    act_unused = act.copy()
    for e in exp:
        hit = None
        for a in act_unused:
            # Wild‑card coords ⇒ auto‑match on any label
            if e["start"] is None or e["end"] is None:
                hit = a
                break
            # Exact coordinate match on any label
            if a["start"] == e["start"] and a["end"] == e["end"]:
                hit = a
                break
        if hit:
            tp += 1
            act_unused.remove(hit)
        else:
            fn += 1
    fp = len(act_unused)
    return tp, fp, fn


def match_factual(exp: List[Dict], act: List[Dict]) -> Tuple[int, int, int]:
    """
    Factual tier: strict label equality + containment/overlap + redaction_type when provided.
    Larger actual spans that fully cover expected are treated as matches.
    """
    tp = fp = fn = 0
    act_unused = act.copy()
    for e in exp:
        found = None
        for a in act_unused:
            # Strict label equality for Factual tier
            if e["label"] != a["label"]:
                continue
            if e.get("redaction_type") is not None and a.get("redaction_type") != e["redaction_type"]:
                continue
            if e["start"] is None or e["end"] is None:
                found = a
                break
            if expected_within_actual(e, a):
                found = a
                break
            if e["start"] < a["end"] and a["start"] < e["end"]:
                found = a
                break
        if found:
            tp += 1
            act_unused.remove(found)
        else:
            fn += 1
    fp = len(act_unused)
    return tp, fp, fn


def parse_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts, ignoring blank/invalid lines."""
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" not in obj:
                    print(f"[WARN] {path}:{idx} missing 'text' field – skipped",
                          file=sys.stderr)
                    continue
                if "expected_entities" not in obj and "entities" not in obj:
                    print(f"[WARN] {path}:{idx} missing expected entity info – skipped",
                          file=sys.stderr)
                items.append(obj)
            except json.JSONDecodeError as err:
                print(f"[WARN] {path}:{idx} invalid JSON – {err} – skipped",
                      file=sys.stderr)
    return items


def outcome_counts(expected: Set[str], actual: Set[str],
                   include_partials: bool) -> Tuple[int, int, int, bool]:
    """
    Calculate TP/FP/FN for one example.
    Returns (tp, fp, fn, is_match_fully_correct)
    Partial/mismatch examples yield zero counts when include_partials=False.
    """
    tp = len(expected & actual)
    fp = len(actual - expected)
    fn = len(expected - actual)

    fully_correct = (fp == 0 and fn == 0)
    if not include_partials and not fully_correct:
        # ignore in strict metrics if partials are excluded
        return 0, 0, 0, fully_correct
    return tp, fp, fn, fully_correct


def metrics(tp: int, fp: int, fn: int, correct: int, total: int, duration_sum: float) -> Dict:
    """
    Return a dict with common evaluation statistics.

    tp / fp / fn  – counts for this tier
    correct       – # examples where *all* entities were correct (no fp/fn)
    total         – total examples considered
    duration_sum  – cumulative API latency in seconds

    Keys:
        accuracy, precision, recall, f1, specificity,
        fp_rate, fn_rate, avg_duration,
        true_positives, false_positives, false_negatives,
        examples_considered, examples_fully_correct
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = correct / total if total else 0.0
    specificity = (
        (total - fp - fn - tp) / (total - fp) if (total - fp) else 0.0
    )  # simple TN calc
    fp_rate = fp / (fp + (total - tp - fp - fn)) if (fp + (total - tp - fp - fn)) else 0.0
    fn_rate = fn / (tp + fn) if (tp + fn) else 0.0
    avg_dur = duration_sum / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "avg_duration": avg_dur,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "examples_considered": total,
        "examples_fully_correct": correct,
    }

def rate_limited(max_per_second):
    min_interval = 1.0 / float(max_per_second) if max_per_second > 0 else 0.0
    def decorator(func):
        last_call = [0.0]                 # mutable closure
        def wrapped(*args, **kwargs):
            if min_interval:
                wait = min_interval - (time.time() - last_call[0])
                if wait > 0:
                    time.sleep(wait)
            result = func(*args, **kwargs)
            last_call[0] = time.time()
            return result
        return wrapped
    return decorator

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute efficacy metrics for the Pangea Redact service."
    )
    ap.add_argument("-i", "--input_files", nargs="+", required=True,
                    help="One or more JSONL files containing test examples.")
    ap.add_argument("--output_metrics", help="Path to write summary metrics (JSON format).")
    ap.add_argument("--fp_out", help="Write false positives JSONL to this path.")
    ap.add_argument("--fn_out", help="Write false negatives JSONL to this path.")
    ap.add_argument("--include_partials", action="store_true",
                    help="Include PARTIAL / MISMATCH examples in metrics.")
    ap.add_argument("--rps", type=float, default=1.0,
                    help="Requests per second rate‑limit (0 = unlimited).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print detailed FP/FN examples to stdout.")
    ap.add_argument(
        "--redaction_type",
        default="replacement",
        help="Redaction type override for all provided rules (default: replacement). When 'replacement' is used, the tool automatically supplies a placeholder like <EMAIL_ADDRESS>.",
    )
    ap.add_argument(
        "--config_id",
        help="Optional Pangea config_id to include in each request.",
    )
    args = ap.parse_args()

    # ---------- threaded execution setup ----------
    max_workers = int(args.rps) if args.rps >= 1 else 1
    semaphore = Semaphore(max_workers)

    @rate_limited(args.rps)
    def process_example(idx_ex, ex, total):
        # acquire a slot in the pool
        with semaphore:
            nonlocal tp_eff, fp_eff, fn_eff
            nonlocal tp_cor, fp_cor, fn_cor, tp_fac, fp_fac, fn_fac
            nonlocal correct_eff_cnt, correct_cor_cnt, correct_fac_cnt
            nonlocal considered_cnt, duration_sum, remote_latencies
            nonlocal args

            # progress bar
            prog = (idx_ex + 1) / total * 100
            print("\r\033[2K", end="")
            print(f"{prog:.2f}%", end="\r", flush=True)

            text = ex["text"]

            # Build expected_entities list of dicts (same logic as original loop)
            if "entities" in ex:
                expected_entities = ex["entities"]
            elif "expected_entities" in ex:
                exp_val = ex["expected_entities"]
                if exp_val and isinstance(exp_val[0], dict) and "label" in exp_val[0]:
                    expected_entities = exp_val
                else:
                    expected_entities = [
                        {"label": lbl, "text": None, "start": None, "end": None}
                        for lbl in exp_val
                    ]
            else:
                expected_entities = []

            # Derive redact rules from expected_entities for this example
            rule_list = list({
                ent["label"] for ent in expected_entities
                if isinstance(ent, dict) and ent.get("label")
            })
            try:
                start_time = time.time()
                actual_entities, api_raw = call_redact_api(
                    base_url,
                    token,
                    text,
                    rules=rule_list,
                    redaction_type=args.redaction_type,
                    config_id=args.config_id,
                )

                # Validate removal of expected text
                redacted_output = api_raw.get("result", {}).get("redacted_text", "") or ""
                for ent in expected_entities:
                    ent_txt = ent.get("text")
                    if not ent_txt:
                        continue
                    if ent_txt not in text:
                        print(f"[WARN] Example #{idx_ex+1}: expected entity text '{ent_txt}' not found in original input.",
                              file=sys.stderr)
                    elif ent_txt in redacted_output and args.verbose:
                        print(f"[WARN] Example #{idx_ex+1}: expected entity text '{ent_txt}' still present after redaction.",
                              file=sys.stderr)

                # Measure latency
                duration = time.time() - start_time
                duration_sum += duration
                req_ts = api_raw.get("request_time") or api_raw.get("meta", {}).get("request_time")
                resp_ts = api_raw.get("response_time") or api_raw.get("meta", {}).get("response_time")
                try:
                    if req_ts and resp_ts:
                        req_dt = _dt.datetime.fromisoformat(req_ts.replace("Z", "+00:00"))
                        resp_dt = _dt.datetime.fromisoformat(resp_ts.replace("Z", "+00:00"))
                        remote_latencies.append((resp_dt - req_dt).total_seconds())
                except Exception:
                    pass
            except Exception as e:
                print(f"[ERROR] Example #{idx_ex+1} failed: {e}", file=sys.stderr)
                return

            # Tiered metrics
            eff_tp, eff_fp, eff_fn = match_effective(expected_entities, actual_entities, redacted_output)
            cor_tp, cor_fp, cor_fn = match_correct(expected_entities, actual_entities)
            fac_tp, fac_fp, fac_fn = match_factual(expected_entities, actual_entities)

            tp_eff += eff_tp; fp_eff += eff_fp; fn_eff += eff_fn
            tp_cor += cor_tp; fp_cor += cor_fp; fn_cor += cor_fn
            tp_fac += fac_tp; fp_fac += fac_fp; fn_fac += fac_fn

            tp_e, fp_e, fn_e, exp_unmatched, act_unmatched = match_entities(expected_entities, actual_entities)

            considered_cnt += 1
            full_ok_eff = (eff_fp == 0 and eff_fn == 0)
            full_ok_cor = (cor_fp == 0 and cor_fn == 0)
            full_ok_fac = (fac_fp == 0 and fac_fn == 0)

            if args.include_partials or full_ok_eff:
                correct_eff_cnt += 1 if full_ok_eff else 0
            if args.include_partials or full_ok_cor:
                correct_cor_cnt += 1 if full_ok_cor else 0
            if args.include_partials or full_ok_fac:
                correct_fac_cnt += 1 if full_ok_fac else 0

            fp_labels = [a["label"] for a in act_unmatched]
            fn_labels = [e["label"] for e in exp_unmatched]

            if not fp_labels and cor_fp:
                fp_labels = [a["label"] for a in actual_entities
                             if (a["label"], a["start"], a["end"]) not in
                             {(e["label"], e["start"], e["end"]) for e in expected_entities}]

            # ----------------------------------------------------------------
            # Persist FP / FN details per efficacy tier
            def _write(handle, labels):
                if handle and labels:
                    handle.write(json.dumps({
                        "text": text,
                        "redacted_text": api_raw.get("result", {}).get("redacted_text"),
                        "expected_entities": expected_entities,
                        "actual_entities": actual_entities,
                        "labels": labels
                    }, ensure_ascii=False) + "\n")

            # Base tier
            _write(base_fp_handle, fp_labels if eff_fp else [])
            _write(base_fn_handle, fn_labels if eff_fn else [])

            # Correct tier
            _write(correct_fp_handle, fp_labels if cor_fp else [])
            _write(correct_fn_handle, fn_labels if cor_fn else [])

            # Factual tier
            _write(factual_fp_handle, fp_labels if fac_fp else [])
            _write(factual_fn_handle, fn_labels if fac_fn else [])

            if args.verbose and (fp_e or fn_e):
                if fp_e:
                    print(f"[FP] #{idx_ex+1}: {fp_labels} was redacted | '{text}'")
                if fn_e:
                    print(f"[FN] #{idx_ex+1}: {fn_labels} was not redacted in | '{text}'")

    base_url = os.getenv("PANGEA_BASE_URL") or os.getenv("PANGEA_REDACT_BASE_URL")
    token = os.getenv("PANGEA_REDACT_TOKEN")
    if not base_url or not token:
        print("Error: PANGEA_BASE_URL and PANGEA_REDACT_TOKEN env vars are required.",
              file=sys.stderr)
        sys.exit(1)

    # Load datasets
    examples: List[Dict] = []
    for path in args.input_files:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}", file=sys.stderr)
            continue
        examples.extend(parse_jsonl(path))
    if not examples:
        print("No valid examples to process – exiting.", file=sys.stderr)
        sys.exit(1)
    total_examples = len(examples)
    print(f"Loaded {total_examples} test case{'s' if total_examples != 1 else ''}.")

    # ------------------------------------------------------------------
    # Tier‑specific FP / FN output handles
    def _tier_file(base_path: str, tier: str, kind: str):
        """
        Derive filenames like  redact_test.base.fps.jsonl  or  redact_test.factual.fns.jsonl
        where *tier* ∈ {base, correct, factual} and *kind* ∈ {fps, fns}.
        """
        stem = base_path.rsplit(".", 1)[0]
        return open(f"{stem}.{tier}.{kind}.jsonl", "w", encoding="utf-8")

    if args.fp_out:
        base_fp_handle     = _tier_file(args.fp_out, "base",    "fps")
        correct_fp_handle  = _tier_file(args.fp_out, "correct", "fps")
        factual_fp_handle  = _tier_file(args.fp_out, "factual", "fps")
    else:
        base_fp_handle = correct_fp_handle = factual_fp_handle = None

    if args.fn_out:
        base_fn_handle     = _tier_file(args.fn_out, "base",    "fns")
        correct_fn_handle  = _tier_file(args.fn_out, "correct", "fns")
        factual_fn_handle  = _tier_file(args.fn_out, "factual", "fns")
    else:
        base_fn_handle = correct_fn_handle = factual_fn_handle = None

    tp_sum = fp_sum = fn_sum = 0
    considered_cnt = 0
    duration_sum = 0.0
    remote_latencies: List[float] = []  # collect each (response_time - request_time)

    # Multi-tier counters
    tp_eff=fp_eff=fn_eff=0
    tp_cor=fp_cor=fn_cor=0
    tp_fac=fp_fac=fn_fac=0
    correct_eff_cnt = correct_cor_cnt = correct_fac_cnt = 0

    # ---------- run in ThreadPoolExecutor ----------
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [
            pool.submit(process_example, i, ex, total_examples)
            for i, ex in enumerate(examples)
        ]
        for _ in as_completed(futs):
            pass        # simply wait for all to finish

    for _h in [base_fp_handle, correct_fp_handle, factual_fp_handle,
               base_fn_handle, correct_fn_handle, factual_fn_handle]:
        if _h:
            _h.close()

    # Move to new line after progress indicator
    print()

    def make_stats(tp, fp, fn, correct_val):
        return metrics(tp, fp, fn, correct_val, considered_cnt, duration_sum)

    # Remote latency percentiles
    if remote_latencies:
        remote_latencies.sort()
        p50_latency = _stats.median(remote_latencies)
        p95_latency = remote_latencies[int(len(remote_latencies)*0.95) - 1]
        p99_latency = remote_latencies[int(len(remote_latencies)*0.99) - 1]
    else:
        p50_latency = p95_latency = p99_latency = 0.0

    stats_eff = make_stats(tp_eff, fp_eff, fn_eff, correct_eff_cnt)
    stats_cor = make_stats(tp_cor, fp_cor, fn_cor, correct_cor_cnt)
    stats_fac = make_stats(tp_fac, fp_fac, fn_fac, correct_fac_cnt)

    # Pretty print summary
    print("\n=== Pangea Redact Efficacy ===")
    def pr(name,s):
        print(f"{name:<10}  Acc {s['accuracy']*100:.2f}%  P {s['precision']*100:.2f}%  "
              f"R {s['recall']*100:.2f}%  F1 {s['f1']*100:.2f}%  "
              f"(TP={s['true_positives']}, FP={s['false_positives']}, FN={s['false_negatives']})")
    pr("Base",stats_eff)
    pr("Correct",  stats_cor)
    pr("Factual",  stats_fac)

    # -----------------------------------------------------------------------
    # Explain tiers once so users know how to read the numbers
    print("\nLegend:")
    print("  Base    – Label match & partial span ok")
    print("  Correct – Exact span & label match")
    print("  Factual – Label match, span containment/overlap, correct redaction type")
    print("  Acc     – Examples fully correct | P – Precision | R – Recall | F1 – F‑score")

    print(f"\nAPI latency (server‑side): "
          f"P50 {p50_latency*1000:.1f} ms  |  "
          f"P95 {p95_latency*1000:.1f} ms  |  "
          f"P99 {p99_latency*1000:.1f} ms")

    if args.output_metrics:
        with open(args.output_metrics, "w", encoding="utf-8") as fh:
            json.dump({
                "effective": stats_eff,
                "correct": stats_cor,
                "factual": stats_fac,
                "latency_p50": p50_latency,
                "latency_p95": p95_latency,
                "latency_p99": p99_latency
            }, fh, indent=2)
        print(f"Metrics saved to {args.output_metrics}")

    print("Done.")


if __name__ == "__main__":
    main()