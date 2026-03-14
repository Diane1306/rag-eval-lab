#!/usr/bin/env python3
# src/eval_experiment_day10.py
#
# Purpose:
#   Run a controlled offline experiment (A/B test) over a small gold set, aligned with
#   common LLM Data Analyst job requirements:
#     - define variants
#     - run consistent evaluation
#     - compute actionable metrics
#     - write analysis-ready outputs for dashboards
#
# Experiment (Day 10):
#   Variant A: dedupe_field = "doc_id"
#   Variant B: dedupe_field = "title"
#
# Gold set:
#   eval/gold_v1.jsonl
#   Each line has: qid, type, turns, answerable, expected_refusal, notes
#
# Multi-turn query formation rule:
#   final_query = "Conversation:\n1) ...\n2) ...\n...\n\nAnswer the final user request."
#
# Outputs:
#   eval/outputs/day10_outputs.jsonl  (one row per qid x variant)
#   Printed summary per variant:
#     - expected_refusal accuracy
#     - refusal rates (answerable/unanswerable)
#     - citation rate when not refusing
#     - latency p95
#     - diversity (unique doc_ids)
#
# How to run:
#   uv run python src/eval_experiment_day10.py
#
# Success criteria:
#   - Output JSONL exists with 2 * N rows (N is #gold items)
#   - Summary prints meaningful differences between variants (even if small)

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import your retrieval + deterministic answering
from src.rag_retrieve import retrieve
from src.rag_answer_local import answer_from_retrieval

# -----------------------------
# File paths
# -----------------------------

GOLD_PATH = Path("eval/gold_v1.jsonl")                 # Gold set
OUT_PATH = Path("eval/outputs/day10_outputs.jsonl")    # Experiment outputs

# -----------------------------
# Experiment configuration
# -----------------------------

VARIANTS = [
    {"variant": "doc_id", "dedupe_field": "doc_id"},
    {"variant": "title",  "dedupe_field": "title"},
]

TOP_K = 5
CANDIDATE_MULTIPLIER = 5

# -----------------------------
# Helper functions
# -----------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_final_query(turns: List[str]) -> str:
    """
    Concatenate ALL turns into a deterministic multi-turn query.

    Output format is stable and readable, which is helpful for debugging.
    """
    lines = ["Conversation:"]
    for i, t in enumerate(turns, start=1):
        lines.append(f"{i}) {t}")
    lines.append("")
    lines.append("Answer the final user request (the last turn).")
    return "\n".join(lines)


def percentile(values: List[int], q: float) -> int:
    """
    Simple percentile without extra libs.
    q in [0, 1]; e.g., 0.95 for p95.
    """
    if not values:
        return 0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return xs[idx]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute key KPIs for a list of evaluation rows (single variant).

    KPIs:
      - expected_refusal accuracy
      - refusal rates stratified by answerable
      - citation rate among non-refusals
      - p95 latency
      - avg diversity
    """
    n = len(rows)
    if n == 0:
        return {}

    # expected_refusal accuracy: did model refusal match expected_refusal?
    exp_acc = sum(1 for r in rows if bool(r["refusal"]) == bool(r["expected_refusal"])) / n

    # Stratified refusal rates
    ans_rows = [r for r in rows if bool(r["answerable"])]
    unans_rows = [r for r in rows if not bool(r["answerable"])]

    def rate_refusal(subset: List[Dict[str, Any]]) -> float:
        return sum(1 for r in subset if bool(r["refusal"])) / len(subset) if subset else 0.0

    refusal_rate_answerable = rate_refusal(ans_rows)
    refusal_rate_unanswerable = rate_refusal(unans_rows)

    # Citation rate among non-refusals
    nonref = [r for r in rows if not bool(r["refusal"])]
    citation_rate_nonrefusal = (
        sum(1 for r in nonref if int(r["n_citations"]) > 0) / len(nonref)
        if nonref else 0.0
    )

    # Latency p95
    latencies = [int(r["latency_ms"]) for r in rows]
    p95 = percentile(latencies, 0.95)

    # Diversity average
    diversity = [int(r["diversity_unique_doc_ids"]) for r in rows]
    avg_div = sum(diversity) / len(diversity) if diversity else 0.0

    return {
        "n": n,
        "expected_refusal_accuracy": exp_acc,
        "refusal_rate_answerable": refusal_rate_answerable,
        "refusal_rate_unanswerable": refusal_rate_unanswerable,
        "citation_rate_nonrefusal": citation_rate_nonrefusal,
        "latency_p95_ms": p95,
        "diversity_avg_unique_doc_ids": avg_div,
    }


def main() -> None:
    # 1) Ensure gold set exists
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Missing {GOLD_PATH}. Create it first.")

    # 2) Load gold items
    gold = load_jsonl(GOLD_PATH)

    # 3) Run timestamp (for traceability)
    run_ts = datetime.now(timezone.utc).isoformat()

    all_outputs: List[Dict[str, Any]] = []

    # 4) Evaluate each item under each variant
    for item in gold:
        qid = item["qid"]
        qtype = item.get("type", "single")
        turns = item["turns"]
        answerable = bool(item["answerable"])
        expected_refusal = bool(item["expected_refusal"])
        notes = item.get("notes", "")

        # Build deterministic final query from ALL turns
        final_query = build_final_query(turns)

        for v in VARIANTS:
            variant_name = v["variant"]
            dedupe_field = v["dedupe_field"]

            # Retrieval
            r = retrieve(
                query=final_query,
                k=TOP_K,
                dedupe_field=dedupe_field,
                candidate_multiplier=CANDIDATE_MULTIPLIER,
            )

            # Compute top_score (for refusal gating in answerer)
            top_score = r["results"][0]["score"] if r["results"] else None

            # Deterministic grounded answer
            ans = answer_from_retrieval(final_query, r["results"], top_score=top_score)

            # Collect IDs for analysis
            doc_ids = [x.get("doc_id", "") for x in r["results"]]
            diversity_unique_doc_ids = len(set(doc_ids))

            # Output row (analysis-ready)
            out = {
                "run_ts": run_ts,
                "qid": qid,
                "type": qtype,
                "variant": variant_name,
                "dedupe_field": dedupe_field,
                "final_query": final_query,
                "answerable": answerable,
                "expected_refusal": expected_refusal,
                "notes": notes,
                "refusal": bool(ans["refusal"]),
                "refusal_reason": ans.get("refusal_reason", ""),
                "n_citations": len(ans.get("citations", [])),
                "latency_ms": int(r["latency_ms"]),
                "top_score": float(top_score) if top_score is not None else None,
                "diversity_unique_doc_ids": diversity_unique_doc_ids,
            }
            all_outputs.append(out)

    # 5) Write outputs JSONL
    write_jsonl(OUT_PATH, all_outputs)

    # 6) Print summaries per variant
    print("=" * 80)
    print("DAY 10 EXPERIMENT SUMMARY")
    print(f"run_ts: {run_ts}")
    print(f"gold_n: {len(gold)}  (rows written: {len(all_outputs)})")
    print("-" * 80)

    for v in VARIANTS:
        variant_name = v["variant"]
        rows = [r for r in all_outputs if r["variant"] == variant_name]
        s = summarize(rows)
        print(f"variant: {variant_name}")
        print(f"  n: {s['n']}")
        print(f"  expected_refusal_accuracy: {s['expected_refusal_accuracy']:.2f}")
        print(f"  refusal_rate_answerable:   {s['refusal_rate_answerable']:.2f}")
        print(f"  refusal_rate_unanswerable: {s['refusal_rate_unanswerable']:.2f}")
        print(f"  citation_rate_nonrefusal:  {s['citation_rate_nonrefusal']:.2f}")
        print(f"  latency_p95_ms:            {s['latency_p95_ms']}")
        print(f"  diversity_avg_unique_docs: {s['diversity_avg_unique_doc_ids']:.2f}")
        print("-" * 80)

    print(f"Wrote outputs -> {OUT_PATH}")
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
