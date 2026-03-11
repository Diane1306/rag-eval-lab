#!/usr/bin/env python3
# src/eval_run_day8.py
#
# Purpose:
#   Offline evaluation runner for your RAG pipeline.
#   It runs the same retrieval + deterministic answering used in Streamlit, but in batch.
#
# What it measures:
#   - refusal rates overall + by answerable/unanswerable
#   - citation rate when not refusing
#   - latency p50/p95
#   - diversity proxy (unique doc_ids in returned results)
#   - comparison: dedupe_field = doc_id vs title
#
# Inputs:
#   - data/eval/eval_questions.jsonl (20 lines)
#
# Outputs:
#   - data/eval/eval_outputs_day8.jsonl
#   - terminal summary

import json
import os
import statistics
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.rag_retrieve import retrieve
from src.rag_answer_local import answer_from_retrieval

EVAL_PATH = Path("data/eval/eval_questions.jsonl")
OUT_PATH = Path("data/eval/eval_outputs_day8.jsonl")

K = 5
CANDIDATE_MULTIPLIER = 5


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pctl(values: List[int], q: float) -> int:
    """
    Approx percentile without external libs.
    q in [0,1], e.g. 0.50 or 0.95.
    """
    if not values:
        return 0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return xs[idx]


def run_eval(dedupe_field: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run eval over all questions for a given dedupe_field."""
    questions = load_jsonl(EVAL_PATH)

    outputs = []
    latencies = []
    refusal_all = []
    refusal_answerable = []
    refusal_unanswerable = []
    citation_present_nonrefusal = []
    diversity_docids = []

    for item in questions:
        qid = item["qid"]
        query = item["query"]
        answerable = bool(item["answerable"])

        r = retrieve(
            query=query,
            k=K,
            dedupe_field=dedupe_field,
            candidate_multiplier=CANDIDATE_MULTIPLIER,
        )

        top_score = r["results"][0]["score"] if r["results"] else None
        ans = answer_from_retrieval(query, r["results"], top_score=top_score)

        # Metrics
        latencies.append(int(r["latency_ms"]))
        refusal = bool(ans["refusal"])
        refusal_all.append(refusal)
        if answerable:
            refusal_answerable.append(refusal)
        else:
            refusal_unanswerable.append(refusal)

        n_citations = len(ans["citations"])
        if not refusal:
            citation_present_nonrefusal.append(n_citations > 0)

        # Diversity proxy: how many unique doc_ids in returned results
        doc_ids = [x.get("doc_id", "") for x in r["results"]]
        diversity_docids.append(len(set(doc_ids)))

        outputs.append(
            {
                "qid": qid,
                "query": query,
                "answerable": answerable,
                "dedupe_field": dedupe_field,
                "answer": ans["answer"],
                "refusal": refusal,
                "refusal_reason": ans.get("refusal_reason", ""),
                "n_citations": n_citations,
                "citations": ans["citations"],
                "latency_ms": int(r["latency_ms"]),
                "k_candidates": int(r["k_candidates"]),
                "top_doc_ids": doc_ids,
                "diversity_unique_doc_ids": len(set(doc_ids)),
            }
        )

    # Summaries
    def rate(bools: List[bool]) -> float:
        return sum(1 for b in bools if b) / len(bools) if bools else 0.0

    summary = {
        "dedupe_field": dedupe_field,
        "n": len(outputs),
        "refusal_rate_overall": rate(refusal_all),
        "refusal_rate_answerable": rate(refusal_answerable),
        "refusal_rate_unanswerable": rate(refusal_unanswerable),
        "citation_rate_nonrefusal": rate([not x for x in []])  # placeholder
    }

    # Citation rate among non-refusals
    if citation_present_nonrefusal:
        summary["citation_rate_nonrefusal"] = sum(1 for x in citation_present_nonrefusal if x) / len(citation_present_nonrefusal)
    else:
        summary["citation_rate_nonrefusal"] = 0.0

    summary["latency_p50_ms"] = pctl(latencies, 0.50)
    summary["latency_p95_ms"] = pctl(latencies, 0.95)
    summary["diversity_avg_unique_doc_ids"] = sum(diversity_docids) / len(diversity_docids) if diversity_docids else 0.0

    return outputs, summary


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write list of dicts to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_PATH}. Create it first.")

    # Run eval twice for comparison
    out_doc, sum_doc = run_eval("doc_id")
    out_title, sum_title = run_eval("title")

    # Combine outputs (so you can filter later by dedupe_field)
    combined = out_doc + out_title
    write_jsonl(OUT_PATH, combined)

    # Print summaries
    print("=" * 80)
    print("DAY 8 EVAL SUMMARY")
    print("-" * 80)
    for s in [sum_doc, sum_title]:
        print(f"dedupe_field: {s['dedupe_field']}")
        print(f"  n: {s['n']}")
        print(f"  refusal_rate_overall:      {s['refusal_rate_overall']:.2f}")
        print(f"  refusal_rate_answerable:   {s['refusal_rate_answerable']:.2f}")
        print(f"  refusal_rate_unanswerable: {s['refusal_rate_unanswerable']:.2f}")
        print(f"  citation_rate_nonrefusal:  {s['citation_rate_nonrefusal']:.2f}")
        print(f"  latency_p50_ms:            {s['latency_p50_ms']}")
        print(f"  latency_p95_ms:            {s['latency_p95_ms']}")
        print(f"  diversity_avg_unique_docs: {s['diversity_avg_unique_doc_ids']:.2f}")
        print("-" * 80)

    # Show 3 examples where doc_id vs title behaved differently (by diversity)
    # (Heuristic: compare diversity_unique_doc_ids for same qid between two runs.)
    diff = []
    doc_by_qid = {r["qid"]: r for r in out_doc}
    title_by_qid = {r["qid"]: r for r in out_title}
    for qid in doc_by_qid:
        d = doc_by_qid[qid]["diversity_unique_doc_ids"] - title_by_qid[qid]["diversity_unique_doc_ids"]
        diff.append((abs(d), d, qid))
    diff.sort(reverse=True)

    print("Top 3 qids where diversity differs most (doc_id vs title):")
    for _, d, qid in diff[:3]:
        print(f"  {qid}: doc_id diversity - title diversity = {d}")

    print("=" * 80)
    print(f"Wrote outputs -> {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
