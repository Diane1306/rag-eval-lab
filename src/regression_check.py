# src/regression_check.py
#
# Purpose:
#   Enforce regression thresholds on evaluation summaries.
#   If metrics violate thresholds, this script exits with code 1 (failure).
#   Otherwise exits with code 0 (success).
#
# Input:
#   - reports/eval_summary.json
#
# Output:
#   - prints PASS/FAIL messages and exits with appropriate status code
#
# How to run:
#   uv run python src/regression_check.py

import json
import sys
from pathlib import Path

SUMMARY_PATH = Path("reports/eval_summary.json")

# -----------------------------
# Threshold configuration
# -----------------------------
# These thresholds are based on your CURRENT baseline:
# doc_id: refusal_answerable=0.10, refusal_unanswerable=0.40, citation=1.00, p95=39, diversity=5.0
# title : refusal_answerable=0.10, refusal_unanswerable=0.40, citation=1.00, p95=25, diversity=3.7

THRESHOLDS = {
    "doc_id": {
        "citation_rate_nonrefusal_min": 0.95,
        "latency_p95_ms_max": 80,
        "diversity_avg_unique_docs_min": 4.5,
        "refusal_rate_answerable_max": 0.20,
        "refusal_rate_unanswerable_min": 0.30,
    },
    "title": {
        "citation_rate_nonrefusal_min": 0.95,
        "latency_p95_ms_max": 60,
        "diversity_avg_unique_docs_min": 3.0,
        "refusal_rate_answerable_max": 0.20,
        "refusal_rate_unanswerable_min": 0.30,
    },
}


def fail(msg: str) -> None:
    """Print failure message and exit non-zero."""
    print("FAIL:", msg)
    sys.exit(1)


def main() -> None:
    # 1) Ensure summary exists.
    if not SUMMARY_PATH.exists():
        fail(f"Missing {SUMMARY_PATH}. Run the eval runner to generate it first.")

    # 2) Load summary JSON.
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    runs = blob.get("runs", [])
    if not runs:
        fail("Summary JSON has no 'runs' array.")

    # 3) Index runs by dedupe_field.
    by_field = {}
    for r in runs:
        field = r.get("dedupe_field")
        if field:
            by_field[field] = r

    # 4) Validate we have the required fields present.
    for field, th in THRESHOLDS.items():
        if field not in by_field:
            fail(f"Missing dedupe_field='{field}' in summary runs. Found: {list(by_field.keys())}")

        r = by_field[field]

        # Extract metrics
        citation = float(r.get("citation_rate_nonrefusal", 0.0))
        p95 = int(r.get("latency_p95_ms", 10**9))
        # Accept either field name (older/newer eval runner versions)
        diversity = float(
            r.get("diversity_avg_unique_docs",
                  r.get("diversity_avg_unique_doc_ids", 0.0))
        )
        refuse_ans = float(r.get("refusal_rate_answerable", 1.0))
        refuse_unans = float(r.get("refusal_rate_unanswerable", 0.0))

        # Check thresholds
        if citation < th["citation_rate_nonrefusal_min"]:
            fail(f"{field}: citation_rate_nonrefusal {citation:.2f} < {th['citation_rate_nonrefusal_min']}")

        if p95 > th["latency_p95_ms_max"]:
            fail(f"{field}: latency_p95_ms {p95} > {th['latency_p95_ms_max']}")

        if diversity < th["diversity_avg_unique_docs_min"]:
            fail(f"{field}: diversity_avg_unique_docs {diversity:.2f} < {th['diversity_avg_unique_docs_min']}")

        if refuse_ans > th["refusal_rate_answerable_max"]:
            fail(f"{field}: refusal_rate_answerable {refuse_ans:.2f} > {th['refusal_rate_answerable_max']}")

        if refuse_unans < th["refusal_rate_unanswerable_min"]:
            fail(f"{field}: refusal_rate_unanswerable {refuse_unans:.2f} < {th['refusal_rate_unanswerable_min']}")

        print(f"PASS: {field} meets thresholds.")

    print("ALL PASS: regression checks successful.")
    sys.exit(0)


if __name__ == "__main__":
    main()