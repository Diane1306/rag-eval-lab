# src/bq_load_eval_outputs_day10.py
#
# Purpose:
#   Load eval/outputs/day10_outputs.jsonl into BigQuery table:
#     rag_eval_lab.eval_outputs_day10
#
# Why load job:
#   - More scalable and reliable than streaming inserts
#   - Avoids request-size errors
#
# How to run:
#   uv run python src/bq_load_eval_outputs_day10.py
#
# Success criteria:
#   - Table exists and has expected row count (e.g., 48)
#   - Schema types look reasonable in BigQuery

from pathlib import Path
from google.cloud import bigquery

DATASET = "rag_eval_lab"
TABLE = "eval_outputs_day10"
SRC = Path("eval/outputs/day10_outputs.jsonl")

def main() -> None:
    # 1) Check source exists
    if not SRC.exists():
        raise FileNotFoundError(f"Missing {SRC}. Run the Day 10 experiment first.")

    # 2) BigQuery client (uses ADC)
    client = bigquery.Client()

    # 3) Fully qualified table id
    table_id = f"{client.project}.{DATASET}.{TABLE}"

    # 4) Configure a JSONL load job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        autodetect=True,                 # Let BigQuery infer schema (fine for prototyping)
        write_disposition="WRITE_TRUNCATE",  # Replace table each run (idempotent)
    )

    # 5) Run the load job from local file
    with SRC.open("rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)

    # 6) Wait for completion
    job.result()

    # 7) Print results
    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows} rows into {table_id}")
    print("Done.")

if __name__ == "__main__":
    main()
