# src/bq_upload_docs_short.py
# Upload docs_short.jsonl into BigQuery table rag_eval_lab.docs_short.
# IMPORTANT BEHAVIOR:
#   - This script TRUNCATES the target table before inserting.
#   - This makes the script idempotent: re-running it will not create duplicates.
#   - Inserts are batched to avoid BigQuery streaming request size limits (413 errors).

import json  # Parse JSONL lines into Python dicts (rows).
from pathlib import Path  # Robust filesystem paths.

from google.cloud import bigquery  # BigQuery client library.

# -----------------------------
# Configuration
# -----------------------------

PROJECT_ID = None  # None => use the default project from Application Default Credentials (ADC).
DATASET = "rag_eval_lab"  # BigQuery dataset name.
TABLE = "docs_short"  # BigQuery table name.

SRC = Path("data/processed/docs_short.jsonl")  # Input file produced by src/ingest_short.py.

# -----------------------------
# BigQuery Schema (table columns)
# -----------------------------
# We explicitly define the schema so table creation is deterministic.

SCHEMA = [
    bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),  # Unique document identifier.
    bigquery.SchemaField("title", "STRING", mode="NULLABLE"),  # Optional title.
    bigquery.SchemaField("source", "STRING", mode="NULLABLE"),  # Dataset/source label.
    bigquery.SchemaField("text", "STRING", mode="REQUIRED"),  # Document text payload.
    bigquery.SchemaField("url", "STRING", mode="NULLABLE"),  # Optional URL (often blank here).
]


def ensure_table(client: bigquery.Client, dataset_id: str, table_id: str) -> str:
    """
    Ensure the BigQuery table exists; create it if missing.

    Inputs:
      - client: authenticated BigQuery client
      - dataset_id: dataset name (e.g., rag_eval_lab)
      - table_id: table name (e.g., docs_short)

    Output:
      - fully qualified table id: "<project>.<dataset>.<table>"
    """

    # Build a fully qualified table id using the client's project.
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"

    try:
        # Attempt to retrieve table metadata; if this succeeds, table exists.
        client.get_table(full_table_id)
        return full_table_id
    except Exception:
        # If table doesn't exist (usually NotFound), create it with the schema.
        table = bigquery.Table(full_table_id, schema=SCHEMA)
        table = client.create_table(table)
        return table.full_table_id


def truncate_table(client: bigquery.Client, full_table_id: str) -> None:
    """
    TRUNCATE the target table (delete all rows, keep schema).

    Why:
      - BigQuery streaming inserts do not deduplicate automatically.
      - If you run the same insert twice, you'll get duplicates.
      - TRUNCATE ensures each run starts from a clean table.

    Inputs:
      - client: authenticated BigQuery client
      - full_table_id: "<project>.<dataset>.<table>"
    """

    # Construct a TRUNCATE SQL statement.
    # Backticks are required in BigQuery for fully qualified identifiers.
    sql = f"TRUNCATE TABLE `{full_table_id}`"

    # Execute the query as a job.
    job = client.query(sql)

    # Wait for completion so we know truncation finished before inserting new rows.
    job.result()

    print(f"Truncated table: {full_table_id}")


def main(max_rows: int = 20000, batch_size: int = 200) -> None:
    """
    Main upload routine.

    Steps:
      1) Ensure input JSONL exists
      2) Create BigQuery client
      3) Ensure table exists
      4) TRUNCATE table to avoid duplicates
      5) Read JSONL rows
      6) Insert rows in batches using streaming inserts

    Inputs:
      - max_rows: number of records to upload (default 20000)
      - batch_size: rows per API request (default 200, prevents 413)
    """

    # 1) Verify input exists.
    assert SRC.exists(), f"Missing {SRC}. Run src/ingest_short.py first."

    # 2) Create a BigQuery client (uses ADC by default).
    client = bigquery.Client(project=PROJECT_ID)

    # 3) Ensure the table exists.
    full_table_id = ensure_table(client, DATASET, TABLE)

    # 4) TRUNCATE the table before inserting to avoid duplicates.
    truncate_table(client, full_table_id)

    # 5) Read JSONL into a list of row dicts.
    rows = []
    with SRC.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            # Parse each JSON line into a dict.
            rows.append(json.loads(line))

    # 6) Insert in batches to avoid request entity too large (413).
    total_inserted = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]

        # insert_rows_json returns [] on success; otherwise it returns error details.
        errors = client.insert_rows_json(full_table_id, batch)

        if errors:
            # If errors occur, stop and show a small subset to debug.
            raise RuntimeError(f"Insert errors near row {start} (first 3): {errors[:3]}")

        total_inserted += len(batch)

        # Print progress every 2000 rows so you can see it moving.
        if total_inserted % 2000 == 0:
            print(f"Inserted {total_inserted}/{len(rows)}...")

    print(f"Inserted {total_inserted} rows into {full_table_id}")


if __name__ == "__main__":
    # Optional: add CLI arguments later if you want.
    # For now, defaults are max_rows=20000 and batch_size=200.
    main()
