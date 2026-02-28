# src/bq_upload_docs_short.py
# Upload docs_short.jsonl into BigQuery table rag_eval_lab.docs_short using batched streaming inserts.

import json  # To parse JSONL lines into Python dicts.
from pathlib import Path  # For filesystem path handling.

from google.cloud import bigquery  # BigQuery client library.

# -----------------------------
# Configuration
# -----------------------------

PROJECT_ID = None  # None means: use the default project from your ADC credentials.

DATASET = "rag_eval_lab"  # Your BigQuery dataset name.
TABLE = "docs_short"      # Target table name inside the dataset.

SRC = Path("data/processed/docs_short.jsonl")  # Input JSONL file created by ingest_short.py.

# -----------------------------
# BigQuery table schema
# -----------------------------
# We define a schema so BigQuery knows the column types and required fields.

SCHEMA = [
    bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),    # Unique doc identifier.
    bigquery.SchemaField("title", "STRING", mode="NULLABLE"),     # Optional title.
    bigquery.SchemaField("source", "STRING", mode="NULLABLE"),    # Dataset/source label.
    bigquery.SchemaField("text", "STRING", mode="REQUIRED"),      # Full doc text (Q/A + context).
    bigquery.SchemaField("url", "STRING", mode="NULLABLE"),       # Optional URL (empty in our dataset).
]


def ensure_table(client: bigquery.Client, dataset_id: str, table_id: str) -> str:
    """
    Create the BigQuery table if it doesn't exist; otherwise return its ID.

    Inputs:
      - client: BigQuery Client authenticated via ADC.
      - dataset_id: dataset name (e.g., rag_eval_lab).
      - table_id: table name (e.g., docs_short).

    Output:
      - full table id string: "<project>.<dataset>.<table>"
    """

    # Construct the fully qualified table ID using the client's active project.
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"

    try:
        # Attempt to fetch the table metadata; this succeeds if the table exists.
        client.get_table(full_table_id)
        return full_table_id  # If table exists, return its ID.
    except Exception:
        # If get_table fails (usually NotFound), we create the table with the schema.
        table = bigquery.Table(full_table_id, schema=SCHEMA)  # Define the table object.
        table = client.create_table(table)  # Create the table in BigQuery.
        return table.full_table_id  # Return the created table's ID.


def main(max_rows: int = 20000, batch_size: int = 200) -> None:
    """
    Upload JSONL docs to BigQuery using batched streaming inserts.

    Why batching matters:
      - BigQuery streaming insert endpoint has request size limits.
      - Sending 20k rows in one request can exceed the limit (413 error).

    Inputs:
      - max_rows: maximum number of JSONL rows to upload.
      - batch_size: number of rows per API call (smaller avoids 413).

    Output:
      - BigQuery table filled with rows.
    """

    # Ensure the input file exists.
    assert SRC.exists(), f"Missing {SRC}. Run Day 2 ingest first."

    # Create a BigQuery client (uses ADC credentials by default).
    client = bigquery.Client(project=PROJECT_ID)

    # Ensure the target table exists and get its full ID.
    full_table_id = ensure_table(client, DATASET, TABLE)

    # Read JSONL rows into memory (20k is fine for a laptop).
    rows = []  # List of dicts (each dict = one row).
    with SRC.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:  # Stop once we reach the maximum requested rows.
                break
            rows.append(json.loads(line))  # Parse JSON line to dict and store.

    total = 0  # Track how many rows have been successfully inserted so far.

    # Loop over the rows list in chunks of size batch_size.
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]  # Slice out the next batch of rows.

        # Insert this batch using streaming inserts.
        # Returns [] if successful; otherwise returns a list of error details.
        errors = client.insert_rows_json(full_table_id, batch)

        # If BigQuery reports errors, stop and show some error detail.
        if errors:
            raise RuntimeError(f"Insert errors near row {start} (first 3): {errors[:3]}")

        total += len(batch)  # Update inserted row count.

        # Periodically print progress so you know it's working.
        if total % 2000 == 0:
            print(f"Inserted {total}/{len(rows)}...")

    # Final success message.
    print(f"Inserted {total} rows into {full_table_id}")


if __name__ == "__main__":
    # Run main with defaults (20k rows, batched by 200).
    main()
