# src/bq_log_event.py
#
# Purpose:
#   Log retrieval events into BigQuery table rag_eval_lab.rag_events.
#   Each log row represents one user query / retrieval run.
#   We keep logs small and analytics-friendly (no huge context blobs).

from datetime import datetime, timezone  # Create a timestamp in UTC.
from google.cloud import bigquery  # BigQuery client.

# Dataset/table names (must match what you created).
DATASET = "rag_eval_lab"
TABLE = "rag_events"

# BigQuery schema for the rag_events table.
SCHEMA = [
    bigquery.SchemaField("event_ts", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("turn_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("query", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("dedupe_field", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("top_k", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("candidate_multiplier", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("k_candidates", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("retrieved_chunk_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("retrieved_doc_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("latency_ms", "INT64", mode="REQUIRED"),
]


def ensure_table(client: bigquery.Client) -> str:
    """
    Ensure rag_events table exists; create it if missing.

    Returns:
      - fully qualified table id string: "<project>.<dataset>.<table>"
    """
    # Build references using the official BigQuery reference objects (avoids colon-format bugs).
    dataset_ref = bigquery.DatasetReference(client.project, DATASET)
    table_ref = dataset_ref.table(TABLE)

    try:
        client.get_table(table_ref)  # succeeds if table exists
    except Exception:
        table = bigquery.Table(table_ref, schema=SCHEMA)

        # Partition by event_ts (DAY) for cost control
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="event_ts",
        )

        client.create_table(table)

    # Return a standard dot-separated id (safe for insert_rows_json)
    return f"{client.project}.{DATASET}.{TABLE}"

def log_event(
    session_id: str,
    turn_id: int,
    query: str,
    dedupe_field: str,
    top_k: int,
    candidate_multiplier: int,
    k_candidates: int,
    retrieved_chunk_ids: list[str],
    retrieved_doc_ids: list[str],
    latency_ms: int,
) -> None:
    """
    Insert one event row into BigQuery.

    Notes:
      - This is a streaming insert (small, one row).
      - For your scale, cost is tiny and performance is fine.

    Inputs:
      - session_id: unique session identifier
      - turn_id: per-session turn counter
      - query: user query text
      - dedupe_field: which field you deduped on (doc_id or title)
      - top_k: number of results returned after dedupe
      - candidate_multiplier: retrieval multiplier used
      - k_candidates: raw FAISS candidates retrieved
      - retrieved_chunk_ids/doc_ids: IDs for the returned results
      - latency_ms: end-to-end retrieval latency in milliseconds
    """

    client = bigquery.Client()
    full_table_id = ensure_table(client)

    # Build row payload.
    row = {
        "event_ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "turn_id": turn_id,
        "query": query,
        "dedupe_field": dedupe_field,
        "top_k": top_k,
        "candidate_multiplier": candidate_multiplier,
        "k_candidates": k_candidates,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "latency_ms": latency_ms,
    }

    # Streaming insert of one row.
    errors = client.insert_rows_json(full_table_id, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert error (first 1): {errors[:1]}")
