# src/bq_log_event_v2.py
#
# Purpose:
#   Log retrieval + answer telemetry into BigQuery table: rag_eval_lab.rag_events_v2
#   This is a "v2" schema that adds fields for deterministic grounded answers:
#     - answer_len, has_citations, n_citations, refusal
#
# Why v2:
#   BigQuery table schemas are not always convenient to evolve in place during early prototyping.
#   Creating a new v2 table avoids schema migration complexity and keeps your Day 6 data intact.
#
# How it is used:
#   - Called once per user turn from Streamlit.
#   - Inserts exactly one row per turn (small streaming insert).
#
# Success criteria:
#   - BigQuery table exists and row count increases as you chat.
#   - Fields populate correctly (especially has_citations/refusal).

from datetime import datetime, timezone  # Generate a UTC timestamp.
from google.cloud import bigquery  # BigQuery client.

# -----------------------------
# BigQuery dataset/table naming
# -----------------------------

DATASET = "rag_eval_lab"     # Your dataset
TABLE = "rag_events_v2"      # New v2 table name

# -----------------------------
# BigQuery schema for v2 table
# -----------------------------

SCHEMA = [
    # Time + identifiers
    bigquery.SchemaField("event_ts", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("turn_id", "INT64", mode="REQUIRED"),

    # Retrieval inputs
    bigquery.SchemaField("query", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("dedupe_field", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("top_k", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("candidate_multiplier", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("k_candidates", "INT64", mode="REQUIRED"),

    # Retrieval outputs (IDs only; keep logs small)
    bigquery.SchemaField("retrieved_chunk_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("retrieved_doc_ids", "STRING", mode="REPEATED"),

    # Performance
    bigquery.SchemaField("latency_ms", "INT64", mode="REQUIRED"),

    # Answer telemetry (new in v2)
    bigquery.SchemaField("answer_len", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("has_citations", "BOOL", mode="REQUIRED"),
    bigquery.SchemaField("n_citations", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("refusal", "BOOL", mode="REQUIRED"),
]


def ensure_table(client: bigquery.Client) -> str:
    """
    Ensure rag_events_v2 table exists; create it if missing.

    Returns:
      - A dot-separated full table id: "<project>.<dataset>.<table>"
    """

    # Build references (robust; avoids colon-format issues).
    dataset_ref = bigquery.DatasetReference(client.project, DATASET)
    table_ref = dataset_ref.table(TABLE)

    try:
        # If table exists, this succeeds.
        client.get_table(table_ref)
    except Exception:
        # If not, create table with schema.
        table = bigquery.Table(table_ref, schema=SCHEMA)

        # Partition by day on event_ts to control cost and speed up time-filtered queries.
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="event_ts",
        )

        client.create_table(table)

    # Return standard dot-separated ID for streaming inserts.
    return f"{client.project}.{DATASET}.{TABLE}"


def log_event_v2(
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
    answer_len: int,
    has_citations: bool,
    n_citations: int,
    refusal: bool,
) -> None:
    """
    Insert one telemetry row into rag_events_v2.

    Inputs are intentionally "small":
      - We log IDs rather than large text blobs.
      - This keeps storage and query scans cheap.

    Raises:
      - RuntimeError if BigQuery returns insert errors.
    """

    # Create client (uses ADC).
    client = bigquery.Client()

    # Ensure table exists.
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
        "answer_len": answer_len,
        "has_citations": has_citations,
        "n_citations": n_citations,
        "refusal": refusal,
    }

    # Streaming insert (single row). This is fine for your scale.
    errors = client.insert_rows_json(full_table_id, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert error (v2, first 1): {errors[:1]}")
