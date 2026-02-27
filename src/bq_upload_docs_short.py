import json
from pathlib import Path
from google.cloud import bigquery

PROJECT_ID = None  # use ADC default
DATASET = "rag_eval_lab"
TABLE = "docs_short"
SRC = Path("data/processed/docs_short.jsonl")

SCHEMA = [
    bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("title", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("url", "STRING", mode="NULLABLE"),
]

def ensure_table(client: bigquery.Client, dataset_id: str, table_id: str):
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    try:
        client.get_table(full_table_id)
        return full_table_id
    except Exception:
        table = bigquery.Table(full_table_id, schema=SCHEMA)
        table = client.create_table(table)
        return table.full_table_id

def main(max_rows: int = 20000, batch_size: int = 200) -> None:
    assert SRC.exists(), f"Missing {SRC}. Run Day 2 ingest first."

    client = bigquery.Client(project=PROJECT_ID)
    full_table_id = ensure_table(client, DATASET, TABLE)

    # Read JSONL
    rows = []
    with SRC.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(json.loads(line))

    # Batch streaming inserts
    total = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        errors = client.insert_rows_json(full_table_id, batch)
        if errors:
            raise RuntimeError(f"Insert errors near row {start} (first 3): {errors[:3]}")
        total += len(batch)
        if total % 2000 == 0:
            print(f"Inserted {total}/{len(rows)}...")

    print(f"Inserted {total} rows into {full_table_id}")

if __name__ == "__main__":
    main()
