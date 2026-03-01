# src/duckdb_parquet_demo.py
# Teaching-mode example: run DuckDB SQL from Python against a Parquet file.

import duckdb  # DuckDB Python package gives you an embedded analytical database.
from pathlib import Path  # For safe file paths.

# -----------------------------
# Configuration: where is the Parquet file?
# -----------------------------

PARQUET_PATH = Path("data/processed/chunks_short.parquet")  # The Parquet you created on Day 3.


def run_query(con: duckdb.DuckDBPyConnection, sql: str, title: str) -> None:
    """
    Helper function to run SQL and print results.

    Inputs:
      - con: an active DuckDB connection
      - sql: SQL string to execute
      - title: label to print before results (for readability)

    Output:
      - prints results to the terminal
    """
    print("\n" + "=" * 80)
    print(title)
    print("- SQL:")
    print(sql)

    # Execute the SQL and fetch results as a pandas DataFrame for nice printing.
    df = con.execute(sql).df()

    print("- Result preview:")
    print(df.head(20))  # Print up to first 20 rows (keeps terminal output manageable).


def main() -> None:
    """
    Main entry: connect to DuckDB and run a few Parquet queries.
    """

    # 1) Ensure the Parquet file exists; otherwise you need to run chunking first.
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PARQUET_PATH}. Run: uv run python src/chunk_short.py"
        )

    # 2) Create an in-memory DuckDB connection.
    # You can also connect to a persistent database file like duckdb.connect("rag.duckdb").
    con = duckdb.connect(database=":memory:")

    # 3) Turn the path into a string for SQL.
    parquet_str = str(PARQUET_PATH)

    # 4) Query 1: read a few doc_ids (projection pushdown concept).
    sql1 = f"""
    SELECT doc_id
    FROM read_parquet('{parquet_str}')
    LIMIT 5;
    """
    run_query(con, sql1, "Query 1: Select only doc_id (projection pushdown)")

    # 5) Query 2: filter pushdown example (count only SQuAD source chunks).
    sql2 = f"""
    SELECT COUNT(*) AS n_chunks
    FROM read_parquet('{parquet_str}')
    WHERE source = 'squad_v2';
    """
    run_query(con, sql2, "Query 2: Filter by source (filter pushdown)")

    # 6) Query 3: aggregation example (top docs by chunk count).
    sql3 = f"""
    SELECT doc_id, COUNT(*) AS n_chunks
    FROM read_parquet('{parquet_str}')
    GROUP BY doc_id
    ORDER BY n_chunks DESC
    LIMIT 10;
    """
    run_query(con, sql3, "Query 3: Aggregate chunks per doc (GROUP BY)")

    # 7) Close connection (good hygiene).
    con.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
