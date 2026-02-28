# src/chunk_short.py
# This script turns docs_short.jsonl into chunks_short.parquet (chunked text rows).

import json  # For reading each JSON object per line in a JSONL file.
from pathlib import Path  # For OS-independent file path handling.

import pandas as pd  # For building a table of chunks and writing Parquet.

# -----------------------------
# Input/Output file locations
# -----------------------------

INP = Path("data/processed/docs_short.jsonl")  # Input JSONL produced on Day 2.
OUT = Path("data/processed/chunks_short.parquet")  # Output Parquet with chunk rows.

# -----------------------------
# Chunking parameters
# -----------------------------
# We chunk by characters to keep it simple and deterministic. We'll tune later.

CHUNK_SIZE = 800   # The target number of characters per chunk.
OVERLAP = 150      # How many characters overlap between consecutive chunks.


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """
    Split a string into overlapping character chunks.

    Inputs:
      - text: the full document text to chunk.
      - chunk_size: max characters per chunk.
      - overlap: characters repeated between chunks to preserve context.

    Output:
      - list of (start_index, end_index, chunk_string)
    """

    text = text.strip()  # Remove leading/trailing whitespace for consistent chunk boundaries.

    if not text:  # If empty after stripping, there is nothing to chunk.
        return []  # Return an empty list (no chunks).

    chunks = []  # This will store tuples of (start, end, chunk).

    start = 0  # Start index of the current chunk window.
    n = len(text)  # Total number of characters in the document text.

    # Keep making chunks until the start pointer reaches the end of the string.
    while start < n:
        end = min(start + chunk_size, n)  # End index is chunk_size ahead, but not beyond n.
        chunk = text[start:end]  # Slice the text to create the chunk content.

        # Save a tuple containing the chunk location (start/end) plus the chunk text itself.
        chunks.append((start, end, chunk))

        # If we reached the end of the text, stop; no more chunks are needed.
        if end == n:
            break

        # Move the start forward for the next chunk, but keep an overlap.
        # Example: if chunk_size=800 and overlap=150, next start is end-150.
        start = max(0, end - overlap)

    return chunks  # Return the list of chunk tuples.


def main(limit_docs: int | None = None) -> None:
    """
    Read JSONL docs, chunk them, and write chunks to Parquet.

    Inputs:
      - limit_docs: optional cap on how many documents to process (useful for quick tests).

    Output:
      - data/processed/chunks_short.parquet
    """

    # Ensure the input file exists; otherwise the user likely skipped Day 2.
    assert INP.exists(), f"Missing {INP}. Run Day 2 ingest first."

    rows = []  # We'll accumulate a list of dicts, one dict per chunk row.

    # Open the JSONL file. Each line is a JSON object representing one document.
    with INP.open("r", encoding="utf-8") as f:
        # Enumerate gives us the doc counter i (0,1,2,...) plus the line content.
        for i, line in enumerate(f):
            # If the user provided a limit and we've reached it, stop early.
            if limit_docs is not None and i >= limit_docs:
                break

            doc = json.loads(line)  # Parse the JSON string into a Python dict.

            doc_id = doc["doc_id"]  # Unique ID for the doc (required by our schema).
            title = doc.get("title", "")  # Optional title; default to empty string.
            text = doc.get("text", "")  # Main text content we will chunk.

            # Generate chunks for this document, with character offsets.
            for j, (s, e, c) in enumerate(chunk_text(text)):
                # Append one chunk "row" (dict) to the list.
                rows.append(
                    {
                        "chunk_id": f"{doc_id}_c{j}",  # Unique chunk ID: doc + chunk index.
                        "doc_id": doc_id,              # Link chunk back to its document.
                        "title": title,                # Carry title for convenience in UI/debug.
                        "chunk_index": j,              # Chunk order within the doc (0,1,2,...).
                        "char_start": s,               # Start character offset in original text.
                        "char_end": e,                 # End character offset in original text.
                        "text": c,                     # The chunk content itself.
                        "source": doc.get("source", "unknown"),  # Dataset/source label.
                    }
                )

    df = pd.DataFrame(rows)  # Convert chunk rows into a DataFrame table.

    OUT.parent.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists.

    df.to_parquet(OUT, index=False)  # Write chunks table to Parquet (fast + compact).

    # Print basic stats so you can sanity check the pipeline output.
    print(f"Wrote chunks -> {OUT}")
    print(f"Docs processed: {df['doc_id'].nunique()}")
    print(f"Chunks: {len(df)}")
    print(f"Avg chunk chars: {df['text'].str.len().mean():.1f}")


if __name__ == "__main__":
    # When run as a script, execute main() on the full dataset.
    main()
