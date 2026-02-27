import json
from pathlib import Path
import pandas as pd

INP = Path("data/processed/docs_short.jsonl")
OUT = Path("data/processed/chunks_short.parquet")

# Chunking parameters (you can tune later)
CHUNK_SIZE = 800     # characters
OVERLAP = 150        # characters


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def main(limit_docs: int | None = None) -> None:
    assert INP.exists(), f"Missing {INP}. Run Day 2 ingest first."

    rows = []
    with INP.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit_docs is not None and i >= limit_docs:
                break
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")

            for j, (s, e, c) in enumerate(chunk_text(text)):
                rows.append(
                    {
                        "chunk_id": f"{doc_id}_c{j}",
                        "doc_id": doc_id,
                        "title": title,
                        "chunk_index": j,
                        "char_start": s,
                        "char_end": e,
                        "text": c,
                        "source": doc.get("source", "unknown"),
                    }
                )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"Wrote chunks -> {OUT}")
    print(f"Docs processed: {df['doc_id'].nunique()}")
    print(f"Chunks: {len(df)}")
    print(f"Avg chunk chars: {df['text'].str.len().mean():.1f}")


if __name__ == "__main__":
    main()
