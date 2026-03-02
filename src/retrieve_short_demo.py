# src/retrieve_short_demo.py
#
# Purpose:
#   This script demonstrates *vector retrieval* over your chunk corpus using:
#     1) a FAISS index built from chunk embeddings
#     2) a metadata table aligned with the FAISS index rows
#
#   It:
#     - embeds a user query using the same Sentence-Transformers model as indexing
#     - retrieves top candidates from FAISS
#     - deduplicates results by a user-chosen field (doc_id by default; title optional)
#     - prints the final top-k results
#
# Inputs:
#   - FAISS index: data/processed/index_short/faiss.index
#   - metadata:   data/processed/index_short/chunks_meta.parquet
#   - CLI args:   --query, --k, --dedupe_field, --candidate_multiplier
#
# Outputs:
#   - prints ranked retrieval results to stdout
#
# How to run:
#   # Default dedupe (doc_id)
#   uv run python src/retrieve_short_demo.py --query "How did Beyonce become popular?" --k 5
#
#   # Dedupe by title (more topic-diverse, but may return <k if neighbors share titles)
#   uv run python src/retrieve_short_demo.py --query "How did Beyonce become popular?" --k 5 --dedupe_field title
#
# Success criteria:
#   - The script prints results and the dedupe key is unique across printed rows (as much as data allows)


import argparse  # Parse command-line arguments like --query, --k, --dedupe_field.
from pathlib import Path  # Work with filesystem paths safely.

import numpy as np  # Handle vectors and ensure correct dtype for FAISS.
import pandas as pd  # Load metadata (Parquet).
import faiss  # Load/search the FAISS vector index.
from sentence_transformers import SentenceTransformer  # Embed the query into a dense vector.

# -----------------------------
# Configuration: file paths
# -----------------------------

OUT_DIR = Path("data/processed/index_short")  # Output directory from embed_index_short.py
FAISS_INDEX_PATH = OUT_DIR / "faiss.index"  # FAISS index file
META_PATH = OUT_DIR / "chunks_meta.parquet"  # Metadata table aligned to index rows

# -----------------------------
# Configuration: embedding model
# -----------------------------

MODEL_NAME = "all-mpnet-base-v2"  # Must match embed_index_short.py


def safe_preview(text: str, max_chars: int = 240) -> str:
    """
    Create a compact one-line preview of a text block.

    Why:
      - Chunk texts can be long and multi-line.
      - We want readable terminal output.

    Inputs:
      - text: original text
      - max_chars: maximum number of characters to display

    Output:
      - One-line truncated preview string
    """
    s = str(text).replace("\n", "\\n")  # Escape newlines so it stays on one terminal line.
    return s if len(s) <= max_chars else s[:max_chars] + "..."  # Truncate if needed.


def main(query: str, k: int, dedupe_field: str, candidate_multiplier: int) -> None:
    """
    Run retrieval demo.

    Steps:
      1) Validate index and metadata files exist
      2) Load FAISS index
      3) Load metadata
      4) Validate dedupe_field exists in metadata
      5) Embed query
      6) Retrieve candidates from FAISS
      7) Deduplicate by dedupe_field
      8) Print results

    Inputs:
      - query: query string to embed and search
      - k: number of final results to return (after deduplication)
      - dedupe_field: metadata column name used to enforce diversity (e.g., doc_id, title)
      - candidate_multiplier: retrieve k * multiplier before dedupe to increase diversity
    """

    # -----------------------------
    # Step 1: sanity-check files exist
    # -----------------------------

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FAISS_INDEX_PATH}. Run: uv run python src/embed_index_short.py"
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Missing {META_PATH}. Run: uv run python src/embed_index_short.py"
        )

    # -----------------------------
    # Step 2: load index and metadata
    # -----------------------------

    index = faiss.read_index(str(FAISS_INDEX_PATH))  # Load FAISS index from disk
    meta = pd.read_parquet(META_PATH)  # Load metadata aligned with FAISS row order

    # -----------------------------
    # Step 3: validate dedupe_field
    # -----------------------------

    if dedupe_field not in meta.columns:
        raise ValueError(
            f"dedupe_field='{dedupe_field}' not in metadata columns.\n"
            f"Available columns: {list(meta.columns)}\n"
            f"Tip: try --dedupe_field doc_id or --dedupe_field title"
        )

    # -----------------------------
    # Step 4: embed query
    # -----------------------------

    model = SentenceTransformer(MODEL_NAME)  # Load embedding model (cached after first download)
    q_vec = model.encode(
        [query],                   # encode expects list[str]; we embed one query
        convert_to_numpy=True,      # return numpy array
        normalize_embeddings=True   # normalize vectors so cosine == dot product
    ).astype(np.float32)           # FAISS expects float32

    # -----------------------------
    # Step 5: retrieve candidates
    # -----------------------------

    # Retrieve more than k so deduplication has room to pick diverse results.
    k_candidates = max(k * candidate_multiplier, k)

    # FAISS search returns (scores, indices) shaped (1, k_candidates) for one query.
    scores, idxs = index.search(q_vec, k_candidates)
    scores = scores[0]
    idxs = idxs[0]

    # -----------------------------
    # Step 6: deduplicate while preserving rank order
    # -----------------------------

    seen_keys = set()  # Track already-used dedupe keys.
    results = []       # Store chosen results as tuples (score, idx, row).

    for score, idx in zip(scores, idxs):
        if idx < 0:  # Defensive: FAISS can return -1 if not enough neighbors
            continue

        row = meta.iloc[int(idx)]  # Map FAISS index row -> metadata row
        key = str(row.get(dedupe_field, ""))  # Extract the dedupe key and normalize to string

        if not key:  # Skip empty keys (rare)
            continue

        if key in seen_keys:  # Skip duplicates of the chosen field
            continue

        seen_keys.add(key)
        results.append((float(score), int(idx), row))

        if len(results) >= k:  # Stop once we have k unique results
            break

    # -----------------------------
    # Step 7: print results
    # -----------------------------

    print("=" * 80)
    print(f"Query: {query}")
    print(f"Desired top-k: {k}")
    print(f"Deduplication field: {dedupe_field}")
    print(f"Candidates retrieved from FAISS: {k_candidates}")
    print(f"Results returned after dedupe: {len(results)}")
    print("-" * 80)

    for rank, (score, idx, row) in enumerate(results, start=1):
        print(
            f"[{rank}] score={score:.4f} "
            f"chunk_id={row.get('chunk_id', '')} "
            f"doc_id={row.get('doc_id', '')}"
        )
        print(f"     title={row.get('title', '')}")
        print(f"     dedupe_key({dedupe_field})={row.get(dedupe_field, '')}")
        print(f"     text_preview={safe_preview(row.get('text', ''))}")

    if len(results) < k:
        print("-" * 80)
        print(
            "Note: Returned fewer than k unique results.\n"
            "Possible reasons:\n"
            "  - Many top neighbors share the same dedupe key (low diversity near the query)\n"
            "  - k is large relative to dataset diversity\n"
            "  - Increase --candidate_multiplier to retrieve more candidates before dedupe\n"
        )

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    # -----------------------------
    # CLI interface
    # -----------------------------

    parser = argparse.ArgumentParser(
        description="Demo: retrieve top-k unique results using FAISS + Sentence-Transformers embeddings."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query string to search for (required)."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return after deduplication (default: 5)."
    )

    parser.add_argument(
        "--dedupe_field",
        type=str,
        default="doc_id",
        help="Metadata column used for deduplication (default: doc_id). Try: title"
    )

    parser.add_argument(
        "--candidate_multiplier",
        type=int,
        default=5,
        help="Retrieve k * multiplier candidates before deduplication (default: 5)."
    )

    args = parser.parse_args()

    main(
        query=args.query,
        k=args.k,
        dedupe_field=args.dedupe_field,
        candidate_multiplier=args.candidate_multiplier
    )
