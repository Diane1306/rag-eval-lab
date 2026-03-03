# src/rag_retrieve.py
#
# Purpose:
#   Provide a reusable retrieval function for the Streamlit app.
#   This module:
#     - lazily loads the Sentence-Transformers model (once per Python process)
#     - lazily loads the FAISS index + metadata (once per Python process)
#     - embeds a query
#     - searches FAISS for nearest neighbors
#     - deduplicates results by a chosen metadata field (default: doc_id)
#     - returns a structured list of results + latency info
#
# Why this design works well with Streamlit:
#   Streamlit reruns your script on each interaction, but the Python process often persists.
#   Module-level cached objects avoid expensive reloads on each rerun.

import time  # Measure retrieval latency.
from pathlib import Path  # Handle paths robustly.

import numpy as np  # Query vector dtype conversion for FAISS.
import pandas as pd  # Load Parquet metadata.
import faiss  # Load/search FAISS index.
from sentence_transformers import SentenceTransformer  # Embed query text.

# -----------------------------
# Configuration: where your index lives
# -----------------------------

OUT_DIR = Path("data/processed/index_short")  # Default index folder
FAISS_INDEX_PATH = OUT_DIR / "faiss.index"  # FAISS index file
META_PATH = OUT_DIR / "chunks_meta.parquet"  # metadata aligned with index rows

MODEL_NAME = "all-mpnet-base-v2"  # Must match indexing model

# -----------------------------
# Module-level caches (persist across Streamlit reruns)
# -----------------------------

_cached_model = None  # Will hold the SentenceTransformer model after first load
_cached_index = None  # Will hold the FAISS index after first load
_cached_meta = None   # Will hold the metadata DataFrame after first load


def _load_model() -> SentenceTransformer:
    """
    Lazily load the embedding model and cache it.

    Output:
      - SentenceTransformer model object
    """
    global _cached_model

    # If already loaded, return it immediately.
    if _cached_model is not None:
        return _cached_model

    # Otherwise load from disk/cache (may download on first ever use).
    _cached_model = SentenceTransformer(MODEL_NAME)
    return _cached_model


def _load_index_and_meta():
    """
    Lazily load FAISS index + metadata and cache them.

    Output:
      - (faiss_index, metadata_dataframe)
    """
    global _cached_index, _cached_meta

    # If already loaded, return cached objects.
    if _cached_index is not None and _cached_meta is not None:
        return _cached_index, _cached_meta

    # Ensure files exist; fail with clear error messages if not.
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FAISS_INDEX_PATH}. Run: uv run python src/embed_index_short.py"
        )
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Missing {META_PATH}. Run: uv run python src/embed_index_short.py"
        )

    # Load index from disk.
    _cached_index = faiss.read_index(str(FAISS_INDEX_PATH))

    # Load metadata aligned with index row order.
    _cached_meta = pd.read_parquet(META_PATH)

    return _cached_index, _cached_meta


def retrieve(
    query: str,
    k: int = 5,
    dedupe_field: str = "doc_id",
    candidate_multiplier: int = 5,
    preview_chars: int = 240,
):
    """
    Retrieve top-k results for a query, with deduplication.

    Inputs:
      - query: query string to search
      - k: number of results to return after dedupe
      - dedupe_field: metadata field to dedupe by (doc_id default; can use title)
      - candidate_multiplier: retrieve k*multiplier before dedupe to increase diversity
      - preview_chars: how many characters to include in text previews

    Outputs:
      - dict with:
          results: list[dict] each containing score, chunk_id, doc_id, title, text_preview
          latency_ms: int
          k_candidates: int
          dedupe_field: str
    """

    # Start timing.
    t0 = time.time()

    # Load cached resources.
    model = _load_model()
    index, meta = _load_index_and_meta()

    # Validate dedupe field exists.
    if dedupe_field not in meta.columns:
        raise ValueError(
            f"dedupe_field='{dedupe_field}' not in metadata. "
            f"Available: {list(meta.columns)}"
        )

    # Embed query as a normalized float32 vector.
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Retrieve more candidates than k for diversity before dedupe.
    k_candidates = max(k * candidate_multiplier, k)

    # FAISS search returns (scores, idxs) shaped (1, k_candidates).
    scores, idxs = index.search(q_vec, k_candidates)
    scores = scores[0]
    idxs = idxs[0]

    # Deduplicate while preserving rank order.
    seen = set()
    results = []

    for score, idx in zip(scores, idxs):
        if idx < 0:
            continue

        row = meta.iloc[int(idx)]
        key = str(row.get(dedupe_field, ""))

        if not key or key in seen:
            continue

        seen.add(key)

        # Build a short preview with newlines escaped.
        text = str(row.get("text", "")).replace("\n", "\\n")
        text_preview = text if len(text) <= preview_chars else text[:preview_chars] + "..."

        results.append(
            {
                "score": float(score),
                "chunk_id": row.get("chunk_id", ""),
                "doc_id": row.get("doc_id", ""),
                "title": row.get("title", ""),
                "dedupe_key": key,
                "text_preview": text_preview,
            }
        )

        if len(results) >= k:
            break

    # Stop timing.
    latency_ms = int((time.time() - t0) * 1000)

    return {
        "results": results,
        "latency_ms": latency_ms,
        "k_candidates": k_candidates,
        "dedupe_field": dedupe_field,
    }
