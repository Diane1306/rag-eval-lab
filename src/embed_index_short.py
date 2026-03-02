# src/embed_index_short.py
# Build embeddings for chunk texts and create a FAISS index (local retrieval backend).

import time  # For measuring runtime and printing useful timing info.
from pathlib import Path  # For robust path manipulation.

import numpy as np  # For working with embedding arrays.
import pandas as pd  # For loading/saving Parquet metadata tables.
import faiss  # FAISS library for fast similarity search over vectors.
from sentence_transformers import SentenceTransformer  # Local embedding model loader.

# -----------------------------
# Configuration
# -----------------------------

# Where the chunk table lives (produced by src/chunk_short.py).
CHUNKS_PARQUET = Path("data/processed/chunks_short.parquet")

# Where we will store the index + metadata.
OUT_DIR = Path("data/processed/index_short")
FAISS_INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH = OUT_DIR / "chunks_meta.parquet"

# Embedding model choice (you selected this).
MODEL_NAME = "all-mpnet-base-v2"

# Batch size for embedding computation.
# Larger batches can be faster but use more memory; adjust if needed.
BATCH_SIZE = 64

# Number of chunks to process (None means "all chunks").
# You can set this to a smaller number for quick debugging, e.g., 5000.
LIMIT_CHUNKS = None


def main() -> None:
    """
    Main entrypoint:
      1) Load chunk records from Parquet
      2) Compute embeddings for each chunk's text
      3) Build a FAISS index
      4) Save the index + metadata so retrieval demos can use them
    """

    # 1) Sanity check that the chunk parquet exists.
    if not CHUNKS_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {CHUNKS_PARQUET}. Run: uv run python src/chunk_short.py"
        )

    # 2) Ensure output directory exists.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3) Load chunk table from Parquet into a pandas DataFrame.
    # This table contains one row per chunk: chunk_id, doc_id, title, text, etc.
    df = pd.read_parquet(CHUNKS_PARQUET)

    # 4) Optionally limit the number of chunks (useful for debugging).
    if LIMIT_CHUNKS is not None:
        df = df.iloc[:LIMIT_CHUNKS].copy()

    # 5) Extract the chunk texts we want to embed.
    # Ensure they are strings (defensive).
    texts = df["text"].astype(str).tolist()

    # 6) Load the Sentence-Transformers embedding model.
    # This will download weights the first time and cache them locally.
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    # Prevent multiprocessing instability on some Mac setups
    model._target_device = "cpu"
    
    # 7) Compute embeddings in batches.
    # We request numpy output and normalize embeddings so cosine similarity works well.
    t0 = time.time()
    print(f"Encoding {len(texts)} chunks (batch_size={BATCH_SIZE})...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,   # Show a progress bar in the terminal.
        convert_to_numpy=True,    # Return embeddings as a numpy array.
        normalize_embeddings=True # Normalize vectors; enables cosine via dot product.
    )

    # 8) Ensure embeddings are float32, which FAISS expects for most index types.
    embeddings = embeddings.astype(np.float32)

    t1 = time.time()
    print(f"Embeddings shape: {embeddings.shape} (computed in {t1 - t0:.1f}s)")

    # 9) Build a FAISS index.
    # For normalized vectors, inner product (dot product) corresponds to cosine similarity.
    dim = embeddings.shape[1]  # Embedding dimension (e.g., 768 for mpnet).
    index = faiss.IndexFlatIP(dim)  # Flat (exact) index using inner product similarity.

    # 10) Add vectors to the index.
    # The order of vectors in FAISS is exactly the order we add them.
    index.add(embeddings)

    print(f"FAISS index built. Total vectors: {index.ntotal}")

    # 11) Save the FAISS index to disk.
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Saved FAISS index to: {FAISS_INDEX_PATH}")

    # 12) Save metadata aligned to the index rows.
    # Keep only columns you want at retrieval time.
    meta_cols = ["chunk_id", "doc_id", "source", "title", "chunk_index", "char_start", "char_end", "text"]
    meta = df[meta_cols].copy()

    meta.to_parquet(META_PATH, index=False)
    print(f"Saved metadata to: {META_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
