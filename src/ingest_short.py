# src/ingest_short.py
# This script ingests a support-like Q&A dataset (SQuAD v2) and writes it to a normalized JSONL file.

import json  # Used to serialize Python dicts into JSON strings (one per line).
from pathlib import Path  # Used for robust file paths across operating systems.

from datasets import load_dataset  # Hugging Face datasets loader for pulling SQuAD v2.

# -----------------------------
# Output file location
# -----------------------------
# We'll write a JSONL file where each line is a single JSON object (a "document").
OUT = Path("data/processed/docs_short.jsonl")


def pick_first_answer(answers: dict) -> str:
    """
    Select the first available answer text from the SQuAD 'answers' field.

    Why:
      - SQuAD stores answers as a dict containing a list of answer strings.
      - Many questions have multiple acceptable answers; we keep the first for simplicity.

    Input:
      - answers: dict like {"text": [...], "answer_start": [...]}

    Output:
      - A single answer string (possibly empty if not present).
    """

    # Defensive programming: if answers is None or empty, return an empty string.
    if not answers:
        return ""

    # SQuAD stores answer texts in answers["text"] as a list.
    texts = answers.get("text", [])

    # If the list exists and has at least one element, choose the first; otherwise empty string.
    first = texts[0] if texts else ""

    # Strip whitespace so output is cleaner and consistent.
    return first.strip()


def main(n: int = 20000) -> None:
    """
    Download SQuAD v2 and write the first n examples in normalized JSONL format.

    Inputs:
      - n: number of dataset rows to write

    Outputs:
      - data/processed/docs_short.jsonl
    """

    # Ensure the output directory exists (create parent dirs if needed).
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Load the dataset split.
    # - "squad_v2" is the dataset name on Hugging Face
    # - split="train" selects the training set
    ds = load_dataset("squad_v2", split="train")

    # Avoid requesting more rows than exist.
    n = min(n, len(ds))

    # We'll compute simple length statistics for your dev log and sanity checking.
    total_chars = 0               # Sum of character lengths across all docs.
    min_len = 10**9               # Very large initial value so any real doc is smaller.
    max_len = 0                   # Initial max length.

    # Open the output file for writing text in UTF-8.
    # We'll write one JSON object per line (JSONL).
    with OUT.open("w", encoding="utf-8") as f:
        # Loop over the first n dataset items.
        for i in range(n):
            row = ds[i]  # This returns a dict-like object with keys: title, question, context, answers, etc.

            # Extract fields safely and normalize whitespace.
            title = (row.get("title") or "").strip()         # Article title
            question = (row.get("question") or "").strip()   # Question string
            context = (row.get("context") or "").strip()     # Passage/context string

            # Extract one answer (SQuAD v2 can have multiple answers; some are unanswerable).
            answer = pick_first_answer(row.get("answers") or {})

            # Build the final text blob that will be chunked and indexed later.
            # This formatting makes it "support-like" (question + answer) while keeping grounding context.
            full_text = f"Question: {question}\nAnswer: {answer}\n\nContext:\n{context}".strip()

            # Build a normalized document record.
            # doc_id must be unique; we include the dataset + split + index.
            doc = {
                "doc_id": f"squad_v2_train_{i}",  # Unique doc ID
                "title": title[:200],             # Keep title reasonably short
                "source": "squad_v2",             # Source label for debugging/filtering
                "text": full_text,                # Main payload text
                "url": "",                         # No URL for this dataset
            }

            # Compute text length for statistics.
            s = len(full_text)
            total_chars += s
            min_len = min(min_len, s)
            max_len = max(max_len, s)

            # Write JSON object as a single line.
            # ensure_ascii=False preserves non-ASCII characters rather than escaping them.
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Compute average length for the report.
    avg_len = total_chars / max(n, 1)

    # Print a summary so you can confirm success in the terminal.
    print(f"Wrote {n} docs -> {OUT}")
    print(f"Chars: avg={avg_len:.1f}, min={min_len}, max={max_len}")


if __name__ == "__main__":
    # Provide a command-line interface so you can choose n without editing the file.
    import argparse  # Standard library tool for parsing CLI arguments.

    # Create a parser that understands "--n".
    p = argparse.ArgumentParser(description="Ingest SQuAD v2 into normalized JSONL for RAG.")
    p.add_argument("--n", type=int, default=20000, help="Number of rows to ingest from SQuAD v2 train split.")

    # Parse args from the command line.
    args = p.parse_args()

    # Run the ingestion with the chosen n value.
    main(args.n)
