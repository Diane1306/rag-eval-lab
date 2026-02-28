# src/inspect_jsonl.py
# This script sanity-checks a JSONL file (one JSON object per line).
# It prints:
#   - total number of records (lines)
#   - the union of keys present across sampled records
#   - text-length statistics (avg/min/max) if a "text" field exists
#   - a short preview of the first N records

import argparse  # For parsing command-line arguments like --path and --n.
import json  # For loading each line (JSON string) into a Python dict.
from pathlib import Path  # For robust path handling.

# -----------------------------
# Default settings
# -----------------------------

DEFAULT_PATH = Path("data/processed/docs_short.jsonl")  # Default JSONL to inspect.
DEFAULT_PREVIEW_N = 3  # Default number of records to print as a preview.
DEFAULT_SAMPLE_FOR_KEYS = 200  # How many records to scan to infer keys (keeps it fast).


def safe_preview(text: str, max_chars: int = 200) -> str:
    """
    Return a safe, short preview string.

    Why:
      - JSONL "text" fields can be long.
      - We want to avoid dumping huge blocks into the terminal.

    Input:
      - text: the full text
      - max_chars: maximum characters to show

    Output:
      - a trimmed preview with "..." if truncated
    """
    if text is None:
        return ""
    text = str(text).replace("\n", "\\n")  # Replace newlines so previews print on one line.
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main(path: Path, preview_n: int, sample_for_keys: int) -> None:
    """
    Inspect a JSONL file and print sanity-check information.

    Inputs:
      - path: JSONL file path to inspect
      - preview_n: number of records to preview (printed)
      - sample_for_keys: how many records to scan to infer keys/stats quickly

    Output:
      - prints to stdout
    """

    # 1) Verify the file exists before doing anything else.
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}\n"
            f"Tip: run your ingestion script first, or pass --path to a valid JSONL file."
        )

    # 2) Initialize counters and containers for summary stats.
    total_lines = 0  # Total number of records in the file (one record per line).
    key_union = set()  # Union of keys found across sampled records.
    text_lengths = []  # Store lengths of the "text" field when present (for sampled records).
    previews = []  # Store the first preview_n parsed records for printing later.

    # 3) Read the file line-by-line (streaming) to handle large files safely.
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            total_lines += 1  # Count every line as a record.

            # Skip empty lines defensively (shouldn't happen, but avoids errors).
            if not line.strip():
                continue

            # Parse one JSON object from the line.
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                # If the JSON is malformed, show a helpful error with line number.
                raise ValueError(f"Invalid JSON on line {line_idx + 1}: {e}") from e

            # Save the first few objects for preview printing.
            if len(previews) < preview_n:
                previews.append(obj)

            # For key union + text stats, only sample the first `sample_for_keys` records for speed.
            if line_idx < sample_for_keys:
                # Update the union of keys (works if obj is a dict).
                if isinstance(obj, dict):
                    key_union.update(obj.keys())

                    # If there's a "text" field, collect its length.
                    if "text" in obj and obj["text"] is not None:
                        text_lengths.append(len(str(obj["text"])))

            # If we've already collected enough previews AND scanned enough for keys,
            # we still continue counting total_lines, because the total count matters.
            # (But note: we are only collecting key/text stats from the sampled subset.)

    # 4) Print summary statistics.
    print("=" * 60)
    print(f"JSONL path: {path}")
    print(f"Total records (lines): {total_lines}")

    # Print keys discovered in the sampled records (sorted for stable output).
    if key_union:
        print(f"Keys found (sampled first {min(sample_for_keys, total_lines)} records):")
        for k in sorted(key_union):
            print(f"  - {k}")
    else:
        print("Keys found: (none detected in sampled records)")

    # Print text length stats if we collected any.
    if text_lengths:
        avg_len = sum(text_lengths) / len(text_lengths)
        print(f"Text length stats (sampled {len(text_lengths)} records with 'text'):")
        print(f"  avg chars: {avg_len:.1f}")
        print(f"  min chars: {min(text_lengths)}")
        print(f"  max chars: {max(text_lengths)}")
    else:
        print("Text length stats: no 'text' field found in sampled records (or all were empty).")

    # 5) Print preview of the first few records.
    print("-" * 60)
    print(f"Preview: first {len(previews)} record(s)")
    for i, obj in enumerate(previews, start=1):
        # Extract common fields if they exist; otherwise show placeholders.
        doc_id = obj.get("doc_id", "(no doc_id)") if isinstance(obj, dict) else "(non-dict record)"
        title = obj.get("title", "") if isinstance(obj, dict) else ""
        source = obj.get("source", "") if isinstance(obj, dict) else ""

        # Print a compact header.
        print(f"[{i}] doc_id={doc_id}  source={source}  title={safe_preview(title, 80)}")

        # Print a short preview of the text if present.
        if isinstance(obj, dict) and "text" in obj:
            print(f"    text_preview: {safe_preview(obj.get('text', ''), 240)}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    # Build a CLI so you can inspect any JSONL in your project, not just docs_short.jsonl.
    parser = argparse.ArgumentParser(description="Sanity-check a JSONL file (one JSON object per line).")

    # --path lets you choose which JSONL to inspect.
    parser.add_argument(
        "--path",
        type=str,
        default=str(DEFAULT_PATH),
        help=f"Path to JSONL file (default: {DEFAULT_PATH})",
    )

    # --n lets you choose how many records to print as preview.
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_PREVIEW_N,
        help=f"Number of records to preview (default: {DEFAULT_PREVIEW_N})",
    )

    # --sample_for_keys controls how many lines we sample for key/text stats.
    parser.add_argument(
        "--sample_for_keys",
        type=int,
        default=DEFAULT_SAMPLE_FOR_KEYS,
        help=f"How many records to sample for keys/text stats (default: {DEFAULT_SAMPLE_FOR_KEYS})",
    )

    # Parse args from the command line.
    args = parser.parse_args()

    # Convert the path string into a Path object.
    path_obj = Path(args.path)

    # Run the inspector.
    main(path=path_obj, preview_n=args.n, sample_for_keys=args.sample_for_keys)
