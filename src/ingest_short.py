import json
from pathlib import Path
from datasets import load_dataset

OUT = Path("data/processed/docs_short.jsonl")

def pick_first_answer(answers) -> str:
    # answers is a dict with 'text' list for SQuAD
    if not answers:
        return ""
    texts = answers.get("text", [])
    return (texts[0] if texts else "").strip()

def main(n: int = 5000) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("squad_v2", split="train")
    n = min(n, len(ds))

    total_chars = 0
    min_len = 10**9
    max_len = 0

    with OUT.open("w", encoding="utf-8") as f:
        for i in range(n):
            row = ds[i]
            title = (row.get("title") or "").strip()
            question = (row.get("question") or "").strip()
            context = (row.get("context") or "").strip()
            answer = pick_first_answer(row.get("answers") or {})

            # Some SQuAD v2 questions are unanswerable; keep them but mark empty answer
            full = f"Question: {question}\nAnswer: {answer}\n\nContext:\n{context}".strip()

            doc = {
                "doc_id": f"squad_v2_train_{i}",
                "title": title[:200],
                "source": "squad_v2",
                "text": full,
                "url": "",
            }

            s = len(full)
            total_chars += s
            min_len = min(min_len, s)
            max_len = max(max_len, s)

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    avg_len = total_chars / max(n, 1)
    print(f"Wrote {n} docs -> {OUT}")
    print(f"Chars: avg={avg_len:.1f}, min={min_len}, max={max_len}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    args = p.parse_args()
    main(args.n)
