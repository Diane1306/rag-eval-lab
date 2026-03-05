# src/rag_answer_local.py
#
# Purpose:
#   Produce a deterministic, "grounded" answer from retrieved chunks WITHOUT an LLM.
#   This is a robust baseline that:
#     - never hallucinates beyond retrieved text
#     - always produces citations to retrieved chunk IDs
#     - supports a later swap to OpenAI generation (same function signature)
#
# Inputs:
#   - query: user question (string)
#   - retrieved: list of retrieval dicts (each with chunk_id/doc_id/title/text_preview/text)
#
# Outputs:
#   - dict with:
#       answer: str
#       citations: list of {doc_id, chunk_id}
#       refusal: bool
#       refusal_reason: str

import re  # Used for simple sentence splitting and cleanup.
from typing import List, Dict, Any  # Type hints for clarity.


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a simple heuristic.

    Why:
      - We want a short extractive summary.
      - Deterministic, no external NLP dependency.

    Output:
      - list of sentence strings
    """
    # Normalize whitespace to make splitting cleaner.
    t = re.sub(r"\s+", " ", str(text)).strip()

    # Split on sentence-ending punctuation followed by a space.
    # This is simplistic but works well enough for a baseline.
    sents = re.split(r"(?<=[.!?])\s+", t)

    # Remove empty sentences.
    return [s.strip() for s in sents if s.strip()]


def _pick_support_sentences(query: str, text: str, max_sentences: int = 2) -> List[str]:
    """
    Pick a small number of sentences that likely support the query.

    Heuristic:
      - Prefer sentences that share words with the query.
      - Fall back to the first sentences if no overlap.

    Output:
      - list of selected sentences (length <= max_sentences)
    """
    # Basic tokenization of query words (lowercase alphanumeric tokens).
    q_words = set(re.findall(r"[a-z0-9]+", query.lower()))

    sents = _split_sentences(text)

    if not sents:
        return []

    # Score each sentence by how many query words it contains.
    scored = []
    for s in sents:
        s_words = set(re.findall(r"[a-z0-9]+", s.lower()))
        score = len(q_words.intersection(s_words))
        scored.append((score, s))

    # Sort by score descending, then keep top sentences.
    scored.sort(key=lambda x: x[0], reverse=True)

    # If the best score is 0, fallback to the first sentences (extractive baseline).
    if scored[0][0] == 0:
        return sents[:max_sentences]

    return [s for _, s in scored[:max_sentences]]


def answer_from_retrieval(query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministically generate an answer + citations from retrieved chunks.

    Logic:
      - If no retrieved results, refuse.
      - Otherwise, extract 1–2 support sentences from top chunks.
      - Return a short answer with citations.

    Returns:
      - {answer, citations, refusal, refusal_reason}
    """

    # If nothing retrieved, we cannot ground an answer.
    if not retrieved:
        return {
            "answer": "I don’t know based on the provided sources.",
            "citations": [],
            "refusal": True,
            "refusal_reason": "No retrieved chunks were available to support an answer.",
        }

    # Use up to the top 3 retrieved chunks to build a short answer.
    top = retrieved[:3]

    # Collect support sentences and citations.
    answer_parts = []
    citations = []

    for r in top:
        doc_id = str(r.get("doc_id", ""))
        chunk_id = str(r.get("chunk_id", ""))

        # Prefer full text if available, else fall back to preview.
        text = r.get("text", r.get("text_preview", ""))

        # Pick a couple supporting sentences.
        support = _pick_support_sentences(query=query, text=text, max_sentences=2)

        # If we got useful text, add it to the answer.
        if support:
            answer_parts.append(" ".join(support))

        # Add a citation entry (even if support was weak; it’s still the source we referenced).
        citations.append({"doc_id": doc_id, "chunk_id": chunk_id})

    # If we somehow extracted nothing, refuse conservatively.
    if not answer_parts:
        return {
            "answer": "I don’t know based on the provided sources.",
            "citations": citations,
            "refusal": True,
            "refusal_reason": "Retrieved chunks did not contain extractable supporting sentences.",
        }

    # Join extracted sentences into one concise answer paragraph.
    answer = " ".join(answer_parts).strip()

    return {
        "answer": answer,
        "citations": citations,
        "refusal": False,
        "refusal_reason": "",
    }
