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

# -----------------------------
# Evidence-gating configuration
# -----------------------------

# If the embedding score is extremely low AND keyword overlap is weak, we refuse.
# We set this low because you observed valid answers around ~0.49.
LOW_SCORE_THRESHOLD = 0.40

# Require at least this many query keywords to appear in retrieved text to consider it supported.
MIN_KEYWORD_HITS = 2

# A small stopword list (keep it minimal; we only want to remove very common words).
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "what", "when", "where", "who", "why", "how", "which",
}

def _extract_keywords(query: str) -> list[str]:
    """
    Extract simple keywords from the query.

    Rule:
      - lowercase alphanumeric tokens
      - length >= 5 (filters out tiny words)
      - not in STOPWORDS

    Output:
      - list of keyword strings
    """
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    keywords = [t for t in tokens if len(t) >= 5 and t not in STOPWORDS]
    return keywords


def _keyword_hit_count(keywords: list[str], retrieved_text: str) -> int:
    """
    Count how many query keywords appear (exact substring match) in retrieved_text.

    Note:
      - This is intentionally simple and deterministic.
      - We use exact substring presence, not stemming.

    Output:
      - integer count of matched keywords (unique keyword hits)
    """
    haystack = retrieved_text.lower()
    hits = 0
    for kw in set(keywords):
        if kw in haystack:
            hits += 1
    return hits

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


def answer_from_retrieval(query: str, retrieved: List[Dict[str, Any]], top_score: float | None = None) -> Dict[str, Any]:
    """
    Deterministically generate an answer + citations from retrieved chunks,
    with an "evidence gate" to refuse when support is weak.

    Inputs:
      - query: user question
      - retrieved: retrieval results list (each dict has chunk_id/doc_id/title/text_preview/text/score)
      - top_score: similarity score of the best retrieved item (optional but recommended)

    Evidence gate (refusal):
      - If retrieved is empty -> refuse
      - Else compute keyword overlap between query and concatenated retrieved text
      - Refuse only when:
          keyword_hits < MIN_KEYWORD_HITS  AND  (top_score is not None and top_score < LOW_SCORE_THRESHOLD)

    Returns:
      - {answer, citations, refusal, refusal_reason}
    """

    # 1) If nothing retrieved, we cannot ground an answer.
    if not retrieved:
        return {
            "answer": "I don’t know based on the provided sources.",
            "citations": [],
            "refusal": True,
            "refusal_reason": "No retrieved chunks were available to support an answer.",
        }

    # 2) Build a combined text blob from the top-k retrieved results for overlap checks.
    # Prefer full text if present; otherwise use text_preview.
    combined = []
    for r in retrieved:
        combined.append(str(r.get("text", r.get("text_preview", ""))))
    combined_text = "\n".join(combined)

    # 3) Extract query keywords and compute how many appear in retrieved text.
    keywords = _extract_keywords(query)
    keyword_hits = _keyword_hit_count(keywords, combined_text)

    # 4) Evidence gate:
    # Refuse only if overlap is weak AND score is extremely low.
    if top_score is not None:
        if keyword_hits < MIN_KEYWORD_HITS and top_score < LOW_SCORE_THRESHOLD:
            return {
                "answer": "I don’t know based on the provided sources.",
                "citations": [],
                "refusal": True,
                "refusal_reason": (
                    f"Weak evidence: keyword_hits={keyword_hits} (<{MIN_KEYWORD_HITS}) "
                    f"and top_score={top_score:.3f} (<{LOW_SCORE_THRESHOLD})."
                ),
            }

    # 5) Otherwise, proceed with the original deterministic extractive answer logic.
    top = retrieved[:3]  # Use top 3 results to form a concise answer.

    answer_parts = []
    citations = []

    for r in top:
        doc_id = str(r.get("doc_id", ""))
        chunk_id = str(r.get("chunk_id", ""))

        text = r.get("text", r.get("text_preview", ""))

        support = _pick_support_sentences(query=query, text=text, max_sentences=2)
        if support:
            answer_parts.append(" ".join(support))

        citations.append({"doc_id": doc_id, "chunk_id": chunk_id})

    if not answer_parts:
        return {
            "answer": "I don’t know based on the provided sources.",
            "citations": citations,
            "refusal": True,
            "refusal_reason": "Retrieved chunks did not contain extractable supporting sentences.",
        }

    answer = " ".join(answer_parts).strip()

    return {
        "answer": answer,
        "citations": citations,
        "refusal": False,
        "refusal_reason": "",
    }
