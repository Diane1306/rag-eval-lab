Grounded answering (core concept)
	•	A “grounded” RAG answer should be generated only from retrieved context; if the needed info is not present in the retrieved chunks, the assistant should abstain (say “I don’t know based on the provided sources”) rather than hallucinate.
	•	The purpose of grounding is to make outputs auditable: a reviewer can trace each claim back to a retrieved chunk.

Prompt / instruction principles (even if you don’t use an LLM today)
	•	Enforce a strict rule: “Use only the provided context.”
	•	Require citations: every key statement should cite at least one retrieved source.
	•	Add a refusal condition: if no chunk supports the answer → return “insufficient evidence” and optionally suggest what info is missing.

Citation format (simple + practical)
	•	Use a compact citation tag like: [doc_id:chunk_id]
	•	Track citations as structured data too (list of {doc_id, chunk_id}) so you can count them and log them.

What “good” looks like in the UI
	•	Left chat: assistant message includes:
	•	Answer
	•	Citations line (2–5 citations max)
	•	Right debug panel: shows the retrieved chunks that match those citations so you can visually verify grounding.

What to log (BigQuery) for engineering rigor
	•	Log whether the answer is grounded:
	•	has_citations (bool)
	•	n_citations (int)
	•	Log answer verbosity:
	•	answer_len (int)
	•	Keep logs small: store IDs, not full contexts.

Why p50/p95 latency matters (for your RAG system)
	•	p50 latency = typical user experience
	•	p95 latency = tail latency; highlights slow cases (index load, long queries, expensive reruns)

