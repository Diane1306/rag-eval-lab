	•	what a regression test prevents in RAG: Regression testing in RAG (Retrieval-Augmented Generation) prevents performance degradation and ensures that system updates—such as prompt changes, new chunking strategies, or index updates—do not break existing functionality.

	•	what metric you care about most (refusal rate / citation rate / latency p95): refusal rate
	•	one “failure mode” you saw so far: return the same contents with two doc_ids
	•	Added evidence gate (keyword overlap + low-score fallback) to reduce hallucinations and enable refusals.
	•	Created eval_summary.json and regression_check.py to prevent quality regressions.
	•	Current baseline: refusal_answerable=0.10, refusal_unanswerable=0.40, citation_rate=1.00, p95_latency≈39ms, diversity≈5 (doc_id).
