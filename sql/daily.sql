-- Count docs
SELECT COUNT(*) AS n_docs
FROM `rag_eval_lab.docs_short`;

-- Average length
SELECT
  AVG(LENGTH(text)) AS avg_chars,
  MIN(LENGTH(text)) AS min_chars,
  MAX(LENGTH(text)) AS max_chars
FROM `rag_eval_lab.docs_short`;

-- Sources distribution
SELECT source, COUNT(*) AS n
FROM `rag_eval_lab.docs_short`
GROUP BY source
ORDER BY n DESC;
