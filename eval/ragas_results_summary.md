# RAGAS End-to-End Evaluation

LLM-as-judge evaluation of the full A→R→G chain. Each question is answered by the production pipeline (augmentation → retrieval → Gemini generation), then scored by Gemini 2.0 Flash as judge using RAGAS metrics.

| Config | Faithfulness | Answer Relevancy | Context Precision | Retrieval ms | Generation ms |
|---|---:|---:|---:|---:|---:|
| F — all + BM25 | 0.897 | 0.655 | 0.692 | 5856 | 3675 |

## Reading the metrics

- **Faithfulness** [0–1] — fraction of claims in the answer that are grounded in the retrieved contexts. Penalises hallucination.
- **Answer Relevancy** [0–1] — semantic similarity between the answer and the question, controlled for off-topic verbosity.
- **Context Precision** [0–1] — RAGAS reference-free variant: how much of the retrieved context is actually useful for answering the question.
- **Context Recall** [0–1, optional] — fraction of facts in the ground-truth answer that are present in the retrieved contexts. Only computed when an `eval/golden_answers.json` is supplied.
- **Retrieval / Generation ms** — wall-clock latency per stage; useful for understanding the latency-vs-quality trade-off across configs (G in particular).
