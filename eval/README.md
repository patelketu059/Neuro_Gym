# Evaluation Suite

End-to-end evaluation of the Neuro Gym RAG pipeline across all 8 retrieval configurations.

## Files

| File | Purpose |
|------|---------|
| `question_bank.py` | **Single source of truth** — 50 golden questions, all intents/levels |
| `sample_athletes.py` | Validate athlete IDs exist in Qdrant; sample replacements if needed |
| `generate_references.py` | Pull raw Qdrant data → Gemini → `golden_answers.json` |
| `run_server_eval.py` | **Main eval runner** — POST to live server, collect RAGAS + retrieval metrics |
| `visualize.py` | Generate 6 portfolio-quality plots from eval results |
| `retrieval_eval.py` | Library-mode retrieval-only eval (Hit@K, MRR, NDCG) |
| `ragas_eval.py` | Library-mode full-pipeline RAGAS eval |

## Full Workflow

### Step 0 — Validate athletes (one-time)
```bash
python eval/sample_athletes.py
```
Confirms that all 7 athlete IDs in `question_bank.py` exist in your Qdrant instance.
If any are missing, the script samples replacements and you update `question_bank.py`.

### Step 1 — Generate reference answers (one-time)
```bash
python eval/generate_references.py
```
Queries Qdrant for each athlete's raw session data, then uses Gemini 2.0 Flash to
synthesise a 2–4 sentence reference answer per question. Writes `eval/golden_answers.json`.

**Review the output against your PDFs before proceeding.**

### Step 2 — Start the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 3 — Run the full evaluation
```bash
# All 8 configs × 50 questions with RAGAS + retrieval metrics
python eval/run_server_eval.py --references eval/golden_answers.json

# Quick smoke test (5 questions, 2 configs)
python eval/run_server_eval.py \
    --configs "F — all + BM25" "G — hybrid + rerank" \
    --limit 5 \
    --no-ragas
```

Outputs land in `eval/server_results/`:
- `server_results_raw.json` — every (config, question) row
- `server_results_raw.csv` — same, spreadsheet-friendly
- `server_results_summary.json` — aggregated metrics per config
- `server_results_summary.md` — RAGAS markdown table
- `server_results_retrieval.md` — retrieval markdown table

### Step 4 — Generate plots
```bash
python eval/visualize.py
```

Writes 6 PNGs to `eval/plots/`:

| Plot | Description |
|------|-------------|
| `radar_chart.png` | RAGAS 4-metric profile overlay for all 8 configs |
| `faithfulness_bar.png` | Faithfulness per config (primary anti-hallucination metric) |
| `intent_heatmap.png` | Faithfulness heatmap: config × intent |
| `latency_scatter.png` | Retrieval latency vs. faithfulness, bubble = Hit@5 |
| `intent_breakdown.png` | Per-intent faithfulness + answer relevancy, all configs |
| `retrieval_metrics.png` | Hit@5, MRR, NDCG@5, Recall@5 across configs |

## Metrics Reference

### RAGAS (LLM-as-judge, Gemini 2.0 Flash)

| Metric | Range | What it measures |
|--------|-------|-----------------|
| Faithfulness | 0–1 ↑ | Fraction of answer claims supported by retrieved context (hallucination detector) |
| Answer Relevancy | 0–1 ↑ | Semantic match between answer and question |
| Context Precision | 0–1 ↑ | Fraction of retrieved chunks that are actually useful |
| Context Recall | 0–1 ↑ | Fraction of ground-truth evidence present in retrieved context (requires `golden_answers.json`) |

### Retrieval (athlete-level)

| Metric | Range | What it measures |
|--------|-------|-----------------|
| Hit@K | 0–1 ↑ | Fraction of questions where a correct athlete appears in top K |
| MRR | 0–1 ↑ | Mean reciprocal rank of first correct athlete |
| NDCG@5 | 0–1 ↑ | Position-weighted retrieval quality |
| Recall@5 | 0–1 ↑ | Fraction of ground-truth athletes recovered in top 5 |

## Question Bank Distribution

| Intent | Count | Notes |
|--------|-------|-------|
| Factual | 15 | Single data-point look-ups |
| Trend | 10 | Week-by-week progression |
| Comparison | 10 | Cross-athlete + cross-level |
| Coaching | 10 | Open-ended advice |
| Visual | 5 | Chart/PDF description |
| **Total** | **50** | |

## Config Reference

| Config | Collections | BM25 | Reranker |
|--------|-------------|------|----------|
| A — images only | gym_images | No | No |
| B — text only | gym_text | No | No |
| C — tables only | gym_tables | No | No |
| D — all dense | all 3 | No | No |
| E — tables + BM25 | gym_tables | Yes | No |
| **F — all + BM25** | **all 3** | **Yes** | **No** |
| G — hybrid + rerank | all 3 | Yes | Yes |
| H — BM25 only | — | Yes | No |

Config **F** is the production default.
