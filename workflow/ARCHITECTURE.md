# Neuro Gym RAG — Architecture Reference

This document is the single source of truth for design decisions, data flow,
model choices, and configuration in the Neuro Gym multimodal RAG system.
Update it whenever a meaningful architectural change is made.

---

## System Overview

Neuro Gym is a **multimodal Retrieval-Augmented Generation (RAG)** coaching chatbot
for powerlifting. It synthesises structured training data (session logs, PDFs, weekly
tables) from 5,000 synthetic athletes and answers free-form natural-language questions
via a Gemini 2.5 Flash generation back-end.

The system is split into two phases:

- **Ingestion** (one-time, GPU-heavy, runs on Colab T4): chunk → embed → index
- **Query-time** (real-time, CPU-OK): augment → retrieve → generate → stream

---

## Architecture Diagram

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                       USER (Streamlit UI)                           │
 └────────────────────────┬────────────────────────────────────────────┘
                          │  SSE stream / JSON
                          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │              FastAPI  app/main.py                                   │
 │   POST /chat (JSON)   ·   POST /chat/stream (SSE)                   │
 │   DELETE /chat/{id}   ·   GET /health                               │
 └──────────┬──────────────────────────────────────────────────────────┘
            │  run_chain() / run_chain_stream()
            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  1. AUGMENTATION   app/augmentation.py                              │
 │     ├─ EntityRegister: pronoun resolution (athlete_\d{5} regex)     │
 │     ├─ Query analysis: Gemini 2.0 Flash (temp=0.0, max 200 tokens)  │
 │     │    → intent · rewritten_query · athlete_ids · training_levels │
 │     │    → sub_queries (comparison intent only)                     │
 │     └─ HyDE: Gemini 2.0 Flash (temp=0.4, max 400 tokens)           │
 │          Only for intent ∈ {trend, coaching}  — gated by rag_config │
 └──────────┬──────────────────────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  2. RETRIEVAL   pipeline/retrieval/retrieve.py                      │
 │                                                                     │
 │  Embed query → OpenRouter (nvidia/llama-nemotron-embed-vl-1b-v2)    │
 │  LRU-cached by (text, mode, api_key) — 512 slots                    │
 │                                                                     │
 │  Dense search (Qdrant, COSINE, 2048-dim):                           │
 │    gym_text   — coaching text chunks                                │
 │    gym_tables — weekly session records                              │
 │    gym_images — PDF pages  [skipped for factual/trend intents]      │
 │  Filter: athlete_id (MatchAny) + training_level (MatchAny)          │
 │                                                                     │
 │  Sparse search (BM25Okapi):                                         │
 │    Post-filter by athlete_id (20× oversample) or level (4× oversam) │
 │                                                                     │
 │  Sub-query dense search (multi_retrieve, ThreadPoolExecutor ×4):    │
 │    One embed+search per sub_query for comparison intents            │
 │                                                                     │
 │  Fusion: Reciprocal Rank Fusion (k=60) across all result lists      │
 │  Reranker: nvidia/llama-nemotron-rerank-vl-1b-v2 (opt, config G)   │
 │  Dedup: keep best-scoring result per athlete_id (top 5 athletes)    │
 │  Context assembly: token-budgeted text block (per rag_config)       │
 └──────────┬──────────────────────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  3. GENERATION   app/chain.py:generation()                          │
 │     Model: Gemini 2.5 Flash                                         │
 │     Prompt: system_instruction + history + context + intent hint    │
 │     PDF pages: embed relevant page (from payload page_number)       │
 │     Thinking: enabled for comparison/trend (budget=1024 tokens)     │
 │     Streaming: generate_content_stream() → SSE → Streamlit          │
 └──────────┬──────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  4. MEMORY   app/memory.py                                           │
 │     ConversationSummaryBufferMemory (k=8 turns, 2000-token budget)  │
 │     Overflow → Gemini 2.0 Flash JSON summary (preserves athlete IDs) │
 │     Persistence: Redis (primary) → in-memory dict (fallback)        │
 │     TTL: 1 hour idle; in-memory eviction + Redis TTL                │
 └─────────────────────────────────────────────────────────────────────┘

 VECTOR STORE (Qdrant, local port 6333)
 ┌──────────────┬───────────┬──────────────────────────────────────────┐
 │ Collection   │ Dim / Dst │ Payload indexes                          │
 ├──────────────┼───────────┼──────────────────────────────────────────┤
 │ gym_images   │ 2048 COS  │ athlete_id, training_level, page_number  │
 │ gym_text     │ 2048 COS  │ athlete_id, training_level, chunk_index  │
 │ gym_tables   │ 2048 COS  │ athlete_id, training_level, week, phase  │
 └──────────────┴───────────┴──────────────────────────────────────────┘

 SPARSE INDEX (BM25, in-memory)
   File: hf_pull/k2p/gym-rag-embeddings/{BM_index.pkl, BM_corpus.json}
   ~60 K documents — one per athlete-week + one summary record (week=0)
```

---

## Model Registry

| Component | Model | Where | Temp | Max tokens | Notes |
|-----------|-------|-------|------|------------|-------|
| Query embedding | nvidia/llama-nemotron-embed-vl-1b-v2 | OpenRouter API | — | — | 2048-dim, L2-normalised; LRU-cached |
| Document embedding | nvidia/llama-nemotron-embed-vl-1b-v2 | Local GPU (Colab T4) | — | — | float16, sdpa attn; ingestion only |
| Reranker (opt.) | nvidia/llama-nemotron-rerank-vl-1b-v2 | Local GPU | — | — | Config G only; loaded at startup |
| Query analysis | gemini-2.0-flash | Google AI API | 0.0 | 200 | Returns structured JSON |
| HyDE synthesis | gemini-2.0-flash | Google AI API | 0.4 | 400 | Runs only for trend/coaching |
| Memory summary | gemini-2.0-flash | Google AI API | 0.1 | 400 | JSON-structured summary |
| Generation | gemini-2.5-flash | Google AI API | 0.3 | 1024 | 1.0/2048 when thinking active |

All model IDs are defined in `config/model_settings.py`. Do not hardcode them elsewhere.

---

## Data Flow — Ingestion (one-time)

```
OpenPowerlifting CSV  +  Boostcamp 600K exercises
         │
         ▼  pipeline/dataset/dataset_main.py
5,000 synthetic athletes — stratified by DOTS score
  sessions.csv  (granular weekly rows)
  block_summary.csv
  pdfs/athlete_XXXXX.pdf (one per athlete, 12 pages)
         │
         ├─▶ pipeline/ingestion/chunking.py
         │     build_all_nl_strings() → NL text per athlete-week
         │     + coaching summary record (week=0) per athlete
         │
         ├─▶ pipeline/ingestion/bm_index.py
         │     BM25Okapi on tokenised text field
         │     → BM_index.pkl + BM_corpus.json  → HF + local
         │
         └─▶ Colab T4 (embedder.py + collection.py)
               embed_pdf_pages_batch()   → gym_images vectors
               embed_text_batch()        → gym_text vectors
               embed_pil_batch()         → gym_tables vectors
               load_from_numpy.py        → upsert to Qdrant (batch 256)
```

---

## Data Flow — Query Time

```
User query (+ optional image)
      │
      ▼  augment()
  pronoun_resolved  ──▶  _call_combined()  [Gemini 2.0, temp=0.0]
                          intent · rewritten_query · athlete_ids
                          training_levels · sub_queries
                    ──▶  _generate_hyde_document()  [if trend/coaching]
                          → hyde_vector (passage-mode embed)
      │
      ▼  retrieval()
  embed_query()  [OpenRouter, LRU-cached]
      │
      ├─▶ dense_search_all()  [Qdrant, COSINE]
      │    Filter: athlete_id (MatchAny) ∩ training_level (MatchAny)
      │    Uses hyde_vector (document space) if available, else query_vector
      │    Skips gym_images for factual/trend intent
      │
      ├─▶ sparse_search()  [BM25, in-memory]
      │    Oversamples candidates before post-filter
      │
      ├─▶ multi_retrieve()  [per sub_query, ThreadPoolExecutor×4]
      │    For comparison intent only
      │
      ▼  RRF(all_lists)  [k=60, rank-only fusion]
      ▼  rerank()         [optional, Nemotron reranker]
      ▼  deduplicate_athlete()  [keep best per athlete_id, top 5]
      ▼  assemble_context()     [token-budgeted, char/3 ratio]
      │
      ▼  generation()  /  run_chain_stream()
  _build_content_parts()
    history + retrieved context + intent hint + PDF pages + user image
      │
      ▼  gemini.models.generate_content_stream()
  Streamed SSE tokens → Streamlit st.write_stream()
  Final event: metadata dict  (__done__: true)
```

---

## Retrieval Configurations

| Key | Collections | BM25 | Reranker | Use case |
|-----|-------------|------|----------|---------|
| A — images only | gym_images | ✗ | ✗ | Visual ablation |
| B — text only | gym_text | ✗ | ✗ | Coaching text ablation |
| C — tables only | gym_tables | ✗ | ✗ | Session data ablation |
| D — all dense | all | ✗ | ✗ | Dense-only baseline |
| E — tables + BM25 | gym_tables | ✓ | ✗ | Keyword + structured |
| **F — all + BM25** | **all** | **✓** | **✗** | **Production default** |
| G — hybrid + rerank | all | ✓ | ✓ | Best quality, higher latency |
| H — BM25 only | — | ✓ | ✗ | Keyword-only baseline |

---

## Intent Routing

Query intent is classified by Gemini 2.0 Flash at augmentation time:

| Intent | HyDE | Image collection | Context budget | Thinking |
|--------|------|-----------------|---------------|---------|
| factual | ✗ | ✗ | 8 192 tokens | ✗ |
| trend | ✓ | ✗ | 16 384 tokens | ✓ |
| comparison | ✗ | ✓ | 32 768 tokens | ✓ |
| coaching | ✓ | ✓ | 16 384 tokens | ✗ |
| visual | ✗ | ✓ | 8 192 tokens | ✗ |

All values are defined in `config/rag_config.py`. Change them there.

---

## Key Design Decisions

### Why three Qdrant collections instead of one?
Different modalities require different ingestion pipelines and have different
payload schemas. Keeping them separate allows per-collection top-k tuning and
lets configs A/B/C serve as clean ablation baselines without touching production.

### Why BM25 alongside dense vectors?
Dense vectors excel at semantic similarity but fail on exact token matches
(athlete IDs like `athlete_00042`, week numbers, exact lift names). BM25 is
deterministic and handles exact-match recall. RRF fusion combines both signals
without tuning a blending weight.

### Why HyDE only for trend/coaching?
HyDE generates a synthetic training record passage and embeds it in document
space. For *factual* queries ("what was X's squat in week 8?") the answer is
exact-match — HyDE adds noise. For *trend* and *coaching* queries the semantic
gap between query phrasing and document style is large, so HyDE helps. Saves
2 API calls per factual/comparison/visual query.

### Why athlete_id as a Qdrant hard-filter?
When augmentation extracts explicit athlete IDs from a query, running semantic
search across all 5,000 athletes returns irrelevant results that crowd out the
target athlete. A `MatchAny` filter on `athlete_id` reduces the search space
from 5,000 athletes to 1–4, dramatically improving precision for factual queries.

### Why RRF instead of learned fusion weights?
Learned weights require labelled query-document pairs to train. RRF is
parameter-free, robust to score scale differences between BM25 and cosine
similarity, and empirically competitive with learned fusion for domain-specific
corpora. The `k=60` constant is the standard default from the RRF paper.

### Why gemini-2.5-flash with thinking for comparison/trend?
These intents require multi-step reasoning: finding relevant athletes, aligning
week numbers, computing progressions. Thinking tokens let the model plan before
generating the answer. Factual and visual queries don't benefit enough to justify
the latency and the mandatory temperature=1.0 constraint.

### Why Redis for session persistence?
In-process `dict` sessions are lost on every server restart. Redis provides:
- Persistence across restarts (sessions survive deploys)
- Automatic TTL expiry (no memory leak)
- Foundation for horizontal scaling (multiple API workers share state)
- Transparent fallback to in-memory if Redis isn't running

### Why SSE for streaming instead of WebSocket?
SSE is uni-directional (server → client), stateless (standard HTTP), and natively
supported by `requests` (client-side `iter_lines`) and FastAPI `StreamingResponse`.
WebSocket would require a persistent connection and adds complexity for a flow
that's already sequential (one query → one streamed response).

### Why embed queries via OpenRouter rather than local inference?
Ingestion runs on a Colab T4 (GPU available). Query-time runs wherever the API
server runs. Keeping the embedding model local at query time would require a GPU
always attached to the API server. OpenRouter's free tier is sufficient for
development. **Known risk**: model version divergence between local ingestion and
API embeddings could silently degrade retrieval quality. Mitigation: pin the
`nvidia/llama-nemotron-embed-vl-1b-v2` model ID in both paths and run the
integration test in `eval/retrieval_eval.py` after any embedding API change.

---

## Configuration Guide

| What to change | File | Key |
|---------------|------|-----|
| Context token budgets | `config/rag_config.py` | `INTENT_CONTEXT_TOKENS` |
| HyDE intents | `config/rag_config.py` | `HYDE_INTENTS` |
| Text-only intents (skip images) | `config/rag_config.py` | `TEXT_ONLY_INTENTS` |
| BM25 oversampling | `config/rag_config.py` | `BM25_OVERSAMPLE_*` |
| Thinking budget | `config/rag_config.py` | `THINKING_BUDGET`, `THINKING_INTENTS` |
| Generation temperature | `config/rag_config.py` | `GENERATION_TEMPERATURE` |
| Session TTL | `config/rag_config.py` | `SESSION_TTL_SECONDS` |
| Redis URL | env var `REDIS_URL` or `config/rag_config.py` | `REDIS_URL_DEFAULT` |
| Gemini model IDs | `config/model_settings.py` | `GEMINI_*_MODEL` |
| Embedding model ID | `config/model_settings.py` | `EMBEDDING_MODEL_ID` |
| Qdrant host/port | env vars `QDRANT_HOST`, `QDRANT_PORT` | — |

---

## Known Limitations & Accepted Tradeoffs

| Limitation | Impact | Mitigation / Accepted because |
|-----------|--------|-------------------------------|
| Query embedding via OpenRouter free tier | Version drift risk vs ingestion model | Document risk; add integration test |
| No GPU at query time | Reranker (Config G) is slow on CPU | CPU-OK for dev; use Config F in prod |
| BM25 loaded entirely in RAM | ~200 MB for 60 K documents | Acceptable for a single-process server |
| Sessions in-process + Redis | Redis restart clears Redis TTL | Restart is rare; TTL is 1 h anyway |
| PDF rendered per query | +200–500 ms for PDF-heavy responses | Cache as pre-rendered PNGs in future |
| Char/3 token estimation | May over-count; actual limit is model-side | Gemini will truncate silently; true fix is `count_tokens` API |
| No horizontal scaling yet | One FastAPI process | Redis sessions enable this; stateless workers next step |

---

## Development Workflow

```bash
# 1 — Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# 2 — (optional) Start Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 3 — Install deps
pip install -r requirements.txt

# 4 — Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5 — Start UI (separate terminal)
streamlit run ui/streamlit_app.py

# 6 — Run retrieval eval (smoke, 5 questions)
python eval/retrieval_eval.py --configs F --limit 5 --no-chart
```

### Environment variables (`.env` at repo root)

```
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_URL=redis://localhost:6379
HF_TOKEN=...
```

### Ingestion (Colab — only needed when re-generating dataset)

```
notebooks/colab_ingestion.ipynb
  Cell 1–3:  Clone repo, pull HF dataset
  Cell 4:    Create Qdrant collections
  Cell 5–9:  Embed images / text / tables
  Cell 10:   Build BM25 index
  Cell 11:   Push embeddings to HF
```
