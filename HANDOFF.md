# Neuro Gym — Session Handoff

> This document is written for the next Claude session (or developer) taking over. Read it top-to-bottom before touching any code. It captures the current state of the project, every decision made in the most recent session, and exactly where to pick up.

---

## Project at a Glance

**What it is:** A multimodal RAG chatbot for powerlifting coaching. Users ask questions about synthetic athlete training data; the system retrieves relevant records from Qdrant + BM25 and generates grounded answers with Gemini.

**Portfolio purpose:** Mid-level AI/ML Engineer showcase — demonstrates Qdrant, BM25 hybrid retrieval, multimodal embeddings (NVIDIA Nemotron), FastAPI, streaming SSE, CI/CD, and RAGAS evaluation.

**Live demo:**
- Frontend: `https://k2p-neuro-gym-ui.hf.space` (Streamlit)
- Backend API: `https://k2p-neuro-gym-rag.hf.space` (FastAPI Docker)

**GitHub:** `https://github.com/patelketu059/Neuro_Gym`

---

## Repository Layout

```
Neuro_Gym/
├── app/                    ← FastAPI backend
│   ├── main.py             ← startup: BM25 download, Qdrant connect, Gemini init
│   ├── chain.py            ← RAG loop: augment → retrieve → generate (streaming + blocking)
│   ├── augmentation.py     ← intent classification + HyDE query expansion
│   ├── memory.py           ← ConversationSummaryBufferMemory (Redis / in-memory)
│   ├── session_store.py    ← session eviction logic
│   └── routes/
│       ├── chat.py         ← /chat (blocking) and /chat/stream (SSE) endpoints
│       ├── health_status.py← /health endpoint
│       └── pdf.py          ← /pdf/page and /pdf/pages endpoints
│
├── pipeline/
│   ├── dataset/            ← Phase 1: synthetic data generation
│   │   ├── dataset_main.py ← entry point (parallel with ProcessPoolExecutor)
│   │   ├── athlete_generator.py
│   │   ├── opl_loader.py   ← OpenPowerlifting CSV
│   │   ├── gym_600k_loader.py
│   │   ├── periodization.py
│   │   ├── generate_pdfs.py
│   │   └── export.py
│   ├── ingestion/          ← Phase 2: embed + index
│   │   ├── chunking.py     ← 3 chunk types: coaching text, table rows, PDF images
│   │   ├── embedder.py     ← NVIDIA Nemotron via OpenRouter API
│   │   ├── collection.py   ← Qdrant collection setup
│   │   ├── ingest.py       ← orchestrates embed → upsert
│   │   └── bm_index.py     ← builds + loads BM25 index
│   └── retrieval/          ← Phase 4: query-time search
│       ├── retrieve.py     ← retrieval configs (A–H), orchestrates hybrid search
│       ├── dense_search.py ← Qdrant ANN
│       ├── sparse_search.py← BM25 keyword
│       ├── fusion_search.py← RRF merge
│       └── reranker.py     ← NVIDIA reranker (config G only)
│
├── config/
│   ├── model_settings.py   ← ALL model IDs (single source of truth)
│   ├── rag_config.py       ← TOP_K, context budgets, temperatures
│   └── settings.py         ← file paths, env vars
│
├── ui/
│   └── streamlit_app.py    ← Streamlit frontend
│
├── eval/
│   ├── ragas_eval.py       ← RAGAS evaluation runner
│   ├── ragas_results_summary.json  ← latest scores
│   └── ragas_results_summary.md
│
├── scripts/
│   ├── push_to_hf.py       ← uploads BM25 index to HF Hub dataset repo
│   └── e2e_test.py         ← end-to-end smoke tests against live API
│
├── .github/workflows/
│   └── deploy.yml          ← CI/CD: auto-deploys to two HF Spaces on push to main
│
├── PIPELINE.md             ← full pipeline walkthrough (written this session)
├── ARCHITECTURE.md         ← architecture reference
├── MASTER_PLAN.md          ← original design decisions + rationale
└── HANDOFF.md              ← this file
```

---

## What Was Done in This Session

### Bug 1 — UI showing "Retrieval: 0 ms · Generation: 0 ms" with blank responses

**Root cause:** The SSE stream event ordering in `_stream_chat_api()` checked `__done__` **before** `__error__`. Backend errors arrive as `{"__error__": "...", "__done__": True}` — both keys present. The `__done__` branch fired first, silently capturing the error dict into `metadata_sink` and yielding nothing, so the user saw a blank response with 0 ms timing.

**Fix (`ui/streamlit_app.py`):** Reordered the if-checks so `__error__` is evaluated before `__done__`.

```python
# CORRECT order:
if parsed.get("__ping__"):   continue
if parsed.get("__error__"):  yield f"\n\n⚠️ Backend error: {parsed['__error__']}"; return
if parsed.get("__done__"):   metadata_sink.append(parsed); return
if isinstance(parsed, str):  yield parsed
```

### Bug 2 — SSE connection dropped during retrieval

**Root cause:** Retrieval + augmentation takes 5–10 s. HuggingFace's Nginx reverse proxy closes "idle" SSE connections before the first byte arrives.

**Fix (`app/chain.py`):** Added a heartbeat as the very first yield in `run_chain_stream()`:
```python
yield _json.dumps({"__ping__": True})   # keep HF proxy alive
# ... augment + retrieve runs here ...
```

### Bug 3 — 500 Server Error from backend

**Root cause:** Gemini prepaid credits were exhausted (drained by ~50-question RAGAS eval × 5–8 Gemini calls each).

**Fix (`config/model_settings.py`):** Switched all models to free-tier:
```python
GEMINI_GENERATION_MODEL = "gemini-2.0-flash"       # 15 RPM / 1,500 RPD free
GEMINI_FALLBACK_MODEL   = "gemini-2.0-flash-lite"  # 30 RPM / 1,500 RPD free
GEMINI_AUX_MODEL        = "gemini-2.0-flash-lite"
GEMINI_JUDGE_MODEL      = "gemini-2.0-flash-lite"
```

Added automatic fallback in `app/chain.py`:
- `_is_rate_limit(exc)` — detects 429 / RESOURCE_EXHAUSTED / quota errors
- `_model_order()` — returns `[primary, fallback]` deduplicated
- Both `generation()` and `run_chain_stream()` loop through models, retrying on rate limits

### Bug 4 — ThinkingConfig crash on non-2.5 models

**Root cause:** `gemini-2.0-flash` does not support `ThinkingConfig`; passing it causes an API error.

**Fix (`app/chain.py`):** Gated thinking on a whitelist:
```python
_THINKING_CAPABLE = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"}

def _build_gen_config(intent: str, model: str = GEMINI_GENERATION_MODEL):
    supports_thinking = model in _THINKING_CAPABLE
    use_thinking = supports_thinking and intent in THINKING_INTENTS
    # ThinkingConfig only added when use_thinking is True
```

### Feature — Non-streaming fallback

Added `_call_chat_blocking()` in `ui/streamlit_app.py` which calls the `/chat` endpoint (not `/chat/stream`). If streaming returns nothing and captures no metadata, the UI automatically retries blocking. Added a sidebar checkbox to disable streaming entirely.

### Documentation

- `PIPELINE.md` — comprehensive end-to-end pipeline walkthrough with ASCII diagram, written for a new contributor. Committed as `a721193`.

---

## Current State of the Codebase

### What works ✅
- Full RAG pipeline: augment → retrieve → generate
- Streaming SSE with heartbeat (survives HF proxy timeouts)
- Automatic fallback: `gemini-2.0-flash` → `gemini-2.0-flash-lite` on rate limits
- Non-streaming `/chat` endpoint as backup
- SSE error events surfaced properly in UI
- Memory: ConversationSummaryBufferMemory with Redis/in-memory fallback
- CI/CD: GitHub Actions deploys both HF Spaces on push to `main`
- RAGAS evaluation runner (`eval/ragas_eval.py`)
- PDF viewer in Streamlit (right-hand panel, paginated)
- Image upload → multimodal query

### Known issues / watch-outs ⚠️
- **Free-tier rate limits are real.** `gemini-2.0-flash` gives 1,500 requests/day. Running RAGAS (50 questions × ~5 Gemini calls = 250+ calls) can eat through 15–20% of the daily quota in one run. If you switch back to a paid model, update `GEMINI_GENERATION_MODEL` in `config/model_settings.py` and add `"gemini-2.5-flash"` (or whichever) to `_THINKING_CAPABLE` in `app/chain.py`.
- **Streaming fallback yields partial tokens.** If the primary model starts streaming and then gets rate-limited mid-stream, `full_answer` is reset to `""` and the fallback starts fresh — but any tokens already `yield`ed to the SSE client cannot be un-sent. The UI will show a partial response followed by the complete retry. This is acceptable but worth noting.
- **BM25 oversample factor.** When filtering by `athlete_id`, BM25 fetches 20× more candidates than needed because one athlete is ~0.02% of 65K docs. If you significantly change the corpus size, revisit `BM25_OVERSAMPLE_ATHLETE` in `config/rag_config.py`.
- **`context_recall` is always null in RAGAS.** This metric requires ground-truth reference answers. The eval set uses no references (`refs=off`), so context_recall stays null. That's intentional.
- **Redis unavailable on HF Spaces.** Memory falls back to in-memory dict automatically. Sessions are stored per-process — if HF restarts the container, conversation history is lost. This is acceptable for a demo.

---

## Environment Variables

### Required everywhere (local + HF Spaces)
```
GEMINI_API_KEY        — Google AI Studio key (free tier sufficient)
QDRANT_URL            — Qdrant Cloud cluster URL (e.g. https://xxx.qdrant.io:6333)
QDRANT_API_KEY        — Qdrant Cloud API key
OPENROUTER_API_KEY    — For NVIDIA Nemotron embedding model
HF_TOKEN              — HuggingFace token (read access to k2p/gym-rag-embeddings dataset)
```

### Optional
```
REDIS_URL             — Falls back to in-memory if absent
ARTIFACT_DIR          — Where BM25 index is cached on disk (default: /tmp/gym-rag-artifacts)
EMBED_REPO            — HF dataset repo for BM25 (default: k2p/gym-rag-embeddings)
```

### GitHub Secrets (CI/CD)
```
HF_TOKEN              — Must have write access to both HF Spaces
```

### Local `.env` file (project root, gitignored)
The app loads `.env` automatically via `python-dotenv` if it exists at the project root. Create one:
```
GEMINI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

---

## Running Locally

```bash
# 1. Activate venv
.\.neuro_gym\Scripts\activate          # Windows
source .neuro_gym/bin/activate         # Mac/Linux

# 2. Start backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Start frontend (separate terminal, backend must be running first)
streamlit run ui/streamlit_app.py

# 4. Smoke test against live HF backend
python scripts/e2e_test.py

# 5. Run RAGAS eval (25 questions, default config)
python eval/ragas_eval.py --n_questions 25 --configs "F — all + BM25"
```

---

## Recent Commits

| Hash | Message |
|------|---------|
| `a721193` | docs: add comprehensive pipeline walkthrough (PIPELINE.md) |
| `c9abfb0` | feat: switch to free-tier Gemini models with rate-limit fallback |
| `8bfedb0` | fix(ui): surface backend errors and add non-streaming fallback for HuggingFace |
| `344da57` | fix: eliminate Streamlit UI jitter on HuggingFace |
| `5843f1f` | fix: working RAGAS eval + regenerated portfolio plots |

---

## Current RAGAS Scores (Config F — all + BM25, 50 questions)

| Metric | Score | Notes |
|--------|-------|-------|
| Faithfulness | **0.897** | Very good — minimal hallucination |
| Answer Relevancy | **0.655** | Weakest overall; comparison + visual drag it down |
| Context Precision | **0.692** | Good — retrieved chunks are mostly relevant |
| Context Recall | null | Requires ground-truth refs; intentionally disabled |

**By intent:**

| Intent | n | Faithfulness | Answer Relevancy |
|--------|---|:---:|:---:|
| Factual | 15 | 0.928 | 0.778 |
| Trend | 10 | 0.975 | 0.816 |
| Comparison | 10 | 0.886 | 0.445 |
| Coaching | 10 | 0.759 | 0.589 |
| Visual | 5 | 0.963 | 0.517 |

Comparison and coaching answer relevancy are the weakest points. The retrieval step pulls good chunks (high faithfulness) but generation doesn't always synthesise them into a directly on-topic answer. Likely caused by the comparison system prompt instructions not being strong enough.

---

## Key Config Values (Quick Reference)

**`config/model_settings.py`**
```python
GEMINI_GENERATION_MODEL = "gemini-2.0-flash"       # primary
GEMINI_FALLBACK_MODEL   = "gemini-2.0-flash-lite"  # rate-limit fallback
GEMINI_AUX_MODEL        = "gemini-2.0-flash-lite"  # intent classification + HyDE
GEMINI_JUDGE_MODEL      = "gemini-2.0-flash-lite"  # RAGAS LLM-as-judge
EMBEDDING_MODEL_ID      = "nvidia/llama-nemotron-embed-vl-1b-v2"
```

**`config/rag_config.py`**
```python
TOP_K_HYBRID             = 50        # candidates before RRF merge
TOP_K_RERANK             = 20        # candidates before reranker (config G)
RRF_K                    = 60        # RRF denominator
BM25_OVERSAMPLE_ATHLETE  = 20        # fetch 20× when filtering by athlete_id
GENERATION_TEMPERATURE   = 0.3
GENERATION_MAX_TOKENS    = 1024
THINKING_INTENTS         = {"comparison", "trend"}  # intents that use extended thinking
```

---

## Suggested Next Steps

These are ordered by impact, not difficulty:

### High priority
1. **Improve comparison + coaching answer relevancy** — the RAGAS scores show these two intents are weakest (0.445 and 0.589). The system prompt in `app/chain.py` has generic instructions for comparison; a more structured prompt ("List athlete A stats, then athlete B stats, then compare directly") would help.

2. **Add ground-truth eval set for context_recall** — right now context_recall is always null. Creating 20–30 (question, ground_truth_answer) pairs in `eval/` would unlock this metric and give a more complete picture.

3. **Upgrade back to Gemini 2.5 Flash when budget allows** — the free 2.0 models are capable but lack extended thinking. Thinking was designed for trend/comparison intents and made a measurable quality difference. When switching, just update `GEMINI_GENERATION_MODEL` in `config/model_settings.py` and the `_THINKING_CAPABLE` set in `app/chain.py`.

### Medium priority
4. **Reranker evaluation** — Config G (hybrid + rerank) uses the NVIDIA reranker but has never been formally benchmarked against Config F. Run RAGAS on both and compare. The reranker adds latency (~1–2 s) so the trade-off needs to be quantified.

5. **Add LangSmith tracing** — `MASTER_PLAN.md` specifies LangSmith for observability. It was planned but not implemented. Adding `@traceable` decorators to `augmentation()`, `retrieval()`, and `generation()` in `app/chain.py` would give full trace trees in the LangSmith dashboard.

6. **Redis for production memory** — conversations are lost on HF container restarts. A free-tier Redis Cloud instance (30 MB, sufficient for session data) wired up via `REDIS_URL` env var would make memory persistent.

### Low priority / polish
7. **Retrieval cache TTL surfacing** — there's a 5-min in-memory retrieval cache in `chain.py`. The UI has no indicator that a result came from cache. Adding a `"cached": true` field to the SSE metadata and displaying it in the sidebar would be a nice debug tool.

8. **Athlete count** — currently ~300 athletes. Adding more would improve coverage for comparison queries (some athlete ID combos miss each other in retrieval). Re-running `dataset_main.py` with a higher `N_ATHLETES` in `config/settings.py` is the only change needed.

---

## How the SSE Protocol Works (for debugging)

The `/chat/stream` endpoint emits `data: {json}\n\n` lines. Event sequence:

```
data: {"__ping__": true}              ← sent immediately; client ignores
data: "Athlete 42 "                   ← text token
data: "started at "                   ← text token
data: "120kg in week 1."              ← text token
...
data: {"__done__": true, "retrieval_ms": 4200, "generation_ms": 3100, ...}
```

On error:
```
data: {"__ping__": true}
data: {"__error__": "429 RESOURCE_EXHAUSTED ...", "__done__": true}
```

The UI's `_stream_chat_api()` in `ui/streamlit_app.py` handles all these cases. The ordering of `if` checks matters — `__error__` must come before `__done__`.

---

## Architecture Decision Record (Quick Reminder)

| Decision | Choice | Reason |
|----------|--------|--------|
| Vector DB | Qdrant Cloud | Native hybrid search, RRF, no vector limit on free tier |
| Embedding | NVIDIA Nemotron-Embed-VL-1B | Only free model that embeds text + images in the same space |
| Generation | Gemini 2.0 Flash | Free tier, multimodal, adequate quality |
| Sparse index | BM25 (rank_bm25) | Catches exact numerical matches that dense embeddings blur |
| Memory | ConversationSummaryBufferMemory | Auto-compresses; keeps last 8 turns verbatim |
| Frontend | Streamlit | Free HF Spaces deploy, native image upload, fragment reruns |
| CI/CD | GitHub Actions → HF Spaces | Fully automated; no manual deploys needed |

Full rationale for every decision is in `MASTER_PLAN.md`.

---

## Files to Read First

If you're just picking this up, read in this order:

1. `MASTER_PLAN.md` — the "why" behind every architectural choice
2. `PIPELINE.md` — end-to-end walkthrough of all 6 phases
3. `config/model_settings.py` — all model IDs
4. `config/rag_config.py` — all tuning knobs
5. `app/chain.py` — the core RAG loop
6. `app/augmentation.py` — how queries are expanded before retrieval

*Last updated: May 2026 — after session fixing SSE error handling, rate-limit fallback, and free-tier model migration.*
