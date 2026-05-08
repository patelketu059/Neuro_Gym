# Neuro Gym RAG — Architecture & Design Reference

Quick-reference for codebase decisions, data flow, and tuning parameters.
Read this before editing any pipeline file so you understand the **why** behind each design choice.

---

## 1. System Overview

Multimodal Retrieval-Augmented Generation (RAG) chatbot for a powerlifting gym.
Answers natural-language questions about athlete training records using:
- Dense vector search (NVIDIA Nemotron-Embed-VL-1B-v2 via OpenRouter)
- BM25 keyword search (rank-bm25, in-process)
- Reciprocal Rank Fusion (RRF k=60) to merge ranked lists
- Optional cross-encoder reranking
- Gemini 2.5 Flash for generation (with extended thinking on complex intents)

**Scale**: ~60,000 documents, ~4,000 athletes, 3 Qdrant collections.

---

## 2. File Map

```
app/
  main.py              FastAPI app, lifespan startup/shutdown
  chain.py             Orchestration: augment → retrieve → generate
  augmentation.py      Query rewrite, intent classification, HyDE, pronoun resolution
  memory.py            ConversationSummaryBufferMemory, session registry
  session_store.py     Redis-backed session persistence with in-memory fallback
  routes/
    chat.py            /chat (sync) and /chat/stream (SSE) endpoints
    health_status.py   /health

pipeline/
  ingestion/
    ingest.py          CLI: load CSVs → build NL strings → BM25 index
    chunking.py        Build natural-language strings from session rows
    embedder.py        embed_query() for query-time; embed_text_batch() for ingestion
    bm_index.py        Build and load BM25 index (rank-bm25 + pickle)
    collection.py      Qdrant collection creation and client factory
  retrieval/
    retrieve.py        Main retrieve() and multi_retrieve() functions
    sparse_search.py   BM25 search with athlete_id / training_level post-filter
    fusion_search.py   RRF implementation
    context.py         deduplicate_athlete(), assemble_context(), token counting

config/
  settings.py          File paths (DATA_DIR, SESSIONS_PATH, etc.)
  model_settings.py    Model IDs, RETRIEVAL_CONFIGS dict, ALL_COLLECTIONS
  rag_config.py        All numeric tuning constants (see §6)

scripts/
  push_to_hf.py        Upload data/ or embeddings/ to HuggingFace datasets
```

---

## 3. Query Lifecycle (end-to-end)

```
User query
  │
  ▼
[augmentation.py — augment()]
  1. _normalize_athlete_refs()      regex normalise "athlete 89" → "athlete_00089"
  2. EntityRegister.resolve_pronouns() pronoun → most-recent athlete ID from history
  3. _call_combined()               LangChain with_structured_output(QueryAnalysis)
                                     → intent, rewritten_query, athlete_ids,
                                       sub_queries, training_levels
  4. Merge regex IDs + LLM IDs      ground-truth wins; LLM fills gaps
  5. HyDE (if intent in HYDE_INTENTS) Gemini synthesises a passage, embed as dense vector
  6. Sub-query fallback for comparison intent
  │
  ▼
[chain.py — retrieval()]
  Builds retrieval_filters from athlete_ids + training_levels
  → multi_retrieve() for sub_queries (parallel dense calls)
  → retrieve() for primary query:
      dense search across collections (gym_text, gym_tables, gym_images)
      + BM25 sparse_search() with post-filter
      → RRF fusion (k=60)
      → optional cross-encoder rerank
      → deduplicate_athlete() (top 5 athletes, best chunk per athlete)
      → assemble_context() (tiktoken budget, profile headers, passage blocks)
  │
  ▼
[chain.py — generation()]  or  run_chain_stream()
  _build_content_parts():
    conversation history + retrieved context + PDF page images + user image + query
  _build_gen_config():
    thinking mode ON for {comparison, trend} intents (ThinkingBudget=1024, T=1.0)
    standard mode for {factual, coaching, visual} (T=0.3, max_tokens=1024)
  gemini.models.generate_content() with retry (3 attempts, delays 1/3/9 s)
  │
  ▼
Response + metadata (sources, athlete_ids, retrieval_ms, generation_ms, intent, …)
```

---

## 4. Collections

| Collection    | Content                              | Vector source                     |
|---------------|--------------------------------------|-----------------------------------|
| `gym_text`    | Coaching notes (NL text per athlete) | Nemotron-Embed text mode          |
| `gym_tables`  | Session rows (week, lift, RPE, etc.) | Nemotron-Embed text mode          |
| `gym_images`  | PDF chart pages (image + OCR text)   | Nemotron-Embed multimodal mode    |

All vectors are L2-normalised float16, stored as float32 in Qdrant.
Dimension: 2048 (Nemotron-Embed-VL-1B-v2).

---

## 5. Intent System

Classified by `augmentation.py` using LangChain `with_structured_output(QueryAnalysis)`.

| Intent       | Description                                  | HyDE | Thinking | Context tokens |
|--------------|----------------------------------------------|------|----------|----------------|
| `factual`    | Single data point for one athlete            | No   | No       | 8,192          |
| `trend`      | Week-by-week progression for one athlete     | Yes  | Yes      | 16,384         |
| `comparison` | Multiple athletes or levels compared         | No   | Yes      | 32,768         |
| `coaching`   | Open-ended advice grounded in data           | Yes  | No       | 16,384         |
| `visual`     | Questions about charts / PDF pages           | No   | No       | 8,192          |

`TEXT_ONLY_INTENTS = {factual, trend}` — gym_images collection is skipped for these.

---

## 6. Retrieval Configuration Presets

Defined in `config/model_settings.py → RETRIEVAL_CONFIGS`. Selected per-request via the
`config_name` form field on `/chat`.

| Key                   | Label                        | Collections         | BM25 | Reranker |
|-----------------------|------------------------------|---------------------|------|----------|
| `dense_images_only`   | Dense — images only          | gym_images          | No   | No       |
| `dense_text_only`     | Dense — text only            | gym_text            | No   | No       |
| `dense_tables_only`   | Dense — tables only          | gym_tables          | No   | No       |
| `dense_all`           | Dense — all collections      | all three           | No   | No       |
| `hybrid_tables`       | Hybrid — tables + BM25       | gym_tables          | Yes  | No       |
| `hybrid_all`          | Hybrid — all + BM25 + RRF    | all three           | Yes  | No       |
| `hybrid_all_reranked` | Hybrid + reranker            | all three           | Yes  | Yes      |

Default config: `hybrid_all` (BM25 + dense + RRF, no reranker — best balance).

---

## 7. Numeric Tuning Constants (`config/rag_config.py`)

| Constant                 | Value  | Meaning                                          |
|--------------------------|--------|--------------------------------------------------|
| `TOP_K_HYBRID`           | 50     | Candidates from each dense/BM25 list before RRF  |
| `TOP_K_RERANK`           | 20     | Retained after cross-encoder reranker            |
| `TOP_K_ATHLETES`         | 5      | Unique athletes after deduplication              |
| `RRF_K`                  | 60     | RRF denominator (higher → less aggressive bias)  |
| `BM25_OVERSAMPLE_ATHLETE`| 20×    | Fetch factor when filtering by `athlete_id`      |
| `BM25_OVERSAMPLE_LEVEL`  | 4×     | Fetch factor when filtering by `training_level`  |
| `GENERATION_TEMPERATURE` | 0.3    | Non-thinking intents                             |
| `GENERATION_MAX_TOKENS`  | 1024   | Non-thinking intents                             |
| `THINKING_BUDGET`        | 1024   | Thinking tokens for comparison/trend             |
| `THINKING_TEMPERATURE`   | 1.0    | Required by Gemini SDK when thinking is enabled  |
| `THINKING_MAX_TOKENS`    | 2048   | Output tokens for thinking intents               |
| `EMBED_CACHE_SIZE`       | 512    | LRU slots for text embedding cache               |
| `SESSION_TTL_SECONDS`    | 3600   | Idle seconds before session eviction             |

---

## 8. BM25 Pipeline

`pipeline/ingestion/bm_index.py` — builds at ingest time, loaded into process memory at startup.

- Corpus: `list[dict]` where each dict has `{"text": ..., "athlete_id": ..., "training_level": ...}`
- Index: `BM25Okapi` object, serialised with pickle (`BM_index.pkl`)
- Corpus: saved as JSON (`BM_corpus.json`)

`pipeline/retrieval/sparse_search.py` — query-time:
- Oversample by `BM25_OVERSAMPLE_ATHLETE` (20×) when `athlete_id` filter is present
- Post-filter results by `athlete_id` and/or `training_level`
- Returns same dict structure as Qdrant results (`{"payload": ..., "score": ..., "collection": "bm25"}`)

**Why oversample**: One athlete's ~12 weekly records are ~0.02% of the 60K corpus.
Without oversampling, BM25 returns top-50 which may miss all records for a low-frequency athlete.

---

## 9. HyDE (Hypothetical Document Embeddings)

Active for `trend` and `coaching` intents only.

1. Gemini (`gemini-2.0-flash`, T=0.4) generates a synthetic 100-word training log passage
2. Passage is embedded via `embed_query(doc, mode="passage")`
3. Resulting vector (`hyde_vector`) is passed to `retrieve()` alongside the query vector
4. RRF fuses: dense(query), dense(hyde), BM25 — three ranked lists merged

**Why not for factual**: HyDE benefits open-ended queries; factual queries have exact keywords
that BM25 handles better than a hallucinated passage.

---

## 10. Athlete ID Normalisation

Three-layer guarantee that athlete IDs are always `athlete_NNNNN` format:

1. **Regex pre-normalisation** (`_normalize_athlete_refs`): Replaces informal refs in the raw
   query before anything else. Pattern: `\bathlete[\s_]*#?\s*(\d{1,5})\b`
2. **Pydantic `@field_validator`** (`QueryAnalysis.normalize_ids`): Normalises whatever the
   LLM returns in `athlete_ids`, accepting bare numbers, short-padded IDs, informal formats.
3. **Merge step**: `informal_ids + analysis.athlete_ids` deduped — regex is ground truth,
   LLM output fills gaps (e.g. athlete IDs mentioned only in rewritten query).

**Why three layers**: LangChain `with_structured_output` eliminates brittle JSON parsing,
but the LLM can still zero-pad incorrectly. The validator is the hard guarantee.

---

## 11. Session Memory

`app/memory.py — ConversationSummaryBufferMemory`:
- Keeps last `k=8` messages in a live buffer
- When `buffer_tokens() > max_token_budget (2000)`, oldest half is compressed via Gemini
  into a JSON summary (`{"athletes": ..., "open_questions": ..., "theme": ...}`)
- Summary is prepended to history as a synthetic message tagged `[Summary of earlier conversation]`
- Token counting uses tiktoken (`cl100k_base` encoding) — accurate, not chars/4 estimate

`app/session_store.py — SessionStore`:
- Redis backend when available (`REDIS_URL` env var, default `redis://localhost:6379`)
- Transparent in-memory dict fallback when Redis is unavailable
- TTL: 3600 s (Redis handles expiry automatically; in-memory eviction runs every 5 min via FastAPI background task)
- `evict_stale()` is called by an `asyncio.create_task` loop in `app/main.py` lifespan

---

## 12. Embedding Cache

`pipeline/ingestion/embedder.py — _cached_text_embed(text, mode)`:
- `@functools.lru_cache(maxsize=512)` — ~8 KB per slot × 512 = ~4 MB
- Cache key is `(text, mode)` only — **API key is NOT part of the key** to prevent
  fragmentation and avoid storing credentials in cache entries
- API key is resolved from `OPENROUTER_API_KEY` env var at call time
- Image queries bypass the cache (unique bytes, not worth caching)

---

## 13. API Endpoints

| Method | Path               | Description                                   |
|--------|--------------------|-----------------------------------------------|
| POST   | `/chat`            | Synchronous — returns full response JSON      |
| POST   | `/chat/stream`     | SSE — streams tokens then final metadata      |
| DELETE | `/chat/{session_id}` | Clear conversation history for a session  |
| GET    | `/health`          | Liveness check                                |

**SSE protocol** (`/chat/stream`):
- Each non-final event: `data: "token text"\n\n` (JSON-encoded string)
- Final event: `data: {"__done__": true, "sources": [...], "generation_ms": ..., ...}\n\n`
- Error event: `data: {"__error__": "message", "__done__": true}\n\n`
  (always includes `__done__: true` so client parser terminates)

**Input validation** (both endpoints):
- `query` max 2,000 characters (HTTP 422 if exceeded)
- Image max 10 MB (HTTP 413 if exceeded)
- Image MIME must be `image/jpeg`, `image/png`, `image/gif`, or `image/webp` (HTTP 422)

---

## 14. Generation & Retry

`app/chain.py — _generate_with_retry()`:
- Retries on `ResourceExhausted`, `ServiceUnavailable`, `QuotaExceeded`, `Unavailable`,
  HTTP 429, HTTP 503
- Backoff delays: 1 s, 3 s, 9 s (3 attempts total)
- `run_chain_stream()` uses the same retry logic wrapping the full stream iteration;
  if the stream fails mid-flight and the error is retryable, it retries from scratch

---

## 15. Key Design Decisions

### D1 — LangChain `with_structured_output` for query analysis
Rather than parsing LLM JSON with regex, use LangChain function-calling mode backed by a
Pydantic model. This makes malformed athlete IDs, wrong intents, and missing fields
impossible — the schema validates and normalises at deserialisation time.

### D2 — RRF instead of score normalisation
Dense scores (cosine similarity) and BM25 scores are on incompatible scales. RRF is
rank-based and immune to this. k=60 was tuned empirically to avoid over-weighting
top-1 documents while still being responsive to rank.

### D3 — Per-intent context token budgets
A factual lookup doesn't need a 32K context window. Tiered budgets reduce Gemini input
cost and latency by 4× for the most common intent (factual).

### D4 — Thinking mode only for comparison and trend
Thinking tokens cost extra and add latency. Empirically, factual and coaching queries
don't benefit; comparison and multi-week trend analysis do.

### D5 — tiktoken for token counting
`chars / 3` is systematically wrong — it underestimates Unicode-heavy text and
overestimates ASCII. tiktoken `cl100k_base` gives the correct GPT-family tokenization
which closely approximates Gemini's tokenizer, keeping context budgets accurate.

### D6 — BM25 oversampling, not pre-filtering
Qdrant supports payload filtering, but BM25 is an in-process numpy operation.
Post-filtering with a 20× oversample is simpler and just as accurate for the ~12
records per athlete case.

### D7 — Redis optional, not required
The app boots and is fully functional without Redis. The in-memory fallback means
local development and Colab demos work without Docker. Redis adds persistence across
restarts and multi-worker scaling.

---

## 16. Environment Variables

| Variable            | Required | Default                | Description                        |
|---------------------|----------|------------------------|------------------------------------|
| `GEMINI_API_KEY`    | Yes      | —                      | Gemini 2.5 Flash + 2.0 Flash       |
| `OPENROUTER_API_KEY`| Yes      | —                      | NVIDIA Nemotron-Embed-VL via OR    |
| `QDRANT_URL`        | Cloud    | —                      | Cloud Qdrant endpoint              |
| `QDRANT_API_KEY`    | Cloud    | —                      | Cloud Qdrant API key               |
| `QDRANT_HOST`       | Local    | `localhost`            | Local Qdrant host                  |
| `QDRANT_PORT`       | Local    | `6333`                 | Local Qdrant port                  |
| `HF_TOKEN`          | Push only| —                      | HuggingFace write token            |
| `REDIS_URL`         | No       | `redis://localhost:6379`| Session persistence               |

---

## 17. Common Gotchas

- **CSV duplicate headers**: The generation pipeline appends to CSV with checkpoint flushes,
  creating rows where `athlete_id == 'athlete_id'`. `ingest.py` and the notebook cell both
  drop these before any numeric coercion.

- **pandas ArrowDtype**: `pyarrow` as a pandas dependency causes mixed CSV columns to infer
  as `StringDtype`. Always call `pd.to_numeric(col, errors='coerce')` after `read_csv`.

- **Gemini streaming + thinking**: When thinking mode is active, `chunk.text` may be empty
  for the thinking-phase chunks. The chain handles this with `if chunk.text:` guard.

- **HyDE embedding call site**: `embed_query(hyde_document, mode="passage")` is called in
  `augmentation.py`, not in `retrieve.py`. The resulting vector is passed as `hyde_vector`
  through the inputs dict.

- **`_chain_cache` in augmentation.py**: LangChain chains are cached per API key to avoid
  rebuilding the LLM client on every request. Cache is module-level dict — no TTL, but
  chains are stateless so this is safe.
