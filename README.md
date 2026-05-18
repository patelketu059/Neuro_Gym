# Neuro Gym — Full Pipeline Walkthrough

> **Who this is for:** Someone taking over the project from scratch who needs to understand every moving part — from raw CSV files all the way to the live chatbot demo. No assumptions made about prior knowledge.

---

## The Big Picture

Neuro Gym is a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about synthetic powerlifting athletes. You ask "How did athlete 42's squat progress in week 8?" and the bot retrieves the right data from a vector database, then generates a natural language answer using an LLM (Gemini).

Here's the entire system at a glance:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        NEURO GYM PIPELINE                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PHASE 1 — DATA GENERATION                                                  ║
║  ┌─────────────────┐   ┌──────────────────┐                                 ║
║  │ OpenPowerlifting│   │  600K Workout    │                                 ║
║  │    CSV (real)   │   │  Dataset (real)  │                                 ║
║  └────────┬────────┘   └────────┬─────────┘                                ║
║           └──────────┬──────────┘                                           ║
║                      ▼                                                       ║
║            athlete_generator.py                                              ║
║            (generates 300+ synthetic athletes,                               ║
║             each with a 12-week training block)                              ║
║                      │                                                       ║
║              ┌───────┴────────┐                                             ║
║              ▼                ▼                                              ║
║         sessions.csv      PDFs (one per athlete)                            ║
║         (65 000 rows)     (session log pages)                               ║
║                                                                              ║
║  PHASE 2 — EMBEDDING & INDEXING                                             ║
║              │                ▼                                              ║
║              ▼          generate_pdfs.py                                    ║
║          chunking.py    → athlete_NNN.pdf                                   ║
║          (3 chunk types)                                                     ║
║              │                                                               ║
║              ▼                                                               ║
║          embedder.py   ←── NVIDIA Nemotron-Embed-VL-1B                     ║
║          (2 048-dim vectors)                                                 ║
║              │                                                               ║
║    ┌─────────┼─────────┐                                                    ║
║    ▼         ▼         ▼                                                    ║
║ gym_text  gym_tables gym_images    ← 3 Qdrant collections                  ║
║ (coaching (session   (PDF page                                              ║
║  summaries) rows)    screenshots)                                           ║
║              │                                                               ║
║          bm_index.py → BM25 sparse index (65 000 docs)                     ║
║                                                                              ║
║  PHASE 3 — CLOUD UPLOAD                                                     ║
║              │                                                               ║
║    ┌─────────┴──────────┐                                                   ║
║    ▼                    ▼                                                    ║
║ HuggingFace Hub     Qdrant Cloud                                            ║
║ (BM25 index,        (3 collections,                                         ║
║  raw data files)     ~300K vectors)                                         ║
║                                                                              ║
║  PHASE 4 — FASTAPI BACKEND (app/)                                           ║
║  ┌──────────────────────────────────────────────────┐                       ║
║  │  User Query                                      │                       ║
║  │       ↓                                          │                       ║
║  │  augmentation.py  ← intent classification        │                       ║
║  │  (factual/trend/comparison/coaching/visual)      │                       ║
║  │  + HyDE query expansion                          │                       ║
║  │       ↓                                          │                       ║
║  │  retrieve.py                                     │                       ║
║  │  ├─ dense_search.py  → Qdrant ANN               │                       ║
║  │  ├─ sparse_search.py → BM25                     │                       ║
║  │  └─ fusion_search.py → RRF merge → top-K docs   │                       ║
║  │       ↓                                          │                       ║
║  │  chain.py (generation)                           │                       ║
║  │  ├─ gemini-2.0-flash (primary, free tier)        │                       ║
║  │  └─ gemini-2.0-flash-lite (rate-limit fallback)  │                       ║
║  │       ↓                                          │                       ║
║  │  memory.py (conversation history)                │                       ║
║  │  SSE stream → HTTP response                      │                       ║
║  └──────────────────────────────────────────────────┘                       ║
║                                                                              ║
║  PHASE 5 — CI/CD (.github/workflows/deploy.yml)                            ║
║  ┌────────────────────────────────────────────────────┐                     ║
║  │  git push → main                                   │                     ║
║  │       ↓ (GitHub Actions triggered)                 │                     ║
║  │  Job 1: sync app/ → HF Space: neuro-gym-rag        │                     ║
║  │  Job 2: sync ui/  → HF Space: neuro-gym-ui         │                     ║
║  └────────────────────────────────────────────────────┘                     ║
║                                                                              ║
║  PHASE 6 — LIVE DEMO                                                        ║
║  ┌──────────────────────┐     ┌───────────────────────┐                    ║
║  │  neuro-gym-ui        │────▶│  neuro-gym-rag        │                    ║
║  │  (Streamlit UI)      │     │  (FastAPI backend)    │                    ║
║  │  k2p/neuro-gym-ui    │     │  k2p/neuro-gym-rag    │                    ║
║  └──────────────────────┘     └───────────────────────┘                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Phase 1 — Data Generation

### What's happening
We don't have real gym data, so we **synthesise** it by combining two real-world datasets:

| Dataset | What it provides |
|---------|-----------------|
| **OpenPowerlifting** (`openpowerlifting.csv`) | Real competition results: squat/bench/deadlift totals, bodyweight, DOTS scores, sex |
| **MuscleWiki 600K Workout Dataset** | Real exercise names, muscle groups, set/rep schemes for accessory work |

The generator samples real athletes from OPL and attaches realistic synthetic training blocks to them.

### Key files
```
pipeline/dataset/
  opl_loader.py          — loads & filters OpenPowerlifting CSV
  gym_600k_loader.py     — loads workout dataset, builds exercise pools by muscle group
  periodization.py       — defines 12-week block templates (linear, undulating, etc.)
  athlete_generator.py   — combines everything → one athlete's full training block
  dataset_main.py        — orchestrates parallel generation across 300 athletes
  generate_pdfs.py       — converts session data → PDF pages (one PDF per athlete)
  export.py              — writes sessions.csv and block_summary.csv
```

### Step by step
1. **`opl_loader.py`** reads `data/openpowerlifting.csv`, filters to athletes with valid competition totals, classifies each by DOTS score into beginner / intermediate / advanced / elite
2. **`gym_600k_loader.py`** reads `data/gym_exercises_600k.csv`, groups exercises by muscle group, builds pools for accessory selection
3. **`periodization.py`** defines templates like "12 weeks, RPE 6→9, linear load progression" — the shape of the training block
4. **`athlete_generator.py`** runs for each athlete:
   - Picks a real competition result from OPL
   - Assigns a periodization template
   - Generates 12 weeks × multiple sessions per week
   - Each session: main lift (squat / bench / deadlift) with sets, reps, kg, RPE + 3–5 accessory exercises
5. **`dataset_main.py`** runs `generate_one_athlete()` in parallel using `ProcessPoolExecutor` (up to 8 workers) and writes checkpoints every N athletes so a crash doesn't lose everything
6. **`export.py`** saves `data/sessions.csv` (~65 000 rows, one row per set) and `data/block_summary.csv`
7. **`generate_pdfs.py`** turns session data into PDF files (`data/athlete_NNN.pdf`) — these become the **image collection** source

### Output
- `data/sessions.csv` — 65 000 rows of training data
- `data/block_summary.csv` — one row per athlete with peak lifts, DOTS, program info
- `data/athlete_NNN.pdf` — one PDF per athlete (scanned-document style pages)

---

## Phase 2 — Embedding & Indexing

### What's happening
Raw data is useless for retrieval. We need to convert it into **vectors** (lists of 2 048 numbers) that capture semantic meaning, then store those in a database that can find similar vectors fast. We also build a keyword index (BM25) as a backup retrieval method.

### The three chunk types
The pipeline creates three very different kinds of documents from the same source data:

```
sessions.csv ──┬─→ coaching texts  → gym_text collection
               ├─→ table rows      → gym_tables collection
               └─→ (PDF pages)     → gym_images collection
```

| Collection | What's stored | Best for |
|------------|--------------|----------|
| `gym_text` | Prose summaries: "Athlete 42 is an intermediate male powerlifter, 84.2kg, DOTS 342..." | Factual / coaching questions |
| `gym_tables` | Structured session rows: "Week 3, Squat, 120kg, 3×5, RPE 7.5" | Trend / comparison questions |
| `gym_images` | Screenshot images of PDF pages | Visual / document questions |

### The embedding model
**NVIDIA Nemotron-Embed-VL-1B-v2** (via OpenRouter API)
- "VL" = Vision-Language — handles both text and images
- Outputs 2 048-dimensional vectors
- Both the query at search time and the stored chunks use this same model — so "similar meaning" = "nearby in vector space"

### Key files
```
pipeline/ingestion/
  chunking.py     — splits sessions.csv into the three chunk types
  embedder.py     — calls OpenRouter API to embed each chunk (batched)
  collection.py   — creates Qdrant collections with correct vector config
  ingest.py       — orchestrates: chunk → embed → upsert to Qdrant
  bm_index.py     — builds BM25 index from gym_tables text corpus
  checkpoint.py   — saves progress so re-runs skip already-embedded chunks
```

### Step by step
1. **`chunking.py`** reads `sessions.csv` and produces three datasets:
   - Coaching texts (one per athlete — a prose paragraph)
   - Table rows (one per training set — structured text)
   - PDF page images (rendered via PyMuPDF from the generated PDFs)
2. **`embedder.py`** sends each chunk to the OpenRouter API in batches of 32, receives 2 048-dim vectors back
3. **`collection.py`** creates (or resets) the three Qdrant collections with `COSINE` distance metric and 2 048-dim vectors
4. **`ingest.py`** upserts each (vector, payload) pair into the right collection. The payload carries metadata: `athlete_id`, `week`, `lift`, `kg`, `rpe`, etc. — used for filtering at query time
5. **`bm_index.py`** tokenises all gym_tables text rows and builds a BM25 (Best Match 25) sparse index. BM25 is like a smarter version of keyword search — it ranks documents by term frequency × rarity

### Why BM25 alongside dense vectors?
Dense vectors are great at semantic similarity ("what does the athlete bench?" ≈ "upper body pressing load") but struggle with **exact numbers** ("120kg" vs "121kg" look identical in embedding space). BM25 is great at exact term matching. Together they cover each other's blind spots.

---

## Phase 3 — Cloud Upload

### What's happening
The embedded data needs to be accessible from HuggingFace Spaces where the backend runs. Two cloud services are used:

```
Local machine
├─→ Qdrant Cloud  ← vectors (dense embeddings)
└─→ HuggingFace Hub ← BM25 index files + raw data CSVs
```

### Key files
```
scripts/
  push_to_hf.py   — uploads BM25 index, sessions.csv, etc. to HF Hub dataset repo
  
pipeline/ingestion/
  ingest.py       — contains the Qdrant upsert logic (also the upload step)
```

### HuggingFace Hub dataset repo
The HF dataset repo (`k2p/neuro-gym-data` or similar) stores:
- `bm25_index.pkl` — serialised BM25 index (~30 MB)
- `bm25_corpus.pkl` — the raw text documents BM25 scores against
- `sessions.csv` — raw training data (for reference)

At startup, the FastAPI backend downloads the BM25 index from HF Hub to a local cache so it doesn't need to rebuild it (which takes ~2 min).

### Qdrant Cloud
Qdrant is a **vector database** — think of it as Postgres but optimised for finding the nearest vectors. Three collections live here permanently:
- `gym_text` (~300 coaching summaries)
- `gym_tables` (~65 000 session rows)
- `gym_images` (~2 400 PDF page images)

---

## Phase 4 — FastAPI Backend

### What's happening
This is the brain. When a user sends a question, the backend:
1. **Understands** the question (intent classification)
2. **Expands** the question (HyDE)
3. **Retrieves** relevant documents from Qdrant + BM25
4. **Generates** an answer using Gemini
5. **Remembers** the conversation for follow-up questions

### Key files
```
app/
  main.py          — FastAPI app startup, loads BM25, connects Qdrant, initialises Gemini
  augmentation.py  — intent classification + HyDE query expansion
  chain.py         — orchestrates retrieval → generation, handles streaming + fallback
  memory.py        — ConversationSummaryBufferMemory (Redis or in-memory)
  routes/
    chat.py        — /chat and /stream endpoints
    health.py      — /health endpoint (used by HF to know the space is alive)
config/
  model_settings.py — all model IDs in one place
  rag_config.py     — retrieval knobs (TOP_K, context budget, etc.)
  settings.py       — file paths, environment variable loading
```

### The augmentation step (`augmentation.py`)
Before retrieval, the query goes through two transformations:

**1. Intent classification**
The query is classified into one of 5 intents using Gemini (AUX model):

| Intent | Example question | Effect |
|--------|-----------------|--------|
| `factual` | "What did athlete 42 squat in week 3?" | Small context window, no HyDE |
| `trend` | "How did athlete 42's squat progress?" | Large context, HyDE on |
| `comparison` | "Compare athlete 42 vs 117 on squat" | Multi-athlete retrieval |
| `coaching` | "What program should athlete 42 follow?" | Coaching text collection priority |
| `visual` | "Show me athlete 42's training log page" | Image collection only |

**2. HyDE (Hypothetical Document Embeddings)**
For trend/comparison queries, instead of embedding the raw question, Gemini *generates a fake answer first* ("Athlete 42's squat started at 120kg in week 1 and peaked at 145kg in week 11..."). That fake answer is then embedded. The idea: a fake answer's embedding is closer to the real answer chunks than the question's embedding was.

### The retrieval step (`pipeline/retrieval/retrieve.py`)
```
Query embedding
      │
      ├──→ dense_search.py  → Qdrant ANN search (top 30 per collection)
      │                        filtered by athlete_id if mentioned in query
      │
      └──→ sparse_search.py → BM25 keyword search (top 20 per athlete)
                               *fetches 20× more when filtering, because one
                                athlete is only ~0.02% of 65K corpus*

Both ranked lists → fusion_search.py
                    RRF merge: score = Σ 1/(60 + rank)
                    → top-K final docs (usually 15–17)
```

**Retrieval configs** (selectable from the UI sidebar):

| Config | Collections used | BM25? | Use case |
|--------|-----------------|-------|----------|
| A — images only | gym_images | No | Visual questions |
| B — text only | gym_text | No | Coaching |
| C — tables only | gym_tables | No | Factual/simple |
| D — all dense | all three | No | Balanced |
| E — tables + BM25 | gym_tables | Yes | Factual + keywords |
| **F — all + BM25** | all three | Yes | **Default (best overall)** |
| G — hybrid + rerank | all three | Yes + reranker | Highest quality, slowest |
| H — BM25 only | none (BM25 only) | Yes | Keyword-only baseline |

### The generation step (`app/chain.py`)
Retrieved documents are formatted into a context string and sent to Gemini along with the conversation history:

```
System prompt: "You are a powerlifting coach AI..."
+ Conversation history (last 8 turns, auto-summarised)
+ Context: [retrieved documents]
+ User query
      ↓
gemini-2.0-flash (primary, free tier: 15 RPM / 1,500 RPD)
      │
      ├── If rate-limited (429 / RESOURCE_EXHAUSTED):
      │   └──→ gemini-2.0-flash-lite (fallback, free tier: 30 RPM / 1,500 RPD)
      │
      └── Response streamed back via SSE
```

**Streaming (SSE — Server-Sent Events):**
The response arrives token by token. The backend yields:
- `{"__ping__": true}` — heartbeat sent immediately to keep the connection alive during the slow retrieval phase
- `"token text"` — each text chunk as Gemini produces it
- `{"__done__": true, "retrieval_ms": ..., "generation_ms": ...}` — final metadata

**Why streaming?** Retrieval + generation can take 5–10 seconds. Without streaming the user stares at a blank screen. With streaming, text starts appearing after ~1 second.

### The memory step (`app/memory.py`)
The bot remembers the conversation so you can ask follow-ups:
- **LangChain `ConversationSummaryBufferMemory`** — keeps the last 8 messages verbatim
- When buffer > 2 000 tokens, Gemini compresses the oldest half into a summary paragraph
- Backed by **Redis** (production) or in-memory dict (fallback if Redis is unavailable)
- Sessions are keyed by `session_id` (UUID generated per browser session)

---

## Phase 5 — CI/CD

### What's happening
Every time you push code to the `main` branch on GitHub, **GitHub Actions** automatically deploys the changes to two HuggingFace Spaces. You never have to manually copy files.

### Key file
```
.github/workflows/deploy.yml
```

### Two deployment jobs

**Job 1: Backend (`neuro-gym-rag` space)**
- Triggered when any of these paths change: `app/**`, `pipeline/**`, `config/**`, `requirements.txt`, `Dockerfile`
- Clones the HF Space repo (`k2p/neuro-gym-rag`)
- `rsync`s the changed files in
- Commits and pushes → HF rebuilds the Docker container

**Job 2: Frontend (`neuro-gym-ui` space)**
- Triggered when `ui/streamlit_app.py` changes
- Copies `ui/streamlit_app.py` → `app.py` (HF Spaces expects the main file at root as `app.py`)
- Patches the `FASTAPI_URL` constant to point at the deployed backend space URL
- Pushes → HF rebuilds the Streamlit container

### Secrets needed in GitHub
```
HF_TOKEN          — HuggingFace token with write access to both spaces
```

### How to verify CI/CD ran
```bash
gh run list --limit 5
# Should show "completed success" for recent pushes
```

---

## Phase 6 — Live Demo

### Two HuggingFace Spaces

| Space | URL | What it is |
|-------|-----|-----------|
| `k2p/neuro-gym-rag` | `https://k2p-neuro-gym-rag.hf.space` | FastAPI backend (Docker) |
| `k2p/neuro-gym-ui` | `https://k2p-neuro-gym-ui.hf.space` | Streamlit frontend |

### Backend startup sequence (`app/main.py`)
When the Docker container starts:
1. Downloads `bm25_index.pkl` and `bm25_corpus.pkl` from HF Hub → `data/` cache
2. Connects to Qdrant Cloud (URL + API key from environment variables)
3. Initialises Gemini client (API key from environment)
4. Starts FastAPI on port 7860
5. `/health` endpoint returns `{"status":"ok","bm25_loaded":true}` — HF monitors this

### Frontend features (`ui/streamlit_app.py`)
- **Chat interface** — type questions, get streamed answers
- **Image upload** — attach a chart or training log image; it's embedded with the multimodal model and used in retrieval
- **Sidebar controls:**
  - Retrieval config selector (A through H)
  - Streaming toggle (disable if SSE has proxy issues)
  - BM25 status indicator
  - Response timing display (retrieval ms + generation ms)
- **Non-streaming fallback** — if SSE fails silently, automatically retries via the blocking `/chat` endpoint

### Key environment variables (set in HF Space secrets)
```
GEMINI_API_KEY      — Google AI Studio free key
QDRANT_URL          — Qdrant Cloud cluster URL
QDRANT_API_KEY      — Qdrant Cloud API key
OPENROUTER_API_KEY  — for NVIDIA embedding model
HF_TOKEN            — for downloading BM25 index from Hub
REDIS_URL           — optional; falls back to in-memory if absent
```

---

## Data Flow — 14 Steps

1. Raw CSVs (`openpowerlifting.csv` + `gym_exercises_600k.csv`) are loaded and filtered
2. 300+ synthetic athlete profiles are generated with 12-week training blocks
3. Session data is exported to `sessions.csv` (~65 000 rows)
4. PDF pages are rendered from session data using PyMuPDF
5. `chunking.py` splits sessions into three chunk types (coaching text, table rows, image pages)
6. `embedder.py` calls NVIDIA Nemotron-Embed-VL-1B via OpenRouter, producing 2 048-dim vectors
7. Vectors + payloads are upserted into three Qdrant Cloud collections
8. `bm_index.py` builds a BM25 keyword index over all gym_tables text
9. BM25 index is uploaded to HuggingFace Hub dataset repo
10. A user question arrives at the Streamlit UI → POSTed to the FastAPI backend
11. `augmentation.py` classifies intent and optionally generates a HyDE expansion
12. `retrieve.py` runs dense search (Qdrant) + sparse search (BM25), merges with RRF
13. `chain.py` sends [system prompt + history + context + query] to Gemini, streams the response
14. SSE chunks flow back to Streamlit, appearing word-by-word; `memory.py` saves the exchange

---

## Key Files Cheat Sheet

| File | What it does | When to touch it |
|------|-------------|-----------------|
| `pipeline/dataset/dataset_main.py` | Generates all athlete data | Adding more athletes or changing data shape |
| `pipeline/ingestion/chunking.py` | Defines what goes into each Qdrant collection | Changing chunk structure |
| `pipeline/ingestion/embedder.py` | Calls embedding API | Switching embedding model |
| `pipeline/ingestion/ingest.py` | Uploads vectors to Qdrant | Re-indexing after schema changes |
| `pipeline/ingestion/bm_index.py` | Builds BM25 index | Re-indexing keyword search |
| `pipeline/retrieval/retrieve.py` | Orchestrates hybrid search | Tuning retrieval configs |
| `pipeline/retrieval/fusion_search.py` | RRF merge logic | Changing ranking algorithm |
| `app/augmentation.py` | Intent classification + HyDE | Adding new intent types |
| `app/chain.py` | Core RAG loop + streaming | Changing generation behaviour |
| `app/memory.py` | Conversation history | Changing memory strategy |
| `config/model_settings.py` | All model IDs | Switching LLMs |
| `config/rag_config.py` | TOP_K, context budgets | Tuning retrieval quality |
| `ui/streamlit_app.py` | Frontend UI | UI changes |
| `.github/workflows/deploy.yml` | CI/CD | Deployment changes |
| `scripts/push_to_hf.py` | Uploads data to HF Hub | After re-generating BM25 index |

---

## Common Tasks

### Re-generate the dataset from scratch
```bash
python pipeline/dataset/dataset_main.py
python pipeline/dataset/generate_pdfs.py
```

### Re-embed and re-index everything
```bash
python pipeline/ingestion/ingest.py          # vectors → Qdrant
python pipeline/ingestion/bm_index.py        # BM25 index
python scripts/push_to_hf.py                 # upload BM25 to HF Hub
```

### Run the backend locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the frontend locally (against local backend)
```bash
# Edit FASTAPI_URL in ui/streamlit_app.py to "http://localhost:8000"
streamlit run ui/streamlit_app.py
```

### Run the RAGAS evaluation
```bash
python eval/ragas_eval.py --n_questions 50 --configs "F — all + BM25"
# Results written to eval/ragas_results_summary.json
```

### Deploy manually (skip CI/CD)
```bash
# Backend
cd hf_pull/space && git pull && rsync -av ../../app/ ./app/ && git add -A && git commit -m "manual deploy" && git push

# Frontend  
cd hf_pull/ui && git pull && cp ../../ui/streamlit_app.py ./app.py && git add -A && git commit -m "manual deploy" && git push
```

---

## RAGAS Evaluation Results (as of last run)

The pipeline quality is measured with **RAGAS** (Retrieval Augmented Generation Assessment):

| Config | Faithfulness | Answer Relevancy | Context Precision | Retrieval (ms) | Generation (ms) |
|--------|:-----------:|:----------------:|:-----------------:|:--------------:|:---------------:|
| F — all + BM25 | 0.897 | 0.655 | 0.692 | 5 856 | 3 675 |

**By intent (Config F, 50 questions):**

| Intent | n | Faithfulness | Answer Relevancy |
|--------|---|:------------:|:----------------:|
| Factual | 15 | 0.928 | 0.778 |
| Trend | 10 | 0.975 | 0.816 |
| Comparison | 10 | 0.886 | 0.445 |
| Coaching | 10 | 0.759 | 0.589 |
| Visual | 5 | 0.963 | 0.517 |

**What the metrics mean:**
- **Faithfulness** — does the answer only say things that are in the retrieved context? (1.0 = perfect, no hallucination)
- **Answer Relevancy** — does the answer actually address the question? (1.0 = perfectly on-topic)
- **Context Precision** — are the retrieved chunks actually useful? (1.0 = no noise retrieved)

**Takeaway:** The system is very faithful (doesn't hallucinate much) but answer relevancy for comparison and coaching questions is lower — those are the hardest query types and good targets for improvement.

---

*Last updated: May 2026*
