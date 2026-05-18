# Neuro Gym — Multimodal RAG Chatbot for Powerlifting

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about synthetic powerlifting athletes. Ask about training loads, RPE progressions, program comparisons, or attach a chart image — the system retrieves the right data and generates grounded, cited answers.

**Portfolio project** targeting mid-level AI/ML Engineer roles. Every architectural decision is intentional and documented.

---

## Live Demo

| | URL |
|---|---|
| **Frontend** (Streamlit) | https://k2p-neuro-gym-ui.hf.space |
| **Backend API** (FastAPI) | https://k2p-neuro-gym-rag.hf.space/health |

> Free-tier HuggingFace Spaces — first load after idle may take ~30 s to warm up.

---

## Features

- **Hybrid retrieval** — dense (Qdrant ANN) + sparse (BM25) merged with Reciprocal Rank Fusion
- **Multimodal** — attach a training chart image; NVIDIA Nemotron embeds it alongside text for visual search
- **Streaming responses** — SSE token-by-token with heartbeat to survive proxy timeouts; auto-falls back to blocking `/chat` if SSE fails
- **Intent-aware pipeline** — classifies queries as factual / trend / comparison / coaching / visual and adjusts context budget, HyDE, and thinking mode accordingly
- **HyDE query expansion** — generates a hypothetical answer, embeds it, uses that vector for retrieval
- **Conversation memory** — `ConversationSummaryBufferMemory`: keeps last 8 turns verbatim, auto-compresses overflow
- **Multi-model fallback** — `gemini-2.5-flash → gemini-2.5-flash-lite → gemini-2.0-flash` on rate limits or quota errors
- **PDF viewer** — side panel renders matched athlete training log pages inline
- **8 retrieval configs** — A through H, selectable from the UI (images-only → full hybrid + reranker)
- **CI/CD** — GitHub Actions deploys both HF Spaces on every push to `main`

---

## Stack

| Layer | Choice |
|---|---|
| Embedding | NVIDIA Nemotron-Embed-VL-1B-v2 (2 048-dim, text + image, via OpenRouter) |
| Vector DB | Qdrant Cloud — 3 collections: `gym_text`, `gym_tables`, `gym_images` |
| Sparse index | BM25 (`rank_bm25`, in-memory, 65 000 docs) |
| Reranker | NVIDIA Nemotron-Rerank-VL-1B-v2 (config G only) |
| LLM | Gemini 2.5 Flash (free tier, AI Studio key) |
| Memory | LangChain ConversationSummaryBufferMemory + Redis/in-memory fallback |
| Backend | FastAPI + SSE streaming |
| Frontend | Streamlit |
| CI/CD | GitHub Actions → HuggingFace Spaces |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                       NEURO GYM PIPELINE                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  DATA GENERATION                                                     ║
║  OpenPowerlifting CSV + 600K Workout Dataset                        ║
║       → athlete_generator.py (300+ synthetic athletes)              ║
║       → sessions.csv (65 000 rows) + athlete PDFs                   ║
║                                                                      ║
║  EMBEDDING & INDEXING                                                ║
║  chunking.py → 3 chunk types                                        ║
║       → NVIDIA Nemotron-Embed-VL-1B (2 048-dim)                    ║
║       → Qdrant: gym_text | gym_tables | gym_images                  ║
║       → BM25 sparse index (65 000 docs)                             ║
║                                                                      ║
║  FASTAPI BACKEND                                                     ║
║  User query + optional image                                        ║
║       → augmentation.py  (intent classification + HyDE)             ║
║       → retrieve.py      (dense + BM25 → RRF → top-K docs)         ║
║       → chain.py         (Gemini 2.5 Flash, SSE stream)             ║
║         fallback chain:  2.5-flash → 2.5-flash-lite → 2.0-flash    ║
║       → memory.py        (conversation history)                     ║
║                                                                      ║
║  CI/CD                                                               ║
║  git push → GitHub Actions → neuro-gym-rag + neuro-gym-ui (HF)     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## RAGAS Evaluation (Config F — all + BM25, 50 questions)

| Metric | Score |
|--------|:-----:|
| Faithfulness | **0.897** |
| Answer Relevancy | **0.655** |
| Context Precision | **0.692** |

| Intent | n | Faithfulness | Answer Relevancy |
|--------|---|:---:|:---:|
| Factual | 15 | 0.928 | 0.778 |
| Trend | 10 | 0.975 | 0.816 |
| Comparison | 10 | 0.886 | 0.445 |
| Coaching | 10 | 0.759 | 0.589 |
| Visual | 5 | 0.963 | 0.517 |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/patelketu059/Neuro_Gym.git
cd Neuro_Gym
python -m venv .neuro_gym && .neuro_gym/Scripts/activate  # Windows
pip install -r requirements.txt

# 2. Set environment variables (.env at project root)
GEMINI_API_KEY=...        # Google AI Studio — free tier key (no billing project)
QDRANT_URL=...            # Qdrant Cloud cluster URL
QDRANT_API_KEY=...        # Qdrant Cloud API key
OPENROUTER_API_KEY=...    # NVIDIA embedding model
HF_TOKEN=...              # HuggingFace read token

# 3. Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Run frontend (separate terminal)
streamlit run ui/streamlit_app.py
```

---

## Project Structure

```
Neuro_Gym/
├── app/                    # FastAPI backend
│   ├── chain.py            # RAG loop — augment → retrieve → generate
│   ├── augmentation.py     # Intent classification + HyDE
│   ├── memory.py           # Conversation memory
│   └── routes/             # /chat, /chat/stream, /health, /pdf
├── pipeline/
│   ├── dataset/            # Synthetic data generation
│   ├── ingestion/          # Embedding + Qdrant indexing + BM25
│   └── retrieval/          # Dense, sparse, fusion, reranker
├── ui/
│   └── streamlit_app.py    # Frontend
├── config/
│   ├── model_settings.py   # All model IDs
│   └── rag_config.py       # TOP_K, context budgets, temperatures
├── eval/                   # RAGAS evaluation scripts + results
├── scripts/                # push_to_hf.py, e2e_test.py
└── .github/workflows/
    ├── deploy.yml          # CI/CD — deploys both HF Spaces on push to main
    └── keep_alive.yml      # Pings HF Spaces every 12h to prevent cold starts
```

---

## Documentation

| Doc | What's in it |
|-----|-------------|
| [`PIPELINE.md`](PIPELINE.md) | Full end-to-end walkthrough — data generation through live demo, with diagrams |
| [`HANDOFF.md`](HANDOFF.md) | Current project state, recent changes, known issues, next steps |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Architecture decision record — why each technology was chosen |
| [`MASTER_PLAN.md`](MASTER_PLAN.md) | Original design spec and resume gap analysis |

---

## Environment Variables

| Variable | Required | Used by |
|----------|:--------:|---------|
| `GEMINI_API_KEY` | ✅ | Generation, intent classification, memory summarisation |
| `QDRANT_URL` | ✅ | Vector search |
| `QDRANT_API_KEY` | ✅ | Vector search |
| `OPENROUTER_API_KEY` | ✅ | NVIDIA Nemotron embedding |
| `HF_TOKEN` | ✅ | Download BM25 index from HF Hub on startup |
| `REDIS_URL` | ⬜ | Session memory (falls back to in-memory if absent) |

> **Gemini key note:** Must be created at [aistudio.google.com](https://aistudio.google.com) as **"Create API key in new project"** — not an existing billing-enabled GCP project. Billing-enabled projects set free-tier quota to 0.
