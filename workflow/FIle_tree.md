gym_rag/
│
├── config/                             shared across all days
│   ├── settings.py                     all constants, paths, cited parameters
│   ├── hf_config.toml                  HuggingFace repo IDs and push config
│   └── pdf_config.toml                 PDF generation randomisation controls
│
├── pipeline/
│   │
│   ├── dataset/                        ── Day 1 ──────────────────────────────
│   │   ├── models_day1.py              dataclass definitions (AthletePersona, SessionLog, Exercise, ...)
│   │   ├── opl_loader_day1.py          load + filter OpenPowerlifting CSV, derive amplitude
│   │   ├── gym_loader_day1.py          load 600K dataset, precompute accessory pools, query_accessories()
│   │   ├── periodization_day1.py       build periodization templates from OPL + literature constants
│   │   ├── athlete_generator_day1.py   sample_athlete_persona(), build_training_block()
│   │   ├── export_day1.py              flatten records → sessions.csv + block_summary.csv
│   │   └── dataset_main_day1.py        CLI entry point, checkpoint loop, orchestrates all Day 1 modules
│   │
│   ├── ingestion/                      ── Day 2 ──────────────────────────────
│   │   ├── chunking_day2.py            chunk_text() · build_all_nl_strings()        local + Colab
│   │   ├── bm25_index_day2.py          build_bm25_index() · load_bm25_index()       local + Colab
│   │   ├── ingest_day2.py              local entry point · extract_coaching_texts()  local + Colab
│   │   ├── collections_day2.py         Qdrant collection creation + payload indexes  Colab only
│   │   ├── checkpoint_day2.py          Drive-backed progress tracking                Colab only
│   │   └── embedder_day2.py            load_model() · embed_pdf_page() · embed_text_batch() · embed_query_api()  Colab + Day 3/4
│   │
│   └── retrieval/                      ── Day 3 ──────────────────────────────
│       ├── dense_day3.py               Qdrant cosine search across all 3 collections
│       ├── sparse_day3.py              BM25 keyword search wrapper
│       ├── fusion_day3.py              reciprocal_rank_fusion() · hybrid_search()
│       ├── reranker_day3.py            cross-encoder reranking (passthrough in prod)
│       ├── context_day3.py             Option A dedup by athlete_id · assemble_context()
│       └── retrieve_day3.py            top-level entry point tying all retrieval steps together
│
├── app/                                ── Day 4 ──────────────────────────────
│   ├── memory_day4.py                  ConversationSummaryBufferMemory · session registry
│   ├── chain_day4.py                   @traceable retrieval_step + generation_step · run_chain()
│   ├── main_day4.py                    FastAPI app · lifespan (loads Qdrant + BM25) · CORS
│   └── routes/
│       ├── chat_day4.py                POST /chat · DELETE /chat/{session_id}
│       └── health_day4.py              GET /health → Qdrant counts + BM25 status
│
├── ui/                                 ── Day 4 ──────────────────────────────
│   └── streamlit_app_day4.py           chat UI · image uploader · source expanders · FastAPI client
│
├── notebooks/
│   └── colab_ingestion_day2.ipynb      14-cell T4 GPU embedding notebook
│
├── eval/                               ── Day 5 (empty) ──────────────────────
│
├── data/
│   ├── raw/                            gitignored — source CSVs go here (opl.csv, etc.)
│   ├── output/                         gitignored — sessions.csv, block_summary.csv
│   ├── pdfs/                           gitignored — 4000 generated athlete PDFs
│   ├── bm25_corpus.json                pushed to HF, pulled into Colab
│   ├── bm25_index.pkl                  built locally + rebuilt in Colab Cell 10
│   └── qdrant_storage/                 gitignored — local Qdrant data volume
│
├── push_to_hf.py                       push dataset or embeddings to HuggingFace
├── requirements.txt                    all project dependencies (no tiktoken)
├── ARCHITECTURE.md                     system design decisions and data provenance
└── DAY2_INGESTION.md                   Day 2 process documentation

#########################################################

Day 1 local
  dataset_main_day1 → opl_loader · gym_loader · periodization
                     → athlete_generator → export
                     → sessions.csv + block_summary.csv + 4000 PDFs

Day 2 local (before Colab)
  ingest_day2 → chunking_day2 → bm25_corpus.json
              → bm25_index_day2 → bm25_index.pkl
  push_to_hf → gym-rag-dataset (sessions, block_summary, corpus, PDFs)

Day 2 Colab (14 cells)
  clone repo → pull HF data
  collections_day2 → empty Qdrant collections
  checkpoint_day2 → resume state from Drive
  embedder_day2 → model loaded into T4
  embedder_day2 + checkpoint_day2 → gym_images (16K vectors)
  ingest_day2 + chunking_day2 + embedder_day2 → gym_text (20K vectors)
  embedder_day2 → gym_tables (48K vectors)
  bm25_index_day2 → bm25_index.pkl
  push snapshots → gym-rag-embeddings

Day 3 local
  retrieve_day3 → dense_day3 + sparse_day3 → fusion_day3
               → reranker_day3 → context_day3
               uses embedder_day2.embed_query_api() at query time

Day 4 local
  main_day4 → loads Qdrant + BM25 at startup
  chain_day4 → retrieval_step (retrieve_day3) + generation_step (Gemini)
             → memory_day4 per session
  routes → chat_day4 + health_day4
  streamlit_app_day4 → POSTs to FastAPI

Day 5 (eval/ empty — to be built)