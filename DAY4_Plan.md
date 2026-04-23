
Layer 1 — Retrieval pipeline modifications
No app dependencies. Safe to implement and test in isolation with a Python REPL.
1. pipeline/retrieval/dense_day3.py (modified)
Added ThreadPoolExecutor parallel search. The 3 Qdrant queries for gym_images, gym_text, gym_tables now run concurrently instead of sequentially. Also added collections parameter so configs A/B/C can restrict to a single collection. Implement this first because retrieve_day3.py depends on it.
2. pipeline/retrieval/context_day3.py (modified)
Upgraded context assembly with two new functions: _athlete_profile_block() prepends a structured header (training level, Dots, peak lifts) before each athlete's first passage; _passage_block() wraps each retrieved passage with --- SOURCE: athlete_id | week N · phase · collection | score --- so Gemini can cite correctly. Also added payload dict to each source entry for the Streamlit athlete profile cards. Implement before retrieve_day3.py since it's imported there.
3. pipeline/retrieval/retrieve_day3.py (modified)
Two changes. First, added RetrievalConfig dataclass and the 8 named CONFIGS dict (A through H) — this is the config system the entire app uses. Second, implemented correct image/text vector routing: text collections (gym_text, gym_tables) always use embed_query(query), gym_images uses embed_query(query, image_path=...) when an image is uploaded (multimodal via OpenRouter), or falls back to the text vector when no image is present. Returns retrieval_ms and config_name in the context dict.

Layer 2 — Application layer
Depends on Layer 1. Implement after retrieval pipeline is stable.
4. app/augmentation_day4.py (new)
The explicit A step in A→R→G. Passthrough on first turn. On subsequent turns calls gemini-2.0-flash at temperature=0 with the last 6 history messages to resolve pronouns and compress follow-up questions into a self-contained retrieval query. Sets retrieval_query in inputs (used for embedding) while preserving query (used for generation). Graceful fallback to raw query on any exception. Implement before chain_day4.py which imports it.
5. app/memory_day4.py (pre-existing — no changes needed)
ConversationSummaryBufferMemory was already complete. Keeps last k=8 messages in buffer, summarises overflow every 16 messages using Gemini. Session registry with get_or_create_memory() and clear_memory(). No implementation work required — just verify it imports cleanly.
6. app/chain_day4.py (new)
Wires the explicit A→R→G chain. augmentation_step() imports and calls augmentation_day4.augmentation_step directly (no wrapper — avoids double @traceable). retrieval_step() uses inputs["retrieval_query"] for embedding. generation_step() uses inputs["query"] (original) for response naturalness. Contains two image helpers: _pdf_page_to_b64() for rendering corpus PDF pages via fitz, and _image_file_to_b64() for reading user-uploaded PNG/JPEG directly (preserves mime type, no fitz overhead). Gemini model received from inputs["gemini"] — not recreated per request.

Layer 3 — FastAPI application
Depends on Layer 2. Implement once chain is stable.
7. app/main_day4.py (new)
FastAPI lifespan loads three things once at startup and stores them on app.state: Qdrant client, BM25 index + corpus, and genai.GenerativeModel("gemini-2.5-flash"). The Gemini model being instantiated here (not per-request) is the key optimisation. Also stores app.state.pdf_dir pointing to the local PDF directory.
8. app/routes/health_day4.py (new)
GET /health. Uses getattr(info, "points_count", None) or getattr(info, "vectors_count", None) to handle both old and new qdrant-client versions. Returns Qdrant collection counts, BM25 loaded status, Gemini loaded status, and active session count.
9. app/routes/chat_day4.py (new)
POST /chat with form fields query, session_id, and config_name (defaults to "F — all + BM25"). Saves uploaded image to a temp file, passes path to run_chain(), deletes it in finally. ChatResponse includes retrieval_ms, generation_ms, pdf_paths, athlete_ids, and config_name so the UI has everything it needs in one response.

Layer 4 — Frontend
Depends on Layer 3. Implement last — it consumes the complete API.
10. ui/streamlit_app_day4.py (new)
Two-column layout: chat (left 3/5), PDF viewer (right 2/5). Sidebar contains: retrieval config dropdown with descriptions, golden eval question loader (all 25 questions), image upload widget, session controls, and health status. PDF viewer shows top-5 retrieved athlete PDFs with left/right arrow navigation, page slider, and position dots. Each assistant message shows: response text, latency bar (retrieval ms · generation ms · config name), color-coded sources panel (blue=gym_images, green=gym_text, amber=gym_tables, pink=BM25), and athlete profile cards (Dots, level, S/B/D peaks, program).