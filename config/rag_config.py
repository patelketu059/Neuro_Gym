"""RAG pipeline tuning constants.

All numeric thresholds, token budgets, and intent-routing tables live here
so any tuning change is a single-line edit rather than a grep-and-replace
across multiple pipeline files.
"""
from __future__ import annotations
from typing import FrozenSet

# ── Context window budgets (tokens) per query intent ─────────────────────────
# Gemini 2.5 Flash supports 1 M input tokens.  Budgets are tiered by intent
# so a factual look-up doesn't pay the latency/cost of a 32 K window, while
# multi-athlete comparisons have room for full 12-week profiles.
INTENT_CONTEXT_TOKENS: dict[str, int] = {
    "factual":     8_192,   # one athlete, one week
    "trend":      16_384,   # full 12-week block for one athlete
    "comparison": 32_768,   # multiple athletes in parallel
    "coaching":   16_384,
    "visual":      8_192,
}
DEFAULT_CONTEXT_TOKENS: int = 8_192   # fallback for unrecognised intents

# ── Retrieval candidate counts ────────────────────────────────────────────────
TOP_K_HYBRID:             int = 50  # candidates from each dense/BM25 list before RRF
TOP_K_RERANK:             int = 20  # kept after cross-encoder reranker
TOP_K_ATHLETES:           int = 5   # unique athletes after deduplication
TOP_K_PER_ATHLETE_CHUNKS: int = 3   # max week-chunks kept per athlete (allows multi-week context)

# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
RRF_K: int = 60   # denominator — higher → less aggressive rank bias

# ── BM25 oversampling ratios ──────────────────────────────────────────────────
# One athlete's 12 weekly records ≈ 0.02 % of the 60 K-document corpus.
# Multiply fetch_k so all their weeks survive post-filter.
BM25_OVERSAMPLE_ATHLETE: int = 20   # when filtering by athlete_id
BM25_OVERSAMPLE_LEVEL:   int = 4    # when filtering by training_level only

# ── HyDE (Hypothetical Document Embeddings) ───────────────────────────────────
# HyDE adds one Gemini call + one embedding API call per query.  Only run it
# for open-ended intents where the benefit justifies the latency.
HYDE_INTENTS: FrozenSet[str] = frozenset({"trend", "coaching"})

# ── Intent-adaptive collection routing ───────────────────────────────────────
# Factual and trend queries are answered purely from text records — images add
# noise and latency without improving answer quality for these intents.
TEXT_ONLY_INTENTS: FrozenSet[str] = frozenset({"factual", "trend"})

# ── Gemini generation ─────────────────────────────────────────────────────────
GENERATION_TEMPERATURE: float = 0.3
GENERATION_MAX_TOKENS:  int   = 1024

# Thinking mode — Gemini 2.5 Flash extended thinking.
# Requires temperature = 1.0 when enabled (SDK requirement).
THINKING_INTENTS: FrozenSet[str] = frozenset({"comparison", "trend"})
THINKING_BUDGET:       int   = 1024   # thinking tokens allocated
THINKING_TEMPERATURE:  float = 1.0
THINKING_MAX_TOKENS:   int   = 2048   # output tokens (larger to accommodate reasoning)

# ── Embedding cache ───────────────────────────────────────────────────────────
EMBED_CACHE_SIZE: int = 512   # LRU slots; each ≈ 2048 × 4 bytes ≈ 8 KB

# ── Session management ────────────────────────────────────────────────────────
SESSION_TTL_SECONDS: float = 3600.0          # idle seconds before eviction
REDIS_URL_DEFAULT:   str   = "redis://localhost:6379"
