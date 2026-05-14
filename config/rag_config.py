"""RAG pipeline tuning constants — single source for all thresholds and budgets."""
from __future__ import annotations
from typing import FrozenSet

INTENT_CONTEXT_TOKENS: dict[str, int] = {
    "factual":     8_192,   # one athlete, one week
    "trend":      16_384,   # full 12-week block for one athlete
    "comparison": 32_768,   # multiple athletes in parallel
    "coaching":   16_384,
    "visual":      8_192,
}
DEFAULT_CONTEXT_TOKENS: int = 8_192

TOP_K_HYBRID:             int = 50
TOP_K_RERANK:             int = 20
TOP_K_ATHLETES:           int = 5
TOP_K_PER_ATHLETE_CHUNKS: int = 3

RRF_K: int = 60

# Oversample BM25 so all of an athlete's 12 weekly records survive post-filter.
BM25_OVERSAMPLE_ATHLETE: int = 20
BM25_OVERSAMPLE_LEVEL:   int = 4

HYDE_INTENTS: FrozenSet[str] = frozenset({"trend", "coaching"})

# "trend" skips gym_images; images add noise for week-by-week aggregation queries.
TEXT_ONLY_INTENTS: FrozenSet[str] = frozenset({"trend"})

GENERATION_TEMPERATURE: float = 0.3
GENERATION_MAX_TOKENS:  int   = 1024

# Extended thinking requires temperature=1.0 (SDK requirement).
THINKING_INTENTS: FrozenSet[str] = frozenset({"comparison", "trend"})
THINKING_BUDGET:       int   = 512
THINKING_TEMPERATURE:  float = 1.0
THINKING_MAX_TOKENS:   int   = 2048

EMBED_CACHE_SIZE: int = 512

SESSION_TTL_SECONDS: float = 3600.0
REDIS_URL_DEFAULT:   str   = "redis://localhost:6379"
