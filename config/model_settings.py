from pathlib import Path
import os

EMBEDDING_MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"

# Gemini model identifiers — centralised so chain.py, augmentation.py and
# memory.py stay in sync without scattered string literals.
#
# Tier strategy:
#   GENERATION_MODEL  — primary answer model
#   FALLBACK_CHAIN    — tried in order when primary hits a 429 / RESOURCE_EXHAUSTED.
#                       All entries are free-tier eligible with an AI Studio key.
#                       Add a paid model (e.g. "gemini-2.5-flash") at the end of
#                       FALLBACK_CHAIN if you want a paid last-resort.
#   AUX_MODEL         — intent classification + HyDE; lighter, higher RPM
#   JUDGE_MODEL       — RAGAS LLM-as-judge; eval-only
#
# Free-tier availability (AI Studio key, no billing project — May 2026):
#   gemini-2.5-flash      free tier — RPM limited, supports ThinkingConfig
#   gemini-2.5-flash-lite free tier — lighter fallback
#
# NOTE: gemini-2.0-flash and gemini-2.0-flash-lite have limit:0 on free-tier
#       AI Studio projects (removed from free tier in 2026). Do not use as primary.
# NOTE: gemini-1.5-x models discontinued Feb 2025 — return 404.

GEMINI_GENERATION_MODEL = "gemini-2.5-flash"

GEMINI_FALLBACK_CHAIN: list[str] = [
    "gemini-2.5-flash-lite",  # lighter 2.5 variant — higher RPM on free tier
    "gemini-2.0-flash",       # kept as last resort; may work on some free projects
]

# Kept for any code that still imports GEMINI_FALLBACK_MODEL directly.
GEMINI_FALLBACK_MODEL = GEMINI_FALLBACK_CHAIN[0]

GEMINI_AUX_MODEL   = "gemini-2.5-flash-lite"  # query analysis + HyDE
GEMINI_JUDGE_MODEL = "gemini-2.5-flash-lite"  # RAGAS LLM-as-judge


def _get_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


DEVICE = _get_device()
