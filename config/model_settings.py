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
# Free-tier daily limits per key (AI Studio, no billing project):
#   gemini-2.0-flash      15 RPM / 1 500 RPD
#   gemini-2.0-flash-lite 30 RPM / 1 500 RPD
#
# NOTE: gemini-1.5-flash and gemini-1.5-flash-8b were discontinued Feb 2025
#       and now return 404. Do not add them back.

GEMINI_GENERATION_MODEL = "gemini-2.0-flash"

GEMINI_FALLBACK_CHAIN: list[str] = [
    "gemini-2.0-flash-lite",  # 30 RPM / 1 500 RPD — higher rate limit, confirmed available
    # "gemini-2.5-flash",     # uncomment to add a paid/preview fallback at the very end
]

# Kept for any code that still imports GEMINI_FALLBACK_MODEL directly.
GEMINI_FALLBACK_MODEL = GEMINI_FALLBACK_CHAIN[0]

GEMINI_AUX_MODEL   = "gemini-2.0-flash-lite"  # query analysis + HyDE
GEMINI_JUDGE_MODEL = "gemini-2.0-flash-lite"  # RAGAS LLM-as-judge


def _get_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


DEVICE = _get_device()
