from pathlib import Path
import os

EMBEDDING_MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"

# Gemini model identifiers — centralised so chain.py, augmentation.py and
# memory.py stay in sync without scattered string literals.
#
# Tier strategy:
#   GENERATION_MODEL — primary answer model; free-tier eligible (15 RPM / 1 500 RPD)
#   FALLBACK_MODEL   — used automatically when the primary hits a 429 rate-limit
#   AUX_MODEL        — intent classification + HyDE; lighter tasks, free-tier (30 RPM)
#   JUDGE_MODEL      — RAGAS LLM-as-judge; free-tier, eval-only
GEMINI_GENERATION_MODEL = "gemini-2.0-flash"       # primary — free tier
GEMINI_FALLBACK_MODEL   = "gemini-2.0-flash-lite"  # rate-limit fallback — free tier
GEMINI_AUX_MODEL        = "gemini-2.0-flash-lite"  # query analysis + HyDE
GEMINI_JUDGE_MODEL      = "gemini-2.0-flash-lite"  # RAGAS LLM-as-judge


def _get_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


DEVICE = _get_device()
