from pathlib import Path
import os

EMBEDDING_MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"

# Gemini model identifiers — centralised so chain.py, augmentation.py and
# memory.py stay in sync without scattered string literals.
GEMINI_GENERATION_MODEL = "gemini-2.5-flash"   # main answer model — paid tier
GEMINI_AUX_MODEL        = "gemini-2.5-flash"   # query analysis + HyDE
GEMINI_JUDGE_MODEL      = "gemini-2.5-flash"   # RAGAS LLM-as-judge for eval scoring


def _get_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


DEVICE = _get_device()
