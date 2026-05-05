from pathlib import Path
import os

EMBEDDING_MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"

# Gemini model identifiers — centralised so chain.py, augmentation.py and
# memory.py stay in sync without scattered string literals.
GEMINI_GENERATION_MODEL = "gemini-2.5-flash"
GEMINI_AUX_MODEL        = "gemini-2.0-flash"


def _get_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


DEVICE = _get_device()


ALL_COLLECTIONS = ["gym_images", "gym_text", "gym_tables"]

RETRIEVAL_CONFIGS: dict[str, dict] = {
    "dense_images_only": {
        "label":       "Dense — images only",
        "collections": ["gym_images"],
        "use_bm25":    False,
        "reranker":    False,
    },
    "dense_text_only": {
        "label":       "Dense — text only",
        "collections": ["gym_text"],
        "use_bm25":    False,
        "reranker":    False,
    },
    "dense_tables_only": {
        "label":       "Dense — tables only",
        "collections": ["gym_tables"],
        "use_bm25":    False,
        "reranker":    False,
    },
    "dense_all": {
        "label":       "Dense — all collections",
        "collections": ALL_COLLECTIONS,
        "use_bm25":    False,
        "reranker":    False,
    },
    "hybrid_tables": {
        "label":       "Hybrid — tables + BM25",
        "collections": ["gym_tables"],
        "use_bm25":    True,
        "reranker":    False,
    },
    "hybrid_all": {
        "label":       "Hybrid — all + BM25 + RRF",
        "collections": ALL_COLLECTIONS,
        "use_bm25":    True,
        "reranker":    False,
    },
    "hybrid_all_reranked": {
        "label":       "Hybrid + reranker",
        "collections": ALL_COLLECTIONS,
        "use_bm25":    True,
        "reranker":    True,
    },
}
