from __future__ import annotations
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Start with: 
# uvicorn app.main_day4:app --host 0.0.0.0 --port 8000 --reload

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_env() -> None:
    env_path = ROOT / '.env'
    if not env_path.is_file():
        return
    from dotenv import load_dotenv
    load_dotenv(env_path, override=False)


def _resolve_embed_dir() -> Path:
    """Return the path to BM25 + text_vectors artifacts.

    Local dev  : uses hf_pull/k2p/gym-rag-embeddings when BM_index.pkl exists there.
    HF Spaces  : downloads BM_index.pkl and BM_corpus.json from HF Hub into
                 /tmp/gym-rag-artifacts on first boot; subsequent boots reuse the
                 cached files for the lifetime of the container.
    """
    local = ROOT / "hf_pull" / "k2p" / "gym-rag-embeddings"
    if (local / "BM_index.pkl").exists():
        return local

    artifact_dir = Path(os.environ.get("ARTIFACT_DIR", "/tmp/gym-rag-artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bm_index  = artifact_dir / "BM_index.pkl"
    bm_corpus = artifact_dir / "BM_corpus.json"

    if not bm_index.exists() or not bm_corpus.exists():
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN", "") or None
        embed_repo = os.environ.get("EMBED_REPO", "k2p/gym-rag-embeddings")
        print(f"[INFO-APP] - Downloading BM25 artifacts from {embed_repo} …")
        for filename in ("BM_index.pkl", "BM_corpus.json"):
            hf_hub_download(
                repo_id   = embed_repo,
                repo_type = "dataset",
                filename  = filename,
                local_dir = str(artifact_dir),
                token     = hf_token,
            )
        print(f"[INFO-APP] - BM25 artifacts ready at {artifact_dir}")

    return artifact_dir

    
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_env()

    from pipeline.ingestion.collection import get_client
    _qdrant_url     = os.environ.get("QDRANT_URL", "")
    _qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
    app.state.qdrant = get_client(
        host    = os.environ.get("QDRANT_HOST", "localhost"),
        port    = int(os.environ.get("QDRANT_PORT", "6333")),
        url     = _qdrant_url     or None,
        api_key = _qdrant_api_key or None,
    )
    _qdrant_target = _qdrant_url or f"{os.environ.get('QDRANT_HOST','localhost')}:{os.environ.get('QDRANT_PORT','6333')}"
    print(f"[INFO-APP] - Qdrant connected → {_qdrant_target}")


    from pipeline.ingestion.bm_index import load_bm_index, build_athlete_peaks, patch_corpus_with_peaks
    _embed_dir = _resolve_embed_dir()
    app.state.bm25, app.state.corpus = load_bm_index(
        index_path  = _embed_dir / 'BM_index.pkl',
        corpus_path = _embed_dir / 'BM_corpus.json',
    )
    print(f"[INFO-APP] - BM25 Loaded | {len(app.state.corpus)}")

    # Patch corpus with competition lift peaks parsed from gym_text coaching
    # summaries — build_athlete_peaks returns {} gracefully if text_vectors/ is absent.
    _peaks = build_athlete_peaks(_embed_dir / 'text_vectors')
    _patched = patch_corpus_with_peaks(app.state.corpus, _peaks)
    print(f"[INFO-APP] - Corpus patched with lift peaks | {len(_peaks)} athletes, {_patched} records updated")


    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    # print(api_key)
    if api_key:
        app.state.gemini = genai.Client(api_key = api_key)
        print(f"[INFO-APP] - Gemini 2.5 Flash connected")

    else:
        app.state.gemini = None
        print(f"[INFO-APP] - Gemini Key not set")

    app.state.pdf_dir = str(ROOT / 'data')

    async def _eviction_loop():
        from app.session_store import get_store
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            try:
                get_store().evict_stale()
            except Exception:
                pass

    eviction_task = asyncio.create_task(_eviction_loop())

    yield

    eviction_task.cancel()
    print(f"[INFO-APP] - Shutdown!")


app = FastAPI(
    title = "Neuro Gym RAG API",
    description = "Multimodal RAG Chatbot for Lifting",
    version = "0.1.0",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

from app.routes.chat import router as chat_router
from app.routes.health_status import router as health_router

app.include_router(chat_router)
app.include_router(health_router)