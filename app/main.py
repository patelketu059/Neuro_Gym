from __future__ import annotations
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
    if not env_path.is_file(): return 

    for line in env_path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue
        
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

    
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_env()

    from pipeline.ingestion.collection import get_client
    app.state.qdrant = get_client(
        host = os.environ.get("QDRANT_HOST", "localhost"),
        port = int(os.environ.get("QDRANT_PORT", "6333"))
    )
    print(f"[INFO-APP] - Qdrant Instantiated")


    from pipeline.ingestion.bm_index import load_bm_index
    app.state.bm25, app.state.corpus = load_bm_index(
        index_path = ROOT / 'hf_pull' / 'k2p' / 'gym-rag-embeddings' / 'BM_index.pkl',
        corpus_path = ROOT / 'hf_pull'/ 'k2p' / 'gym-rag-embeddings' / 'BM_corpus.json'
    )
    print(f"[INFO-APP] - BM25 Loaded | {len(app.state.corpus)}")


    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    # print(api_key)
    if api_key:
        app.state.gemini = genai.Client(api_key = api_key)
        print(f"[INFO-APP] - Gemini 2.5 Flash connected")

    else:
        app.state.gemini - None
        print(f"[INFO-APP] - Gemini Key not set")

    app.state.pdf_dir = str(ROOT / 'data' / 'pdfs' / "pdfs_archive")
    yield 
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