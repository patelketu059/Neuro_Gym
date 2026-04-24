from __future__ import annotations
import tempfile
from pathlib import Path
from fastapi import APIRouter, Form, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

class ChatResponse(BaseModel):
    response: str
    sources: list[dict]
    pdf_paths: list[str]
    athlete_ids: list[str]
    config_name: str
    retrieval_ms : int
    generation_ms: int
    session_id: str


@router.post("/chat", response_model = ChatResponse)
async def chat(
    request:            Request,
    query:              str         = Form(...),
    session_id:         str         = Form(default = 'default'),
    config_name:        str         = Form(default = 'F - all + BM25'),
    image:              UploadFile  = File(default = None)
):
    state = request.app.state
    tmp_image_path: str | None = None

    try:
        if image and image.filename:
            suffix = Path(image.filename).suffix or '.png'
            with tempfile.NamedTemporaryFile(delete = False, suffix = suffix) as tmp: 
                tmp.write(await image.read())
                tmp_image_path = tmp.name

            from app.chain import run_chain
            result = run_chain(
                            query            = query,
                            session_id       = session_id,
                            bm25             = state.bm25,
                            corpus           = state.corpus,
                            client           = state.qdrant,
                            gemini           = state.gemini,
                            config_name      = config_name,
                            query_image_path = tmp_image_path,
                            pdf_dir          = state.pdf_dir,
            )
            return ChatResponse(**result)
    finally:
        if tmp_image_path:
            Path(tmp_image_path).unlink(missing_ok = True)



@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    from app.memory import clear_memory
    clear_memory(session_id)
    return JSONResponse({"cleared": session_id})