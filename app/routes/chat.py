from __future__ import annotations
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatResponse(BaseModel):
    response: str
    sources: list[dict]
    pdf_paths: list[str]
    athlete_ids: list[str]
    session_id: str

    retrieval_ms: int
    generation_ms: int

    config_name: str
    intent: str = 'factual'
    query_rewritten: bool = False
    retrieval_query: str = ""
    hyde_document: str = ""
    sub_queries: list[str] = []


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request:     Request,
    query:       str        = Form(...),
    session_id:  str        = Form(default='default'),
    config_name: str        = Form(default='F — all + BM25'),
    image:       UploadFile = File(default=None),
):
    state = request.app.state
    tmp_image_path: str | None = None

    try:
        if image and image.filename:
            suffix = Path(image.filename).suffix or '.png'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_image_path:
            Path(tmp_image_path).unlink(missing_ok=True)


@router.post("/chat/stream")
async def chat_stream(
    request:     Request,
    query:       str        = Form(...),
    session_id:  str        = Form(default='default'),
    config_name: str        = Form(default='F — all + BM25'),
    image:       UploadFile = File(default=None),
):
    """Server-Sent Events endpoint.

    Each event is ``data: <json>\\n\\n``.  Non-final events are JSON-encoded
    text tokens (``"Hello"``).  The final event is a JSON object with
    ``"__done__": true`` and all metadata (sources, timing, athlete_ids, …).
    """
    state = request.app.state
    tmp_image_path: str | None = None

    if image and image.filename:
        suffix = Path(image.filename).suffix or '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            tmp_image_path = tmp.name

    from app.chain import run_chain_stream

    def event_generator():
        try:
            for chunk in run_chain_stream(
                query            = query,
                session_id       = session_id,
                bm25             = state.bm25,
                corpus           = state.corpus,
                client           = state.qdrant,
                gemini           = state.gemini,
                config_name      = config_name,
                query_image_path = tmp_image_path,
                pdf_dir          = state.pdf_dir,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'__error__': str(e)})}\n\n"
        finally:
            if tmp_image_path:
                Path(tmp_image_path).unlink(missing_ok=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
        },
    )


@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    from app.memory import clear_memory
    clear_memory(session_id)
    return JSONResponse({"cleared": session_id})
