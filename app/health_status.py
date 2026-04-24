from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get('/health')
async def health(request: Request):
    
    state = request.app.state
    collections = {}
    qdrant_ok = False

    try:
        for col in ["gym_images", "gym_text", "gym_tables"]:
            info = state.qdrant.get_collection(col)
            count = (
                getattr(info, "points_count", None)
                or getattr(info, "vectors_count", None)
                or 0
            )

            collections[col] = count
        qdrant_ok = True
    except Exception as e:
        collections = {"error": str(e)}

    
    from app.memory import active_sessions
    return JSONResponse({
        "status":          "ok" if qdrant_ok else "degraded",
        "qdrant":          collections,
        "bm25_loaded":     getattr(state, "bm25", None) is not None,
        "gemini_loaded":   getattr(state, "gemini", None) is not None,
        "active_sessions": len(active_sessions()),
    })