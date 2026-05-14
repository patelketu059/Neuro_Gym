from __future__ import annotations
import os
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response

router = APIRouter()

_PDF_CACHE_DIR = Path(os.environ.get("PDF_CACHE_DIR", "/tmp/gym-rag-pdfs"))


def _get_pdf_path(filename: str) -> Path:
    """Return local path to PDF, downloading from HF Hub on first request."""
    cached = _PDF_CACHE_DIR / filename
    if cached.is_file():
        return cached

    _PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hf_token     = (os.environ.get("HF_TOKEN", "") or "").strip() or None
    dataset_repo = os.environ.get("DATASET_REPO", "k2p/gym-rag-dataset")

    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id   = dataset_repo,
            repo_type = "dataset",
            filename  = f"pdfs/{filename}",
            local_dir = str(_PDF_CACHE_DIR),
            token     = hf_token,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found in dataset repo: pdfs/{filename} ({exc})",
        )

    # hf_hub_download mirrors the remote path: _PDF_CACHE_DIR/pdfs/<filename>
    nested = _PDF_CACHE_DIR / "pdfs" / filename
    if nested.is_file():
        return nested

    if cached.is_file():
        return cached

    raise HTTPException(status_code=404, detail=f"PDF not found after download: {filename}")


@router.get("/pdf/page")
async def pdf_page(
    request: Request,
    path: str,
    page: int = 0,
    dpi: int = 150,
):
    """Render one PDF page and return as image/png."""
    filename = Path(path).name

    local = Path(request.app.state.pdf_dir) / "pdfs" / filename
    if not local.is_file():
        local = _get_pdf_path(filename)

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(local))
        if page >= len(doc):
            page = 0
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page].get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Render failed: {exc}")

    return Response(content=png_bytes, media_type="image/png")


@router.get("/pdf/pages")
async def pdf_page_count(request: Request, path: str):
    """Return total page count for a PDF."""
    filename = Path(path).name

    local = Path(request.app.state.pdf_dir) / "pdfs" / filename
    if not local.is_file():
        local = _get_pdf_path(filename)

    try:
        import fitz
        doc = fitz.open(str(local))
        return {"filename": filename, "pages": len(doc)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
