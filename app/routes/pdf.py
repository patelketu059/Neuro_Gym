from __future__ import annotations
import os
import threading
import zipfile
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response

router = APIRouter()

_PDF_CACHE_DIR = Path(os.environ.get("PDF_CACHE_DIR", "/tmp/gym-rag-pdfs"))
_EXTRACT_LOCK  = threading.Lock()   # one download/extract at a time
_ZIP_EXTRACTED = False              # module-level flag — set once per container boot


def _ensure_extracted() -> None:
    """Download pdf/pdf_archive.zip from HF Hub and extract it once per boot."""
    global _ZIP_EXTRACTED

    if _ZIP_EXTRACTED:
        return

    with _EXTRACT_LOCK:
        if _ZIP_EXTRACTED:           # double-checked inside the lock
            return

        _PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # If we already have individual PDFs from a previous boot (persistent
        # volume) skip the download entirely.
        existing = list(_PDF_CACHE_DIR.glob("athlete_*.pdf"))
        if existing:
            print(f"[PDF] Found {len(existing)} cached PDFs — skipping download.")
            _ZIP_EXTRACTED = True
            return

        hf_token     = (os.environ.get("HF_TOKEN", "") or "").strip() or None
        dataset_repo = os.environ.get("DATASET_REPO", "k2p/gym-rag-dataset")

        zip_path = _PDF_CACHE_DIR / "pdf_archive.zip"

        try:
            from huggingface_hub import hf_hub_download
            print(f"[PDF] Downloading pdf/pdf_archive.zip from {dataset_repo} …")
            hf_hub_download(
                repo_id   = dataset_repo,
                repo_type = "dataset",
                filename  = "pdf/pdf_archive.zip",
                local_dir = str(_PDF_CACHE_DIR),
                token     = hf_token,
            )
            # hf_hub_download mirrors the HF path structure:
            # _PDF_CACHE_DIR/pdf/pdf_archive.zip
            nested_zip = _PDF_CACHE_DIR / "pdf" / "pdf_archive.zip"
            if nested_zip.is_file() and not zip_path.is_file():
                zip_path = nested_zip

        except Exception as exc:
            raise RuntimeError(f"Failed to download pdf_archive.zip: {exc}") from exc

        print(f"[PDF] Extracting {zip_path} → {_PDF_CACHE_DIR} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(_PDF_CACHE_DIR)

        extracted = list(_PDF_CACHE_DIR.glob("**/*.pdf"))
        print(f"[PDF] Extracted {len(extracted)} PDFs.")

        _ZIP_EXTRACTED = True


def _get_pdf_path(filename: str) -> Path:
    """Return a local path to the PDF, extracting the archive if needed."""
    # Direct hit
    candidate = _PDF_CACHE_DIR / filename
    if candidate.is_file():
        return candidate

    # May be nested inside a subdirectory from the zip
    matches = list(_PDF_CACHE_DIR.rglob(filename))
    if matches:
        return matches[0]

    # Archive not yet extracted — do it now (first request only)
    _ensure_extracted()

    # Re-check after extraction
    candidate = _PDF_CACHE_DIR / filename
    if candidate.is_file():
        return candidate

    matches = list(_PDF_CACHE_DIR.rglob(filename))
    if matches:
        return matches[0]

    raise HTTPException(status_code=404, detail=f"PDF not found after extraction: {filename}")


@router.get("/pdf/page")
async def pdf_page(
    request: Request,
    path: str,
    page: int = 0,
    dpi: int = 150,
):
    """Render one page of an athlete PDF and return it as image/png.

    Parameters
    ----------
    path : str
        Value as returned by the /chat endpoint, e.g. ``pdfs/athlete_00250.pdf``
        or just ``athlete_00250.pdf``.
    page : int
        Zero-based page index (default 0).
    dpi : int
        Render DPI (default 150).
    """
    filename = Path(path).name   # strip any 'pdfs/' prefix

    # Local dev: data/pdfs/<filename> exists
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
    """Return the total page count for a PDF."""
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
