from __future__ import annotations
from pathlib import Path
import sys
from PIL import Image as PIL
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PDF_DIR


def load_reranker():
    import torch
    from transformers import AutoModel, AutoProcessor

    model_id = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO-Reranker] - Loading reranker {model_id} ON {device} ...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code = True
    )

    processor.max_input_tiles = 6
    processor.use_thumbnail = True

    model = AutoModel.from_pretrained(
        model_id,
        dtype = torch.float16,
        trust_remote_code = True,
        attn_implementation= 'sdpa',
        device_map = 'auto' if device == 'cuda' else None
    ).eval()

    if device == 'cuda':
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"[INFO-Reranker] - Reranker Ready | VRAM: {vram} GB")

    return {"model": model, 
            "processor": processor,
            "device": device}


def _rasterize_page(
        pdf_path: str,
        page_number: int,
        dpi: int = 200
):
    import fitz
    from PIL import Image as PIL

    doc = fitz.open(pdf_path)
    page = doc[page_number]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix = mat)
    img = PIL.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img



def _score_text_pairs(
        query: str,
        pairs: list[tuple[int, str]],
        reranker: dict
) -> dict[int, float]:
    
    import torch
    model = reranker['model']
    processor = reranker['processor']
    device = reranker['device']
    processor.p_max_length = 8192


    queries = [query] * len(pairs)
    texts = [t for _, t in pairs]
    indices = [i for i, _ in pairs]

    inputs = processor(
        text = list(zip(queries, texts)),
        return_tensors = 'pt',
        padding = True,
        truncation = True
    ).to(device)

    with torch.inference_mode():
        logits = model(**inputs).logits.squeeze(-1)

    scores = logits.float().cpu().tolist()

    return dict(zip(indices, scores))



def _score_image_pairs(
        query: str,
        pairs: list[tuple[int, "PIL"]],
        reranker: dict
) -> dict[int, float]:
    
    import torch
    model = reranker['model']
    processor = reranker['processor']
    device = reranker['device']

    processor.p_max_length = 10240


    queries = [query] * len(pairs)
    images = [img for _,img in pairs]
    indices = [i for i, _ in pairs]

    inputs = processor(
        text = queries,
        images = images,
        return_tensors = 'pt',
        padding = True,
        truncation = True
    ).to(device)

    with torch.inference_mode():
        logits = model(**inputs).logits.squeeze(-1)

    scores = logits.float().cpu().tolist()

    return dict(zip(indices, scores))

def _metadata_text(result: dict) -> str:
    p     = result.get("payload", {})
    parts = [f"Athlete {p.get('athlete_id', '')}"]

    if p.get("training_level"):
        parts.append(p["training_level"].capitalize())

    for lift in ["squat_peak_kg", "bench_peak_kg", "deadlift_peak_kg"]:
        if p.get(lift):
            name = lift.replace("_peak_kg", "").capitalize()
            parts.append(f"{name} {p[lift]:.1f}kg")

    if p.get("primary_program"):
        parts.append(f"Program: {p['primary_program']}")

    if p.get("page_number") is not None:
        parts.append(f"Page {p['page_number']}")
    return " · ".join(parts)


def rerank(
        query: str,
        candidates: list[dict],
        model = None,
        top_k: int = 10,
        pdf_base_dir: str | None = None
) -> list[dict]:
    
    if model is None:
        return [
            {**c, 'rerank_score': c.get('rrf_score', c.get('score', 0.0))}
            for c in candidates[:top_k]
        ]
    
    pool = candidates[: top_k * 3]
    if pdf_base_dir is None:
        pdf_base_dir = PDF_DIR

    image_pairs: list[tuple[int, object]] = []
    text_pairs: list[tuple[int, str]] = []

    for idx, result in enumerate(pool):
        collection = result.get('collection', '')
        payload = result.get('payload', {})

        if collection == 'gym_images':
            pdf_path = str(Path(pdf_base_dir) / Path(payload.get('pdf_path', '')).name)
            page_num = int(payload.get('page_number', 0))
            img = _rasterize_page(pdf_path, page_num)

            if img is not None:
                image_pairs.append((idx, img))
            else:
                text_pairs.append((idx, _metadata_text(result)))

        else:
            text = payload.get('text', '').strip() or _metadata_text(result)
            text_pairs.append((idx, text))

    scores: dict[int, float] = {}
    scores.update(_score_text_pairs(query, text_pairs, model))
    scores.update(_score_image_pairs(query, image_pairs, model))

    scored = [
        {**pool[idx], "rerank_score": scores.get(idx, -999.0)}
        for idx in range(len(pool))
        if idx in scores
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored[:top_k]