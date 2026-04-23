from __future__ import annotations
import os
from pathlib import Path


from config.model_settings import (
    EMBEDDING_MODEL_ID,
    DEVICE
)


def _l2_normalise(x):
    return x / (x.norm(p = 2, dim = -1, keepdim = True) + 1e-12)


def load_model():
    import torch
    from transformers import AutoModel


    model_id = EMBEDDING_MODEL_ID
    print(f"[INFO-EMBEDDER] - Loading {model_id} Model for Embedding Generation")
    print(f"[INFO-EMBEDDER] - Device: {DEVICE}")

    model = AutoModel.from_pretrained(
        model_id,
        dtype = torch.float16,
        trust_remote_code = True,
        attn_implementation = 'sdpa',
        device_map = 'auto' if DEVICE == 'cuda' else None,
    ).eval()

    model.processor.p_max_length = 10240
    model.processor.max_input_tiles = 6
    model.processor.use_thumbnail = True


    if DEVICE == 'cuda':
        vram_used  = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Model ready  |  VRAM: {vram_used:.1f}/{vram_total:.1f} GB")
    else:
        print("  Model ready  |  Running on CPU")

    return model

def embed_pdf_pages_batch(
        doc,
        page_nums: int,
        model,
        dpi: int = 200
) -> list[list[float]]:
    

    import torch
    import fitz
    from PIL import Image
    model.processor.p_max_length = 10240

    images: list = []
    texts: list[str] = []

    for page_num in page_nums:
        page = doc[page_num]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix = matrix)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        images.append(img)
        texts.append(page.get_text().strip())

    has_text = any(texts)

    with torch.inference_mode():
        if has_text:
            vecs = model.encode_documents(images = images, texts = texts)
        else:
            vecs = model.encode_documents(images = images)

    vecs = _l2_normalise(vecs)
    return vecs.float().cpu().tolist()



def embed_pil_batch(
        images: list,
        texts: list[str],
        model,
) -> list[list[float]]:
    
    import torch
    model.processor.p_max_length = 10240
    has_text = any(t.strip() for t in texts)

    with torch.inference_mode():
        if has_text:
            vecs = model.encode_documents(images=images, texts=texts)
        else:
            vecs = model.encode_documents(images=images)

    vecs = _l2_normalise(vecs)
    return vecs.float().cpu().tolist()


def embed_text_batch(
        texts: list[str],
        model,
        batch_size: int = 32
) -> list[list[float]]:
    
    import torch

    if not texts: return []
    model.processor.p_max_length = 8192
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.inference_mode():
            vecs = model.encode_documents(texts = batch)
        vecs = _l2_normalise(vecs)
        all_vecs.extend(vecs.float().cpu().tolist())

    return all_vecs


def embed_query_api(
        text: str,
        image_path: str | None = None,
        api_key: str | None = None,
        mode: str = 'query'
) -> list[float]:
    
    import requests
    import base64

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key: 
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set"
        )
    
    if image_path is None:
        payload = {
            "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
            "input": f"{mode}: {text}",
            "encoding_format": "float"
        }
    else:
        image_data   = Path(image_path).read_bytes()
        b64          = base64.b64encode(image_data).decode("utf-8")
        ext          = Path(image_path).suffix.lower().lstrip(".")
        mime_map     = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                        "gif": "gif",  "webp": "webp"}
        mime         = f"image/{mime_map.get(ext, 'jpeg')}"
        data_url     = f"data:{mime};base64,{b64}"

        payload = {
            "model":            "nvidia/llama-nemotron-embed-vl-1b-v2:free",
            "encoding_format":  "float",
            "input": [
                {
                    "content": [
                        {"type": "text",
                         "text": f"{mode}: {text}"},
                        {"type": "image_url",
                         "image_url": {"url": data_url}},
                    ]
                }
            ],
        }

        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers = {
                "Authorization": {f"Bearer {api_key}"},
                "Content-Type": "application/json"
            },
            json = payload,
            timeout = 30
        )

        resp.raise_for_status()
        return resp.json()["data"][0]['embedding']


def embed_query(
    text: str,
    image_path: str | None = None,
    api_key: str | None = None,
    mode: str = 'query'
) -> list[float]:

    return embed_query_api(
        text, 
        image_path = image_path,
        api_key = api_key)