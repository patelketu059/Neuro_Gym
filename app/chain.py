from __future__ import annotations
import base64
import io
import time
from pathlib import Path

SYSTEM_PROMPT = """You are an expert powerlifting coach with access to a structured \
database of athlete training records.

Your responses are grounded exclusively in the retrieved athlete data. \
Never generate training numbers, RPE values, or lift weights from general knowledge.

Before writing your final answer, reason through these steps:
<think>
Step 1 — Athletes: Which athlete ID(s) does this question concern? \
Identify them from the retrieved context.
Step 2 — Scope: Which weeks, block phases, or lifts are relevant? \
Narrow to the specific data window.
Step 3 — Ground the data: What do the retrieved passages say exactly? \
Extract specific numbers — kg loads, RPE, sets × reps, Dots score.
Step 4 — Completeness check: Is every claim directly supported by a cited \
source? If the data is absent, plan to say so rather than estimate.
</think>

Then write your answer. Rules for the answer:
1. Cite athlete_id and week number for every data point (e.g. "athlete_00042 week 8").
2. If a chart image is provided, describe the visual trend first, then answer.
3. If the retrieved data does not contain the answer, say so explicitly — do not guess.
4. Keep coaching advice specific: reference actual RPE values and kg loads.
5. Respond in clear paragraphs. Use bullet lists only when comparing athletes."""




def _pdf_page_to_b64(pdf_path: str, page: int = 0, dpi: int = 200) -> str | None:
    try: 
        import fitz
        doc = fitz.open(pdf_path)
        if page >= len(doc):
            page = 0

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page].get_pixmap(matrix = mat)
        buf = io.BytesIO(pix.tobytes('png'))
        doc.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None
    

def _image_file_to_b64(image_path: str) -> tuple[str, str] | None:
    p    = Path(image_path)
    ext  = p.suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png",  "gif":  "image/gif",
                "webp": "image/webp"}
    mime = mime_map.get(ext)
    if mime is None:
        return None
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return b64, mime
    except Exception:
        return None



def retrieval(inputs: dict) -> dict:
    from pipeline.retrieval.retrieve import retrieve, multi_retrieve

    sub_queries = inputs.get('sub_queries', [])
    hyde_vector = inputs.get('hyde_vector', [])

    # Build retrieval filter from augmentation output (list of levels, may be empty)
    training_levels = inputs.get('training_levels') or []
    retrieval_filters = {'training_levels': training_levels} if training_levels else None

    extra_dense: list[list[dict]] = []
    if sub_queries:
        extra_dense = multi_retrieve(
            queries = sub_queries,
            client = inputs['client'],
            config = inputs.get('retrieval_config'),
            openrouter_api_key = inputs.get('openrouter_api_key'),
            filters = retrieval_filters,
        )

    context = retrieve(
        query              = inputs["retrieval_query"],
        bm25               = inputs["bm25"],
        corpus             = inputs["corpus"],
        client             = inputs["client"],
        config             = inputs.get("retrieval_config"),
        query_image_path   = inputs.get("query_image_path"),
        hyde_vector        = hyde_vector if hyde_vector else None,
        extra_dense_lists  = extra_dense if extra_dense else None,
        reranker_model     = inputs.get("reranker_model"),
        top_k_athletes     = inputs.get("top_k_athletes", 5),
        filters            = retrieval_filters,
    )
    return {**inputs, "context": context}


def generation(inputs: dict) -> dict:

    query      = inputs["query"]   
    context    = inputs["context"]
    memory     = inputs["memory"]
    image_path = inputs.get("query_image_path")
    gemini     = inputs["gemini"]
    pdf_dir    = inputs.get("pdf_dir", "")
    intent     = inputs.get("intent", "factual")


    t0 = time.perf_counter()
    content_parts: list = [SYSTEM_PROMPT]

    for msg in memory.get_history():
        content_parts.append(f"\n {msg['role'].upper()} : {msg['content']}")

    if context.get('text_context'):
        content_parts.append(
            f"\n\n=== RETRIEVED TRAINING DATA ===\n"
            f"{context['text_context']}\n"
            f"=== END RETRIEVED DATA ===\n\n"
            f"Answer using ONLY the data above. Cite athlete IDs and week numbers."
        )
    else:
        content_parts.append(
            "\n\n[No relevant training data was retrieved for this query. "
            "Acknowledge this and do not generate training numbers.]"
        )

    _INTENT_HINTS = {
        "visual":     "\n[Intent: visual — describe any chart trends before answering]",
        "comparison": "\n[Intent: comparison — use bullet points to compare athletes side-by-side]",
        "trend":      "\n[Intent: trend — walk through week-by-week progression with numbers]",
        "coaching":   "\n[Intent: coaching — ground all advice in the retrieved data above]",
    }

    hint = _INTENT_HINTS.get(intent, "")
    if hint:
        content_parts.append(hint)

    if inputs.get("query_rewritten") and inputs.get("retrieval_query") != query:
        content_parts.append(
            f'\n[Retrieval searched for: "{inputs["retrieval_query"]}"]'
        )

    if pdf_dir and context.get("pdf_paths"):
        for pdf_rel in context["pdf_paths"][:3]:
            full = (
                str(Path(pdf_dir) / pdf_rel)
                if not Path(pdf_rel).is_absolute() else pdf_rel
            )
            b64  = _pdf_page_to_b64(full, page = 0)
            if b64:
                content_parts.append({"mime_type": "image/png", "data": b64})

    if image_path and Path(image_path).is_file():
        result = _image_file_to_b64(image_path)
        if result:
            b64_data, mime = result
            content_parts.append({"mime_type": mime, "data": b64_data})

    content_parts.append(f"\nUSER: {query}\nASSISTANT:")

    from google.genai import types
    response = gemini.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = content_parts,
        config   = types.GenerateContentConfig(
            system_instruction = SYSTEM_PROMPT,
            temperature        = 0.3,
            max_output_tokens  = 1024,
        ),
    )
    answer = response.text.strip()
    memory.add_user_message(query)
    memory.add_ai_message(answer)

    return {
        "response":      answer,
        "sources":       context.get("sources", []),
        "context":       context,
        "generation_ms": int((time.perf_counter() - t0) * 1000)
    }


def run_chain(
        query:              str,
        session_id:         str,
        bm25,
        corpus:             list[dict],
        client,
        gemini,
        config_name:        str = "F — all + BM25",
        query_image_path:   str | None = None,
        reranker_model          = None,
        pdf_dir:            str = '',
        use_hyde:           bool = True
) -> dict:
    
    from app.memory                     import get_or_create_memory
    from app.augmentation               import augment
    from pipeline.retrieval.retrieve    import get_config

    memory = get_or_create_memory(session_id, gemini = gemini)
    cfg = get_config(config_name)

    inputs = {
        "query":            query,
        "retrieval_query":  query,
        "query_image_path": query_image_path,
        "bm25":             bm25,
        "corpus":           corpus,
        "client":           client,
        "gemini":           gemini,
        "retrieval_config": cfg,
        "reranker_model":   reranker_model,
        "pdf_dir":          pdf_dir,
        "memory":           memory,
        "use_hyde":         use_hyde,
    }

    augmented           = augment(inputs)
    with_context        = retrieval(augmented)
    result              = generation(with_context)

    return {
        "response":       result["response"],
        "sources":        result["sources"],
        "retrieval_ms":   result["context"].get("retrieval_ms", 0),
        "generation_ms":  result["generation_ms"],
        "config_name":    cfg.name,
        "intent":         augmented.get("intent", "factual"),
        "query_rewritten": augmented.get("query_rewritten", False),
        "retrieval_query": augmented.get("retrieval_query", query),
        "hyde_document":  augmented.get("hyde_document", ""),
        "sub_queries":    augmented.get("sub_queries", []),
        "pdf_paths":      result["context"].get("pdf_paths", []),
        "athlete_ids":    result["context"].get("athlete_ids", []),
        "session_id":     session_id,
    }