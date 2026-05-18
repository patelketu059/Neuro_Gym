from __future__ import annotations
import base64
import functools
import hashlib
import io
import json as _json
import logging
import time
from pathlib import Path
from typing import Generator

log = logging.getLogger(__name__)

from config.model_settings import GEMINI_GENERATION_MODEL, GEMINI_FALLBACK_MODEL, GEMINI_FALLBACK_CHAIN
from config.rag_config import (
    GENERATION_TEMPERATURE, GENERATION_MAX_TOKENS,
    THINKING_INTENTS, THINKING_BUDGET, THINKING_TEMPERATURE, THINKING_MAX_TOKENS,
)

# Models that support extended thinking via ThinkingConfig (Gemini 2.5+).
_THINKING_CAPABLE = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"}


def _should_try_next(exc: Exception) -> bool:
    """Return True when the chain should skip to the next model rather than raising.

    Covers two cases:
    - Rate / quota exhaustion (429, RESOURCE_EXHAUSTED): transient, next model may have headroom.
    - Model not found (404, NOT_FOUND): model was deprecated; skip it silently.
    """
    msg = str(exc)
    return (
        "429"                in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "rate_limit"      in msg.lower()
        or "rate limit"      in msg.lower()
        or "quota"           in msg.lower()
        or "404"             in msg
        or "NOT_FOUND"       in msg
    )

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


_INTENT_HINTS = {
    "visual":     "\n[Intent: visual — describe any chart trends before answering]",
    "comparison": "\n[Intent: comparison — use bullet points to compare athletes side-by-side]",
    "trend":      "\n[Intent: trend — walk through week-by-week progression with numbers]",
    "coaching":   "\n[Intent: coaching — ground all advice in the retrieved data above]",
}


@functools.lru_cache(maxsize=64)
def _pdf_page_to_b64(pdf_path: str, page: int = 0, dpi: int = 200) -> str | None:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if page >= len(doc):
            page = 0
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page].get_pixmap(matrix=mat)
        buf = io.BytesIO(pix.tobytes('png'))
        doc.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


def _image_file_to_b64(image_path: str) -> tuple[str, str] | None:
    p = Path(image_path)
    ext = p.suffix.lower().lstrip(".")
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


def _build_content_parts(inputs: dict) -> list:
    """Assemble multimodal content parts for Gemini from pipeline state dict."""
    from google.genai import types

    query      = inputs["query"]
    context    = inputs["context"]
    memory     = inputs["memory"]
    image_path = inputs.get("query_image_path")
    pdf_dir    = inputs.get("pdf_dir", "")
    intent     = inputs.get("intent", "factual")

    parts: list = []

    for msg in memory.get_history():
        parts.append(f"\n {msg['role'].upper()} : {msg['content']}")

    if context.get('text_context'):
        parts.append(
            f"\n\n=== RETRIEVED TRAINING DATA ===\n"
            f"{context['text_context']}\n"
            f"=== END RETRIEVED DATA ===\n\n"
            f"Answer using ONLY the data above. Cite athlete IDs and week numbers."
        )
    else:
        parts.append(
            "\n\n[No relevant training data was retrieved for this query. "
            "Acknowledge this and do not generate training numbers.]"
        )

    hint = _INTENT_HINTS.get(intent, "")
    if hint:
        parts.append(hint)

    if inputs.get("query_rewritten") and inputs.get("retrieval_query") != query:
        parts.append(f'\n[Retrieval searched for: "{inputs["retrieval_query"]}"]')

    if pdf_dir and context.get("pdf_paths"):
        sources = context.get("sources", [])

        # Only gym_images payloads carry correct page_number; gym_tables/text default to 0.
        image_pages: dict[str, list[int]] = {}
        for s in sources:
            if s.get("collection") == "gym_images" and s.get("pdf_path"):
                pdf_rel = s["pdf_path"]
                pg = int(s.get("page_number") or 0)
                image_pages.setdefault(pdf_rel, [])
                if pg not in image_pages[pdf_rel]:
                    image_pages[pdf_rel].append(pg)

        total_pages = 0
        for pdf_rel in context["pdf_paths"][:3]:  # cap: 3 athletes
            full = (
                str(Path(pdf_dir) / pdf_rel)
                if not Path(pdf_rel).is_absolute() else pdf_rel
            )
            # Use all gym_images-retrieved pages for this PDF; fall back to page 0
            pages_to_render = sorted(image_pages.get(pdf_rel, [0]))[:4]
            for page_num in pages_to_render:
                if total_pages >= 6:  # hard cap: 6 pages total
                    break
                b64 = _pdf_page_to_b64(full, page=page_num)
                if b64:
                    parts.append(types.Part.from_bytes(
                        data=base64.b64decode(b64), mime_type="image/png"
                    ))
                    total_pages += 1

    if image_path and Path(image_path).is_file():
        result = _image_file_to_b64(image_path)
        if result:
            b64_data, mime = result
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(b64_data), mime_type=mime
            ))

    parts.append(f"\nUSER: {query}\nASSISTANT:")
    return parts


def _build_gen_config(intent: str, model: str = GEMINI_GENERATION_MODEL):
    from google.genai import types

    # Extended thinking is only supported on Gemini 2.5+ models.
    supports_thinking = model in _THINKING_CAPABLE
    use_thinking = supports_thinking and intent in THINKING_INTENTS
    kwargs: dict = dict(
        system_instruction = SYSTEM_PROMPT,
        temperature        = THINKING_TEMPERATURE if use_thinking else GENERATION_TEMPERATURE,
        max_output_tokens  = THINKING_MAX_TOKENS  if use_thinking else GENERATION_MAX_TOKENS,
    )
    if use_thinking:
        kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    return types.GenerateContentConfig(**kwargs)


def _model_order() -> list[str]:
    """Return the full fallback chain, deduplicated, preserving order.

    Starts with GEMINI_GENERATION_MODEL then works through GEMINI_FALLBACK_CHAIN.
    To add a paid last-resort, append it to GEMINI_FALLBACK_CHAIN in model_settings.py.
    """
    seen: set[str] = set()
    result: list[str] = []
    for m in [GEMINI_GENERATION_MODEL] + GEMINI_FALLBACK_CHAIN:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


# Retrieval cache keyed by (session_id, query, config, athlete_ids, levels). TTL = 5 min.
_RETRIEVAL_CACHE: dict[str, tuple[float, dict]] = {}
_RETRIEVAL_TTL: float = 300.0


def _retrieval_cache_key(inputs: dict) -> str:
    parts = (
        inputs.get("session_id", ""),
        inputs.get("retrieval_query", ""),
        inputs.get("retrieval_config") and inputs["retrieval_config"].name or "",
        ",".join(sorted(inputs.get("athlete_ids") or [])),
        ",".join(sorted(inputs.get("training_levels") or [])),
    )
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def retrieval(inputs: dict) -> dict:
    from pipeline.retrieval.retrieve import retrieve, multi_retrieve

    cache_key = _retrieval_cache_key(inputs)
    now = time.monotonic()
    if cache_key in _RETRIEVAL_CACHE:
        ts, cached_context = _RETRIEVAL_CACHE[cache_key]
        if now - ts < _RETRIEVAL_TTL:
            log.debug("[chain] retrieval cache hit")
            return {**inputs, "context": cached_context}
        del _RETRIEVAL_CACHE[cache_key]

    sub_queries = inputs.get('sub_queries', [])
    hyde_vector = inputs.get('hyde_vector', [])

    training_levels = inputs.get('training_levels') or []
    athlete_ids     = inputs.get('athlete_ids') or []

    retrieval_filters: dict | None = {}
    if training_levels:
        retrieval_filters['training_levels'] = training_levels
    if athlete_ids:
        retrieval_filters['athlete_ids'] = athlete_ids
    retrieval_filters = retrieval_filters or None

    extra_dense: list[list[dict]] = []
    if sub_queries:
        extra_dense = multi_retrieve(
            queries            = sub_queries,
            client             = inputs['client'],
            config             = inputs.get('retrieval_config'),
            openrouter_api_key = inputs.get('openrouter_api_key'),
            filters            = retrieval_filters,
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
        intent             = inputs.get('intent', 'factual'),
    )
    _RETRIEVAL_CACHE[cache_key] = (time.monotonic(), context)
    return {**inputs, "context": context}


def generation(inputs: dict) -> dict:
    query   = inputs["query"]
    context = inputs["context"]
    memory  = inputs["memory"]
    gemini  = inputs["gemini"]
    intent  = inputs.get("intent", "factual")

    t0            = time.perf_counter()
    content_parts = _build_content_parts(inputs)

    models   = _model_order()
    response = None
    for i, model in enumerate(models):
        is_last = (i == len(models) - 1)
        try:
            response = gemini.models.generate_content(
                model    = model,
                contents = content_parts,
                config   = _build_gen_config(intent, model),
            )
            break
        except Exception as exc:
            if is_last or not _should_try_next(exc):
                raise
            log.warning("[chain] %s exhausted — trying %s next", model, models[i + 1])

    answer = response.text.strip()

    _um             = response.usage_metadata
    input_tokens    = getattr(_um, "prompt_token_count",     0) or 0
    output_tokens   = getattr(_um, "candidates_token_count", 0) or 0
    thinking_tokens = getattr(_um, "thoughts_token_count",   0) or 0

    memory.add_user_message(query)
    memory.add_ai_message(answer)

    return {
        "response":        answer,
        "sources":         context.get("sources", []),
        "context":         context,
        "generation_ms":   int((time.perf_counter() - t0) * 1000),
        "input_tokens":    input_tokens,
        "output_tokens":   output_tokens,
        "thinking_tokens": thinking_tokens,
    }


def _make_inputs(
    query, session_id, bm25, corpus, client, gemini,
    config_name, query_image_path, reranker_model, pdf_dir, use_hyde,
) -> tuple[dict, object]:
    from app.memory                  import get_or_create_memory
    from app.augmentation            import augment
    from pipeline.retrieval.retrieve import get_config

    memory = get_or_create_memory(session_id, gemini=gemini)
    cfg    = get_config(config_name)

    base = {
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
    augmented    = augment(base)
    with_context = retrieval(augmented)
    return with_context, cfg


def run_chain(
        query:            str,
        session_id:       str,
        bm25,
        corpus:           list[dict],
        client,
        gemini,
        config_name:      str = "F — all + BM25",
        query_image_path: str | None = None,
        reranker_model=None,
        pdf_dir:          str = '',
        use_hyde:         bool = True,
) -> dict:

    with_context, cfg = _make_inputs(
        query, session_id, bm25, corpus, client, gemini,
        config_name, query_image_path, reranker_model, pdf_dir, use_hyde,
    )
    result = generation(with_context)

    return {
        "response":        result["response"],
        "sources":         result["sources"],
        "retrieval_ms":    result["context"].get("retrieval_ms", 0),
        "generation_ms":   result["generation_ms"],
        "config_name":     cfg.name,
        "intent":          with_context.get("intent", "factual"),
        "query_rewritten": with_context.get("query_rewritten", False),
        "retrieval_query": with_context.get("retrieval_query", query),
        "hyde_document":   with_context.get("hyde_document", ""),
        "sub_queries":     with_context.get("sub_queries", []),
        "pdf_paths":       result["context"].get("pdf_paths", []),
        "athlete_ids":     result["context"].get("athlete_ids", []),
        "session_id":      session_id,
        "input_tokens":    result.get("input_tokens", 0),
        "output_tokens":   result.get("output_tokens", 0),
        "thinking_tokens": result.get("thinking_tokens", 0),
        "text_context":    result["context"].get("text_context", ""),
    }


def run_chain_stream(
        query:            str,
        session_id:       str,
        bm25,
        corpus:           list[dict],
        client,
        gemini,
        config_name:      str = "F — all + BM25",
        query_image_path: str | None = None,
        reranker_model=None,
        pdf_dir:          str = '',
        use_hyde:         bool = True,
) -> Generator[str, None, None]:
    """Yield SSE-ready strings: JSON text tokens then a final ``{"__done__": true}`` metadata dict.

    A ``{"__ping__": true}`` heartbeat is emitted immediately so the client
    knows the connection is alive while retrieval (5–10 s) runs.
    """
    # Heartbeat: keeps the SSE connection alive through reverse-proxy idle
    # timeouts while augmentation + retrieval run synchronously below.
    yield _json.dumps({"__ping__": True})

    with_context, cfg = _make_inputs(
        query, session_id, bm25, corpus, client, gemini,
        config_name, query_image_path, reranker_model, pdf_dir, use_hyde,
    )

    intent        = with_context.get("intent", "factual")
    content_parts = _build_content_parts(with_context)
    memory        = with_context["memory"]
    context       = with_context["context"]

    t0          = time.perf_counter()
    full_answer = ""
    total_input_tokens    = 0
    total_output_tokens   = 0
    total_thinking_tokens = 0

    models = _model_order()
    for i, model in enumerate(models):
        is_last = (i == len(models) - 1)
        try:
            for chunk in gemini.models.generate_content_stream(
                model    = model,
                contents = content_parts,
                config   = _build_gen_config(intent, model),
            ):
                if chunk.text:
                    full_answer += chunk.text
                    yield _json.dumps(chunk.text)
                if chunk.usage_metadata:  # populated on the final chunk
                    _um = chunk.usage_metadata
                    total_input_tokens    = getattr(_um, "prompt_token_count",     0) or 0
                    total_output_tokens   = getattr(_um, "candidates_token_count", 0) or 0
                    total_thinking_tokens = getattr(_um, "thoughts_token_count",   0) or 0
            break  # stream completed successfully
        except Exception as exc:
            if is_last or not _should_try_next(exc):
                raise
            log.warning("[chain] %s exhausted — trying %s next", model, models[i + 1])
            full_answer = ""  # discard any partial tokens already yielded

    generation_ms = int((time.perf_counter() - t0) * 1000)
    memory.add_user_message(query)
    memory.add_ai_message(full_answer)

    meta = {
        "__done__":        True,
        "sources":         context.get("sources", []),
        "pdf_paths":       context.get("pdf_paths", []),
        "athlete_ids":     context.get("athlete_ids", []),
        "retrieval_ms":    context.get("retrieval_ms", 0),
        "generation_ms":   generation_ms,
        "config_name":     cfg.name,
        "intent":          intent,
        "query_rewritten": with_context.get("query_rewritten", False),
        "retrieval_query": with_context.get("retrieval_query", query),
        "hyde_document":   with_context.get("hyde_document", ""),
        "sub_queries":     with_context.get("sub_queries", []),
        "session_id":      session_id,
        "input_tokens":    total_input_tokens,
        "output_tokens":   total_output_tokens,
        "thinking_tokens": total_thinking_tokens,
    }
    yield _json.dumps(meta)
