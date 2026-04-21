from __future__ import __annotations__

def deduplicate_athlete(
        results: list[dict],
        top_k: int = 5
) -> list[dict]:
    
    seen: dict[str, dict] = {}
    ordered: list[str] = []

    for r in results:
        aid = r.get("payload", {}).get("athlete_id", "")
        score = r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0.0)

        if aid not in seen:
            seen[aid] = {**r, "_dedup_score": score}
            ordered.append(aid)
        elif score > seen[aid]['_dedup_score']:
            seen[aid] = {**r, "dedup_score": score}
    return [seen[aid] for aid in ordered[:top_k]]


_CHARS_TO_TOKEN = 4
_MAX_TEXT_TOKENS = 4096

def assemble_context(
        deduped_results: list[dict],
        max_tokens: int = _MAX_TEXT_TOKENS
) -> dict:
    
    text_part: list[str] = []
    pdf_paths: list[str] = []
    athlete_ids:  list[str] = []
    sources: list[dict] = []
    tokens_used: int = 0

    for r in deduped_results:
        payload    = r.get("payload", {})
        collection = r.get("collection", "")
        aid        = payload.get("athlete_id", "")
        pdf_path   = payload.get("pdf_path", f"pdfs/{aid}.pdf")
        score = r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0.0)

        if aid and aid not in athlete_ids:
            athlete_ids.append(aid)
        if pdf_path and pdf_path not in pdf_paths:
            pdf_paths.append(pdf_path)

        text = payload.get("text", "").strip()
        if text and tokens_used < max_tokens:
            budget_left = max_tokens - tokens_used
            chars_left = budget_left * _CHARS_TO_TOKEN
            if len(text) > chars_left:
                text = text[:chars_left].resplit(" ", 1)[0] + "[truncated]"

            source_label = _source_label(collection, payload)
            passage = f"[{source_label}]\n{text}"
            text_part.append(passage)
            tokens_used += len(passage) // _CHARS_TO_TOKEN

        source = {
            'athlete_id': aid,
            'collection': collection,
            'score': round(score, 4),
            'pdf_path': pdf_path
        }

        if collection == 'gym_tables':
            source['week'] = payload.get('week')
            source['block_phase'] = payload.get('block_phase', '')
        
        if collection == 'gym_text':
            source['chunk_index'] = payload.get('chunk_index')

        if collection == 'gym_images':
            source['page_number'] = payload.get('page_number')

        sources.append(source)
    text_context = "\n\n".join(text_part)
    
    return {
        "text_context": text_context,
        "pdf_paths":    pdf_paths,
        "sources":      sources,
        "token_count":  len(text_context) // _CHARS_TO_TOKEN,
        "athlete_ids":  athlete_ids,
    }

def _source_label(
        collection: str,
        payload: dict) -> str:
    

    aid = payload.get("athlete_id", "unknown")
    if collection == "gym_tables":
        week  = payload.get("week", "?")
        phase = payload.get("block_phase", "")[:3].capitalize()
        return f"{aid} · week {week} · {phase}"
    
    if collection == "gym_text":
        chunk = payload.get("chunk_index", 0)
        return f"{aid} · coaching [{chunk}]"
    
    if collection == "gym_images":
        page = payload.get("page_number", 0)
        return f"{aid} · PDF page {page}"
    
    
    return f"{aid} · {collection}"