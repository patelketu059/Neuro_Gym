from __future__ import annotations

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


def _build_athlete_profile(payload: dict) -> str:
    aid   = payload.get("athlete_id", "unknown")
    level = payload.get("training_level", "")
    dots  = payload.get("dots", "")
    prog  = payload.get("primary_program", "")
    sq    = payload.get("squat_peak_kg", "")
    be    = payload.get("bench_peak_kg", "")
    dl    = payload.get("deadlift_peak_kg", "")

    lines = [f"ATHLETE PROFILE: {aid}"]
    meta  = []
    if level: meta.append(f"Level: {level}")
    if dots:  meta.append(f"Dots: {dots}")
    if meta:  lines.append("  " + " | ".join(meta))
    lifts = []
    if sq: lifts.append(f"Squat {sq}kg")
    if be: lifts.append(f"Bench {be}kg")
    if dl: lifts.append(f"Deadlift {dl}kg")
    if lifts: lines.append("  Competition lifts: " + " | ".join(lifts))
    if prog:  lines.append(f"  Program: {prog}")

    return "\n".join(lines)


def _passage_block(
        text: str,
        collection: str,
        payload: dict,
        score: float,
        budget_chars: int
) -> str:

    aid    = payload.get("athlete_id", "unknown")
    source = _source_label(collection, payload)
    header = (
        f"--- SOURCE: {aid} | {source} | score {score:.3f} ---"
    )
    if len(text) > budget_chars:
        text = text[:budget_chars].rsplit(" ", 1)[0] + " [truncated]"

    return f"{header}\n{text}"




def assemble_context(
        deduped_results: list[dict],
        max_tokens: int = _MAX_TEXT_TOKENS
) -> dict:
    
    text_part: list[str] = []
    pdf_paths: list[str] = []
    athlete_ids:  list[str] = []
    sources: list[dict] = []
    tokens_used: int = 0
    seen_aids: set[str] = set()

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


        if aid and aid not in seen_aids and tokens_used < max_tokens:
            profile = _build_athlete_profile(payload)
            profile_tokens = len(profile) // _CHARS_TO_TOKEN
            if tokens_used + profile_tokens < max_tokens:
                text_part.append(profile)
                tokens_used += profile_tokens
                seen_aids.add(aid)


        text = payload.get("text", "").strip()
        if text and tokens_used < max_tokens:
            budget_chars = (max_tokens - tokens_used) * _CHARS_TO_TOKEN
            block = _passage_block(text, collection, payload, score, budget_chars)
            block_tokens = len(block) // _CHARS_TO_TOKEN
            if block_tokens > 0:
                text_part.append(block)
                tokens_used += block_tokens

        source: dict = {
            'athlete_id': aid,
            'collection': collection,
            'score': round(score, 4),
            'pdf_path': pdf_path,
            'payload': {
                k: payload.get(k)
                for k in ("training_level", "dots", "squat_peak_kg",
                          "bench_peak_kg", "deadlift_peak_kg", "primary_program")
                if payload.get(k) is not None
            }
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