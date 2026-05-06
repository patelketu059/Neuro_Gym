from __future__ import annotations
from pipeline.ingestion.bm_index import bm25_search
from config.rag_config import BM25_OVERSAMPLE_ATHLETE, BM25_OVERSAMPLE_LEVEL

_VALID_LEVELS = {'elite', 'advanced', 'intermediate', 'novice'}


def sparse_search(
        query: str,
        bm25,
        corpus: list[dict],
        top_k: int = 30,
        filters: dict | None = None
) -> list[dict]:

    level_set: set[str] = set()
    if filters:
        for lvl in filters.get('training_levels', []):
            if lvl in _VALID_LEVELS:
                level_set.add(lvl)

    athlete_ids = set(filters.get('athlete_ids', [])) if filters else set()
    has_id_filter = bool(athlete_ids)
    has_level_filter = bool(level_set)

    if has_id_filter:
        fetch_k = top_k * BM25_OVERSAMPLE_ATHLETE
    elif has_level_filter:
        fetch_k = top_k * BM25_OVERSAMPLE_LEVEL
    else:
        fetch_k = top_k

    raw = bm25_search(query, bm25, corpus, top_k=fetch_k)

    if has_level_filter or has_id_filter:
        filtered: list[dict] = []
        for r in raw:
            if has_level_filter and str(r.get('training_level', '')).lower() not in level_set:
                continue
            if has_id_filter and r.get('athlete_id') not in athlete_ids:
                continue
            filtered.append(r)
        raw = filtered[:top_k]

    return [
        {
            "id": f"{r['athlete_id']}_w{r['week']}",
            "score": float(r.get('bm_score', 0)),
            "collection": "bm25",
            "payload": {
                "athlete_id":       r["athlete_id"],
                "week":             r.get("week"),
                "block_phase":      r.get("block_phase", ""),
                "text":             r.get("text", ""),
                "pdf_path":         f"pdfs/{r['athlete_id']}.pdf",
                "training_level":   r.get("training_level", ""),
                "dots":             r.get("dots", 0),
                "opl_row_index":    r.get("opl_row_index", -1),
                "primary_program":  r.get("primary_program", ""),
                "squat_peak_kg":    r.get("squat_peak_kg", 0),
                "bench_peak_kg":    r.get("bench_peak_kg", 0),
                "deadlift_peak_kg": r.get("deadlift_peak_kg", 0),
            }
        }
        for r in raw
    ]