from __future__ import annotations
from pipeline.ingestion.bm_index import bm25_search

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

    # Fetch more candidates so post-filtering still yields top_k hits
    fetch_k = top_k * 4 if level_set else top_k
    raw = bm25_search(query, bm25, corpus, top_k=fetch_k)

    if level_set:
        raw = [r for r in raw if str(r.get('training_level', '')).lower() in level_set]
        raw = raw[:top_k]

    return [
        {
            "id": f"{r['athlete_id']}_w{[r['week']]}",
            "score": float(r.get('bm25_score', 0)),
            "collection": "bm25",
            "payload": {
                "athlete_id":      r["athlete_id"],
                "week":            r.get("week"),
                "block_phase":     r.get("block_phase", ""),
                "text":            r.get("text", ""),
                "pdf_path":        f"pdfs/{r['athlete_id']}.pdf",
                "training_level":  r.get("training_level", ""),
                "dots":            r.get("dots", 0),
                "opl_row_index":   r.get("opl_row_index", -1),
                "primary_program": r.get("primary_program", ""),
            }
        }
        for r in raw
    ]