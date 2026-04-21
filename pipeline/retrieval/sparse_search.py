from __future__ import annotations
from pipeline.ingestion.bm_index import bm25_search 

def sparse_search(
        query: str,
        bm25,
        corpus: list[dict],
        top_k: int = 30
) -> list[dict]:
    
    raw = bm25_search(
        query,
        bm25,
        corpus,
        top_k = top_k
    )

    return [
        {
            "id": f"{r['athlete_id']}_w{[r['week']]}"
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