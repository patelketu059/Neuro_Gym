from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.model_settings import RETRIEVAL_CONFIGS, COLLECTIONS

def retrieve_for_config(
        query: str,
        config: dict,
        client,
        bm25,
        corpus: list[dict],
        reranker_model = None,
        top_k: int = 20
) -> list[str]:
    
    from pipeline.ingestion.embedder import embed_query
    from pipeline.retrieval.dense_search import dense_search
    from pipeline.retrieval.sparse_search import sparse_search
    from pipeline.retrieval.fusion_search import RRF
    from pipeline.retrieval.reranker import rerank
    from pipeline.retrieval.context import deduplicate_athlete

    query_vector = embed_query(query)

    result_lists = []
    for col in config['collections']:
        results = dense_search(query_vector, col, client, top_k = top_k)
        result_lists.append(results)

    if config['use_bm25']:
        bm25_res = sparse_search(query, bm25, corpus, top_k = top_k)
        result_lists.append(bm25_res)

    if len(result_lists) > 1: 
        fused = RRF(result_lists)
    elif len(result_lists) == 1:
        fused = result_lists[0]
    else: 
        return []

    model = reranker_model if config['reranker'] else None
    ranked = rerank(query, fused, model = model, top_k = top_k)

    deduped = deduplicate_athlete(ranked, top_k = top_k)
    return [
        r.get('payload', {}).get("athlete_id", '') 
        for r in deduped
        if r.get('payload', {}).get('athlete_id')
    ]



def reciprocal_rank(
        athlete_ids: list[str],
        correct: str,
        k: int
) -> float:
    
    for rank, aid in enumerate(athlete_ids[:k], start=1):
        if aid == correct:
            return 1.0 / rank
    return 0.0


def hit(athlete_ids: list[str], correct: str, k: int) -> int:
    return int(correct in athlete_ids[:k])


def run(
        configs_run: list[str],
        golden_path: Path,
        k_values: list[int],
        include_reranker: bool,
        qdrant_host: str,
        qdrant_port: int
) -> None:
    raise NotImplementedError(
        "run() is a placeholder — use eval/retrieval_eval.py for the full harness."
    )
