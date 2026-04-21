from __future__ import annotations

from pipeline.retrieval.dense_search import dense_search_all
from pipeline.retrieval.sparse_search import sparse_search



def RRF(
        results_lists: list[list[dict]],
        k: int = 60
) -> list[dict]:
    
    # Reciprocal rank fusion
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}

    for RL in results_lists:
        for rank, result in enumerate(RL, start = 1):
            rid = result['id']
            rrf_contrib = 1.0 / (k + rank)
            scores[rid] = scores.get(rid, 0.0) + rrf_contrib

            if rid not in best or result['scores'] > best['rid']['score']:
                best[rid] = result

    merged = sorted(scores.keys(), key = lambda x: scores[x], reverse = True)
    return [
        {**best[rid],
         "rrf_score": scores[rid]
        }
        for rid in merged
    ]



def fusion_search(
        query: str,
        query_vector: list[float],
        bm25,
        corpus: list[dict],
        client,
        top_k: int = 50,
        filters: dict | None = None
) -> list[dict]:
    
    dense_results = dense_search_all(
        query_vector,
        client,
        top_k = top_k,
        filters = filters
        )
    
    bm25_results = sparse_search(
        query, 
        bm25,
        corpus,
        top_k = top_k
    )

    all_lists = dense_results + [bm25_results]
