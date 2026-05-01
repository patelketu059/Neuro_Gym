from __future__ import annotations
from qdrant_client import QdrantClient
# from qdrant_client.models import Filter, FieldCondition, MatchValue
from concurrent.futures import ThreadPoolExecutor, as_completed

COLLECTIONS = ['gym_images', 'gym_text', 'gym_tables']


def dense_search(
        query_vector: list[float],
        collection: str,
        client: QdrantClient,
        top_k: int = 20,
        filters: dict | None = None
) -> list[dict]:

    # qdrant-client 1.10+ removed .search(); .query_points() is the
    # replacement universal endpoint. It returns a QueryResponse whose
    # .points field is the list of ScoredPoint (id, score, payload).
    response = client.query_points(
        collection_name = collection,
        query           = query_vector,
        limit           = top_k,
        query_filter    = filters,
        with_payload    = True,
    )

    return [
        {
            "id":         str(res.id),
            "score":      float(res.score),
            "collection": collection,
            "payload":    res.payload or {},
        }
        for res in response.points
    ]




def dense_search_all(
        query_vector: list[float],
        client: QdrantClient,
        collections: list[str] | None = None,
        top_k: int = 20,
        filters: dict | None = None
) -> list[list[dict]]:
    
    cols = collections or COLLECTIONS

    res: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers = len(cols)) as pool:
        futures = {
            pool.submit(dense_search, query_vector, col, client, top_k, filters): col 
            for col in cols
        }
        
        for fut in as_completed(futures):
            col = futures[fut]
            try:
                res[col] = fut.result()
            except Exception as e:
                print(f"[INFO-DENSE-SEARCH] - Failed {col} search: {e}")
                res[col] = []

        return [res.get(col, []) for col in cols]

