from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from concurrent.futures import ThreadPoolExecutor, as_completed

COLLECTIONS = ['gym_images', 'gym_text', 'gym_tables']

_VALID_LEVELS = {'elite', 'advanced', 'intermediate', 'novice'}


def _build_qdrant_filter(filters: dict | None) -> Filter | None:
    if not filters:
        return None
    conditions = []

    levels = [lvl for lvl in filters.get('training_levels', []) if lvl in _VALID_LEVELS]
    if len(levels) == 1:
        conditions.append(FieldCondition(key='training_level', match=MatchValue(value=levels[0])))
    elif len(levels) > 1:
        conditions.append(FieldCondition(key='training_level', match=MatchAny(any=levels)))

    athlete_ids = filters.get('athlete_ids', [])
    if athlete_ids:
        conditions.append(FieldCondition(key='athlete_id', match=MatchAny(any=list(athlete_ids))))

    return Filter(must=conditions) if conditions else None


def dense_search(
        query_vector: list[float],
        collection: str,
        client: QdrantClient,
        top_k: int = 20,
        filters: dict | None = None
) -> list[dict]:

    qdrant_filter = _build_qdrant_filter(filters)

    response = client.query_points(
        collection_name = collection,
        query           = query_vector,
        limit           = top_k,
        query_filter    = qdrant_filter,
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

