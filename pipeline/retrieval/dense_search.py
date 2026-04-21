from __future__ import annotations
from qdrant_client import QdrantClient

COLLECTIONS = ['gym_images', 'gym_text', 'gym_tables']


def dense_search(
        query_vector: list[float],
        collection: str,
        client: QdrantClient,
        top_k: int = 20,
        filters: dict | None = None
) -> list[dict]:
    

    qdrant_filter = filters
    results = client.search(
        collection_name = collection,
        query_vector = query_vector,
        limit = top_k,
        query_filter = qdrant_filter,
        with_payload = True
    )

    return [
        {
            "id": str(res.id),
            "score": float(res.score),
            "collection": collection,
            "payload": res.payload or {}
        }
        for res in results
    ]




def dense_search_all(
        query_vector: list[float],
        client: QdrantClient,
        top_k: int = 20,
        filters: dict | None = None
) -> list[list[dict]]:
    
    res = []
    for col in COLLECTIONS:
        res.append(
            dense_search(
            query_vector,
            collection = col,
            top_k = top_k,
            filters = None
            ))
    return res