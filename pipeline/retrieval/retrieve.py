from __future__ import annotations
from pipeline.ingestion.embedder import embed_query
from pipeline.retrieval.fusion_search import fusion_search
from pipeline.retrieval.context import deduplicate_athlete, assemble_context
from pipeline.retrieval.reranker import rerank


def retrieve(
        query: str,
        bm25,
        corpus: list[dict],
        client,
        query_image_path: str | None = None,
        reranker_model = None,
        pdf_base_dir: str | None = None,
        top_k_hybrid: int = 50,
        top_k_rerank: int = 20,
        top_k_athletes: int = 5,
        max_context_tokens: int = 4096,
        openrouter_api_key: str | None = None,
        filters: dict | None = None
) -> dict:
    

    query_vector = embed_query(
        text = query,
        image_path = query_image_path,
        api_key = openrouter_api_key
    )


    fused = fusion_search(
        query = query,
        query_vector = query_vector,
        bm25 = bm25,
        corpus = corpus,
        client = client,
        top_k = top_k_hybrid,
        filters = filters
    )

    ranked = rerank(
        query = query,
        candidates = fused,
        model = reranker_model,
        top_k = top_k_rerank,
        pdf_base_dir = pdf_base_dir
    )

    deduped = deduplicate_athlete(
        ranked,
        top_k = top_k_athletes
    )

    return assemble_context(
        deduped,
        max_tokens = max_context_tokens
    )