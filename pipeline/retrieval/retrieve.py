from __future__ import annotations
import time 
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


TEXT_COLLECTIONS  = ["gym_text", "gym_tables"]
IMAGE_COLLECTIONS = ["gym_images"]
ALL_COLLECTIONS   = ["gym_images", "gym_text", "gym_tables"]

@dataclass
class RetrievalConfig:
    name: str
    collections: list[str]
    use_bm25: bool
    use_reranker: bool = False

CONFIGS: dict[str, RetrievalConfig] = {
    "A — images only":      RetrievalConfig("A — images only",      ["gym_images"],                  use_bm25=False),
    "B — text only":        RetrievalConfig("B — text only",         ["gym_text"],                    use_bm25=False),
    "C — tables only":      RetrievalConfig("C — tables only",       ["gym_tables"],                  use_bm25=False),
    "D — all dense":        RetrievalConfig("D — all dense",         ALL_COLLECTIONS,                 use_bm25=False),
    "E — tables + BM25":    RetrievalConfig("E — tables + BM25",     ["gym_tables"],                  use_bm25=True),
    "F — all + BM25":       RetrievalConfig("F — all + BM25",        ALL_COLLECTIONS,                 use_bm25=True),
    "G — hybrid + rerank":  RetrievalConfig("G — hybrid + rerank",   ALL_COLLECTIONS,                 use_bm25=True,  use_reranker=True),
    "H — BM25 only":        RetrievalConfig("H — BM25 only",         [],                              use_bm25=True),
}

DEFAULT_CONFIG = "F — all + BM25"  # em-dash, matches CONFIGS keys
def get_config(name: str) -> RetrievalConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Valid: {list(CONFIGS)}")
    return CONFIGS[name]


def multi_retrieve(
        queries: list[str],
        client,
        config: RetrievalConfig | None = None,
        openrouter_api_key: str | None = None,
        top_k: int = 30,
        filters: dict | None = None
) -> list[list[dict]]:
    
    from pipeline.ingestion.embedder import embed_query
    from pipeline.retrieval.dense_search import dense_search_all

    cfg = config or CONFIGS[DEFAULT_CONFIG]
    text_collections = [c for c in cfg.collections if c!= 'gym_images']
    if not text_collections: return []

    def _search_one(q: str) -> list[dict]:
        vec = embed_query(q, api_key = openrouter_api_key)
        if not vec: return []
        results = dense_search_all(
            query_vector = vec,
            client = client,
            collections = text_collections,
            top_k = top_k,
            filters = filters
        )

        flat: list[dict] = []
        for r in results:
            flat.extend(r)
        return flat
    
    results_list: list[list[dict]] = []
    with ThreadPoolExecutor(max_workers = min(len(queries), 4)) as pool:
        futures = {pool.submit(_search_one, q) : q for q in queries}
        for fut in as_completed(futures):
            try:
                hits = fut.result()
                if hits: 
                    results_list.append(hits)
            except Exception as e: 
                print(f"[INFO-Retrieve] - Multi Retreive sub-query failure: {e}")

    return results_list


def retrieve(
        query: str,
        bm25,
        corpus: list[dict],
        client,
        config: RetrievalConfig | None              = None,
        query_image_path: str | None                = None,
        hyde_vector: list[float] | None             = None,
        extra_dense_lists: list[list[dict]] | None  = None,
        reranker_model                              = None,
        top_k_hybrid: int                           = 50,
        top_k_rerank: int                           = 20,
        top_k_athletes: int                         = 5,
        max_context_tokens: int                     = 4096,
        openrouter_api_key: str | None              = None,
        filters: dict | None                        = None
) -> dict:
    
    from pipeline.ingestion.embedder import embed_query
    from pipeline.retrieval.dense_search import dense_search_all, dense_search
    from pipeline.retrieval.sparse_search import sparse_search
    from pipeline.retrieval.fusion_search import RRF 
    from pipeline.retrieval.context import deduplicate_athlete, assemble_context
    from pipeline.retrieval.reranker import rerank


    cfg = config or CONFIGS[DEFAULT_CONFIG]
    t0 = time.perf_counter()

    text_collections = [c for c in cfg.collections if c != "gym_images"]
    image_collections_in_cfg = [c for c in cfg.collections if c == "gym_images"]

    text_vector: list[float] = []
    if text_collections or (image_collections_in_cfg and not query_image_path):
        if cfg.collections:
            text_vector = embed_query(
                query,
                api_key = openrouter_api_key,
            )

    image_vector: list[float] = []
    if query_image_path and image_collections_in_cfg:
        try:
            image_vector = embed_query(
                query,
                image_path = query_image_path,
                api_key = openrouter_api_key
            )
        except Exception as e:
            print(f'[INFO-Retrieve] - Imaged Embed skipped: {e}')
            image_vector = text_vector

    # HyDE vector (passage-mode embedding of a synthesised training-record
    # passage) lives in document space; text_vector is in query space. When
    # HyDE is present, prefer it for searching gym_text/gym_tables — that's
    # the entire point of HyDE.
    search_vector = hyde_vector if hyde_vector else text_vector
    dense_results: list[list[dict]] = []

    if text_collections and search_vector:
        text_results = dense_search_all(
            query_vector = search_vector,
            client = client,
            collections = text_collections,
            top_k = top_k_hybrid,
            filters = filters,
        )
        dense_results.extend(text_results)

    if image_collections_in_cfg:
        vec_image = image_vector if image_vector else text_vector
        if vec_image:
            img_results = dense_search(
                query_vector = vec_image,
                collection = 'gym_images',
                client = client,
                top_k = top_k_hybrid,
                filters = filters
            )

            if img_results:
                dense_results.append(img_results)

    if extra_dense_lists:
        dense_results.extend(extra_dense_lists)

        
    bm25_results: list[dict] = []
    if cfg.use_bm25:
        bm25_results = sparse_search(query, bm25, corpus,
                                     top_k = top_k_hybrid,
                                     filters = filters)
        
    all_lists = dense_results + ([bm25_results])
    fused = RRF(all_lists)[:top_k_hybrid]
    # fused = fusion_search(
    #     query = query,
    #     query_vector = query_vector,
    #     bm25 = bm25,
    #     corpus = corpus,
    #     client = client,
    #     top_k = top_k_hybrid,
    #     filters = filters
    # )

    ranked = rerank(
        query = query,
        candidates = fused,
        model = reranker_model if cfg.use_reranker else None,
        top_k = top_k_rerank,
    )

    deduped = deduplicate_athlete(
        ranked,
        top_k = top_k_athletes
    )

    context = assemble_context(deduped, max_tokens = max_context_tokens)
    context['retrieval_ms'] = int((time.perf_counter() - t0) * 1000)
    context['config_name'] = cfg.name

    return context