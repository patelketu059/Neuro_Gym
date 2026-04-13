from __future__ import annotations
import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from config.settings import DATA_DIR, BM_INDEX_PATH, BM_CORPUS_PATH


def _tokenize(text: str) -> list[str]: 
    text = text.lower()
    tokens = re.findall(r"[\w.x]+", text)
    return [t for t in tokens if re.search(r"[a-z0-9]", t)]


def build_bm_index(
        corpus_records: list[dict],
        index_path: Path = BM_INDEX_PATH,
        corpus_path: Path = BM_CORPUS_PATH
) -> None:
    
    texts = [r['text'] for r in corpus_records]
    tokenized = [_tokenize(t) for t in texts]

    bm25 = BM25Okapi(tokenized)
    corpus_path.parent.mkdir(parents = True, exist_ok = True)
    with open(corpus_path, 'w', encoding = 'utf-8') as f:
        json.dump(corpus_records, f, ensure_ascii = False)

    with open(index_path, 'wb') as f:
        pickle.dump(bm25, f)

    print(f"[INFO-BM25] - Corpus saved -> {corpus_path} - {len(corpus_records)} records")
    print(f"[INFO-BN25] - Index saved -> {index_path}")



def load_bm_index(
        index_path: Path = BM_INDEX_PATH,
        corpus_path: Path = BM_CORPUS_PATH
) -> tuple:
    
    for path in [index_path, corpus_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f"BM25 artifact missing {path}", 
                f"Run ingestion pipeline"
            )

    with open(index_path, 'rb') as f:
        bm25 = pickle.load(f)

    with open(corpus_path, encoding = 'utf-8') as f:
        corpus = json.load(f)

    print(f"[INFO-BM25] - Index and Corpus loaded ({len(corpus)} documents)")
    return bm25, corpus


def bm25_search(
        query: str,
        bm25, 
        corpus:  list[dict],
        top_k: int = 30
) -> list[dict]:

    tokens = _tokenize(query)
    if not tokens: return []

    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] <= 0: break
        entry = dict(corpus[idx])
        entry['bm_score'] = float(scores[idx])
        entry['collection'] = 'bm25'
        results.append(entry)

    return results