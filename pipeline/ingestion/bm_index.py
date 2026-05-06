from __future__ import annotations
import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from config.settings import DATA_DIR, BM_INDEX_PATH, BM_CORPUS_PATH

# Regex to extract competition lifts from gym_text coaching-summary passages.
# Matches: "Competition lifts: Squat 162.5kg  Bench 97.5kg  Deadlift 190.0kg"
_COMP_LIFT_RE = re.compile(
    r'Competition lifts:\s*Squat\s*([\d.]+)kg\s+Bench\s*([\d.]+)kg\s+Deadlift\s*([\d.]+)kg',
    re.IGNORECASE,
)


def build_athlete_peaks(gym_text_dir: Path) -> dict[str, dict]:
    """Parse competition lift peaks from gym_text .npy coaching-summary payloads.

    Iterates every *_text.npy file in *gym_text_dir*, locates the
    "Competition lifts: Squat Xkg Bench Xkg Deadlift Xkg" line embedded in the
    coaching-summary text, and returns a dict keyed by athlete_id.

    This is the authoritative source of peak lift data because the coaching
    summary text was generated from the real competition numbers even when those
    numbers were not stored as separate payload fields.
    """
    import numpy as np

    peaks: dict[str, dict] = {}
    npy_dir = Path(gym_text_dir)
    if not npy_dir.is_dir():
        return peaks

    for npy_file in sorted(npy_dir.glob('*_text.npy')):
        try:
            data = np.load(npy_file, allow_pickle=True).item()
        except Exception:
            continue
        for payload in data.get('payloads', []):
            aid = payload.get('athlete_id', '')
            if not aid or aid in peaks:
                continue
            text = payload.get('text', '')
            m = _COMP_LIFT_RE.search(text)
            if m:
                entry: dict = {
                    'squat_peak_kg':    float(m.group(1)),
                    'bench_peak_kg':    float(m.group(2)),
                    'deadlift_peak_kg': float(m.group(3)),
                }
                prog = payload.get('primary_program', '')
                if prog:
                    entry['primary_program'] = prog
                peaks[aid] = entry

    return peaks


def patch_corpus_with_peaks(corpus: list[dict], peaks: dict[str, dict]) -> int:
    """Patch BM25 corpus records in-place with competition lift peak data.

    Returns the number of records that were updated.
    """
    updated = 0
    for record in corpus:
        aid = record.get('athlete_id', '')
        if aid in peaks:
            record.update(peaks[aid])
            updated += 1
    return updated


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