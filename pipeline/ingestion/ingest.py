from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SESSIONS_PATH, BLOCK_SUMMARY_PATH

from pipeline.ingestion.chunking import build_all_nl_strings #, optimized_build_all_nl_strings
from pipeline.ingestion.bm_index import build_bm_index
from pipeline.ingestion.collection import (
    get_client,
    create_collections, 
    verify_collections
)


def run_local(
        sessions_path: Path,
        summary_path: Path,
        recreate_collections: bool = False,
        qdrant_host: str = 'localhost',
        qdrant_port: int = 6333,
) -> None:
    
    print(f"[INFO-QDRANT] - Intitializing Connection -> Host: {qdrant_host}   |   Port: {qdrant_port}")
    client = get_client(qdrant_host, qdrant_port)
    create_collections(client, recreate = recreate_collections)
    print(f"[INFO-QDRANT] Completed QDRANT Initialization.")


    print(f"\n [INFO-DATA] Loading Sessions and Summary...")
    sessions_df = pd.read_csv(sessions_path)
    summary_df = pd.read_csv(summary_path)
    n_athletes = sessions_df['athlete_id'].nunique()
    print(f"[---] Sessions Shape: {sessions_df.shape}   |   {n_athletes} athletes")
    print(f"[---] Summary Shape: {summary_df.shape}" )

    corpus_records = build_all_nl_strings(sessions_df)
    # corpus_records = optimized_build_all_nl_strings(sessions_df)
    build_bm_index(corpus_records)

    print(f"[INFO-BM] - Built BM25 Data locally")
    print(f"Ready to Push to HF")
        

def main():

    run_local(
        sessions_path = SESSIONS_PATH,
        summary_path = BLOCK_SUMMARY_PATH,
    )

if __name__ == '__main__':
    main()