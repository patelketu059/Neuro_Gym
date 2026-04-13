import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SESSIONS_PATH, BLOCK_SUMMARY_PATH
from pipeline.ingestion.chunking import build_all_nl_strings, optimized_build_all_nl_strings
from pipeline.ingestion.bm_index import build_bm_index


def run_local(
        sessions_path: Path,
        summary_path: Path,
) -> None:
    
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
        summary_path = BLOCK_SUMMARY_PATH
    )

if __name__ == '__main__':
    main()