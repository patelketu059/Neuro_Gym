"""
One-time script to backfill squat_peak_kg / bench_peak_kg / deadlift_peak_kg
into existing Qdrant point payloads.

These fields were stored in the sessions CSV but were never written into the
meta dict that gets spread into BM25 corpus entries or the npy payloads loaded
into Qdrant.  This script patches Qdrant in-place without touching vectors.

Usage:
    python scripts/patch_qdrant_lift_payloads.py [--host localhost] [--port 6333]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ingestion.collection import COLLECTIONS, get_client
from config.settings import SESSIONS_PATH

LIFT_FIELDS = ["squat_peak_kg", "bench_peak_kg", "deadlift_peak_kg"]


def patch(host: str = "localhost", port: int = 6333) -> None:
    client = get_client(host, port)

    print(f"[patch] Loading sessions from {SESSIONS_PATH}")
    df = pd.read_csv(SESSIONS_PATH)

    missing = [f for f in LIFT_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"Sessions CSV is missing columns: {missing}")

    per_athlete = (
        df.groupby("athlete_id")[LIFT_FIELDS]
        .first()
        .reset_index()
    )
    print(f"[patch] {len(per_athlete)} athletes to patch across {COLLECTIONS}")

    for collection in COLLECTIONS:
        print(f"\n[patch] Patching collection: {collection}")
        for _, row in tqdm(per_athlete.iterrows(), total=len(per_athlete)):
            aid = row["athlete_id"]
            payload = {
                "squat_peak_kg":    float(row["squat_peak_kg"] or 0),
                "bench_peak_kg":    float(row["bench_peak_kg"] or 0),
                "deadlift_peak_kg": float(row["deadlift_peak_kg"] or 0),
            }
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                client.set_payload(
                    collection_name=collection,
                    payload=payload,
                    points=Filter(
                        must=[FieldCondition(key="athlete_id", match=MatchValue(value=aid))]
                    ),
                )
            except Exception as exc:
                print(f"  [WARN] {collection}/{aid}: {exc}")

    print("\n[patch] Done. All lift payloads backfilled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    args = parser.parse_args()
    patch(args.host, args.port)
