"""
Sample athlete IDs from Qdrant across all 4 training levels.

Run this ONCE before generate_references.py to:
  1. Confirm the athletes used in question_bank.py actually exist in your Qdrant instance
  2. Optionally surface replacement IDs if any are missing

Usage:
    python eval/sample_athletes.py
    python eval/sample_athletes.py --per-level 5 --out eval/sampled_athletes.json

Output (eval/sampled_athletes.json):
    {
        "elite":        [{"athlete_id": "athlete_00042", "dots": 521, ...}, ...],
        "advanced":     [...],
        "intermediate": [...],
        "novice":       [...]
    }

After reviewing the output, update eval/question_bank.py if any athlete IDs
in the question bank are missing from your instance.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEVELS = ["elite", "advanced", "intermediate", "novice"]

# Athletes used in question_bank.py — validate these first
EXPECTED_ATHLETES = [
    "athlete_00042",
    "athlete_00089",
    "athlete_00033",
    "athlete_00117",
    "athlete_00178",
    "athlete_00250",
    "athlete_00301",
]


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _sample_level(client, level: str, n: int, seed: int) -> list[dict]:
    """Scroll gym_tables filtered by training_level, return n unique athlete dicts."""
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    filt = Filter(must=[FieldCondition(key="training_level", match=MatchAny(any=[level]))])

    seen_ids: set[str] = set()
    athletes: list[dict] = []
    offset = None

    while len(athletes) < n * 5:  # oversample — we'll dedupe and sample below
        try:
            results, offset = client.scroll(
                collection_name="gym_tables",
                scroll_filter=filt,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
        except Exception as e:
            print(f"  [WARN] scroll failed for {level}: {e}")
            break

        for hit in results:
            p = hit.payload or {}
            aid = p.get("athlete_id")
            if aid and aid not in seen_ids:
                seen_ids.add(aid)
                athletes.append({
                    "athlete_id": aid,
                    "training_level": p.get("training_level", level),
                    "dots": p.get("dots"),
                    "squat_peak_kg": p.get("squat_peak_kg"),
                    "bench_peak_kg": p.get("bench_peak_kg"),
                    "deadlift_peak_kg": p.get("deadlift_peak_kg"),
                    "primary_program": p.get("primary_program"),
                    "bodyweight_kg": p.get("bodyweight_kg"),
                })

        if offset is None or not results:
            break

    rng = random.Random(seed)
    rng.shuffle(athletes)
    return athletes[:n]


def _validate_expected(client) -> dict[str, bool]:
    """Check which expected athlete IDs exist in gym_tables."""
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    filt = Filter(must=[FieldCondition(key="athlete_id", match=MatchAny(any=EXPECTED_ATHLETES))])
    found: set[str] = set()

    try:
        results, _ = client.scroll(
            collection_name="gym_tables",
            scroll_filter=filt,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )
        for hit in results:
            aid = (hit.payload or {}).get("athlete_id")
            if aid:
                found.add(aid)
    except Exception as e:
        print(f"  [WARN] Validation scroll failed: {e}")

    return {aid: (aid in found) for aid in EXPECTED_ATHLETES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-level", type=int, default=5,
                        help="Athletes to sample per training level (default 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "sampled_athletes.json")
    args = parser.parse_args()

    _load_env()

    from pipeline.ingestion.collection import get_client
    qdrant = get_client(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
    )

    # ── Step 1: Validate question_bank.py athletes ────────────────────────────
    print("[INFO] Validating athletes used in question_bank.py…")
    status = _validate_expected(qdrant)
    all_ok = True
    for aid, exists in status.items():
        mark = "✓" if exists else "✗ MISSING"
        print(f"  {mark}  {aid}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n[WARN] Some question_bank.py athletes are missing from Qdrant!")
        print("       Update question_bank.py with the sampled IDs below.\n")
    else:
        print("  All question_bank.py athletes confirmed present.\n")

    # ── Step 2: Sample athletes per level ─────────────────────────────────────
    print(f"[INFO] Sampling {args.per_level} athletes per level (seed={args.seed})…")
    output: dict[str, list[dict]] = {}

    for level in LEVELS:
        athletes = _sample_level(qdrant, level, args.per_level, args.seed)
        output[level] = athletes
        print(f"  {level:<14} {len(athletes)} athletes sampled")
        for a in athletes:
            print(
                f"    {a['athlete_id']}  dots={a.get('dots')}  "
                f"sq={a.get('squat_peak_kg')}  "
                f"be={a.get('bench_peak_kg')}  "
                f"dl={a.get('deadlift_peak_kg')}"
            )

    # ── Write output ──────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Written to {args.out}")
    print("[NEXT] Review the athlete IDs, then update question_bank.py if needed.")
    print("       Then run:  python eval/generate_references.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
