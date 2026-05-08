"""
Generate ground-truth reference answers for the golden question bank.

Strategy (no hallucination):
  1. For each question, pull the relevant athlete's raw session rows directly
     from the Qdrant `gym_tables` and `gym_text` collections (payload-only —
     no vector search, no LLM inference at retrieval time).
  2. Feed the raw payload data to Gemini 2.0 Flash with a strict "cite only
     what the data says" prompt.
  3. Write the output to eval/golden_answers.json — one entry per question_id.

This file is run ONCE before evaluation begins. The user should review the
output against the athlete PDFs before running run_server_eval.py.

Usage:
    python eval/generate_references.py
    python eval/generate_references.py --limit 10       # smoke test first 10
    python eval/generate_references.py --out eval/golden_answers.json
    python eval/generate_references.py --overwrite      # regenerate all

Output format:
    {
        "Q01": "athlete_00042 squatted 220 kg at RPE 9.0 in week 8. ...",
        "Q02": "...",
        ...
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.question_bank import GOLDEN_QUESTIONS


# ── Env loading ───────────────────────────────────────────────────────────────

def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ── Qdrant payload fetch (no vector search) ───────────────────────────────────

def _fetch_athlete_payloads(
    client,
    athlete_ids: list[str],
    max_records: int = 50,
) -> list[dict]:
    """Pull session rows and coaching text for the given athletes directly from
    Qdrant — payload-only, no embedding, no vector search."""
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    payloads: list[dict] = []

    for collection in ("gym_tables", "gym_text"):
        try:
            filt = Filter(
                must=[
                    FieldCondition(
                        key="athlete_id",
                        match=MatchAny(any=athlete_ids),
                    )
                ]
            ) if athlete_ids else None

            results, _ = client.scroll(
                collection_name=collection,
                scroll_filter=filt,
                limit=max_records,
                with_payload=True,
                with_vectors=False,
            )
            for hit in results:
                payloads.append({
                    "collection": collection,
                    **hit.payload,
                })
        except Exception as e:
            print(f"  [WARN] {collection} scroll failed: {e}")

    return payloads


def _format_payloads(payloads: list[dict]) -> str:
    """Convert raw Qdrant payloads into a dense text block for the prompt."""
    lines: list[str] = []
    for p in payloads:
        col = p.get("collection", "")
        aid = p.get("athlete_id", "?")
        if col == "gym_tables":
            lines.append(
                f"[TABLE] {aid} | week {p.get('week','?')} | "
                f"{p.get('block_phase','?')} | "
                f"main_lift={p.get('main_lift_kg','?')}kg "
                f"RPE={p.get('main_lift_rpe','?')} "
                f"vol_pct={p.get('volume_pct','?')} | "
                f"squat_peak={p.get('squat_peak_kg','?')}kg "
                f"bench_peak={p.get('bench_peak_kg','?')}kg "
                f"dl_peak={p.get('deadlift_peak_kg','?')}kg | "
                f"dots={p.get('dots','?')} level={p.get('training_level','?')} | "
                f"program={p.get('primary_program','?')}"
            )
        elif col == "gym_text":
            text = p.get("text", "")[:400]
            lines.append(f"[TEXT] {aid} | {text}")
    return "\n".join(lines) if lines else "(no data found)"


# ── Gemini reference synthesis ────────────────────────────────────────────────

_REFERENCE_SYSTEM = """\
You are generating ground-truth reference answers for a RAG system evaluation.

Rules (non-negotiable):
1. Cite ONLY information present in the DATA BLOCK below. Never invent numbers.
2. Always include the athlete_id and, for factual claims, the week number.
3. Be concise — 2-4 sentences max. Evaluators will compare answers against this.
4. If the data does not contain enough to answer the question, say exactly:
   "The data does not contain enough information to answer this question."
5. For level/comparison questions with no specific athletes, summarise patterns
   you observe across the data block.
6. Do not use markdown or bullet points. Plain prose only.
"""

_REFERENCE_HUMAN = """\
DATA BLOCK:
{data}

QUESTION: {query}

REFERENCE ANSWER (cite athlete IDs and week numbers):"""


def _generate_reference(
    query: str,
    data_text: str,
    gemini,
) -> str:
    from google.genai import types

    prompt = _REFERENCE_HUMAN.format(data=data_text, query=query)
    try:
        resp = gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_REFERENCE_SYSTEM,
                temperature=0.0,
                max_output_tokens=300,
            ),
        )
        return resp.text.strip()
    except Exception as e:
        return f"[ERROR generating reference: {e}]"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "golden_answers.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate answers even if they already exist in --out")
    parser.add_argument("--max-records", type=int, default=60,
                        help="Max Qdrant records per athlete per collection")
    args = parser.parse_args()

    _load_env()

    # Load Gemini
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set", file=sys.stderr)
        return 2
    from google import genai
    gemini = genai.Client(api_key=api_key)

    # Load Qdrant client
    from pipeline.ingestion.collection import get_client
    qdrant = get_client(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
    )

    # Load existing answers so we can skip already-done ones
    existing: dict[str, str] = {}
    if args.out.is_file() and not args.overwrite:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"[INFO] Loaded {len(existing)} existing references from {args.out}")

    questions = GOLDEN_QUESTIONS[: args.limit] if args.limit else GOLDEN_QUESTIONS
    answers: dict[str, str] = dict(existing)

    skipped = 0
    generated = 0
    errors = 0

    for q in questions:
        qid = q["id"]
        if qid in existing and not args.overwrite:
            skipped += 1
            print(f"  {qid} [skip — already exists]")
            continue

        athlete_ids = q.get("gt_athlete_ids", [])
        training_levels = q.get("training_levels", [])

        print(f"  {qid} [{q['intent']}/{q['difficulty']}] fetching data for {athlete_ids or training_levels}…")

        # Fetch raw payloads
        if athlete_ids:
            payloads = _fetch_athlete_payloads(qdrant, athlete_ids, args.max_records)
        elif training_levels:
            # For level-comparison questions: sample a few athletes per level
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            level_payloads: list[dict] = []
            for level in training_levels:
                try:
                    filt = Filter(
                        must=[FieldCondition(key="training_level", match=MatchAny(any=[level]))]
                    )
                    results, _ = qdrant.scroll(
                        collection_name="gym_tables",
                        scroll_filter=filt,
                        limit=min(20, args.max_records),
                        with_payload=True,
                        with_vectors=False,
                    )
                    for hit in results:
                        level_payloads.append({"collection": "gym_tables", **hit.payload})
                except Exception as e:
                    print(f"    [WARN] level scroll failed for {level}: {e}")
            payloads = level_payloads
        else:
            payloads = []

        data_text = _format_payloads(payloads)
        if not payloads:
            print(f"    [WARN] no data found — marking as data-unavailable")
            answers[qid] = "The data does not contain enough information to answer this question."
            errors += 1
            continue

        ref = _generate_reference(q["query"], data_text, gemini)
        answers[qid] = ref
        generated += 1
        preview = ref[:120].replace("\n", " ")
        print(f"    → {preview}{'…' if len(ref) > 120 else ''}")

        # Brief pause to respect Gemini free tier rate limits
        time.sleep(0.5)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[INFO] Done — {generated} generated, {skipped} skipped, {errors} data-unavailable")
    print(f"[INFO] Written to {args.out}")
    print(f"\n[NEXT] Review eval/golden_answers.json against your PDFs, then run:")
    print(f"       python eval/run_server_eval.py --references eval/golden_answers.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
