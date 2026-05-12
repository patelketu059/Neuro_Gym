"""
Generate ground-truth reference answers for the golden question bank.

Strategy (no hallucination):
  1. For each question, pull the relevant athlete's raw session rows directly
     from the Qdrant `gym_tables` and `gym_text` collections (payload-only —
     no vector search, no LLM inference at retrieval time).
  2. Feed the raw payload data to Gemini with a strict "cite only
     what the data says" prompt.
  3. Write the output to eval/golden_answers.json — one entry per question_id.

This file is run ONCE before evaluation begins. The user should review the
output against the athlete PDFs before running run_server_eval.py.

Usage:
    python eval/generate_references.py
    python eval/generate_references.py --limit 10       # smoke test first 10
    python eval/generate_references.py --out eval/golden_answers.json
    python eval/generate_references.py --overwrite      # regenerate all
    python eval/generate_references.py --delay 5        # slower pacing for free-tier quota

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
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows cp1252 console fix — force UTF-8 output so Unicode chars don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.model_settings import GEMINI_GENERATION_MODEL
from eval.question_bank import GOLDEN_QUESTIONS

# Batch-script retry policy — eval is offline and benefits from waiting out
# Gemini's per-minute quota windows. Do NOT mirror these into the user-facing
# chain.py path; latency-sensitive code should fail fast.
_RETRY_DELAYS = (12.0, 30.0, 60.0)


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _suggested_delay(exc: Exception) -> float:
    m = re.search(r'retryDelay["\s:\']+(\d+)s', str(exc))
    return float(m.group(1)) if m else 0.0



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
) -> tuple[list[dict], list[dict]]:
    """Pull session rows, coaching text, and PDF page metadata for the given athletes.

    Returns (text_payloads, image_page_records).
    image_page_records is a list of {pdf_path, page_number} dicts sorted by page.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    payloads: list[dict] = []
    filt = Filter(
        must=[FieldCondition(key="athlete_id", match=MatchAny(any=athlete_ids))]
    ) if athlete_ids else None

    for collection in ("gym_tables", "gym_text"):
        try:
            results, _ = client.scroll(
                collection_name=collection,
                scroll_filter=filt,
                limit=max_records,
                with_payload=True,
                with_vectors=False,
            )
            for hit in results:
                payloads.append({"collection": collection, **hit.payload})
        except Exception as e:
            print(f"  [WARN] {collection} scroll failed: {e}")

    # Fetch gym_images page metadata (no image bytes — just pdf_path + page_number)
    image_records: list[dict] = []
    try:
        img_results, _ = client.scroll(
            collection_name="gym_images",
            scroll_filter=filt,
            limit=50,
            with_payload=True,
            with_vectors=False,
        )
        seen: set[tuple] = set()
        for hit in img_results:
            pdf_path = hit.payload.get("pdf_path", "")
            page_num = int(hit.payload.get("page_number", 0))
            key = (pdf_path, page_num)
            if pdf_path and key not in seen:
                image_records.append({"pdf_path": pdf_path, "page_number": page_num})
                seen.add(key)
        image_records.sort(key=lambda r: (r["pdf_path"], r["page_number"]))
    except Exception as e:
        print(f"  [WARN] gym_images scroll failed: {e}")

    return payloads, image_records


def _render_pdf_pages(image_records: list[dict], pdf_base_dir: Path) -> list:
    """Render PDF pages to base64-encoded PNG parts for Gemini.

    Returns a list of google.genai types.Part image objects.
    """
    from google.genai import types
    import base64, io

    parts = []
    try:
        import fitz
    except ImportError:
        return parts  # PyMuPDF not available

    for rec in image_records:
        pdf_rel = rec["pdf_path"]
        page_num = rec["page_number"]
        pdf_full = pdf_base_dir / Path(pdf_rel).name
        if not pdf_full.is_file():
            continue
        try:
            doc = fitz.open(str(pdf_full))
            if page_num >= len(doc):
                page_num = 0
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 dpi — enough for text
            pix = doc[page_num].get_pixmap(matrix=mat)
            buf = io.BytesIO(pix.tobytes("png"))
            doc.close()
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(b64), mime_type="image/png"
            ))
        except Exception as e:
            print(f"  [WARN] PDF render failed ({pdf_full} p{page_num}): {e}")
    return parts


def _format_payloads(payloads: list[dict]) -> str:
    """Convert raw Qdrant payloads into a dense text block for the prompt."""
    lines: list[str] = []
    for p in payloads:
        col = p.get("collection", "")
        aid = p.get("athlete_id", "?")
        if col == "gym_tables":
            # 800 chars — enough to capture main lifts + all 4 accessory day lines
            text_snippet = p.get("text", "")[:800]
            lines.append(
                f"[TABLE] {aid} | week {p.get('week','?')} | "
                f"{p.get('block_phase','?')} | "
                f"squat_peak={p.get('squat_peak_kg','?')}kg "
                f"bench_peak={p.get('bench_peak_kg','?')}kg "
                f"dl_peak={p.get('deadlift_peak_kg','?')}kg | "
                f"dots={p.get('dots','?')} level={p.get('training_level','?')} | "
                f"program={p.get('primary_program','?')} | "
                f"{text_snippet}"
            )
        elif col == "gym_text":
            text = p.get("text", "")[:600]
            lines.append(f"[TEXT] {aid} | {text}")
    return "\n".join(lines) if lines else "(no data found)"


# ── Gemini reference synthesis ────────────────────────────────────────────────

_REFERENCE_SYSTEM = """\
You are generating ground-truth reference answers for a RAG system evaluation.

Rules (non-negotiable):
1. Cite ONLY information present in the DATA BLOCK or PDF images below. Never invent numbers.
2. Always include the athlete_id and, for factual claims, the week number.
3. Be concise — 2-4 sentences max. Evaluators will compare answers against this.
4. If the question asks for a specific value that is partially available, give the
   partial answer (e.g. kg and RPE) and note what is missing (e.g. sets/reps not recorded).
   Only use "The data does not contain enough information to answer this question."
   when NO relevant information exists at all.
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
    image_parts: list | None = None,
) -> str | None:
    from google.genai import types

    text_prompt = _REFERENCE_HUMAN.format(data=data_text, query=query)
    # Build multimodal contents: text block + any PDF page images
    contents: list = [text_prompt]
    if image_parts:
        contents.extend(image_parts)

    config = types.GenerateContentConfig(
        system_instruction=_REFERENCE_SYSTEM,
        temperature=0.0,
        max_output_tokens=1024,
    )

    last_exc: Exception | None = None
    for attempt, fallback_delay in enumerate([0.0, *_RETRY_DELAYS]):
        if fallback_delay:
            time.sleep(fallback_delay)
        try:
            resp = gemini.models.generate_content(
                model=GEMINI_GENERATION_MODEL,
                contents=contents,
                config=config,
            )
            return resp.text.strip()
        except Exception as exc:
            last_exc = exc
            if _is_rate_limited(exc):
                wait = _suggested_delay(exc) or fallback_delay or _RETRY_DELAYS[0]
                print(f"    [RATE LIMIT] waiting {wait:.0f}s (attempt {attempt + 1}/{len(_RETRY_DELAYS) + 1})")
                time.sleep(wait)
            else:
                break

    print(f"    [ERROR] {last_exc}")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "golden_answers.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate answers even if they already exist in --out")
    parser.add_argument("--max-records", type=int, default=60,
                        help="Max Qdrant records per athlete per collection")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds to wait between questions (default 3.0)")
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

    pdf_base_dir = ROOT / "data" / "pdfs"

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

        # Fetch raw payloads + PDF page metadata
        image_parts: list = []
        if athlete_ids:
            payloads, image_records = _fetch_athlete_payloads(qdrant, athlete_ids, args.max_records)
            image_parts = _render_pdf_pages(image_records, pdf_base_dir)
            if image_parts:
                print(f"    [PDF] {len(image_parts)} page(s) rendered for context")
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

        ref = _generate_reference(q["query"], data_text, gemini, image_parts=image_parts)
        if ref is None:
            print(f"    [SKIP] {qid} — generation failed, will retry on next run")
            errors += 1
        else:
            answers[qid] = ref
            generated += 1
            preview = ref[:120].replace("\n", " ")
            print(f"    -> {preview}{'...' if len(ref) > 120 else ''}")

        # Write after every question so progress is never lost on interruption
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")

        time.sleep(args.delay)

    print(f"\n[INFO] Done — {generated} generated, {skipped} skipped, {errors} data-unavailable")
    print(f"[INFO] Written to {args.out}")
    print(f"\n[NEXT] Review eval/golden_answers.json against your PDFs, then run:")
    print(f"       python eval/run_server_eval.py --references eval/golden_answers.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
