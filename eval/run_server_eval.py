"""
Server-mode end-to-end evaluation — realistic production test.

Unlike retrieval_eval.py / ragas_eval.py (which import the pipeline directly),
this script POSTs each question to the live FastAPI server via HTTP — exactly
the same path a real user hits. This catches serialisation bugs, middleware
overhead, and session-store interactions that library-mode tests miss.

Flow:
  1. Health-check the server at --base-url (abort if not reachable)
  2. For each config × each question  →  POST /chat with config_name form field
  3. Collect (answer, retrieved_contexts, retrieval_ms, generation_ms, intent)
  4. Score all rows with RAGAS (LLM-as-judge via Gemini 2.0 Flash)
  5. Write CSV, raw JSON, summary JSON, and markdown table

Industry metrics:
  Retrieval   — Hit@1, Hit@3, Hit@5, MRR, NDCG@5, Recall@5
  Full-chain  — Faithfulness, Answer Relevancy, Context Precision, Context Recall*
               (* only when golden_answers.json is supplied via --references)
  Latency     — retrieval_ms, generation_ms per question per config

Usage:
    # Start the server first:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    # Then run this script:
    python eval/run_server_eval.py
    python eval/run_server_eval.py --configs "F — all + BM25" "G — hybrid + rerank"
    python eval/run_server_eval.py --references eval/golden_answers.json
    python eval/run_server_eval.py --limit 5   # smoke test
    python eval/run_server_eval.py --base-url http://my-railway-app.up.railway.app

Outputs (in eval/server_results/):
    server_results_raw.json       — every (config, question) row
    server_results_summary.json   — aggregated metrics per config
    server_results_summary.md     — markdown table (paste into README / website)
    server_results_retrieval.md   — retrieval-only metrics table
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.question_bank import GOLDEN_QUESTIONS

ALL_CONFIGS = [
    "A — images only",
    "B — text only",
    "C — tables only",
    "D — all dense",
    "E — tables + BM25",
    "F — all + BM25",
    "G — hybrid + rerank",
    "H — BM25 only",
]

DEFAULT_SESSION_ID = "server_eval"
_SOURCE_HDR = "--- SOURCE:"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EvalRow:
    config: str
    question_id: str
    intent: str
    difficulty: str
    query: str
    answer: str
    contexts: list[str]
    gt_athlete_ids: list[str]
    retrieved_athlete_ids: list[str]
    retrieval_ms: int
    generation_ms: int
    query_rewritten: bool
    retrieval_query: str
    reference: str | None = None
    # Retrieval metrics
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    recall_at_5: float = 0.0
    # RAGAS metrics
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    error: str | None = None


@dataclass
class ConfigSummary:
    config: str
    n_questions: int
    n_errors: int
    # Retrieval
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_5: float
    recall_at_5: float
    # RAGAS
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float | None
    # Latency
    mean_retrieval_ms: float
    mean_generation_ms: float
    median_retrieval_ms: float
    # By intent
    by_intent: dict[str, dict[str, float]] = field(default_factory=dict)


# ── Env / server helpers ──────────────────────────────────────────────────────

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


def _health_check(base_url: str, timeout: float = 10.0) -> None:
    url = f"{base_url}/health"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        print(f"[INFO] Server healthy at {base_url}")
    except Exception as e:
        print(f"[ERROR] Server not reachable at {url}: {e}", file=sys.stderr)
        print(
            "       Start with:  uvicorn app.main:app --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )
        sys.exit(1)


# ── HTTP call ─────────────────────────────────────────────────────────────────

def _post_chat(
    base_url: str,
    query: str,
    config_name: str,
    session_id: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """POST to /chat and return the parsed JSON response."""
    url = f"{base_url}/chat"
    data = {
        "query": query,
        "session_id": session_id,
        "config_name": config_name,
    }
    resp = requests.post(url, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _split_contexts(text_context: str) -> list[str]:
    """Split the assembled context string into individual passages."""
    import re
    hdr = re.compile(r"^---\s*SOURCE:.*?---\s*$", re.MULTILINE)
    chunks = hdr.split(text_context)
    return [c.strip() for c in chunks if c and c.strip()]


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def _hit(retrieved: list[str], gt: set[str], k: int) -> int:
    return int(any(a in gt for a in retrieved[:k]))

def _recall(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    return sum(1 for a in retrieved[:k] if a in gt) / len(gt)

def _ndcg(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, a in enumerate(retrieved[:k]) if a in gt)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    return dcg / idcg if idcg else 0.0

def _mrr(retrieved: list[str], gt: set[str]) -> float:
    for i, a in enumerate(retrieved, 1):
        if a in gt:
            return 1.0 / i
    return 0.0


# ── RAGAS scoring ─────────────────────────────────────────────────────────────

def _score_ragas(rows: list[EvalRow]) -> None:
    """Run RAGAS in-place — mutates faithfulness/answer_relevancy/... on each row."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError as e:
        print(f"[WARN] RAGAS scoring skipped — {e}")
        return

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[WARN] GEMINI_API_KEY not set — RAGAS scoring skipped")
        return

    judge = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.0)
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    valid = [r for r in rows if not r.error and r.answer and r.contexts]
    if not valid:
        print("[WARN] No valid rows to score with RAGAS")
        return

    has_refs = any(r.reference for r in valid)
    metrics = [
        Faithfulness(llm=LangchainLLMWrapper(judge)),
        ResponseRelevancy(llm=LangchainLLMWrapper(judge), embeddings=LangchainEmbeddingsWrapper(embedder)),
        LLMContextPrecisionWithoutReference(llm=LangchainLLMWrapper(judge)),
    ]
    if has_refs:
        metrics.append(LLMContextRecall(llm=LangchainLLMWrapper(judge)))

    samples = []
    for r in valid:
        kw: dict = {"user_input": r.query, "response": r.answer, "retrieved_contexts": r.contexts}
        if r.reference:
            kw["reference"] = r.reference
        samples.append(SingleTurnSample(**kw))

    print(f"  [RAGAS] scoring {len(samples)} sample(s), {len(metrics)} metric(s)…")
    result = evaluate(
        dataset=EvaluationDataset(samples),
        metrics=metrics,
        llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(embedder),
    )
    df = result.to_pandas()
    for row, dr in zip(valid, df.to_dict(orient="records")):
        row.faithfulness      = _sf(dr.get("faithfulness"))
        row.answer_relevancy  = _sf(dr.get("answer_relevancy"))
        row.context_precision = _sf(dr.get("llm_context_precision_without_reference") or dr.get("context_precision"))
        row.context_recall    = _sf(dr.get("context_recall") or dr.get("llm_context_recall"))


def _sf(v: Any) -> float | None:
    try:
        f = float(v)
        return None if f != f else round(f, 4)  # filter NaN
    except (TypeError, ValueError):
        return None


# ── Aggregation ───────────────────────────────────────────────────────────────

def _summarise(rows: list[EvalRow]) -> ConfigSummary:
    cfg = rows[0].config
    valid = [r for r in rows if not r.error]
    n = max(len(valid), 1)

    def _mean(attr: str) -> float:
        vals = [getattr(r, attr) for r in valid if getattr(r, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _mean_opt(attr: str) -> float | None:
        vals = [getattr(r, attr) for r in valid if getattr(r, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_intent: dict[str, list[EvalRow]] = {}
    for r in valid:
        by_intent.setdefault(r.intent, []).append(r)

    intent_breakdown = {
        intent: {
            "n": len(rs),
            "hit_at_5":         round(sum(r.hit_at_5 for r in rs) / len(rs), 4),
            "mrr":              round(sum(r.mrr for r in rs) / len(rs), 4),
            "faithfulness":     round(
                sum(r.faithfulness for r in rs if r.faithfulness is not None)
                / max(1, sum(1 for r in rs if r.faithfulness is not None)), 4
            ),
            "answer_relevancy": round(
                sum(r.answer_relevancy for r in rs if r.answer_relevancy is not None)
                / max(1, sum(1 for r in rs if r.answer_relevancy is not None)), 4
            ),
            "mean_retrieval_ms":  round(sum(r.retrieval_ms for r in rs) / len(rs), 1),
            "mean_generation_ms": round(sum(r.generation_ms for r in rs) / len(rs), 1),
        }
        for intent, rs in by_intent.items()
    }

    latencies = [r.retrieval_ms for r in valid]

    return ConfigSummary(
        config=cfg,
        n_questions=len(rows),
        n_errors=len(rows) - len(valid),
        hit_at_1=round(sum(r.hit_at_1 for r in valid) / n, 4),
        hit_at_3=round(sum(r.hit_at_3 for r in valid) / n, 4),
        hit_at_5=round(sum(r.hit_at_5 for r in valid) / n, 4),
        mrr=round(sum(r.mrr for r in valid) / n, 4),
        ndcg_at_5=round(sum(r.ndcg_at_5 for r in valid) / n, 4),
        recall_at_5=round(sum(r.recall_at_5 for r in valid) / n, 4),
        faithfulness=_mean("faithfulness"),
        answer_relevancy=_mean("answer_relevancy"),
        context_precision=_mean("context_precision"),
        context_recall=_mean_opt("context_recall"),
        mean_retrieval_ms=round(statistics.mean(latencies), 1) if latencies else 0,
        mean_generation_ms=round(sum(r.generation_ms for r in valid) / n, 1),
        median_retrieval_ms=round(statistics.median(latencies), 1) if latencies else 0,
        by_intent=intent_breakdown,
    )


# ── Output writers ────────────────────────────────────────────────────────────

def _write_ragas_md(summaries: list[ConfigSummary], path: Path) -> None:
    has_recall = any(s.context_recall is not None for s in summaries)
    headers = ["Config", "Faithfulness ↑", "Ans. Relevancy ↑", "Ctx. Precision ↑"]
    if has_recall:
        headers.append("Ctx. Recall ↑")
    headers += ["Ret. ms ↓", "Gen. ms ↓", "Errors"]
    aligns = ["---"] + ["---:"] * (len(headers) - 1)

    lines = [
        "# RAGAS End-to-End Evaluation (Server Mode)\n",
        "Full pipeline evaluation: augment → retrieve → generate, scored with RAGAS "
        "(LLM-as-judge via Gemini 2.0 Flash). All 8 retrieval configurations tested "
        f"against {summaries[0].n_questions} golden questions.\n\n"
        "Metrics: Faithfulness (hallucination rate), Answer Relevancy, "
        "Context Precision, Context Recall (with reference answers only). "
        "Higher is better (↑); latency lower is better (↓).\n",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for s in summaries:
        cells = [
            f"**{s.config}**" if "F —" in s.config else s.config,
            f"{s.faithfulness:.3f}",
            f"{s.answer_relevancy:.3f}",
            f"{s.context_precision:.3f}",
        ]
        if has_recall:
            cells.append(f"{s.context_recall:.3f}" if s.context_recall is not None else "—")
        cells += [
            f"{s.mean_retrieval_ms:.0f}",
            f"{s.mean_generation_ms:.0f}",
            str(s.n_errors),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "\n## By Intent\n",
        "| Config | Factual Faith. | Trend Faith. | Comparison Faith. | Coaching Faith. | Visual Faith. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    intents = ["factual", "trend", "comparison", "coaching", "visual"]
    for s in summaries:
        row = [s.config]
        for intent in intents:
            cell = s.by_intent.get(intent, {})
            row.append(f"{cell.get('faithfulness', 0):.3f}" if cell else "—")
        lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_retrieval_md(summaries: list[ConfigSummary], path: Path) -> None:
    headers = ["Config", "Hit@1 ↑", "Hit@3 ↑", "Hit@5 ↑", "MRR ↑", "NDCG@5 ↑", "Recall@5 ↑",
               "Mean Ret. ms ↓", "Median Ret. ms ↓"]
    aligns = ["---"] + ["---:"] * (len(headers) - 1)

    lines = [
        "# Retrieval Evaluation (Server Mode)\n",
        "Athlete-level retrieval metrics across all 8 configurations. "
        "'Ground truth' is the set of `gt_athlete_ids` for each question.\n",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for s in summaries:
        lines.append(
            f"| {'**' + s.config + '**' if 'F —' in s.config else s.config} "
            f"| {s.hit_at_1:.2f} | {s.hit_at_3:.2f} | {s.hit_at_5:.2f} "
            f"| {s.mrr:.3f} | {s.ndcg_at_5:.3f} | {s.recall_at_5:.2f} "
            f"| {s.mean_retrieval_ms:.0f} | {s.median_retrieval_ms:.0f} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(rows: list[EvalRow], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the running FastAPI server")
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                        help="Configs to evaluate. Default: all 8.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (smoke test)")
    parser.add_argument("--references", type=Path, default=None,
                        help="Path to golden_answers.json for context_recall scoring")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "server_results",
                        help="Output directory. Default: eval/server_results/")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS scoring (just collect answers and retrieval metrics)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="HTTP request timeout in seconds")
    args = parser.parse_args()

    _load_env()
    _health_check(args.base_url)

    references: dict[str, str] = {}
    if args.references and args.references.is_file():
        references = json.loads(args.references.read_text(encoding="utf-8"))
        print(f"[INFO] Loaded {len(references)} reference answers")

    questions = GOLDEN_QUESTIONS[: args.limit] if args.limit else GOLDEN_QUESTIONS
    configs = args.configs

    print(
        f"[INFO] Running {len(configs)} config(s) × {len(questions)} question(s) "
        f"= {len(configs) * len(questions)} HTTP calls"
    )
    print(f"[INFO] Server: {args.base_url}\n")

    all_rows: list[EvalRow] = []
    summaries: list[ConfigSummary] = []

    for cfg_name in configs:
        print(f"\n=== {cfg_name} ===")
        rows: list[EvalRow] = []

        for q in questions:
            # Use a unique session per (config, question) so conversation history
            # doesn't bleed between questions.
            session_id = f"eval_{cfg_name.split(' ')[0]}_{q['id']}"
            answer = ""
            contexts: list[str] = []
            retrieved_ids: list[str] = []
            ret_ms = gen_ms = 0
            query_rewritten = False
            retrieval_query = q["query"]
            error: str | None = None

            t0 = time.perf_counter()
            try:
                resp = _post_chat(
                    args.base_url, q["query"], cfg_name, session_id, args.timeout
                )
                answer         = resp.get("response", "")
                retrieved_ids  = resp.get("athlete_ids", [])
                ret_ms         = resp.get("retrieval_ms", 0)
                gen_ms         = resp.get("generation_ms", 0)
                query_rewritten= resp.get("query_rewritten", False)
                retrieval_query= resp.get("retrieval_query", q["query"])

                # Reconstruct context list from sources for RAGAS
                sources = resp.get("sources", [])
                contexts = [
                    s.get("payload", {}).get("text", "") or str(s)
                    for s in sources
                    if s.get("payload", {}).get("text")
                ]
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                print(f"  {q['id']} [ERROR] {error}")

            elapsed = int((time.perf_counter() - t0) * 1000)

            gt = set(q["gt_athlete_ids"])
            row = EvalRow(
                config=cfg_name,
                question_id=q["id"],
                intent=q["intent"],
                difficulty=q["difficulty"],
                query=q["query"],
                answer=answer,
                contexts=contexts,
                gt_athlete_ids=q["gt_athlete_ids"],
                retrieved_athlete_ids=retrieved_ids,
                retrieval_ms=ret_ms or elapsed,
                generation_ms=gen_ms,
                query_rewritten=query_rewritten,
                retrieval_query=retrieval_query,
                reference=references.get(q["id"]),
                hit_at_1=_hit(retrieved_ids, gt, 1),
                hit_at_3=_hit(retrieved_ids, gt, 3),
                hit_at_5=_hit(retrieved_ids, gt, 5),
                mrr=round(_mrr(retrieved_ids, gt), 4),
                ndcg_at_5=round(_ndcg(retrieved_ids, gt, 5), 4),
                recall_at_5=round(_recall(retrieved_ids, gt, 5), 4),
                error=error,
            )
            rows.append(row)
            all_rows.append(row)

            hit = "✓" if row.hit_at_1 else ("✓@3" if row.hit_at_3 else ("✓@5" if row.hit_at_5 else "✗"))
            rw  = " [rewritten]" if query_rewritten else ""
            print(
                f"  {q['id']} [{hit}]  ret={ret_ms}ms gen={gen_ms}ms  "
                f"{q['query'][:50]}…{rw}"
            )

        # Score this config with RAGAS
        if not args.no_ragas:
            try:
                _score_ragas(rows)
            except Exception as e:
                print(f"  [WARN] RAGAS scoring failed: {e}")

        summaries.append(_summarise(rows))

    # ── Write outputs ─────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.out_dir / "server_results_raw.json"
    raw_path.write_text(
        json.dumps([asdict(r) for r in all_rows], indent=2), encoding="utf-8"
    )

    summary_path = args.out_dir / "server_results_summary.json"
    summary_path.write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )

    csv_path = args.out_dir / "server_results_raw.csv"
    _write_csv(all_rows, csv_path)

    md_ragas = args.out_dir / "server_results_summary.md"
    _write_ragas_md(summaries, md_ragas)

    md_ret = args.out_dir / "server_results_retrieval.md"
    _write_retrieval_md(summaries, md_ret)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Config':<24} {'Faith':>7} {'AnsR':>7} {'CtxP':>7} {'CtxR':>7} "
          f"{'H@5':>5} {'MRR':>6} {'ret':>6} {'gen':>6}")
    print("-" * 80)
    for s in summaries:
        recall = f"{s.context_recall:.3f}" if s.context_recall is not None else "   —  "
        print(
            f"{s.config:<24} "
            f"{s.faithfulness:>7.3f} {s.answer_relevancy:>7.3f} "
            f"{s.context_precision:>7.3f} {recall:>7} "
            f"{s.hit_at_5:>5.2f} {s.mrr:>6.3f} "
            f"{s.mean_retrieval_ms:>6.0f} {s.mean_generation_ms:>6.0f}"
        )

    print(f"\n[INFO] Written to {args.out_dir}/")
    print(f"[NEXT] Generate plots:  python eval/visualize.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
