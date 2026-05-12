"""
Server-mode end-to-end evaluation — realistic production test.

Unlike retrieval_eval.py / ragas_eval.py (which import the pipeline directly),
this script POSTs each question to the live FastAPI server via HTTP — exactly
the same path a real user hits. This catches serialisation bugs, middleware
overhead, and session-store interactions that library-mode tests miss.

Flow:
  1. Health-check the server at --base-url (abort if not reachable)
  2. For each config x each question  ->  POST /chat with config_name form field
  3. Collect answer, contexts, token usage, latency, and athlete IDs
  4. Score all rows with RAGAS (LLM-as-judge via Gemini)
  5. Write CSV, raw JSON, summary JSON, and five markdown reports

Production metric categories
  Retrieval    -- Hit@1, Hit@3, Hit@5, MRR, NDCG@5, Recall@5
  Quality      -- Faithfulness, Answer Relevancy, Context Precision,
                  Context Recall (requires --references), Citation Rate,
                  Null-Answer Rate, Error Rate
  Latency      -- mean/median/P95/P99 retrieval_ms, generation_ms, total_ms
  Cost         -- input/output/thinking tokens, cost_usd (Gemini 2.5 Flash rates)
  By Breakdown -- per-intent and per-difficulty slices of the above

Usage:
    # Start the server first:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    # Then run this script:
    python eval/run_server_eval.py
    python eval/run_server_eval.py --configs "F -- all + BM25" "G -- hybrid + rerank"
    python eval/run_server_eval.py --references eval/golden_answers.json
    python eval/run_server_eval.py --limit 5   # smoke test
    python eval/run_server_eval.py --base-url http://my-railway-app.up.railway.app
    python eval/run_server_eval.py --no-ragas  # skip LLM-as-judge (much faster)

Outputs (in eval/server_results/):
    server_results_raw.json          -- every (config, question) row
    server_results_summary.json      -- aggregated metrics per config
    server_results_ragas.md          -- RAGAS quality metrics table
    server_results_retrieval.md      -- retrieval-only metrics table
    server_results_latency.md        -- full latency percentile table
    server_results_cost.md           -- token usage and cost breakdown
    server_results_quality.md        -- citation rate, null rate, error rate
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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
from config.model_settings import GEMINI_JUDGE_MODEL

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

# ── Gemini 2.5 Flash pricing (USD / 1M tokens, standard tier < 200K context) ──
# Source: https://ai.google.dev/gemini-api/docs/pricing
_PRICE_INPUT_PER_M    = 0.15   # prompt tokens
_PRICE_OUTPUT_PER_M   = 0.60   # generated tokens
_PRICE_THINKING_PER_M = 3.50   # extended thinking tokens (trend / comparison)


# ── Data classes ───────────────────────────────────────────────────────────────

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
    total_ms: int
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
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    # Answer quality
    athlete_cited: bool = True
    null_answer: bool = False
    text_context: str = ""   # full assembled context string for RAGAS scoring
    error: str | None = None


@dataclass
class ConfigSummary:
    config: str
    n_questions: int
    n_errors: int
    error_rate: float
    # Retrieval
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_5: float
    recall_at_5: float
    # RAGAS quality
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float | None
    # Answer quality
    citation_rate: float
    null_answer_rate: float
    # Latency (ms)
    mean_retrieval_ms: float
    median_retrieval_ms: float
    p95_retrieval_ms: float
    p99_retrieval_ms: float
    mean_generation_ms: float
    mean_total_ms: float
    p95_total_ms: float
    p99_total_ms: float
    # Cost / tokens
    mean_input_tokens: float
    mean_output_tokens: float
    mean_thinking_tokens: float
    mean_cost_usd: float
    total_cost_usd: float
    # Breakdowns
    by_intent: dict[str, dict[str, float]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)


# ── Pricing / quality helpers ──────────────────────────────────────────────────

def _compute_cost(input_tok: int, output_tok: int, thinking_tok: int) -> float:
    return round(
        input_tok    * _PRICE_INPUT_PER_M    / 1_000_000
        + output_tok   * _PRICE_OUTPUT_PER_M   / 1_000_000
        + thinking_tok * _PRICE_THINKING_PER_M / 1_000_000,
        8,
    )


def _check_citation(answer: str, gt_athlete_ids: list[str]) -> bool:
    """True if the answer mentions at least one ground-truth athlete ID.

    Level-only questions (empty gt_athlete_ids) are counted as cited because
    there is no specific athlete to mention.
    """
    if not gt_athlete_ids:
        return True
    al = answer.lower()
    for aid in gt_athlete_ids:
        if aid.lower() in al:
            return True
        # Accept colloquial form: "athlete 42" (leading zeros stripped)
        m = re.search(r"(\d+)$", aid)
        if m and f"athlete {int(m.group(1))}" in al:
            return True
    return False


def _is_null_answer(answer: str) -> bool:
    """True if the answer is empty or a data-not-found refusal."""
    if not answer:
        return True
    lo = answer.lower()
    return any(phrase in lo for phrase in (
        "does not contain enough information",
        "not enough information",
        "no relevant training data",
        "cannot answer",
        "data is not available",
    ))


def _percentile(data: list[float], p: int) -> float:
    """p-th percentile with linear interpolation (matches numpy default)."""
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    if n == 1:
        return round(s[0], 1)
    idx = (p / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 1)


# ── Env / server helpers ───────────────────────────────────────────────────────

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


# ── HTTP call ──────────────────────────────────────────────────────────────────

def _post_chat(
    base_url: str,
    query: str,
    config_name: str,
    session_id: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    url = f"{base_url}/chat"
    resp = requests.post(
        url,
        data={"query": query, "session_id": session_id, "config_name": config_name},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _split_contexts(text_context: str) -> list[str]:
    hdr = re.compile(r"^---\s*SOURCE:.*?---\s*$", re.MULTILINE)
    chunks = hdr.split(text_context)
    return [c.strip() for c in chunks if c and c.strip()]


# ── Retrieval metrics ──────────────────────────────────────────────────────────

def _hit(retrieved: list[str], gt: set[str], k: int) -> int:
    return int(any(a in gt for a in retrieved[:k]))

def _recall(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    return sum(1 for a in retrieved[:k] if a in gt) / len(gt)

def _ndcg(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    dcg  = sum(1.0 / math.log2(i + 2) for i, a in enumerate(retrieved[:k]) if a in gt)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    return dcg / idcg if idcg else 0.0

def _mrr(retrieved: list[str], gt: set[str]) -> float:
    for i, a in enumerate(retrieved, 1):
        if a in gt:
            return 1.0 / i
    return 0.0


# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def _score_ragas(rows: list[EvalRow]) -> None:
    """Run RAGAS in-place — mutates faithfulness/answer_relevancy/... on each row.

    Contexts are built from each row's text_context field (the full assembled
    retrieval string) by splitting on the SOURCE header lines injected by
    pipeline/retrieval/context.py. Falls back to pre-split row.contexts if
    text_context is empty (e.g. rows reconstructed from an older raw JSON).
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        try:
            # ragas >= 0.4: canonical location
            from ragas.metrics.collections import (
                Faithfulness,
                ResponseRelevancy,
                LLMContextPrecisionWithoutReference,
                LLMContextRecall,
            )
        except ImportError:
            # ragas 0.3.x fallback
            from ragas.metrics import (          # type: ignore[no-redef]
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

    judge    = ChatGoogleGenerativeAI(model=GEMINI_JUDGE_MODEL, google_api_key=api_key, temperature=0.0)
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # Build per-row context lists from text_context (preferred) or existing contexts
    def _contexts_for(r: EvalRow) -> list[str]:
        if r.text_context:
            return _split_contexts(r.text_context)
        return r.contexts  # pre-split fallback

    valid = [r for r in rows if not r.error and r.answer and _contexts_for(r)]
    if not valid:
        print("[WARN] No valid rows to score with RAGAS")
        return

    has_refs = any(r.reference for r in valid)
    metrics  = [
        Faithfulness(llm=LangchainLLMWrapper(judge)),
        ResponseRelevancy(llm=LangchainLLMWrapper(judge), embeddings=LangchainEmbeddingsWrapper(embedder)),
        LLMContextPrecisionWithoutReference(llm=LangchainLLMWrapper(judge)),
    ]
    if has_refs:
        metrics.append(LLMContextRecall(llm=LangchainLLMWrapper(judge)))

    samples = []
    for r in valid:
        kw: dict = {
            "user_input":          r.query,
            "response":            r.answer,
            "retrieved_contexts":  _contexts_for(r),
        }
        if r.reference:
            kw["reference"] = r.reference
        samples.append(SingleTurnSample(**kw))

    print(f"  [RAGAS] scoring {len(samples)} samples, {len(metrics)} metrics...")
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
        return None if f != f else round(f, 4)   # filter NaN
    except (TypeError, ValueError):
        return None


# ── Aggregation ────────────────────────────────────────────────────────────────

def _mean_vals(rows: list[EvalRow], attr: str) -> float:
    vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0

def _mean_opt(rows: list[EvalRow], attr: str) -> float | None:
    vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _summarise(rows: list[EvalRow]) -> ConfigSummary:
    cfg   = rows[0].config
    valid = [r for r in rows if not r.error]
    n     = max(len(valid), 1)

    # ── Retrieval latency percentiles ─────────────────────────────────────────
    ret_lats   = [r.retrieval_ms for r in valid]
    gen_lats   = [r.generation_ms for r in valid]
    total_lats = [r.total_ms for r in valid]

    # ── Cost aggregation ──────────────────────────────────────────────────────
    total_cost = round(sum(r.cost_usd for r in valid), 6)

    # ── Quality ───────────────────────────────────────────────────────────────
    citation_rate  = round(sum(r.athlete_cited for r in valid) / n, 4)
    null_rate      = round(sum(r.null_answer   for r in valid) / n, 4)
    error_rate     = round((len(rows) - len(valid)) / max(len(rows), 1), 4)

    # ── Intent breakdown ──────────────────────────────────────────────────────
    by_intent: dict[str, list[EvalRow]] = {}
    for r in valid:
        by_intent.setdefault(r.intent, []).append(r)

    intent_breakdown = {
        intent: {
            "n":                    len(rs),
            "hit_at_5":             round(sum(r.hit_at_5 for r in rs) / len(rs), 4),
            "mrr":                  round(sum(r.mrr      for r in rs) / len(rs), 4),
            "faithfulness":         _mean_vals(rs, "faithfulness"),
            "answer_relevancy":     _mean_vals(rs, "answer_relevancy"),
            "mean_retrieval_ms":    round(sum(r.retrieval_ms  for r in rs) / len(rs), 1),
            "mean_generation_ms":   round(sum(r.generation_ms for r in rs) / len(rs), 1),
            "mean_cost_usd":        round(sum(r.cost_usd      for r in rs) / len(rs), 8),
        }
        for intent, rs in by_intent.items()
    }

    # ── Difficulty breakdown ──────────────────────────────────────────────────
    by_diff: dict[str, list[EvalRow]] = {}
    for r in valid:
        by_diff.setdefault(r.difficulty, []).append(r)

    diff_breakdown = {
        diff: {
            "n":                len(rs),
            "hit_at_5":         round(sum(r.hit_at_5 for r in rs) / len(rs), 4),
            "mrr":              round(sum(r.mrr       for r in rs) / len(rs), 4),
            "faithfulness":     _mean_vals(rs, "faithfulness"),
            "answer_relevancy": _mean_vals(rs, "answer_relevancy"),
            "citation_rate":    round(sum(r.athlete_cited for r in rs) / len(rs), 4),
            "null_rate":        round(sum(r.null_answer   for r in rs) / len(rs), 4),
            "mean_total_ms":    round(sum(r.total_ms      for r in rs) / len(rs), 1),
        }
        for diff, rs in by_diff.items()
    }

    return ConfigSummary(
        config        = cfg,
        n_questions   = len(rows),
        n_errors      = len(rows) - len(valid),
        error_rate    = error_rate,
        # Retrieval
        hit_at_1      = round(sum(r.hit_at_1 for r in valid) / n, 4),
        hit_at_3      = round(sum(r.hit_at_3 for r in valid) / n, 4),
        hit_at_5      = round(sum(r.hit_at_5 for r in valid) / n, 4),
        mrr           = round(sum(r.mrr       for r in valid) / n, 4),
        ndcg_at_5     = round(sum(r.ndcg_at_5 for r in valid) / n, 4),
        recall_at_5   = round(sum(r.recall_at_5 for r in valid) / n, 4),
        # RAGAS
        faithfulness      = _mean_vals(valid, "faithfulness"),
        answer_relevancy  = _mean_vals(valid, "answer_relevancy"),
        context_precision = _mean_vals(valid, "context_precision"),
        context_recall    = _mean_opt(valid,  "context_recall"),
        # Quality
        citation_rate  = citation_rate,
        null_answer_rate = null_rate,
        # Latency
        mean_retrieval_ms   = round(statistics.mean(ret_lats),   1) if ret_lats   else 0,
        median_retrieval_ms = round(statistics.median(ret_lats), 1) if ret_lats   else 0,
        p95_retrieval_ms    = _percentile(ret_lats,   95),
        p99_retrieval_ms    = _percentile(ret_lats,   99),
        mean_generation_ms  = round(sum(gen_lats)   / n, 1),
        mean_total_ms       = round(sum(total_lats) / n, 1),
        p95_total_ms        = _percentile(total_lats, 95),
        p99_total_ms        = _percentile(total_lats, 99),
        # Cost
        mean_input_tokens    = round(sum(r.input_tokens    for r in valid) / n, 1),
        mean_output_tokens   = round(sum(r.output_tokens   for r in valid) / n, 1),
        mean_thinking_tokens = round(sum(r.thinking_tokens for r in valid) / n, 1),
        mean_cost_usd        = round(total_cost / n, 8),
        total_cost_usd       = total_cost,
        # Breakdowns
        by_intent     = intent_breakdown,
        by_difficulty = diff_breakdown,
    )


# ── Markdown writers ───────────────────────────────────────────────────────────

def _write_ragas_md(summaries: list[ConfigSummary], path: Path) -> None:
    has_recall = any(s.context_recall is not None for s in summaries)
    headers = ["Config", "Faithfulness", "Ans. Relevancy", "Ctx. Precision"]
    if has_recall:
        headers.append("Ctx. Recall")
    headers += ["Citation Rate", "Null Rate", "Errors"]
    aligns = ["---"] + ["---:"] * (len(headers) - 1)

    lines = [
        "# RAGAS Quality Evaluation (Server Mode)\n",
        f"LLM-as-judge evaluation of the full augment -> retrieve -> generate chain, "
        f"scored with RAGAS ({GEMINI_JUDGE_MODEL} as judge). "
        f"Tested against {summaries[0].n_questions} golden questions across "
        f"{len(summaries)} retrieval configurations.\n",
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
            f"{s.citation_rate:.1%}",
            f"{s.null_answer_rate:.1%}",
            str(s.n_errors),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "\n## By Intent\n",
        "| Config | Factual Faith. | Trend Faith. | Comparison Faith. | Coaching Faith. | Visual Faith. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        row = [s.config]
        for intent in ("factual", "trend", "comparison", "coaching", "visual"):
            cell = s.by_intent.get(intent, {})
            row.append(f"{cell.get('faithfulness', 0):.3f}" if cell else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "\n## Metric Definitions\n",
        "- **Faithfulness** [0-1]: fraction of answer claims grounded in retrieved contexts; penalises hallucination.",
        "- **Answer Relevancy** [0-1]: semantic alignment between the answer and the question.",
        "- **Context Precision** [0-1]: fraction of retrieved context that is actually relevant to the question.",
        "- **Context Recall** [0-1]: fraction of reference-answer facts covered by the retrieved context (requires --references).",
        "- **Citation Rate**: fraction of answers that explicitly name the ground-truth athlete ID.",
        "- **Null Rate**: fraction of answers that are empty or data-not-found refusals.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_retrieval_md(summaries: list[ConfigSummary], path: Path) -> None:
    headers = ["Config", "Hit@1", "Hit@3", "Hit@5", "MRR", "NDCG@5", "Recall@5",
               "Mean Ret. ms", "Median Ret. ms"]
    aligns = ["---"] + ["---:"] * (len(headers) - 1)

    lines = [
        "# Retrieval Evaluation (Server Mode)\n",
        "Athlete-level retrieval metrics across all configurations. "
        "Ground truth is the `gt_athlete_ids` set for each question.\n",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for s in summaries:
        tag = f"**{s.config}**" if "F —" in s.config else s.config
        lines.append(
            f"| {tag} | {s.hit_at_1:.2f} | {s.hit_at_3:.2f} | {s.hit_at_5:.2f} "
            f"| {s.mrr:.3f} | {s.ndcg_at_5:.3f} | {s.recall_at_5:.2f} "
            f"| {s.mean_retrieval_ms:.0f} | {s.median_retrieval_ms:.0f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latency_md(summaries: list[ConfigSummary], path: Path) -> None:
    headers = ["Config", "Ret P50", "Ret P95", "Ret P99",
               "Gen Mean", "Total Mean", "Total P95", "Total P99"]
    aligns = ["---"] + ["---:"] * (len(headers) - 1)

    lines = [
        "# Latency Evaluation (Server Mode)\n",
        "All values in milliseconds. P95/P99 percentiles reveal tail latency "
        "which matters for production SLA planning. "
        "Total = retrieval + generation time.\n",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for s in summaries:
        tag = f"**{s.config}**" if "F —" in s.config else s.config
        lines.append(
            f"| {tag} "
            f"| {s.median_retrieval_ms:.0f} | {s.p95_retrieval_ms:.0f} | {s.p99_retrieval_ms:.0f} "
            f"| {s.mean_generation_ms:.0f} "
            f"| {s.mean_total_ms:.0f} | {s.p95_total_ms:.0f} | {s.p99_total_ms:.0f} |"
        )

    lines += [
        "\n## By Intent (Mean Total ms)\n",
        "| Config | Factual | Trend | Comparison | Coaching | Visual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        row = [s.config]
        for intent in ("factual", "trend", "comparison", "coaching", "visual"):
            cell = s.by_intent.get(intent, {})
            t = (cell.get("mean_retrieval_ms", 0) + cell.get("mean_generation_ms", 0)) if cell else 0
            row.append(f"{t:.0f}" if cell else "—")
        lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cost_md(summaries: list[ConfigSummary], path: Path, n_questions: int) -> None:
    headers = ["Config", "Avg Input Tok", "Avg Output Tok", "Avg Thinking Tok",
               "Avg Cost (USD)", "Total Cost (USD)"]
    aligns = ["---"] + ["---:"] * (len(headers) - 1)

    lines = [
        "# Cost Evaluation (Server Mode)\n",
        f"Token usage and cost estimates using Gemini 2.5 Flash pricing "
        f"(standard tier, context < 200K tokens): "
        f"input $0.15/1M, output $0.60/1M, thinking $3.50/1M.\n",
        f"Each config ran {n_questions} questions.\n",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for s in summaries:
        tag = f"**{s.config}**" if "F —" in s.config else s.config
        lines.append(
            f"| {tag} "
            f"| {s.mean_input_tokens:,.0f} "
            f"| {s.mean_output_tokens:,.0f} "
            f"| {s.mean_thinking_tokens:,.0f} "
            f"| ${s.mean_cost_usd:.5f} "
            f"| ${s.total_cost_usd:.4f} |"
        )

    lines += [
        "\n## Cost by Intent (Average per query)\n",
        "| Config | Factual | Trend | Comparison | Coaching | Visual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        row = [s.config]
        for intent in ("factual", "trend", "comparison", "coaching", "visual"):
            cell = s.by_intent.get(intent, {})
            row.append(f"${cell.get('mean_cost_usd', 0):.5f}" if cell else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "\n## Cost Drivers\n",
        "- **Thinking tokens** dominate cost for `trend` and `comparison` intents "
        "($3.50/1M vs $0.15/1M for input). "
        "Consider capping `THINKING_BUDGET` in `config/rag_config.py` to reduce spend.",
        "- **Input tokens** grow with context window size — "
        "`comparison` queries retrieve more athletes, increasing prompt size.",
        "- **Retrieval config G** (hybrid + rerank) adds reranker API cost not captured here.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_quality_md(summaries: list[ConfigSummary], path: Path) -> None:
    headers = ["Config", "Faithfulness", "Citation Rate", "Null Rate",
               "Error Rate", "Hit@5", "MRR"]
    aligns = ["---"] + ["---:"] * (len(headers) - 1)

    lines = [
        "# Answer Quality Evaluation (Server Mode)\n",
        "Combined view of RAGAS faithfulness, retrieval accuracy, "
        "and answer completeness metrics.\n",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for s in summaries:
        tag = f"**{s.config}**" if "F —" in s.config else s.config
        lines.append(
            f"| {tag} "
            f"| {s.faithfulness:.3f} "
            f"| {s.citation_rate:.1%} "
            f"| {s.null_answer_rate:.1%} "
            f"| {s.error_rate:.1%} "
            f"| {s.hit_at_5:.2f} "
            f"| {s.mrr:.3f} |"
        )

    lines += [
        "\n## By Difficulty\n",
        "| Config | Difficulty | N | Faith. | Citation | Hit@5 | Null Rate | Avg Total ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        for diff in ("easy", "medium", "hard"):
            cell = s.by_difficulty.get(diff, {})
            if not cell:
                continue
            lines.append(
                f"| {s.config} | {diff} | {cell['n']} "
                f"| {cell.get('faithfulness', 0):.3f} "
                f"| {cell.get('citation_rate', 0):.1%} "
                f"| {cell.get('hit_at_5', 0):.2f} "
                f"| {cell.get('null_rate', 0):.1%} "
                f"| {cell.get('mean_total_ms', 0):.0f} |"
            )

    lines += [
        "\n## Definitions\n",
        "- **Citation Rate**: answers that explicitly mention the expected athlete ID. "
        "Low citation suggests the retriever is pulling wrong athletes or the model ignores retrieved context.",
        "- **Null Rate**: answers that admit the data is unavailable. "
        "High null rate on easy questions signals retrieval failure.",
        "- **Error Rate**: HTTP errors or exceptions during the eval run. "
        "Should be 0% in production.",
        "- **By Difficulty**: easy = single data-point lookups, "
        "medium = multi-step reasoning, hard = open-ended analysis or cross-athlete comparison.",
    ]
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


# ── Main ───────────────────────────────────────────────────────────────────────

def _score_only(args) -> int:
    """Load server_results_raw.json, run RAGAS, rewrite all reports. No HTTP calls."""
    raw_path = args.out_dir / "server_results_raw.json"
    if not raw_path.is_file():
        print(f"[ERROR] {raw_path} not found — run without --score-only first.", file=sys.stderr)
        return 2

    references: dict[str, str] = {}
    if args.references and args.references.is_file():
        references = json.loads(args.references.read_text(encoding="utf-8"))
        print(f"[INFO] Loaded {len(references)} reference answers")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    print(f"[INFO] Loaded {len(raw)} rows from {raw_path}")

    # Reconstruct EvalRow objects, injecting reference answers
    all_rows: list[EvalRow] = []
    for d in raw:
        d["reference"] = references.get(d["question_id"], d.get("reference"))
        # contexts may have been serialised as a plain list — ensure list[str]
        d["contexts"] = [str(c) for c in d.get("contexts") or []]
        all_rows.append(EvalRow(**d))

    # Group by config, preserving original order
    from collections import OrderedDict
    by_config: OrderedDict[str, list[EvalRow]] = OrderedDict()
    for r in all_rows:
        by_config.setdefault(r.config, []).append(r)

    summaries: list[ConfigSummary] = []
    for cfg_name, rows in by_config.items():
        print(f"\n=== {cfg_name} — {len(rows)} rows ===")
        try:
            _score_ragas(rows)
        except Exception as e:
            print(f"  [WARN] RAGAS scoring failed: {e}")
        summaries.append(_summarise(rows))

    # Rewrite raw JSON with populated RAGAS fields
    raw_path.write_text(
        json.dumps([asdict(r) for r in all_rows], indent=2), encoding="utf-8"
    )
    (args.out_dir / "server_results_summary.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )
    _write_csv(all_rows, args.out_dir / "server_results_raw.csv")
    _write_ragas_md(summaries,     args.out_dir / "server_results_ragas.md")
    _write_retrieval_md(summaries, args.out_dir / "server_results_retrieval.md")
    _write_latency_md(summaries,   args.out_dir / "server_results_latency.md")
    _write_cost_md(summaries, args.out_dir / "server_results_cost.md",
                   max(len(rows) for rows in by_config.values()))
    _write_quality_md(summaries,   args.out_dir / "server_results_quality.md")

    print(f"\n[INFO] RAGAS scoring complete. Reports written to {args.out_dir}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (smoke test)")
    parser.add_argument("--references", type=Path, default=None,
                        help="Path to golden_answers.json for context_recall scoring")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "server_results")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS scoring (faster, retrieval metrics only)")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip HTTP calls — load server_results_raw.json and run "
                             "RAGAS scoring only. Use after installing ragas post-run.")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    _load_env()

    if args.score_only:
        return _score_only(args)

    _health_check(args.base_url)

    references: dict[str, str] = {}
    if args.references and args.references.is_file():
        references = json.loads(args.references.read_text(encoding="utf-8"))
        print(f"[INFO] Loaded {len(references)} reference answers")

    questions = GOLDEN_QUESTIONS[: args.limit] if args.limit else GOLDEN_QUESTIONS
    configs   = args.configs

    total_calls = len(configs) * len(questions)
    print(
        f"[INFO] Running {len(configs)} config(s) x {len(questions)} question(s) "
        f"= {total_calls} HTTP calls"
    )
    print(f"[INFO] Server: {args.base_url}\n")

    all_rows:   list[EvalRow]      = []
    summaries:  list[ConfigSummary] = []
    wall_start = time.perf_counter()

    for cfg_name in configs:
        print(f"\n=== {cfg_name} ===")
        rows: list[EvalRow] = []

        for q in questions:
            session_id      = f"eval_{cfg_name.split(' ')[0]}_{q['id']}"
            answer          = ""
            contexts: list[str] = []
            retrieved_ids:  list[str] = []
            ret_ms = gen_ms = 0
            input_tok = output_tok = thinking_tok = 0
            query_rewritten = False
            retrieval_query = q["query"]
            error: str | None = None

            t0 = time.perf_counter()
            try:
                resp = _post_chat(
                    args.base_url, q["query"], cfg_name, session_id, args.timeout
                )
                answer          = resp.get("response", "")
                retrieved_ids   = resp.get("athlete_ids", [])
                ret_ms          = resp.get("retrieval_ms", 0)
                gen_ms          = resp.get("generation_ms", 0)
                query_rewritten = resp.get("query_rewritten", False)
                retrieval_query = resp.get("retrieval_query", q["query"])
                input_tok       = resp.get("input_tokens", 0)
                output_tok      = resp.get("output_tokens", 0)
                thinking_tok    = resp.get("thinking_tokens", 0)
                text_context    = resp.get("text_context", "")
                contexts        = _split_contexts(text_context)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                print(f"  {q['id']} [ERROR] {error}")

            elapsed   = int((time.perf_counter() - t0) * 1000)
            total_ms  = ret_ms + gen_ms if (ret_ms or gen_ms) else elapsed
            cost      = _compute_cost(input_tok, output_tok, thinking_tok)
            gt        = set(q["gt_athlete_ids"])

            row = EvalRow(
                config                = cfg_name,
                question_id           = q["id"],
                intent                = q["intent"],
                difficulty            = q["difficulty"],
                query                 = q["query"],
                answer                = answer,
                contexts              = contexts,
                gt_athlete_ids        = q["gt_athlete_ids"],
                retrieved_athlete_ids = retrieved_ids,
                retrieval_ms          = ret_ms or elapsed,
                generation_ms         = gen_ms,
                total_ms              = total_ms,
                query_rewritten       = query_rewritten,
                retrieval_query       = retrieval_query,
                reference             = references.get(q["id"]),
                hit_at_1              = _hit(retrieved_ids, gt, 1),
                hit_at_3              = _hit(retrieved_ids, gt, 3),
                hit_at_5              = _hit(retrieved_ids, gt, 5),
                mrr                   = round(_mrr(retrieved_ids, gt), 4),
                ndcg_at_5             = round(_ndcg(retrieved_ids, gt, 5), 4),
                recall_at_5           = round(_recall(retrieved_ids, gt, 5), 4),
                input_tokens          = input_tok,
                output_tokens         = output_tok,
                thinking_tokens       = thinking_tok,
                cost_usd              = cost,
                athlete_cited         = _check_citation(answer, q["gt_athlete_ids"]),
                null_answer           = _is_null_answer(answer),
                text_context          = text_context,
                error                 = error,
            )
            rows.append(row)
            all_rows.append(row)

            hit = "H1" if row.hit_at_1 else ("H3" if row.hit_at_3 else ("H5" if row.hit_at_5 else " x"))
            rw  = " [rw]" if query_rewritten else ""
            tok_summary = f" {input_tok}in/{output_tok}out" if input_tok else ""
            print(
                f"  {q['id']} [{hit}]  ret={ret_ms}ms gen={gen_ms}ms"
                f"{tok_summary}  ${cost:.5f}"
                f"  {q['query'][:45]}...{rw}"
            )

        # Score with RAGAS
        if not args.no_ragas:
            try:
                _score_ragas(rows)
            except Exception as e:
                print(f"  [WARN] RAGAS scoring failed: {e}")

        summaries.append(_summarise(rows))

    wall_elapsed = time.perf_counter() - wall_start
    throughput   = round(len(all_rows) / wall_elapsed, 2) if wall_elapsed else 0

    # ── Write outputs ──────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)

    (args.out_dir / "server_results_raw.json").write_text(
        json.dumps([asdict(r) for r in all_rows], indent=2), encoding="utf-8"
    )
    (args.out_dir / "server_results_summary.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )
    _write_csv(all_rows, args.out_dir / "server_results_raw.csv")
    _write_ragas_md(summaries,  args.out_dir / "server_results_ragas.md")
    _write_retrieval_md(summaries, args.out_dir / "server_results_retrieval.md")
    _write_latency_md(summaries,   args.out_dir / "server_results_latency.md")
    _write_cost_md(summaries, args.out_dir / "server_results_cost.md", len(questions))
    _write_quality_md(summaries,   args.out_dir / "server_results_quality.md")

    # ── Console summary table ──────────────────────────────────────────────────
    W = 106
    print("\n" + "=" * W)
    print(
        f"{'Config':<24} {'Faith':>7} {'AnsR':>7} {'CtxP':>7} {'CtxR':>7} "
        f"{'H@5':>5} {'MRR':>6} {'Cite%':>6} "
        f"{'ret':>6} {'gen':>6} {'P95tot':>8} {'$/q':>9}"
    )
    print("-" * W)
    for s in summaries:
        recall  = f"{s.context_recall:.3f}" if s.context_recall is not None else "   —  "
        print(
            f"{s.config:<24} "
            f"{s.faithfulness:>7.3f} {s.answer_relevancy:>7.3f} "
            f"{s.context_precision:>7.3f} {recall:>7} "
            f"{s.hit_at_5:>5.2f} {s.mrr:>6.3f} "
            f"{s.citation_rate:>6.1%} "
            f"{s.mean_retrieval_ms:>6.0f} {s.mean_generation_ms:>6.0f} "
            f"{s.p95_total_ms:>8.0f} "
            f"${s.mean_cost_usd:>8.5f}"
        )

    total_cost_all = sum(s.total_cost_usd for s in summaries)
    print(f"\n[INFO] Throughput: {throughput} queries/s  |  "
          f"Wall time: {wall_elapsed:.0f}s  |  "
          f"Total queries: {len(all_rows)}  |  "
          f"Total cost: ${total_cost_all:.4f}")
    print(f"\n[INFO] Written to {args.out_dir}/")
    print(f"         server_results_ragas.md      -- RAGAS quality metrics")
    print(f"         server_results_retrieval.md  -- Hit/MRR/NDCG per config")
    print(f"         server_results_latency.md    -- P50/P95/P99 latency")
    print(f"         server_results_cost.md       -- token usage and cost")
    print(f"         server_results_quality.md    -- citation/null/error rates")
    print(f"\n[NEXT] Generate plots:  python eval/visualize.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
