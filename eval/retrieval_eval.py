"""
Retrieval evaluation harness for Neuro_Gym.

Runs every retrieval configuration (A through H) against a curated golden set
of athlete-tagged questions and reports per-config Hit@K, MRR, NDCG@K,
Recall@5, and retrieval latency. Outputs raw JSON, an aggregate JSON, a
markdown table for the README, and a comparison bar chart.

Usage (from project root):
    python eval/retrieval_eval.py
    python eval/retrieval_eval.py --configs F G --limit 5
    python eval/retrieval_eval.py --no-chart

Outputs (overwritten on each run):
    eval/results_raw.json       — every (config, question) row
    eval/results_summary.json   — aggregated metrics per config
    eval/results_summary.md     — markdown table, paste straight into README
    eval/results_chart.png      — bar chart (one panel per metric)

This module deliberately does not start the FastAPI server. It imports the
retrieval pipeline directly, which means any improvement to retrieve() shows
up in numbers without HTTP/serialization noise polluting the latency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so `pipeline` / `app` / `config`
# import cleanly regardless of where this script is run from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Golden eval set
# ---------------------------------------------------------------------------
# Each entry is a question with the ground-truth athlete IDs that retrieval
# *must* surface for the answer to be groundable. Questions are drawn from the
# Streamlit golden set — only the ones with unambiguous athlete ground-truth
# are kept, so retrieval metrics are interpretable.

GOLDEN_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "Q01",
        "query": "What was athlete_00042's squat weight and RPE in week 8?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "factual",
        "difficulty": "easy",
    },
    {
        "id": "Q02",
        "query": "How many sets and reps did athlete_00117 do on bench press during the intensification phase?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "factual",
        "difficulty": "medium",
    },
    {
        "id": "Q03",
        "query": "What accessories did athlete_00250 perform on lower body days in week 3?",
        "gt_athlete_ids": ["athlete_00250"],
        "intent": "factual",
        "difficulty": "medium",
    },
    {
        "id": "Q04",
        "query": "What is athlete_00089's Dots score and what training level does that correspond to?",
        "gt_athlete_ids": ["athlete_00089"],
        "intent": "factual",
        "difficulty": "easy",
    },
    {
        "id": "Q05",
        "query": "What are athlete_00033's peak squat, bench, and deadlift competition numbers?",
        "gt_athlete_ids": ["athlete_00033"],
        "intent": "factual",
        "difficulty": "easy",
    },
    {
        "id": "Q06",
        "query": "What primary program did athlete_00178 run and did they have a secondary program?",
        "gt_athlete_ids": ["athlete_00178"],
        "intent": "factual",
        "difficulty": "medium",
    },
    {
        "id": "Q07",
        "query": "What happened to athlete_00042's squat load in week 10 — was it a deload week?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "trend",
        "difficulty": "medium",
    },
    {
        "id": "Q08",
        "query": "What is athlete_00301's bodyweight and which IPF weight class do they compete in?",
        "gt_athlete_ids": ["athlete_00301"],
        "intent": "factual",
        "difficulty": "easy",
    },
    {
        "id": "Q09",
        "query": "How did athlete_00117's deadlift progress from week 1 to week 12?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "trend",
        "difficulty": "hard",
    },
    {
        "id": "Q10",
        "query": "How does athlete_00089's training volume change between accumulation and realisation?",
        "gt_athlete_ids": ["athlete_00089"],
        "intent": "trend",
        "difficulty": "hard",
    },
    {
        "id": "Q11",
        "query": "Can you describe what athlete_00042's progression chart looks like?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "visual",
        "difficulty": "medium",
    },
    {
        "id": "Q12",
        "query": "What does athlete_00117's volume radar chart show about upper vs lower body training?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "visual",
        "difficulty": "medium",
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _first_hit_rank(retrieved: list[str], gt: set[str]) -> int | None:
    """Return the 1-indexed rank of the first retrieved athlete that is in gt,
    or None if no hit in the list."""
    for rank, aid in enumerate(retrieved, start=1):
        if aid in gt:
            return rank
    return None


def _hit_at_k(retrieved: list[str], gt: set[str], k: int) -> int:
    return int(any(aid in gt for aid in retrieved[:k]))


def _recall_at_k(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    found = sum(1 for aid in retrieved[:k] if aid in gt)
    return found / len(gt)


def _ndcg_at_k(retrieved: list[str], gt: set[str], k: int) -> float:
    """Binary-relevance NDCG. Score = sum(rel_i / log2(i+1)) / IDCG."""
    if not gt:
        return 0.0
    dcg = 0.0
    for i, aid in enumerate(retrieved[:k], start=1):
        if aid in gt:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    config: str
    question_id: str
    query: str
    intent: str
    difficulty: str
    gt_athlete_ids: list[str]
    retrieved_athlete_ids: list[str]
    first_hit_rank: int | None
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    recall_at_5: float
    ndcg_at_5: float
    mrr: float
    retrieval_ms: int
    top_score: float
    error: str | None = None


@dataclass
class ConfigSummary:
    config: str
    n_questions: int
    n_errors: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_5: float
    recall_at_5: float
    mean_retrieval_ms: float
    median_retrieval_ms: float
    mean_top_score: float
    by_intent: dict[str, dict[str, float]] = field(default_factory=dict)


def _load_env_from_dotenv() -> None:
    """Load .env into os.environ — same logic as app/main.py:_load_env so the
    eval runs in the same environment as the API."""
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


def _init_pipeline_state():
    """Initialise the heavyweight pipeline objects once. Returns a dict that
    can be passed into retrieve() for every (config, question)."""
    from pipeline.ingestion.collection import get_client
    from pipeline.ingestion.bm_index import load_bm_index

    qdrant = get_client(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
    )

    bm25, corpus = load_bm_index(
        index_path=ROOT / "hf_pull" / "k2p" / "gym-rag-embeddings" / "BM_index.pkl",
        corpus_path=ROOT / "hf_pull" / "k2p" / "gym-rag-embeddings" / "BM_corpus.json",
    )

    return {"qdrant": qdrant, "bm25": bm25, "corpus": corpus}


def _evaluate_one(
    cfg_name: str,
    cfg,
    question: dict[str, Any],
    state: dict[str, Any],
) -> QuestionResult:
    from pipeline.retrieval.retrieve import retrieve

    gt = set(question["gt_athlete_ids"])
    retrieved_ids: list[str] = []
    top_score = 0.0
    error: str | None = None
    t0 = time.perf_counter()

    try:
        context = retrieve(
            query=question["query"],
            bm25=state["bm25"],
            corpus=state["corpus"],
            client=state["qdrant"],
            config=cfg,
            top_k_athletes=5,
        )
        retrieved_ids = context.get("athlete_ids", []) or []
        sources = context.get("sources", []) or []
        if sources:
            top_score = float(sources[0].get("score", 0.0))
    except Exception as e:  # noqa: BLE001 — eval runner must not abort
        error = f"{type(e).__name__}: {e}"

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    rank = _first_hit_rank(retrieved_ids, gt)

    return QuestionResult(
        config=cfg_name,
        question_id=question["id"],
        query=question["query"],
        intent=question["intent"],
        difficulty=question["difficulty"],
        gt_athlete_ids=list(gt),
        retrieved_athlete_ids=retrieved_ids,
        first_hit_rank=rank,
        hit_at_1=_hit_at_k(retrieved_ids, gt, 1),
        hit_at_3=_hit_at_k(retrieved_ids, gt, 3),
        hit_at_5=_hit_at_k(retrieved_ids, gt, 5),
        recall_at_5=round(_recall_at_k(retrieved_ids, gt, 5), 4),
        ndcg_at_5=round(_ndcg_at_k(retrieved_ids, gt, 5), 4),
        mrr=round(1.0 / rank, 4) if rank else 0.0,
        retrieval_ms=elapsed_ms,
        top_score=round(top_score, 4),
        error=error,
    )


def _summarise(rows: list[QuestionResult]) -> ConfigSummary:
    cfg = rows[0].config
    valid = [r for r in rows if r.error is None]
    n = len(valid)

    if n == 0:
        return ConfigSummary(
            config=cfg, n_questions=len(rows), n_errors=len(rows),
            hit_at_1=0, hit_at_3=0, hit_at_5=0, mrr=0,
            ndcg_at_5=0, recall_at_5=0,
            mean_retrieval_ms=0, median_retrieval_ms=0, mean_top_score=0,
        )

    by_intent: dict[str, list[QuestionResult]] = {}
    for r in valid:
        by_intent.setdefault(r.intent, []).append(r)

    intent_breakdown = {
        intent: {
            "n": len(rs),
            "hit_at_5": round(sum(r.hit_at_5 for r in rs) / len(rs), 4),
            "mrr": round(sum(r.mrr for r in rs) / len(rs), 4),
        }
        for intent, rs in by_intent.items()
    }

    latencies = [r.retrieval_ms for r in valid]

    return ConfigSummary(
        config=cfg,
        n_questions=len(rows),
        n_errors=len(rows) - n,
        hit_at_1=round(sum(r.hit_at_1 for r in valid) / n, 4),
        hit_at_3=round(sum(r.hit_at_3 for r in valid) / n, 4),
        hit_at_5=round(sum(r.hit_at_5 for r in valid) / n, 4),
        mrr=round(sum(r.mrr for r in valid) / n, 4),
        ndcg_at_5=round(sum(r.ndcg_at_5 for r in valid) / n, 4),
        recall_at_5=round(sum(r.recall_at_5 for r in valid) / n, 4),
        mean_retrieval_ms=round(statistics.mean(latencies), 1),
        median_retrieval_ms=round(statistics.median(latencies), 1),
        mean_top_score=round(sum(r.top_score for r in valid) / n, 4),
        by_intent=intent_breakdown,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_markdown_table(summaries: list[ConfigSummary], path: Path) -> None:
    """Write a clean markdown table that drops straight into a README."""
    lines: list[str] = []
    lines.append("# Retrieval Evaluation\n")
    lines.append(
        "Evaluation of all 8 retrieval configurations against a curated "
        f"golden set of {summaries[0].n_questions} athlete-tagged questions. "
        "Metrics computed per (config, question), then averaged across the set.\n"
    )
    lines.append("## Results by configuration\n")
    lines.append(
        "| Config | Hit@1 | Hit@3 | Hit@5 | MRR | NDCG@5 | Recall@5 | "
        "Mean latency (ms) | Median latency (ms) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for s in summaries:
        lines.append(
            f"| {s.config} | {s.hit_at_1:.2f} | {s.hit_at_3:.2f} | "
            f"{s.hit_at_5:.2f} | {s.mrr:.3f} | {s.ndcg_at_5:.3f} | "
            f"{s.recall_at_5:.2f} | {s.mean_retrieval_ms:.0f} | "
            f"{s.median_retrieval_ms:.0f} |"
        )

    lines.append("\n## Hit@5 by intent\n")

    intents: list[str] = []
    for s in summaries:
        for k in s.by_intent:
            if k not in intents:
                intents.append(k)
    if intents:
        header = "| Config | " + " | ".join(intents) + " |"
        sep = "|---|" + "|".join(["---:"] * len(intents)) + "|"
        lines.append(header)
        lines.append(sep)
        for s in summaries:
            row = [s.config]
            for intent in intents:
                cell = s.by_intent.get(intent, {})
                row.append(f"{cell.get('hit_at_5', 0):.2f}" if cell else "—")
            lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Reading the table\n")
    lines.append(
        "- **Hit@K** — fraction of questions where at least one ground-truth "
        "athlete appeared in the top-K retrieved.\n"
        "- **MRR** — mean reciprocal rank of the first correct athlete "
        "(0 if not in top-5).\n"
        "- **NDCG@5** — discounted cumulative gain, position-weighted.\n"
        "- **Recall@5** — fraction of ground-truth athletes recovered in top-5.\n"
        "- **Latency** — wall-clock time inside `retrieve()`, excluding network "
        "and FastAPI overhead. Lower is better.\n"
    )
    lines.append(
        "\nConfig F (all collections + BM25 + RRF) is the production default. "
        "Configs A–C and H exist as ablation baselines to quantify what each "
        "retrieval source contributes; G adds a cross-encoder reranker on top "
        "of F and trades latency for ranking quality.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_chart(summaries: list[ConfigSummary], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO-EVAL] matplotlib not installed — skipping chart.")
        return

    cfg_labels = [s.config.split(" — ")[0] for s in summaries]
    metrics = [
        ("Hit@5", [s.hit_at_5 for s in summaries], "tab:blue"),
        ("MRR",      [s.mrr for s in summaries],      "tab:green"),
        ("NDCG@5",   [s.ndcg_at_5 for s in summaries], "tab:orange"),
        ("Mean latency (ms)",
            [s.mean_retrieval_ms for s in summaries], "tab:red"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (name, vals, color) in zip(axes.flat, metrics):
        bars = ax.bar(cfg_labels, vals, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Retrieval config")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            label = f"{val:.2f}" if name != "Mean latency (ms)" else f"{int(val)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(
        "Neuro_Gym retrieval evaluation — 8 configurations",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Subset of config keys to evaluate (e.g. F G). Default: all."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N golden questions (for quick smoke runs)."
    )
    parser.add_argument(
        "--no-chart", action="store_true",
        help="Skip generating the matplotlib comparison chart."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "eval",
        help="Directory to write outputs into. Default: eval/"
    )
    args = parser.parse_args()

    _load_env_from_dotenv()

    print(f"[INFO-EVAL] Loading pipeline state...")
    state = _init_pipeline_state()
    print(f"[INFO-EVAL] BM25 corpus: {len(state['corpus'])} docs")

    from pipeline.retrieval.retrieve import CONFIGS

    cfg_keys = args.configs or list(CONFIGS.keys())
    cfg_keys = [k for k in cfg_keys if k in CONFIGS] or list(CONFIGS.keys())

    questions = GOLDEN_QUESTIONS[: args.limit] if args.limit else GOLDEN_QUESTIONS
    print(
        f"[INFO-EVAL] Evaluating {len(cfg_keys)} config(s) × "
        f"{len(questions)} question(s) = {len(cfg_keys) * len(questions)} runs"
    )

    all_rows: list[QuestionResult] = []
    summaries: list[ConfigSummary] = []

    for cfg_name in cfg_keys:
        cfg = CONFIGS[cfg_name]
        print(f"\n[INFO-EVAL] === {cfg_name} ===")
        rows: list[QuestionResult] = []
        for q in questions:
            row = _evaluate_one(cfg_name, cfg, q, state)
            rows.append(row)
            all_rows.append(row)
            status = "✓" if row.first_hit_rank == 1 else (
                f"#{row.first_hit_rank}" if row.first_hit_rank else "✗"
            )
            err = f"  ERROR: {row.error}" if row.error else ""
            print(
                f"  {q['id']} [{status}]  {row.retrieval_ms}ms  "
                f"{q['query'][:60]}{'…' if len(q['query']) > 60 else ''}{err}"
            )
        summaries.append(_summarise(rows))

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.out_dir / "results_raw.json"
    raw_path.write_text(
        json.dumps([asdict(r) for r in all_rows], indent=2),
        encoding="utf-8",
    )

    summary_path = args.out_dir / "results_summary.json"
    summary_path.write_text(
        json.dumps([asdict(s) for s in summaries], indent=2),
        encoding="utf-8",
    )

    md_path = args.out_dir / "results_summary.md"
    _write_markdown_table(summaries, md_path)

    print("\n[INFO-EVAL] === SUMMARY ===")
    print(
        f"{'Config':<24} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} "
        f"{'MRR':>6} {'NDCG@5':>7} {'mean ms':>8}"
    )
    print("-" * 72)
    for s in summaries:
        print(
            f"{s.config:<24} {s.hit_at_1:>6.2f} {s.hit_at_3:>6.2f} "
            f"{s.hit_at_5:>6.2f} {s.mrr:>6.3f} {s.ndcg_at_5:>7.3f} "
            f"{s.mean_retrieval_ms:>8.0f}"
        )

    print(f"\n[INFO-EVAL] Wrote {raw_path.relative_to(ROOT)}")
    print(f"[INFO-EVAL] Wrote {summary_path.relative_to(ROOT)}")
    print(f"[INFO-EVAL] Wrote {md_path.relative_to(ROOT)}")

    if not args.no_chart:
        chart_path = args.out_dir / "results_chart.png"
        _write_chart(summaries, chart_path)
        if chart_path.is_file():
            print(f"[INFO-EVAL] Wrote {chart_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
