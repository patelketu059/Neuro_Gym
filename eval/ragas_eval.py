"""
RAGAS evaluation harness for the full Neuro_Gym A→R→G chain.

Where retrieval_eval.py judges the *retriever* in isolation (Hit@K / MRR /
NDCG), this module judges the *full pipeline* — retrieval + generation —
using LLM-as-judge metrics from RAGAS:

    - faithfulness          Does the answer make claims grounded in the contexts?
    - answer_relevancy      Does the answer actually address the question?
    - context_precision     Is the retrieved context informative for the question?
                            (reference-free variant is used by default)
    - context_recall        (only if reference answers are provided)
                            Does the retrieved context cover the ground truth?

The judge LLM is Gemini 2.0 Flash via langchain-google-genai. Embeddings use
the same Google embeddings model RAGAS recommends.

Usage (from project root):

    pip install ragas langchain-google-genai datasets
    python eval/ragas_eval.py
    python eval/ragas_eval.py --config "F — all + BM25" --limit 5
    python eval/ragas_eval.py --references eval/golden_answers.json

Reference answers (optional) — JSON file like:
    {
        "Q01": "athlete_00042 squatted 215 kg at RPE 9.5 in week 8.",
        "Q02": "athlete_00117 used 4 sets x 5 reps in the intensification block.",
        ...
    }
Without a references file, context_recall is skipped (the metric requires
ground truth answers to score retrieval coverage).

Outputs (in eval/, overwritten on each run):

    eval/ragas_results_raw.json       — per-question scores + answer + contexts
    eval/ragas_results_summary.json   — aggregate scores + by_intent breakdown
    eval/ragas_results_summary.md     — markdown table for the README

This is the second-stage portfolio piece — retrieval_eval.py shows that the
*retriever* picks the right athlete; this file shows that the *pipeline* gives
a faithful, on-topic answer grounded in real data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the curated golden set rather than duplicating it. Single source of
# truth for which questions both eval suites are scored against.
from eval.retrieval_eval import (  # noqa: E402
    GOLDEN_QUESTIONS,
    _init_pipeline_state,
    _load_env_from_dotenv,
)


_SOURCE_HEADER_RE = re.compile(r"^---\s*SOURCE:.*?---\s*$", re.MULTILINE)


@dataclass
class RagasRow:
    config: str
    question_id: str
    intent: str
    difficulty: str
    query: str
    answer: str
    contexts: list[str]
    retrieval_ms: int
    generation_ms: int
    reference: str | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    error: str | None = None


@dataclass
class RagasSummary:
    config: str
    n_questions: int
    n_errors: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: float | None
    mean_retrieval_ms: float
    mean_generation_ms: float
    by_intent: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline runner — produces (answer, contexts) for one question
# ---------------------------------------------------------------------------

def _split_text_context(text_context: str) -> list[str]:
    """Convert the assembled `text_context` string into a list of passages
    suitable for RAGAS. We split on the SOURCE header lines that
    `assemble_context` injects between passages."""
    if not text_context:
        return []
    chunks = _SOURCE_HEADER_RE.split(text_context)
    cleaned = [c.strip() for c in chunks if c and c.strip()]
    return cleaned or [text_context.strip()]


def _run_chain_capture(
    question: dict[str, Any],
    cfg,
    state: dict[str, Any],
    gemini,
) -> tuple[str, list[str], int, int, str | None]:
    """Run augmentation → retrieval → generation, return (answer, contexts,
    retrieval_ms, generation_ms, error)."""
    from app.augmentation import augment
    from app.chain import generation, retrieval as do_retrieval
    from app.memory import get_or_create_memory

    session_id = f"ragas_{question['id']}"
    memory = get_or_create_memory(session_id, gemini=gemini)

    inputs: dict[str, Any] = {
        "query": question["query"],
        "retrieval_query": question["query"],
        "query_image_path": None,
        "bm25": state["bm25"],
        "corpus": state["corpus"],
        "client": state["qdrant"],
        "gemini": gemini,
        "retrieval_config": cfg,
        "reranker_model": None,
        "pdf_dir": "",
        "memory": memory,
        "use_hyde": True,
    }

    t_retrieval = 0
    answer = ""
    contexts: list[str] = []
    error: str | None = None

    try:
        t0 = time.perf_counter()
        augmented = augment(inputs)
        with_context = do_retrieval(augmented)
        t_retrieval = int((time.perf_counter() - t0) * 1000)

        ctx = with_context["context"]
        contexts = _split_text_context(ctx.get("text_context", ""))

        gen_result = generation(with_context)
        answer = gen_result["response"]
        generation_ms = int(gen_result.get("generation_ms", 0))
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        generation_ms = 0

    return answer, contexts, t_retrieval, generation_ms, error


# ---------------------------------------------------------------------------
# RAGAS scoring
# ---------------------------------------------------------------------------

def _build_dataset(rows: list[RagasRow]):
    """Build a RAGAS EvaluationDataset from collected pipeline outputs."""
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    for r in rows:
        if r.error or not r.answer or not r.contexts:
            continue
        sample_kwargs: dict[str, Any] = {
            "user_input": r.query,
            "response": r.answer,
            "retrieved_contexts": r.contexts,
        }
        if r.reference:
            sample_kwargs["reference"] = r.reference
        samples.append(SingleTurnSample(**sample_kwargs))
    return EvaluationDataset(samples=samples), samples


def _score_with_ragas(rows: list[RagasRow], references: dict[str, str]) -> None:
    """Run RAGAS evaluation in-place — populates .faithfulness / .answer_relevancy
    / .context_precision / .context_recall on each row that has valid pipeline
    output."""
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )

    try:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
    except ImportError as e:
        raise RuntimeError(
            "RAGAS evaluation needs langchain-google-genai. "
            "Install with: pip install langchain-google-genai"
        ) from e

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. RAGAS judge LLM cannot be initialised."
        )

    judge = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.0,
    )
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    evaluator_llm = LangchainLLMWrapper(judge)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedder)

    # Decide metric set — context_recall only if at least one reference
    has_references = any(r.reference for r in rows)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]
    if has_references:
        metrics.append(LLMContextRecall(llm=evaluator_llm))

    dataset, included_samples = _build_dataset(rows)
    print(
        f"[INFO-RAGAS] Scoring {len(included_samples)} sample(s) with "
        f"{len(metrics)} metric(s) (refs={'on' if has_references else 'off'})"
    )

    if not included_samples:
        print("[WARN-RAGAS] No valid samples — skipping evaluation.")
        return

    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # Normalise to a list of dicts so we can map back to rows by index.
    df = eval_result.to_pandas()
    valid_iter = (i for i, r in enumerate(rows) if not r.error and r.answer and r.contexts)
    for valid_idx, df_row in zip(valid_iter, df.to_dict(orient="records")):
        target = rows[valid_idx]
        target.faithfulness = _safe_float(df_row.get("faithfulness"))
        target.answer_relevancy = _safe_float(df_row.get("answer_relevancy"))
        target.context_precision = _safe_float(
            df_row.get("llm_context_precision_without_reference")
            or df_row.get("context_precision")
        )
        target.context_recall = _safe_float(
            df_row.get("context_recall")
            or df_row.get("llm_context_recall")
        )


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _summarise(rows: list[RagasRow]) -> RagasSummary:
    cfg = rows[0].config
    valid = [r for r in rows if r.error is None and r.answer]
    n = len(valid)

    def _mean(attr: str) -> float:
        vals = [getattr(r, attr) for r in valid if getattr(r, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _mean_optional(attr: str) -> float | None:
        vals = [getattr(r, attr) for r in valid if getattr(r, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_intent: dict[str, list[RagasRow]] = {}
    for r in valid:
        by_intent.setdefault(r.intent, []).append(r)
    intent_breakdown = {
        intent: {
            "n": len(rs),
            "faithfulness": round(
                sum(x.faithfulness for x in rs if x.faithfulness is not None)
                / max(1, sum(1 for x in rs if x.faithfulness is not None)), 4),
            "answer_relevancy": round(
                sum(x.answer_relevancy for x in rs if x.answer_relevancy is not None)
                / max(1, sum(1 for x in rs if x.answer_relevancy is not None)), 4),
        }
        for intent, rs in by_intent.items()
    }

    return RagasSummary(
        config=cfg,
        n_questions=len(rows),
        n_errors=len(rows) - n,
        mean_faithfulness=_mean("faithfulness"),
        mean_answer_relevancy=_mean("answer_relevancy"),
        mean_context_precision=_mean("context_precision"),
        mean_context_recall=_mean_optional("context_recall"),
        mean_retrieval_ms=round(sum(r.retrieval_ms for r in valid) / max(1, n), 1),
        mean_generation_ms=round(sum(r.generation_ms for r in valid) / max(1, n), 1),
        by_intent=intent_breakdown,
    )


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def _write_markdown(summaries: list[RagasSummary], path: Path) -> None:
    lines: list[str] = []
    lines.append("# RAGAS End-to-End Evaluation\n")
    lines.append(
        "LLM-as-judge evaluation of the full A→R→G chain. Each question is "
        "answered by the production pipeline (augmentation → retrieval → "
        "Gemini generation), then scored by Gemini 2.0 Flash as judge using "
        "RAGAS metrics.\n"
    )

    has_recall = any(s.mean_context_recall is not None for s in summaries)
    headers = ["Config", "Faithfulness", "Answer Relevancy", "Context Precision"]
    aligns = ["---", "---:", "---:", "---:"]
    if has_recall:
        headers.append("Context Recall")
        aligns.append("---:")
    headers += ["Retrieval ms", "Generation ms"]
    aligns += ["---:", "---:"]

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(aligns) + "|")

    for s in summaries:
        cells = [
            s.config,
            f"{s.mean_faithfulness:.3f}",
            f"{s.mean_answer_relevancy:.3f}",
            f"{s.mean_context_precision:.3f}",
        ]
        if has_recall:
            cells.append(
                f"{s.mean_context_recall:.3f}"
                if s.mean_context_recall is not None else "—"
            )
        cells += [f"{s.mean_retrieval_ms:.0f}", f"{s.mean_generation_ms:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Reading the metrics\n")
    lines.append(
        "- **Faithfulness** [0–1] — fraction of claims in the answer that are "
        "grounded in the retrieved contexts. Penalises hallucination.\n"
        "- **Answer Relevancy** [0–1] — semantic similarity between the "
        "answer and the question, controlled for off-topic verbosity.\n"
        "- **Context Precision** [0–1] — RAGAS reference-free variant: how "
        "much of the retrieved context is actually useful for answering "
        "the question.\n"
        "- **Context Recall** [0–1, optional] — fraction of facts in the "
        "ground-truth answer that are present in the retrieved contexts. "
        "Only computed when an `eval/golden_answers.json` is supplied.\n"
        "- **Retrieval / Generation ms** — wall-clock latency per stage; "
        "useful for understanding the latency-vs-quality trade-off across "
        "configs (G in particular).\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", default=["F — all + BM25"],
        help="Configs to evaluate. Default: F (production). Pass keys exactly "
             "as they appear in CONFIGS — em-dash, not hyphen."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N golden questions."
    )
    parser.add_argument(
        "--references", type=Path, default=None,
        help="Path to a JSON file mapping question_id → ground-truth answer "
             "string. Enables context_recall."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "eval",
        help="Directory to write outputs into. Default: eval/"
    )
    args = parser.parse_args()

    _load_env_from_dotenv()

    references: dict[str, str] = {}
    if args.references and args.references.is_file():
        references = json.loads(args.references.read_text(encoding="utf-8"))
        print(f"[INFO-RAGAS] Loaded {len(references)} reference answers")

    print("[INFO-RAGAS] Loading pipeline state...")
    state = _init_pipeline_state()
    print(f"[INFO-RAGAS] BM25 corpus: {len(state['corpus'])} docs")

    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[ERROR-RAGAS] GEMINI_API_KEY not set", file=sys.stderr)
        return 2
    gemini = genai.Client(api_key=api_key)

    from pipeline.retrieval.retrieve import CONFIGS

    cfg_keys = [k for k in args.configs if k in CONFIGS]
    if not cfg_keys:
        print(
            f"[ERROR-RAGAS] None of {args.configs} matched CONFIGS keys "
            f"{list(CONFIGS)}", file=sys.stderr,
        )
        return 2

    questions = GOLDEN_QUESTIONS[: args.limit] if args.limit else GOLDEN_QUESTIONS
    print(
        f"[INFO-RAGAS] Running {len(cfg_keys)} config(s) × "
        f"{len(questions)} question(s) through full pipeline..."
    )

    all_rows: list[RagasRow] = []
    summaries: list[RagasSummary] = []

    for cfg_name in cfg_keys:
        cfg = CONFIGS[cfg_name]
        print(f"\n[INFO-RAGAS] === {cfg_name} ===")
        rows: list[RagasRow] = []
        for q in questions:
            answer, contexts, r_ms, g_ms, err = _run_chain_capture(
                q, cfg, state, gemini
            )
            row = RagasRow(
                config=cfg_name,
                question_id=q["id"],
                intent=q["intent"],
                difficulty=q["difficulty"],
                query=q["query"],
                answer=answer,
                contexts=contexts,
                retrieval_ms=r_ms,
                generation_ms=g_ms,
                reference=references.get(q["id"]),
                error=err,
            )
            rows.append(row)
            all_rows.append(row)
            status = "✗" if err else f"{len(contexts)}ctx"
            print(
                f"  {q['id']} [{status}]  ret {r_ms}ms  gen {g_ms}ms  "
                f"{q['query'][:55]}…"
            )

        # Score this config's rows with RAGAS
        try:
            _score_with_ragas(rows, references)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR-RAGAS] Scoring failed for {cfg_name}: {e}")

        summaries.append(_summarise(rows))

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.out_dir / "ragas_results_raw.json"
    raw_path.write_text(
        json.dumps([asdict(r) for r in all_rows], indent=2),
        encoding="utf-8",
    )

    summary_path = args.out_dir / "ragas_results_summary.json"
    summary_path.write_text(
        json.dumps([asdict(s) for s in summaries], indent=2),
        encoding="utf-8",
    )

    md_path = args.out_dir / "ragas_results_summary.md"
    _write_markdown(summaries, md_path)

    print("\n[INFO-RAGAS] === SUMMARY ===")
    print(
        f"{'Config':<24} {'Faith':>7} {'AnsRel':>7} {'CtxP':>7} "
        f"{'CtxR':>7} {'r_ms':>6} {'g_ms':>6}"
    )
    print("-" * 72)
    for s in summaries:
        recall = (
            f"{s.mean_context_recall:>7.3f}"
            if s.mean_context_recall is not None else f"{'—':>7}"
        )
        print(
            f"{s.config:<24} "
            f"{s.mean_faithfulness:>7.3f} "
            f"{s.mean_answer_relevancy:>7.3f} "
            f"{s.mean_context_precision:>7.3f} "
            f"{recall} "
            f"{s.mean_retrieval_ms:>6.0f} "
            f"{s.mean_generation_ms:>6.0f}"
        )

    print(f"\n[INFO-RAGAS] Wrote {raw_path.relative_to(ROOT)}")
    print(f"[INFO-RAGAS] Wrote {summary_path.relative_to(ROOT)}")
    print(f"[INFO-RAGAS] Wrote {md_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
