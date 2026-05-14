"""
Portfolio-quality visualization suite for the Neuro Gym RAG evaluation results.

Reads from eval/server_results/server_results_summary.json and generates
5 publication-ready plots for the portfolio website and study showcase.

Plots generated:
  1. radar_chart.png          — RAGAS 4-metric profile overlay for all 8 configs
  2. faithfulness_bar.png     — Faithfulness × config (primary anti-hallucination metric)
  3. intent_heatmap.png       — Faithfulness per (intent × config) heatmap
  4. latency_scatter.png      — Retrieval latency vs. context precision scatter
  5. intent_breakdown.png     — Per-intent grouped bar: faithfulness + answer relevancy

Usage:
    python eval/visualize.py
    python eval/visualize.py --results eval/server_results/server_results_summary.json
    python eval/visualize.py --out-dir eval/plots --dpi 300

All plots use a consistent colour palette and are ready to embed directly
into a portfolio website or PDF report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Field normaliser ──────────────────────────────────────────────────────────
# ragas_results_summary.json uses mean_* prefixes; server_results_summary.json
# uses bare names. Normalise to bare names so all plots work with either file.

def _normalise(s: dict) -> dict:
    mapping = {
        "mean_faithfulness":       "faithfulness",
        "mean_answer_relevancy":   "answer_relevancy",
        "mean_context_precision":  "context_precision",
        "mean_context_recall":     "context_recall",
        "mean_retrieval_ms":       "mean_retrieval_ms",   # same in both
        "mean_generation_ms":      "mean_generation_ms",  # same in both
        "n_questions":             "n_questions",
        "n_errors":                "n_errors",
    }
    out = dict(s)
    for src, dst in mapping.items():
        if src in s and dst not in s:
            out[dst] = s[src]
    # Normalise by_intent: ragas uses bare faithfulness/answer_relevancy already
    return out


# ── Config label helpers ───────────────────────────────────────────────────────

def _short_label(config: str) -> str:
    """'F — all + BM25' → 'F'"""
    return config.split(" ")[0].strip()


def _medium_label(config: str) -> str:
    """'F — all + BM25' → 'F\nall+BM25'"""
    parts = config.split(" — ", 1)
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    return config


# ── Colour palette ─────────────────────────────────────────────────────────────

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]

_HIGHLIGHT = "#C44E52"  # production config F


def _bar_colors(labels: list[str]) -> list[str]:
    """Highlight config F in accent colour, grey out the rest."""
    return [
        _HIGHLIGHT if label.startswith("F") else "#6B9FD4"
        for label in labels
    ]


# ── Plot 1: Radar chart ───────────────────────────────────────────────────────

def _plot_radar(summaries: list[dict], out: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    keys    = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Filter to summaries that have RAGAS data
    valid = [s for s in summaries if s.get("faithfulness", 0) > 0]
    if not valid:
        print("[WARN] No RAGAS data found — skipping radar chart")
        return

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.7)

    for i, s in enumerate(valid):
        values = [s.get(k) or 0.0 for k in keys]
        values += values[:1]
        color = _HIGHLIGHT if s["config"].startswith("F") else _PALETTE[i % len(_PALETTE)]
        lw    = 2.5 if s["config"].startswith("F") else 1.2
        alpha = 0.25 if s["config"].startswith("F") else 0.08
        ax.plot(angles, values, color=color, linewidth=lw, label=_short_label(s["config"]))
        ax.fill(angles, values, color=color, alpha=alpha)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=9, title="Config", title_fontsize=10,
    )
    ax.set_title(
        "RAGAS Quality Profile\nAll 8 Retrieval Configurations",
        fontsize=13, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Plot 2: Faithfulness bar chart ─────────────────────────────────────────────

def _plot_faithfulness_bar(summaries: list[dict], out: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    labels = [_short_label(s["config"]) for s in summaries]
    values = [s.get("faithfulness") or 0.0 for s in summaries]
    colors = _bar_colors(labels)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.65)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Retrieval Configuration", fontsize=11)
    ax.set_ylabel("Faithfulness Score [0–1]", fontsize=11)
    ax.set_title(
        "Faithfulness by Retrieval Config\n"
        "(fraction of answer claims grounded in retrieved context — higher = less hallucination)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation for production default
    prod_idx = next((i for i, l in enumerate(labels) if l == "F"), None)
    if prod_idx is not None:
        ax.annotate(
            "production\ndefault",
            xy=(prod_idx, values[prod_idx]),
            xytext=(prod_idx + 0.6, values[prod_idx] - 0.08),
            arrowprops=dict(arrowstyle="->", color=_HIGHLIGHT),
            fontsize=8, color=_HIGHLIGHT,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Plot 3: Intent × config faithfulness heatmap ──────────────────────────────

def _plot_intent_heatmap(summaries: list[dict], out: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    intents = ["factual", "trend", "comparison", "coaching", "visual"]
    configs = [_short_label(s["config"]) for s in summaries]

    matrix = []
    for s in summaries:
        row = [
            s.get("by_intent", {}).get(intent, {}).get("faithfulness", 0.0) or 0.0
            for intent in intents
        ]
        matrix.append(row)

    data = np.array(matrix)  # shape: (n_configs, n_intents)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(intents)))
    ax.set_xticklabels([i.capitalize() for i in intents], fontsize=11)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)

    # Annotate cells
    for i in range(len(configs)):
        for j in range(len(intents)):
            val = data[i, j]
            text_color = "white" if val < 0.35 or val > 0.80 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Faithfulness", fontsize=10)

    ax.set_xlabel("Query Intent", fontsize=11)
    ax.set_ylabel("Retrieval Config", fontsize=11)
    ax.set_title(
        "Faithfulness Heatmap — Config × Intent\n"
        "(green = high faithfulness / low hallucination)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Plot 4: Latency vs. quality scatter ───────────────────────────────────────

def _plot_latency_scatter(summaries: list[dict], out: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    labels     = [_short_label(s["config"]) for s in summaries]
    latencies  = [s.get("mean_retrieval_ms", 0) for s in summaries]
    faiths     = [s.get("faithfulness") or 0.0 for s in summaries]
    sizes      = [s.get("hit_at_5", 0) * 300 + 50 for s in summaries]
    colors     = [_HIGHLIGHT if l == "F" else "#6B9FD4" for l in labels]

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(latencies, faiths, s=sizes, c=colors, alpha=0.85,
                    edgecolors="white", linewidths=0.8)

    for label, x, y in zip(labels, latencies, faiths):
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(8, 4),
            fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Mean Retrieval Latency (ms) ↓ lower is better", fontsize=11)
    ax.set_ylabel("Faithfulness ↑ higher is better", fontsize=11)
    ax.set_title(
        "Quality vs. Latency Trade-off\n"
        "(bubble size = Hit@5; red = production config F)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Plot 5: Per-intent grouped bar ────────────────────────────────────────────

def _plot_intent_breakdown(summaries: list[dict], out: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    intents = ["factual", "trend", "comparison", "coaching", "visual"]

    fig, axes = plt.subplots(1, len(intents), figsize=(18, 5), sharey=True)
    fig.suptitle(
        "Faithfulness & Answer Relevancy by Intent\n(all configs)",
        fontsize=13, fontweight="bold",
    )

    for ax, intent in zip(axes, intents):
        labels = [_short_label(s["config"]) for s in summaries]
        faiths = [
            s.get("by_intent", {}).get(intent, {}).get("faithfulness", 0.0) or 0.0
            for s in summaries
        ]
        ars = [
            s.get("by_intent", {}).get(intent, {}).get("answer_relevancy", 0.0) or 0.0
            for s in summaries
        ]

        x = np.arange(len(labels))
        w = 0.38

        b1 = ax.bar(x - w / 2, faiths, w, label="Faithfulness", color="#4C72B0", alpha=0.85)
        b2 = ax.bar(x + w / 2, ars,    w, label="Ans. Relevancy", color="#DD8452", alpha=0.85)

        ax.set_title(intent.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Score [0–1]", fontsize=10)
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Bonus: Retrieval metrics bar chart ────────────────────────────────────────

def _plot_retrieval_bars(summaries: list[dict], out: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics_cfg = [
        ("Hit@5",    "hit_at_5",    "tab:blue"),
        ("MRR",      "mrr",         "tab:green"),
        ("NDCG@5",   "ndcg_at_5",   "tab:orange"),
        ("Recall@5", "recall_at_5", "tab:purple"),
    ]
    labels = [_short_label(s["config"]) for s in summaries]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Retrieval Metrics — All 8 Configurations", fontsize=14, fontweight="bold")

    for ax, (name, key, color) in zip(axes.flat, metrics_cfg):
        vals = [s.get(key, 0) or 0 for s in summaries]
        bar_colors = [_HIGHLIGHT if l == "F" else color for l in labels]
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "eval" / "server_results" / "server_results_summary.json",
        help="Path to summary JSON (server_results_summary.json or ragas_results_summary.json)",
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Shortcut: load from eval/ragas_results_summary.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "eval" / "plots",
        help="Output directory for plot PNGs",
    )
    parser.add_argument("--dpi", type=int, default=180,
                        help="Plot resolution (default 180 — good for web)")
    args = parser.parse_args()

    if args.ragas:
        args.results = ROOT / "eval" / "ragas_results_summary.json"

    if not args.results.is_file():
        print(
            f"[ERROR] Results file not found: {args.results}\n"
            f"        Run  python eval/run_server_eval.py  first.",
            file=sys.stderr,
        )
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
    except ImportError:
        print("[ERROR] matplotlib not installed — pip install matplotlib", file=sys.stderr)
        return 1

    summaries: list[dict[str, Any]] = json.loads(args.results.read_text(encoding="utf-8"))
    summaries = [_normalise(s) for s in summaries]
    print(f"[INFO] Loaded {len(summaries)} config summary/ies from {args.results}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing plots to {args.out_dir}/\n")

    _plot_radar(summaries,           args.out_dir / "radar_chart.png",        args.dpi)
    _plot_faithfulness_bar(summaries, args.out_dir / "faithfulness_bar.png",   args.dpi)
    _plot_intent_heatmap(summaries,  args.out_dir / "intent_heatmap.png",     args.dpi)
    _plot_latency_scatter(summaries, args.out_dir / "latency_scatter.png",    args.dpi)
    _plot_intent_breakdown(summaries, args.out_dir / "intent_breakdown.png",  args.dpi)
    _plot_retrieval_bars(summaries,  args.out_dir / "retrieval_metrics.png",  args.dpi)

    print(f"\n[INFO] 6 plots saved to {args.out_dir}/")
    print("[INFO] Upload eval/plots/*.png to your portfolio site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
