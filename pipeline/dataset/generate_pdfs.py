"""
generate_pdfs.py — Athlete PDF Report Generator (Phase 1.5)

Page layout (3 pages):
  Page 1 : header + coaching summary + performance metrics + 2 charts
  Page 2 : 1 chart + week-by-week progression table
  Page 3+: full session log (one block per week, all 4 sessions)

Minimalistic light theme. Per-athlete randomised variation seeded from
athlete_id so every PDF is reproducible and the corpus has genuine diversity.

Config file: pdf_config.toml (optional — all settings have hard-coded defaults)

Usage:
    python generate_pdfs.py
    python generate_pdfs.py --config pdf_config.toml
    python generate_pdfs.py --athlete athlete_00042
    python generate_pdfs.py --n-athletes 50 --no-randomise
"""

from __future__ import annotations

import argparse
import io
import random
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)
from config.settings import (PDF_CONFIG_PATH, PDF_DIR)
# ---------------------------------------------------------------------------
# Fixed palette (non-random)
# ---------------------------------------------------------------------------
PAGE_BG   = "#FFFFFF"
TEXT_PRI  = "#1C1C1E"
TEXT_SEC  = "#5F5E5A"
TEXT_MUTED= "#888780"
BORDER    = "#D3D1C7"
ROW_ALT   = "#F5F5F3"
CARD_BG   = "#F5F5F3"

LIFT_COLORS  = {"Squat": "#C94A2A", "Bench": "#185FA5",
                "Deadlift": "#0F6E56", "OHP": "#BA7517"}
PHASE_COLORS = {"accumulation": "#185FA5", "intensification": "#BA7517",
                "realisation": "#C94A2A", "deload": "#888780"}
PHASE_ORDER  = ["accumulation", "intensification", "realisation", "deload"]
LIFT_ORDER   = ["Squat", "Bench", "Deadlift", "OHP"]

MARGIN       = 14 * mm
PAGE_W, PAGE_H = A4

_WIDE_TYPES = {"rpe_heatmap", "lift_progression"}

# ---------------------------------------------------------------------------
# Config — loaded once, controls all defaults and variation ranges
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "variation": {"randomise": True, "seed_offset": 9999},
    "accent": {
        "default": "#C94A2A",
        "palette": ["#C94A2A", "#185FA5", "#0F6E56", "#BA7517", "#7B3FA0"],
    },
    "charts": {
        "n_charts": 3,
        "chart_types": [
            "lift_progression", "peak_comparison", "rpe_heatmap",
            "load_histogram", "volume_area", "strength_radar",
            "scatter_rpe_volume", "phase_boxplot",
        ],
        "marker_density": "medium",
    },
    "metrics_table": {"default_variant": 0, "show_floor_week": True},
    "progression_table": {"delta_lift": "Squat", "row_height": 13},
    "session_log": {"columns": ["Session", "Main lift", "RPE", "Accessories"]},
    "description": {"para_order": [0, 1, 2, 3]},
    "typography": {
        "body_font_size": 8,
        "section_font_size": 6.5,
        "metrics_row_height": 13,
    },
}


def load_config(path: str | None) -> dict:
    """
    Load pdf_config.toml if provided and merge with defaults.
    Returns a flat-ish nested dict. Missing keys fall back to defaults.
    """
    import copy
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    if not path:
        return cfg
    p = Path(path)
    if not p.is_file():
        print(f"  [WARN] Config file not found: {path} — using defaults")
        return cfg
    try:
        import tomllib                         # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib            # pip install tomli for 3.9/3.10
        except ImportError:
            print("  [WARN] tomllib/tomli not available — using defaults")
            return cfg
    with open(p, "rb") as f:
        user = tomllib.load(f)
    # Deep merge: user values override defaults, missing keys kept from defaults
    for section, values in user.items():
        if section in cfg and isinstance(cfg[section], dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    print(f"  [config] Loaded {path}")
    return cfg


# ---------------------------------------------------------------------------
# Metric table row definitions
# ---------------------------------------------------------------------------
_METRIC_VARIANTS = [
    # 0: competition-focused
    [
        ("Competition 1RM (kg)", lambda f, l: f"{f[l]['competition_1rm_kg']:.1f}"),
        ("Week 1 load (kg)",     lambda f, l: f"{f[l]['week_1_kg']:.1f}"),
        ("Peak load (kg)",       lambda f, l: f"{f[l]['week_peak_kg']:.1f}"),
        ("Peak week",            lambda f, l: f"{int(f[l]['peak_week'])}"),
        ("Total gain (kg)",      lambda f, l: f"{f[l]['total_gain_kg']:.1f}"),
        ("Peak % of 1RM",        lambda f, l: f"{f[l]['peak_pct_of_1rm']*100:.1f}%"),
    ],
    # 1: progression-focused (includes floor)
    [
        ("Competition 1RM (kg)", lambda f, l: f"{f[l]['competition_1rm_kg']:.1f}"),
        ("Week 1 load (kg)",     lambda f, l: f"{f[l]['week_1_kg']:.1f}"),
        ("Peak load (kg)",       lambda f, l: f"{f[l]['week_peak_kg']:.1f}"),
        ("Total gain (kg)",      lambda f, l: f"{f[l]['total_gain_kg']:.1f}"),
        ("Floor load (kg)",      lambda f, l: f"{f[l]['week_floor_kg']:.1f}"),
        ("Block phase at peak",  lambda f, l: str(f[l].get("block_phase_at_peak", ""))),
    ],
    # 2: intensity-focused
    [
        ("Competition 1RM (kg)", lambda f, l: f"{f[l]['competition_1rm_kg']:.1f}"),
        ("Peak load (kg)",       lambda f, l: f"{f[l]['week_peak_kg']:.1f}"),
        ("Peak % of 1RM",        lambda f, l: f"{f[l]['peak_pct_of_1rm']*100:.1f}%"),
        ("Peak week",            lambda f, l: f"{int(f[l]['peak_week'])}"),
        ("Total gain (kg)",      lambda f, l: f"{f[l]['total_gain_kg']:.1f}"),
        ("Floor load (kg)",      lambda f, l: f"{f[l]['week_floor_kg']:.1f}"),
    ],
]

_PARA_ORDERS = [
    [0, 1, 2, 3],
    [0, 2, 1, 3],
    [0, 3, 1, 2],
    [0, 1, 3, 2],
]

_SESSION_LOG_VARIANTS = [
    ["Session", "Main lift", "RPE",  "Accessories"],
    ["Session", "RPE",  "Main lift", "Accessories"],
    ["Session", "Main lift", "Accessories", "RPE"],
]

_DELTA_LIFT_VARIANTS = ["Squat", "Deadlift", "Bench"]

_MARKER_DENSITY = {"low": 1.0, "medium": 2.0, "high": 3.5}


def _resolve_variants(athlete_id: str, cfg: dict) -> dict:
    """
    Compute all per-athlete randomised choices from a single seeded RNG.
    When cfg.variation.randomise is False, every athlete gets the config defaults.
    """
    do_rand = cfg["variation"]["randomise"]
    offset  = cfg["variation"]["seed_offset"]
    rng     = random.Random(
        int.from_bytes(athlete_id.encode(), "big") % (2 ** 31) + offset
    )

    def _pick(pool, default_idx=0):
        return rng.choice(pool) if do_rand else pool[default_idx]

    accent_palette = cfg["accent"]["palette"]
    accent_default = cfg["accent"]["default"]

    chart_types    = cfg["charts"]["chart_types"]
    n_charts       = cfg["charts"]["n_charts"]
    density_name   = cfg["charts"]["marker_density"]

    return {
        # 1. Accent colour
        "accent": _pick(accent_palette, 0) if do_rand else accent_default,

        # 2. Metrics table row-set
        "metric_variant": (rng.randint(0, len(_METRIC_VARIANTS) - 1)
                           if do_rand else cfg["metrics_table"]["default_variant"]),

        # 3. Delta lift in progression table
        "delta_lift": (_pick(_DELTA_LIFT_VARIANTS)
                       if do_rand else cfg["progression_table"]["delta_lift"]),

        # 4. Description paragraph order
        "para_order": (_pick(_PARA_ORDERS)
                       if do_rand else cfg["description"]["para_order"]),

        # 5. Session log column order
        "session_cols": (_pick(_SESSION_LOG_VARIANTS)
                         if do_rand else cfg["session_log"]["columns"]),

        # 6. Chart selection (already seeded differently — kept coherent)
        "chart_types_pool": chart_types,
        "n_charts":         n_charts,

        # 7. Chart marker size
        "marker_size": (_MARKER_DENSITY.get(
                            rng.choice(list(_MARKER_DENSITY.keys())), 2.0)
                        if do_rand else _MARKER_DENSITY.get(density_name, 2.0)),

        # 8. Progression table row height
        "prog_row_height": (rng.choice([11, 12, 13, 14])
                            if do_rand else cfg["progression_table"]["row_height"]),

        # 9. Body font size (slight variation for density feel)
        "body_fs": (rng.choice([7.5, 8.0, 8.5])
                    if do_rand else cfg["typography"]["body_font_size"]),

        # 10. Section font size
        "section_fs": (rng.choice([6.0, 6.5, 7.0])
                       if do_rand else cfg["typography"]["section_font_size"]),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rl(h: str) -> colors.HexColor:
    return colors.HexColor(h)


def _ps(name, fn="Helvetica", fs=8, tc=TEXT_PRI, bg=PAGE_BG, **kw):
    return ParagraphStyle(name, fontName=fn, fontSize=fs,
                          textColor=_rl(tc), backColor=_rl(bg), **kw)


def _make_styles(v: dict) -> dict:
    a  = v["accent"]
    bs = v["body_fs"]
    ss = v["section_fs"]
    return {
        "badge":   _ps("badge",  fn="Helvetica-Bold", fs=6,  tc=a,
                        spaceAfter=1, letterSpacing=1.4),
        "aid":     _ps("aid",    fn="Helvetica-Bold", fs=14, spaceAfter=1),
        "sub":     _ps("sub",    fs=7,  tc=TEXT_MUTED),
        "sec":     _ps("sec",    fn="Helvetica-Bold", fs=ss, tc=a,
                        spaceBefore=4, spaceAfter=2, letterSpacing=1.2),
        "body":    _ps("body",   fs=bs, leading=bs+4,  spaceAfter=3),
        "body_sm": _ps("body_sm",fs=bs-0.5, leading=bs+3, spaceAfter=2),
        "muted":   _ps("muted",  fs=7,  tc=TEXT_MUTED),
    }


# ---------------------------------------------------------------------------
# Chart sizes — derived from margin so they always fit
# ---------------------------------------------------------------------------
CHART_W_P1   = 2.45 * inch
CHART_H_P1   = 1.55 * inch
CHART_W_HALF = 2.60 * inch
CHART_W_FULL = 5.40 * inch
CHART_H      = 1.80 * inch


def _select_charts(athlete_id: str, pool: list[str], n: int) -> list[str]:
    seed = int.from_bytes(athlete_id.encode(), "big") % (2 ** 31)
    take = min(n, len(pool))
    return random.Random(seed).sample(pool, take)


def _make_fig(wide=False, p1=False):
    if wide:   w, h = 5.4, 1.80
    elif p1:   w, h = 2.45, 1.55
    else:      w, h = 2.6, 1.80
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#FFFFFF")
    return fig, ax


def _style_ax(ax, title: str, accent: str = "#C94A2A"):
    ax.set_facecolor("#FFFFFF")
    ax.set_title(title, color=TEXT_PRI, fontsize=7, fontweight="bold", pad=3)
    ax.tick_params(colors=TEXT_SEC, labelsize=5.5)
    ax.xaxis.label.set_color(TEXT_SEC); ax.xaxis.label.set_fontsize(6)
    ax.yaxis.label.set_color(TEXT_SEC); ax.yaxis.label.set_fontsize(6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color(BORDER)
    ax.grid(color=BORDER, alpha=0.6, linewidth=0.4)


def _to_img(fig, wide=False, p1=False) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#FFFFFF", edgecolor="none")
    buf.seek(0); plt.close(fig)
    if wide: return Image(buf, width=CHART_W_FULL, height=CHART_H)
    if p1:   return Image(buf, width=CHART_W_P1,   height=CHART_H_P1)
    return          Image(buf, width=CHART_W_HALF,  height=CHART_H)


# ---------------------------------------------------------------------------
# Chart builders — all accept ms (marker_size) for density control
# ---------------------------------------------------------------------------
def _chart_lift_progression(sessions, summary, ms=2.0, wide=False, p1=False, accent="#C94A2A"):
    fig, ax = _make_fig(wide, p1)
    for lift in LIFT_ORDER:
        row = summary[summary["lift"] == lift]
        if row.empty: continue
        kgs   = [float(x) for x in str(row.iloc[0]["weekly_kg_series"]).split(" | ")]
        weeks = list(range(1, len(kgs)+1))
        ax.plot(weeks, kgs, color=LIFT_COLORS[lift], lw=1.2,
                label=lift, marker="o", ms=ms)
    for xp in [4.5, 8.5, 11.5]:
        ax.axvline(xp, color=BORDER, ls="--", lw=0.6)
    ax.set_xlabel("Week"); ax.set_ylabel("kg"); ax.set_xticks(range(1, 13))
    _style_ax(ax, "Load progression")
    ax.legend(fontsize=4.5, framealpha=0.8, facecolor="#FFFFFF",
              labelcolor=TEXT_PRI, ncol=2)
    fig.tight_layout(pad=0.3); return _to_img(fig, wide, p1)


def _chart_peak_comparison(summary, p1=False, accent="#C94A2A"):
    fig, ax = _make_fig(p1=p1)
    present = [l for l in LIFT_ORDER if l in summary["lift"].values]
    x, w = np.arange(len(present)), 0.35
    for i, lift in enumerate(present):
        row = summary[summary["lift"] == lift].iloc[0]
        c   = LIFT_COLORS[lift]
        ax.bar(x[i]-w/2, row["week_1_kg"],   w, color=c, alpha=0.45)
        ax.bar(x[i]+w/2, row["week_peak_kg"], w, color=c, alpha=0.9)
        ax.text(x[i]-w/2, row["week_1_kg"]   +0.5, f"{row['week_1_kg']:.0f}",
                ha="center", va="bottom", fontsize=4, color=TEXT_SEC)
        ax.text(x[i]+w/2, row["week_peak_kg"] +0.5, f"{row['week_peak_kg']:.0f}",
                ha="center", va="bottom", fontsize=4, color=TEXT_PRI)
        ax.text(x[i], row["week_peak_kg"]+5,
                f"+{row['total_gain_kg']:.0f}", ha="center", va="bottom",
                fontsize=4, color=LIFT_COLORS["Deadlift"])
    ax.set_xticks(x); ax.set_xticklabels(present, fontsize=5.5)
    ax.set_ylabel("kg"); _style_ax(ax, "Start vs peak")
    fig.tight_layout(pad=0.3); return _to_img(fig, p1=p1)


def _chart_rpe_heatmap(sessions, wide=False, p1=False, accent="#C94A2A"):
    from matplotlib.colors import LinearSegmentedColormap
    present = [l for l in LIFT_ORDER if l in sessions["main_lift"].unique()]
    pivot   = sessions.pivot_table(index="main_lift", columns="week",
                                   values="main_lift_rpe", aggfunc="first").reindex(present)
    fig, ax = _make_fig(wide, p1)
    cmap    = LinearSegmentedColormap.from_list("rpe", ["#DDEEFF", accent], N=256)
    data    = pivot.values.astype(float)
    vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for ri in range(data.shape[0]):
        for ci in range(data.shape[1]):
            v = data[ri, ci]
            if not np.isnan(v):
                bright = (v-vmin)/(vmax-vmin+1e-9)
                tc = "#FFFFFF" if bright > 0.55 else TEXT_PRI
                ax.text(ci, ri, f"{v:.1f}", ha="center", va="center",
                        fontsize=3.8, color=tc)
    for xp in [3.5, 7.5, 10.5]:
        ax.axvline(xp, color="#FFFFFF", lw=0.5)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=5.5, color=TEXT_PRI)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(wk) for wk in pivot.columns], fontsize=4.5)
    ax.set_xlabel("Week")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("RPE", color=TEXT_SEC, fontsize=5)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_SEC, fontsize=4.5)
    _style_ax(ax, "RPE heatmap"); fig.tight_layout(pad=0.3)
    return _to_img(fig, wide, p1)


def _chart_load_histogram(sessions, summary, ms=2.0, p1=False, accent="#C94A2A"):
    fig, ax = _make_fig(p1=p1)
    all_kg  = sessions["main_lift_kg"].dropna()
    bins    = np.linspace(all_kg.min(), all_kg.max(), 11)
    for lift in LIFT_ORDER:
        sub = sessions[sessions["main_lift"] == lift]["main_lift_kg"]
        if sub.empty: continue
        ax.hist(sub, bins=bins, color=LIFT_COLORS[lift], alpha=0.6,
                label=lift, lw=0.3, edgecolor="#FFFFFF")
    for lift in LIFT_ORDER:
        rows = summary[summary["lift"] == lift]
        if rows.empty: continue
        ax.axvline(rows.iloc[0]["competition_1rm_kg"], color=LIFT_COLORS[lift],
                   lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("kg"); ax.set_ylabel("Sessions")
    _style_ax(ax, "Load distribution")
    ax.legend(fontsize=4.5, framealpha=0.8, facecolor="#FFFFFF", labelcolor=TEXT_PRI)
    fig.tight_layout(pad=0.3); return _to_img(fig, p1=p1)


def _chart_volume_area(sessions, p1=False, accent="#C94A2A"):
    sq     = sessions[sessions["main_lift"] == "Squat"]
    weekly = sq.groupby("week")["volume_pct"].first().sort_index()
    weeks, vols = weekly.index.tolist(), weekly.values
    fig, ax = _make_fig(p1=p1)
    ax.fill_between(weeks, vols, alpha=0.18, color=accent)
    ax.plot(weeks, vols, color=accent, lw=1.2)
    for start, end, phase in [(1,4,"accumulation"),(5,8,"intensification"),
                               (9,11,"realisation"),(12,12,"deload")]:
        ax.axvspan(start-0.5, end+0.5, alpha=0.06,
                   color=PHASE_COLORS[phase], linewidth=0)
        ax.text((start+end)/2, vols.max()*0.97, phase[:5].capitalize(),
                fontsize=4, color=PHASE_COLORS[phase], ha="center", va="top")
    ax.set_xlabel("Week"); ax.set_ylabel("Volume %")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))
    ax.set_xticks(weeks); _style_ax(ax, "Volume arc")
    fig.tight_layout(pad=0.3); return _to_img(fig, p1=p1)


def _chart_strength_radar(summary, sessions, ratios, p1=False, accent="#C94A2A"):
    bw      = float(sessions["bodyweight_kg"].iloc[0])
    present = [l for l in LIFT_ORDER if l in summary["lift"].values]
    N = len(present); vals_r, labels = [], []
    for lift in present:
        rm    = float(summary[summary["lift"] == lift].iloc[0]["competition_1rm_kg"])
        ratio = rm / bw if bw > 0 else 0.0
        vals_r.append(min(ratio / (ratios.get(lift, 1.0) or 1.0), 1.0))
        labels.append(f"{lift}\n{ratio:.2f}x")
    theta = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    theta += theta[:1]; vals = vals_r + vals_r[:1]
    figw, figh = (2.45, 1.55) if p1 else (2.6, 1.80)
    fig = plt.figure(figsize=(figw, figh)); fig.patch.set_facecolor("#FFFFFF")
    ax  = fig.add_subplot(111, polar=True); ax.set_facecolor("#FFFFFF")
    ax.plot(theta, vals, color=accent, lw=1.2)
    ax.fill(theta, vals, color=accent, alpha=0.15)
    ax.set_thetagrids(np.degrees(theta[:-1]), labels=labels, fontsize=5, color=TEXT_PRI)
    ax.set_ylim(0, 1); ax.yaxis.set_ticklabels([])
    ax.grid(color=BORDER, alpha=0.6, lw=0.4); ax.spines["polar"].set_color(BORDER)
    ax.set_title("Strength profile", color=TEXT_PRI, fontsize=7, fontweight="bold", pad=6)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#FFFFFF", edgecolor="none")
    buf.seek(0); plt.close(fig)
    return Image(buf, width=(CHART_W_P1 if p1 else CHART_W_HALF),
                 height=(CHART_H_P1 if p1 else CHART_H))


def _chart_scatter_rpe_volume(sessions, athlete_id, ms=2.0, p1=False, accent="#C94A2A"):
    fig, ax = _make_fig(p1=p1)
    for phase in PHASE_ORDER:
        sub = sessions[sessions["block_phase"] == phase]
        if sub.empty: continue
        xs  = sub["volume_pct"].values.astype(float)
        ys  = sub["main_lift_rpe"].values.astype(float)
        rng = random.Random(f"{athlete_id}{phase}")
        xs += np.array([rng.uniform(-0.002, 0.002) for _ in range(len(xs))])
        ax.scatter(xs, ys, c=PHASE_COLORS[phase], s=ms*6, alpha=0.65,
                   label=phase[:5].capitalize(), zorder=3)
    all_x = sessions["volume_pct"].dropna().values.astype(float)
    all_y = sessions["main_lift_rpe"].dropna().values.astype(float)
    if len(all_x) > 2:
        z = np.polyfit(all_x, all_y, 1); p = np.poly1d(z)
        xl = np.linspace(all_x.min(), all_x.max(), 50)
        ax.plot(xl, p(xl), color=BORDER, ls="--", lw=0.8, zorder=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))
    ax.set_xlabel("Volume %"); ax.set_ylabel("RPE")
    _style_ax(ax, "Volume vs RPE")
    ax.legend(fontsize=4.5, framealpha=0.8, facecolor="#FFFFFF", labelcolor=TEXT_PRI)
    fig.tight_layout(pad=0.3); return _to_img(fig, p1=p1)


def _chart_phase_boxplot(sessions, p1=False, accent="#C94A2A"):
    abbr = {"accumulation":"Accum.","intensification":"Intensif.",
            "realisation":"Realise","deload":"Deload"}
    fig, ax = _make_fig(p1=p1)
    groups, labels, bclrs = [], [], []
    for phase in PHASE_ORDER:
        sub = sessions[sessions["block_phase"] == phase]["main_lift_kg"].dropna()
        if sub.empty: continue
        groups.append(sub.values); labels.append(abbr[phase])
        bclrs.append(PHASE_COLORS[phase])
    if groups:
        bp = ax.boxplot(groups, patch_artist=True,
            medianprops=dict(color=TEXT_PRI, lw=1.0),
            whiskerprops=dict(color=TEXT_SEC, lw=0.8),
            capprops=dict(color=TEXT_SEC, lw=0.8),
            flierprops=dict(marker="o", color=TEXT_MUTED, ms=2, alpha=0.5),
            boxprops=dict(lw=0.5))
        for patch, c in zip(bp["boxes"], bclrs):
            patch.set_facecolor(c); patch.set_alpha(0.25)
    ax.set_xticklabels(labels, fontsize=5.5); ax.set_ylabel("kg")
    _style_ax(ax, "Load by phase"); fig.tight_layout(pad=0.3)
    return _to_img(fig, p1=p1)


def _build_chart(ct, sessions, summary, athlete_id, ratios, v,
                 wide=False, p1=False) -> Image:
    ms     = v["marker_size"]
    accent = v["accent"]
    d = {
        "lift_progression":   lambda: _chart_lift_progression(sessions, summary, ms, wide, p1, accent),
        "peak_comparison":    lambda: _chart_peak_comparison(summary, p1, accent),
        "rpe_heatmap":        lambda: _chart_rpe_heatmap(sessions, wide, p1, accent),
        "load_histogram":     lambda: _chart_load_histogram(sessions, summary, ms, p1, accent),
        "volume_area":        lambda: _chart_volume_area(sessions, p1, accent),
        "strength_radar":     lambda: _chart_strength_radar(summary, sessions, ratios, p1, accent),
        "scatter_rpe_volume": lambda: _chart_scatter_rpe_volume(sessions, athlete_id, ms, p1, accent),
        "phase_boxplot":      lambda: _chart_phase_boxplot(sessions, p1, accent),
    }
    return d[ct]()


def _compute_dataset_max_ratios(sessions, summary):
    bw     = sessions.drop_duplicates("athlete_id")[["athlete_id","bodyweight_kg"]]
    merged = summary.merge(bw, on="athlete_id")
    # Force numeric — pandas may read these as Arrow string dtype when pyarrow
    # is installed, which causes ArrowInvalid on division.
    merged["competition_1rm_kg"] = pd.to_numeric(merged["competition_1rm_kg"], errors="coerce")
    merged["bodyweight_kg"]      = pd.to_numeric(merged["bodyweight_kg"],      errors="coerce")
    out = {}
    for lift in LIFT_ORDER:
        sub = merged[merged["lift"] == lift]
        out[lift] = float((sub["competition_1rm_kg"]/sub["bodyweight_kg"]).max()) if not sub.empty else 1.0
    return out


# ---------------------------------------------------------------------------
# 600K program lookup
# ---------------------------------------------------------------------------
def _load_prog_descriptions(path) -> dict:
    if not path: return {}
    p = Path(path)
    if not p.is_file():
        print(f"  [WARN] prog-summary not found: {path}"); return {}
    try:
        df = pd.read_csv(p, usecols=["title","description"], low_memory=False)
        df = df.dropna(subset=["title","description"]).drop_duplicates("title")
        return {str(r["title"]).strip(): str(r["description"]).strip()
                for _, r in df.iterrows() if str(r["description"]).strip()}
    except Exception as e:
        print(f"  [WARN] prog-summary load failed ({e})"); return {}


def _get_most_common_program(sessions, lookup):
    if "accessory_titles" not in sessions.columns: return None, None
    titles = []
    for cell in sessions["accessory_titles"].dropna():
        titles.extend(t.strip() for t in str(cell).split("|") if t.strip())
    if not titles: return None, None
    mc   = Counter(titles).most_common(1)[0][0]
    desc = lookup.get(mc, "")
    return (mc, desc) if desc else (mc, None)


# ---------------------------------------------------------------------------
# Coaching description
# ---------------------------------------------------------------------------
def _build_description(persona, lifts, sessions, gym_lookup):
    sq    = lifts["Squat"]
    level = persona["training_level"]
    best  = max(lifts, key=lambda l: lifts[l]["total_gain_kg"])
    worst = min(lifts, key=lambda l: lifts[l]["total_gain_kg"])

    p1 = (
        f"This 12-week block runs accumulation (weeks 1–4) opening at "
        f"{sq['week_1_kg']:.1f} kg squat "
        f"({sq['week_1_kg']/sq['competition_1rm_kg']*100:.0f}% of competition max), "
        f"intensification (weeks 5–8), realisation (weeks 9–11) peaking at "
        f"{sq['week_peak_kg']:.1f} kg ({sq['peak_pct_of_1rm']*100:.1f}% of max), "
        f"and a week-12 deload. "
        f"Peak week training reaches {sq['peak_pct_of_1rm']*100:.1f}% of competition "
        f"best, calibrated from real inter-meet progression data for {level} athletes."
    )
    gains = ", ".join(f"{l} +{lifts[l]['total_gain_kg']:.1f} kg"
                      for l in LIFT_ORDER if l in lifts)
    p2 = (
        f"Load progression: {gains}. "
        f"{best} showed the largest gain, peaking in week "
        f"{int(lifts[best]['peak_week'])} at {lifts[best]['week_peak_kg']:.1f} kg. "
        f"{worst} gained the least (+{lifts[worst]['total_gain_kg']:.1f} kg)."
    )
    rpe_notes = {
        "novice":       "Novice athletes (RPE floor 6.5) are still consolidating motor patterns.",
        "intermediate": "Intermediate athletes (RPE floor 7.0) — RPE tracks closely with percentage of max.",
        "advanced":     "Advanced athletes (RPE floor 7.5) with highly consolidated motor patterns.",
        "elite":        "Elite athletes (RPE floor 8.0) operate closest to absolute limits.",
    }
    p3 = (
        f"RPE rises from {persona['rpe_min']:.1f} to {persona['rpe_max']:.1f} "
        f"across the block, tracking the rising fraction of competition maxes. "
        f"{rpe_notes.get(level, '')}"
    )
    # Para 4 — program identity
    primary   = str(sessions["primary_program"].iloc[0]) if "primary_program" in sessions.columns else ""
    secondary = str(sessions["secondary_program"].iloc[0]) if "secondary_program" in sessions.columns else ""
    if secondary in ("", "nan", "None"): secondary = ""
    all_acc = []
    for cell in sessions["accessories"].dropna():
        all_acc.extend(x.strip() for x in str(cell).split("|") if x.strip())
    top3     = [n for n, _ in Counter(all_acc).most_common(3)]
    n_unique = len(set(all_acc))
    all_titles = []
    if "accessory_titles" in sessions.columns:
        for cell in sessions["accessory_titles"].dropna():
            all_titles.extend(t.strip() for t in str(cell).split("|") if t.strip())
    total_slots     = len(all_titles) or 1
    primary_count   = sum(1 for t in all_titles if t == primary)   if primary   else 0
    secondary_count = sum(1 for t in all_titles if t == secondary) if secondary else 0
    fallback_count  = total_slots - primary_count - secondary_count
    primary_desc    = gym_lookup.get(primary,   "") if primary   else ""
    secondary_desc  = gym_lookup.get(secondary, "") if secondary else ""
    if primary:
        p4_parts = [f"Accessory programming follows '{primary}'"]
        if primary_desc:
            # Ensure description ends with a period before the slot stats follow
            desc_clean = primary_desc.rstrip().rstrip(".")
            p4_parts.append(f" — {desc_clean}.")
        pct = f"{primary_count/total_slots*100:.0f}%"
        p4_parts.append(f" ({pct} of slots; {n_unique} unique exercises; most frequent: {', '.join(top3[:3])}.)")
        if secondary and secondary_count > 0:
            sp = f"{secondary_count/total_slots*100:.0f}%"
            p4_parts.append(f" {sp} of slots supplemented from '{secondary}'")
            if secondary_desc: p4_parts.append(f" — {secondary_desc[:120].rstrip()}")
            p4_parts.append(".")
        elif fallback_count > 0:
            fp = f"{fallback_count/total_slots*100:.0f}%"
            p4_parts.append(f" {fp} of slots used general pool fallback.")
        p4 = "".join(p4_parts)
    else:
        p4 = (f"Accessory work spans {n_unique} unique exercises from the 600K Workout Dataset. "
              f"Most frequent: {', '.join(top3[:3])}.")
    return [p1, p2, p3, p4]


# ---------------------------------------------------------------------------
# Week-by-week progression table
# ---------------------------------------------------------------------------
def _week_progression_table(sessions, lifts, styles, v: dict):
    avail_w      = PAGE_W - 2 * MARGIN
    delta_lift   = v["delta_lift"]
    accent       = v["accent"]
    row_h        = v["prog_row_height"]
    col_ws       = [c/100*avail_w for c in [8, 10, 16, 16, 16, 13, 11, 10]]
    phase_abbr   = {"accumulation":"Acc","intensification":"Int",
                    "realisation":"Rea","deload":"Del"}

    pivot = {}
    for lift in LIFT_ORDER:
        sub = sessions[sessions["main_lift"] == lift].sort_values("week")
        pivot[lift] = {
            int(r["week"]): {"kg": r["main_lift_kg"], "delta": r["main_lift_delta_kg"],
                              "rpe": r["main_lift_rpe"], "phase": r["block_phase"]}
            for _, r in sub.iterrows()
        }
    weeks = sorted({int(w) for w in sessions["week"].unique()})

    def _kg(d):
        v = d.get("kg")
        return f"{v:.1f}" if v is not None and pd.notna(v) else "—"
    def _delta(val):
        if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
        return f"+{val:.1f}" if val > 0 else f"{val:.1f}"

    _DELTA_ABBR = {"Squat": "Sq", "Deadlift": "DL", "Bench": "Be", "OHP": "OH"}
    header = ["Wk","Phase","Squat","Bench","DL","OHP","RPE",
              f"{_DELTA_ABBR.get(delta_lift, delta_lift[:2])} \u0394"]
    rows   = [header]
    for wk in weeks:
        sq   = pivot["Squat"].get(wk, {})
        be   = pivot["Bench"].get(wk, {})
        dl   = pivot["Deadlift"].get(wk, {})
        ohp  = pivot["OHP"].get(wk, {})
        phase = sq.get("phase","")
        rpe   = sq.get("rpe","")
        ddata = pivot.get(delta_lift, {}).get(wk, {})
        rows.append([str(wk), phase_abbr.get(phase, phase[:3]),
                     _kg(sq), _kg(be), _kg(dl), _kg(ohp),
                     f"{rpe:.1f}" if rpe else "—",
                     _delta(ddata.get("delta", None))])

    tbl = Table(rows, colWidths=col_ws)
    ts  = [
        ("FONTSIZE",      (0,0),(-1,-1), 6.5),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,0),(-1,0),  _rl(accent)),
        ("TEXTCOLOR",     (0,0),(-1,0),  _rl("#FFFFFF")),
        ("TEXTCOLOR",     (0,1),(0,-1),  _rl(TEXT_MUTED)),
        ("TEXTCOLOR",     (1,1),(1,-1),  _rl(TEXT_MUTED)),
        ("TEXTCOLOR",     (2,1),(-1,-1), _rl(TEXT_PRI)),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("ALIGN",         (0,0),(1,-1),  "LEFT"),
        ("TOPPADDING",    (0,0),(-1,-1), 2),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2),
        ("LEFTPADDING",   (0,0),(-1,-1), 3),
        ("RIGHTPADDING",  (0,0),(-1,-1), 3),
        ("ROWHEIGHT",     (0,0),(-1,-1), row_h),
        ("LINEBELOW",     (0,0),(-1,0),  0.5, _rl(BORDER)),
    ]
    for i, wk in enumerate(weeks, start=1):
        phase = rows[i][1]
        rev   = {"Acc":"accumulation","Int":"intensification",
                 "Rea":"realisation","Del":"deload"}
        pc    = PHASE_COLORS.get(rev.get(phase,""), TEXT_MUTED)
        ts.append(("TEXTCOLOR",  (1,i),(1,i), _rl(pc)))
        ts.append(("BACKGROUND", (0,i),(-1,i), _rl(PAGE_BG if i%2 else ROW_ALT)))
        dv = rows[i][7]
        if dv.startswith("+"): ts.append(("TEXTCOLOR",(7,i),(7,i),_rl(LIFT_COLORS["Deadlift"])))
        elif dv.startswith("-"): ts.append(("TEXTCOLOR",(7,i),(7,i),_rl(TEXT_MUTED)))
    tbl.setStyle(TableStyle(ts))
    return [Paragraph("WEEK-BY-WEEK PROGRESSION", styles["sec"]), tbl]


# ---------------------------------------------------------------------------
# Full session log
# ---------------------------------------------------------------------------
_SESSION_ORDER = ["Lower A","Upper A","Lower B","Upper B"]

def _acc_text(row) -> str:
    names  = [x.strip() for x in str(row.get("accessories","")).split("|") if x.strip()]
    sets_  = [x.strip() for x in str(row.get("accessory_sets","")).split("|") if x.strip()]
    reps_  = [x.strip() for x in str(row.get("accessory_reps","")).split("|") if x.strip()]
    units_ = [x.strip() for x in str(row.get("accessory_reps_unit","")).split("|") if x.strip()]
    parts  = []
    for i, name in enumerate(names):
        s = sets_[i] if i < len(sets_) else "?"
        r = reps_[i] if i < len(reps_) else "?"
        u = units_[i] if i < len(units_) else "reps"
        try:
            rv = float(r); r = f"{int(rv)}" if rv==int(rv) else f"{rv:.1f}"
        except: pass
        parts.append(f"{name} {s}\xd7{r}{'s' if u=='seconds' else ''}")
    return "  |  ".join(parts) if parts else "\u2014"


def _full_session_log(sessions, styles, v: dict):
    avail_w  = PAGE_W - 2 * MARGIN
    cols     = v["session_cols"]
    accent   = v["accent"]
    _COL_W   = {"Session": 52, "Main lift": 90, "RPE": 28}
    acc_w    = avail_w - sum(_COL_W.get(c, 0) for c in cols)
    col_ws   = [(_COL_W.get(c, acc_w) if c != "Accessories" else acc_w) for c in cols]

    def _cell(txt, fn="Helvetica", fs=6, tc=TEXT_PRI, bg=PAGE_BG, leading=8):
        return Paragraph(txt, ParagraphStyle(
            f"_c{abs(hash(txt+fn))%99999}", fontName=fn, fontSize=fs,
            textColor=_rl(tc), backColor=_rl(bg), leading=leading))

    story = [PageBreak(), Paragraph("FULL SESSION LOG", styles["sec"])]

    for wk in sorted(sessions["week"].unique()):
        wk_data   = sessions[sessions["week"] == wk]
        phase_raw = wk_data["block_phase"].iloc[0] if not wk_data.empty else ""
        phase_clr = PHASE_COLORS.get(phase_raw, TEXT_MUTED)
        story.append(Paragraph(
            f"<b>Week {wk}</b>"
            f"<font color='{phase_clr}'>  {phase_raw.capitalize()}</font>",
            ParagraphStyle(f"_wk{wk}", fontName="Helvetica-Bold", fontSize=7,
                           textColor=_rl(TEXT_PRI), backColor=_rl(PAGE_BG),
                           leading=10, spaceBefore=4, spaceAfter=2)))

        tbl_rows = [[_cell(c, fn="Helvetica-Bold", tc="#FFFFFF", bg=accent)
                     for c in cols]]
        for di, day_label in enumerate(_SESSION_ORDER):
            sess = wk_data[wk_data["day_label"] == day_label]
            bg   = PAGE_BG if di % 2 == 0 else ROW_ALT
            if sess.empty:
                tbl_rows.append([_cell("—", bg=bg)] * len(cols)); continue
            r         = sess.iloc[0]
            lift      = str(r.get("main_lift",""))
            kg        = r.get("main_lift_kg")
            rpe       = r.get("main_lift_rpe")
            delta     = r.get("main_lift_delta_kg")
            lift_clr  = LIFT_COLORS.get(lift, TEXT_PRI)
            kg_str    = f"{kg:.1f} kg" if pd.notna(kg) else "—"
            if pd.notna(delta) and delta != 0:
                kg_str += f"  ({'+' if delta>0 else ''}{delta:.1f})"
            cell_map  = {
                "Session":     _cell(day_label, tc=TEXT_MUTED, bg=bg),
                "Main lift":   _cell(f"<b>{lift}</b>  {kg_str}",
                                     fn="Helvetica-Bold", tc=lift_clr, bg=bg),
                "RPE":         _cell(f"{rpe:.1f}" if pd.notna(rpe) else "—",
                                     tc=TEXT_MUTED, bg=bg),
                "Accessories": _cell(_acc_text(r), bg=bg, leading=8),
            }
            tbl_rows.append([cell_map[c] for c in cols])

        tbl = Table(tbl_rows, colWidths=col_ws)
        tbl.setStyle(TableStyle([
            ("FONTSIZE",      (0,0),(-1,-1), 6),
            ("TOPPADDING",    (0,0),(-1,-1), 2),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ("LINEBELOW",     (0,0),(-1,0),  0.5, _rl(BORDER)),
            ("LINEBELOW",     (0,-1),(-1,-1),0.3, _rl(BORDER)),
        ]))
        story.append(tbl)
    return story


# ---------------------------------------------------------------------------
# Page 1
# ---------------------------------------------------------------------------
def _build_page1(athlete_id, sessions, summary, lifts, gym_lookup, styles, charts_p1, v):
    story   = []
    avail_w = PAGE_W - 2 * MARGIN
    accent  = v["accent"]
    row0    = sessions.iloc[0]
    persona = {
        "training_level": row0["training_level"],
        "sex":            row0["sex"],
        "age":            float(row0["age"]) if pd.notna(row0["age"]) else None,
        "bodyweight_kg":  float(row0["bodyweight_kg"]),
        "dots":           float(row0["dots"]),
        "rpe_min":        float(sessions["main_lift_rpe"].min()),
        "rpe_max":        float(sessions["main_lift_rpe"].max()),
    }

    def _hps(name, **kw):
        return ParagraphStyle(name, backColor=_rl(PAGE_BG),
                              textColor=_rl(kw.pop("tc", TEXT_PRI)),
                              fontName=kw.pop("fn","Helvetica"),
                              fontSize=kw.pop("fs",8), **kw)

    age_str    = f"age {persona['age']:.0f}" if persona["age"] else ""
    lift_names = [l for l in LIFT_ORDER if l in lifts]
    cw_lift    = (avail_w - 160 - 110) / max(len(lift_names), 1)

    id_block   = Table([
        [Paragraph("ATHLETE PROFILE", _hps("bp",fn="Helvetica-Bold",fs=6,tc=accent,letterSpacing=1.4))],
        [Paragraph(athlete_id,         _hps("ba",fn="Helvetica-Bold",fs=13))],
        [Paragraph(f"{persona['training_level'].capitalize()} · {persona['sex']} · {age_str}",
                   _hps("bs",fs=7,tc=TEXT_MUTED))],
    ], colWidths=[160])

    lifts_blk = Table([
        [Paragraph(f'<font color="{LIFT_COLORS[l]}"><b>{l}</b></font>',
                   _hps(f"lh{l}", fn="Helvetica-Bold", fs=6, tc=LIFT_COLORS[l]))
         for l in lift_names],
        [Paragraph(f"<b>{lifts[l]['competition_1rm_kg']:.1f}</b>",
                   _hps(f"lv{l}", fn="Helvetica-Bold", fs=12))
         for l in lift_names],
        [Paragraph("kg", _hps(f"lu{l}", fs=6, tc=TEXT_MUTED)) for l in lift_names],
    ], colWidths=[cw_lift]*len(lift_names))

    sq  = lifts.get("Squat", list(lifts.values())[0])
    pp  = sq["peak_pct_of_1rm"]*100 if sq["peak_pct_of_1rm"] else None
    bw  = persona["bodyweight_kg"]
    wc  = row0["weight_class_kg"]
    stats_blk = Table([
        [Paragraph(f"<b>{bw:.1f} kg</b>", _hps("sv",fn="Helvetica-Bold",fs=10))],
        [Paragraph("bodyweight", _hps("sl",fs=6,tc=TEXT_MUTED))],
        [Paragraph(f"<b>{persona['dots']:.1f}</b> Dots · {wc} kg class",
                   _hps("sd",fs=7,tc=TEXT_MUTED))],
        [Paragraph(f"Peak intensity <b>{pp:.1f}% of 1RM</b>" if pp else "",
                   _hps("sa",fs=6.5,tc=TEXT_MUTED))],
    ], colWidths=[110])

    header = Table([[id_block, lifts_blk, stats_blk]],
                   colWidths=[160, avail_w-160-110, 110])
    header.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), _rl(CARD_BG)),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LINEBELOW",     (0,0),(-1,0),  0.5, _rl(BORDER)),
    ]))
    story += [header, Spacer(1, 7)]

    # ── Coaching summary (paragraph order variant) ────────────────────────────
    story.append(Paragraph("COACHING SUMMARY", styles["sec"]))
    paras = _build_description(persona, lifts, sessions, gym_lookup)
    order = v["para_order"] if len(v["para_order"]) == len(paras) else list(range(len(paras)))
    for i in order:
        if i < len(paras):
            story.append(Paragraph(paras[i], styles["body_sm"]))
    story.append(Spacer(1, 5))

    # ── Metrics table (row-set variant) ───────────────────────────────────────
    story.append(Paragraph("PERFORMANCE METRICS", styles["sec"]))
    present = [l for l in LIFT_ORDER if l in lifts]
    mv      = _METRIC_VARIANTS[v["metric_variant"] % len(_METRIC_VARIANTS)]
    rows    = [["Metric"] + present]
    for label, fn in mv:
        try: rows.append([label] + [fn(lifts, l) for l in present])
        except: pass
    cw  = avail_w / (len(present)+1)
    tbl = Table(rows, colWidths=[cw]*(len(present)+1))
    ts  = [
        ("BACKGROUND",    (0,0),(-1,0),  _rl(accent)),
        ("TEXTCOLOR",     (0,0),(-1,0),  _rl("#FFFFFF")),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 7),
        ("ROWHEIGHT",     (0,0),(-1,-1), v["prog_row_height"]),
        ("TEXTCOLOR",     (0,1),(0,-1),  _rl(TEXT_MUTED)),
        ("TEXTCOLOR",     (1,1),(-1,-1), _rl(TEXT_PRI)),
        ("ALIGN",         (1,0),(-1,-1), "RIGHT"),
        ("ALIGN",         (0,0),(0,-1),  "LEFT"),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("RIGHTPADDING",  (0,0),(-1,-1), 5),
        ("TOPPADDING",    (0,0),(-1,-1), 2),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2),
        ("LINEBELOW",     (0,0),(-1,0),  0.5, _rl(BORDER)),
    ]
    for i in range(1, len(rows)):
        ts.append(("BACKGROUND",(0,i),(-1,i), _rl(PAGE_BG if i%2 else ROW_ALT)))
    tbl.setStyle(TableStyle(ts)); story.append(tbl); story.append(Spacer(1, 6))

    # ── 2 charts ──────────────────────────────────────────────────────────────
    story.append(Paragraph("TRAINING CHARTS", styles["sec"]))
    row = Table([[charts_p1[0], charts_p1[1]]],
                colWidths=[CHART_W_P1, CHART_W_P1], rowHeights=[CHART_H_P1])
    row.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),
                              ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(row)
    return story


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_athlete_pdf(athlete_id, sessions, summary, out_dir,
                         dataset_max_ratios, gym_lookup, cfg):
    out_path = out_dir / f"{athlete_id}.pdf"
    lifts    = {r["lift"]: r for _, r in summary.iterrows()}
    v        = _resolve_variants(athlete_id, cfg)
    styles   = _make_styles(v)

    # Chart selection
    selected = _select_charts(athlete_id, v["chart_types_pool"], v["n_charts"])
    narrow   = [ct for ct in selected if ct not in _WIDE_TYPES]
    wide_cts = [ct for ct in selected if ct in _WIDE_TYPES]

    if wide_cts:
        p1_types = (narrow[:2] if len(narrow) >= 2
                    else (narrow + [narrow[0]] if narrow else ["peak_comparison","load_histogram"]))[:2]
        p2_type  = wide_cts[0]; p2_wide = True
    else:
        p1_types = selected[:2]
        p2_type  = selected[2] if len(selected) > 2 else selected[0]
        p2_wide  = False

    charts_p1 = [_build_chart(ct, sessions, summary, athlete_id,
                               dataset_max_ratios, v, p1=True)
                 for ct in p1_types]
    chart_p2  = _build_chart(p2_type, sessions, summary, athlete_id,
                              dataset_max_ratios, v, wide=p2_wide)

    story = _build_page1(athlete_id, sessions, summary, lifts,
                         gym_lookup, styles, charts_p1, v)
    story.append(PageBreak())

    chart_row = Table([[chart_p2]],
                      colWidths=[CHART_W_FULL if p2_wide else CHART_W_HALF],
                      rowHeights=[CHART_H])
    chart_row.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
    story.append(chart_row)
    story.append(Spacer(1, 8))
    story.extend(_week_progression_table(sessions, lifts, styles, v))
    story.extend(_full_session_log(sessions, styles, v))

    def _bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(_rl(PAGE_BG))
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN)
    doc.build(story, onFirstPage=_bg, onLaterPages=_bg)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Path resolution — generate_pdfs.py lives at gym_rag/ root, so:
#   gym_rag/config/pdf_config.toml  → _DEFAULT_CONFIG_PATH
#   gym_rag/data/pdfs/              → _DEFAULT_PDF_DIR
_SCRIPT_DIR          = Path(__file__).resolve().parent.parent.parent  # project root
_DEFAULT_CONFIG_PATH = _SCRIPT_DIR / "config" / "pdf_config.toml"
_DEFAULT_PDF_DIR     = _SCRIPT_DIR / "data"   / "pdfs"
_DEFAULT_SESSIONS    = _SCRIPT_DIR / "data"   / "output" / "sessions.csv"
_DEFAULT_SUMMARY     = _SCRIPT_DIR / "data"   / "output" / "block_summary.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Generate athlete PDFs with configurable variation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sessions",     default=str(_DEFAULT_SESSIONS))
    parser.add_argument("--summary",      default=str(_DEFAULT_SUMMARY))
    parser.add_argument("--prog-summary", default=None)
    parser.add_argument("--out-dir",      default=str(_DEFAULT_PDF_DIR))
    parser.add_argument("--config",       default=str(_DEFAULT_CONFIG_PATH),
                        help="Path to pdf_config.toml (default: config/pdf_config.toml)")
    parser.add_argument("--n-athletes",   type=int, default=None)
    parser.add_argument("--athlete",      default=None)
    parser.add_argument("--no-randomise", action="store_true",
                        help="Disable per-athlete variation — all PDFs use config defaults")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_randomise:
        cfg["variation"]["randomise"] = False

    # dtype_backend="numpy_nullable" prevents pandas from using Arrow string
    # dtypes (which fail on arithmetic) when pyarrow is installed.
    sessions = pd.read_csv(args.sessions, dtype_backend="numpy_nullable")
    summary  = pd.read_csv(args.summary,  dtype_backend="numpy_nullable")
    out_dir  = PDF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios     = _compute_dataset_max_ratios(sessions, summary)
    gym_lookup = _load_prog_descriptions(args.prog_summary)

    all_ids = sorted(sessions["athlete_id"].unique())
    ids = ([args.athlete] if args.athlete
           else all_ids[:args.n_athletes] if args.n_athletes
           else all_ids)

    failures = 0; total_bytes = 0
    for aid in tqdm(ids, desc="Generating PDFs", unit="pdf"):
        try:
            asess = sessions[sessions["athlete_id"] == aid].copy()
            asum  = summary[summary["athlete_id"] == aid].copy()
            if asess.empty or asum.empty:
                print(f"  [SKIP] {aid}: no data"); failures += 1; continue
            out = generate_athlete_pdf(aid, asess, asum, out_dir,
                                       ratios, gym_lookup, cfg)
            total_bytes += out.stat().st_size
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"\n  [ERROR] {aid}: {exc}"); failures += 1

    n = len(ids) - failures
    print(f"\nGenerated {n} PDFs → {out_dir}/")
    if n:
        print(f"Total: {total_bytes/1024/1024:.1f} MB  |  "
              f"Avg: {total_bytes/n/1024:.1f} KB  |  Failures: {failures}")


if __name__ == "__main__":
    main()