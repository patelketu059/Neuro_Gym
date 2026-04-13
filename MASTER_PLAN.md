# Gym RAG Chatbot — Master Plan

> **For Claude Code:** This document is the single source of truth for the entire
> project. Read it fully before writing any code. Every architectural and data
> decision is documented with explicit reasoning. Do not deviate from decisions
> marked as constraints without reading the documented rationale first.

---

## What we are building

A **multimodal RAG chatbot** for powerlifting and gym coaching. Users ask text
questions or upload images of training charts. The system retrieves relevant
athlete profiles, session logs, and visual charts from a vector database, then
generates grounded coaching responses that cite specific data points.

Portfolio project targeting mid-level AI/ML Engineer roles. Every decision is
intentional and must be defensible in a technical interview.

---

## Resume gaps this project addresses

| Gap | How addressed |
|---|---|
| RAG completely absent | Core architecture — Qdrant + BM25 hybrid retrieval |
| No LangChain experience | Full LangChain chain with RunnableWithMessageHistory |
| No vector store experience | Qdrant with 3 collections, hybrid search, multimodal payloads |
| No deployed projects | FastAPI + Streamlit Community Cloud with public URL |
| No eval/observability | RAGAS scores + LangSmith traces + groundedness evaluator |
| No multimodal experience | `nemotron-embed-vl-1b-v2` embeds chart PNGs natively |
| 11-month gap | Dated GitHub commits, dated resume entry, live demo URL |
| Credential inflation on LLMs | Actual LLM integration with memory, tracing, evaluation |
| No model serving tools | FastAPI wrapping the full pipeline as a REST API |

---

## Full stack

| Layer | Choice | Why | Rejected |
|---|---|---|---|
| Multimodal embedding | `nvidia/llama-nemotron-embed-vl-1b-v2` | Only commercially-licensed model that embeds images and text in the same 2048-dim space. T4 or OpenRouter free tier. | CLIP (not unified), Cohere (text only), OpenAI (paid) |
| Multimodal reranking | `nvidia/llama-nemotron-rerank-vl-1b-v2` | Shared SigLIP2 encoder with embedder — aligned spaces. Cross-encoder adds +7% accuracy. Free. | Cohere Rerank (text only) |
| Sparse index | `rank_bm25` in-memory | Catches exact keyword matches BM25 misses. Zero infrastructure. | Elasticsearch (overkill) |
| Vector store | Qdrant local Docker | No vector limit self-hosted. Native hybrid search + RRF. 326 QPS. | Pinecone (vector limit, wrong dims), Chroma (no hybrid) |
| LLM generation | Gemini 2.5 Flash | Only free-tier natively multimodal model. 1K req/day. | Claude (no free API), GPT-4o (rate limited), Groq (text only) |
| Memory | `ConversationSummaryBufferMemory` | Custom — k=8 buffer, summarises overflow every 16 messages. | Default LangChain memory |
| API | FastAPI | Async-native for concurrent Qdrant + Gemini calls. | Flask (sync) |
| Frontend | Streamlit | Free Community Cloud deploy, image upload native. | Gradio (less control) |
| Observability | LangSmith | 5K free traces/month. Interview-ready trace trees. | W&B (training focus) |
| Evaluation | RAGAS | Industry-standard: faithfulness, answer relevancy, context recall. | Manual only |

---

## Project folder structure

This is the exact structure Claude Code must create. Every file listed here
must be created with the content described in the Day 1 implementation section.

```
gym_rag/
│
├── CLAUDE.md                              ← Claude Code context file
├── ARCHITECTURE.md                        ← Architecture reference document
├── requirements.txt                       ← All dependencies, grouped by phase
│
├── config/
│   ├── __init__.py                        ← exports all settings with `from .settings import *`
│   └── settings.py                        ← ALL global constants — the single source of truth
│
├── pipeline/
│   ├── __init__.py
│   │
│   └── dataset/                           ← Phase 1: master dataset generation
│       ├── __init__.py
│       ├── dataset_main.py                ← Entry point. Run this to generate the dataset.
│       ├── models.py                      ← All dataclasses — no logic, only shapes
│       ├── opl_loader.py                  ← OPL loading (cached) + amplitude derivation
│       ├── gym_loader.py                  ← 600K loading (cached) + accessory selection
│       ├── periodization.py               ← Template construction from OPL + literature
│       ├── athlete_generator.py           ← Persona sampling (no-replacement) + block building
│       └── export.py                      ← Flatten AthleteRecord → sessions_df + block_summary_df
│
├── app/
│   ├── main.py                            ← Phase 4: FastAPI app with lifespan
│   ├── chain.py                           ← Phase 4: LangChain RAG chain
│   ├── memory.py                          ← Phase 4: ConversationSummaryBufferMemory
│   ├── context_assembler.py               ← Phase 3/4: assemble retrieved context
│   └── routes/
│       ├── chat.py                        ← POST /chat, DELETE /chat/{session_id}
│       └── health.py                      ← GET /health
│
├── ui/
│   └── streamlit_app.py                   ← Phase 4: Streamlit frontend
│
├── eval/
│   ├── ragas_eval.py                      ← Phase 5: RAGAS evaluation
│   ├── langsmith_eval.py                  ← Phase 5: LangSmith groundedness eval
│   └── retrieval_metrics.py               ← Phase 3/5: MRR and NDCG@5
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env.example
│
├── notebooks/
│   └── colab_ingestion.ipynb              ← Phase 2: Colab T4 embedding notebook
│
└── data/
    ├── raw/                               ← gitignored — source CSVs go here
    │   ├── opl.csv
    │   ├── programs_detailed_boostcamp_kaggle.csv
    │   └── program_summary.csv
    ├── opl_cleaned.csv                    ← cached cleaned OPL (built on first run)
    ├── combined_600k_dataset.csv          ← cached merged 600K (built on first run)
    ├── checkpoint.json                    ← generation progress: completed IDs + used OPL indices
    ├── bm25_index.pkl
    ├── bm25_corpus.json
    ├── golden_eval_set.json
    └── output/
        ├── sessions.csv                   ← one row per session (12 wks × 4 days × n athletes)
        └── block_summary.csv              ← one row per athlete per lift
```

---

## Data strategy

### Source datasets

**OpenPowerlifting** (`data/raw/opl.csv`)
- Source: https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database
- 800K+ real competition entries
- **Role 1 — Athlete persona**: sample one row per athlete (seeded, without replacement).
  Provides `Best3SquatKg`, `Best3BenchKg`, `Best3DeadliftKg`, `TotalKg`, `Dots`,
  `Age`, `BodyweightKg`, `Sex`, `WeightClassKg`. Numbers are real competition results.
- **Role 2 — Periodization amplitude**: athletes with ≥3 meets → mean inter-meet
  progression per training level. Controls how ambitious each block's load target is.
- `Name` and `Date`: loaded for amplitude derivation only. Dropped before persona
  sampling. Never written to any output file.
- Cleaning: drop rows where any lift ≤ 0, Dots ≤ 0, or required columns null.
- **Caching**: cleaned file saved to `data/opl_cleaned.csv` on first run. Subsequent
  runs load directly — skips the full 800K-row parse.

**600K Fitness Exercise & Workout Program Dataset**
- Source: https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset
- Two files: `programs_detailed_boostcamp_kaggle.csv` (605K exercise rows) +
  `program_summary.csv` (2,598 programs)
- **Role — Accessory selection only**. Evaluated for shape derivation and rejected:
  intensity distribution (71% high, 26% medium, 2.4% low) produces a near-flat curve.
- **Confirmed dtypes** (from live data):
  - `title`, `description`, `level`, `goal`, `equipment`, `exercise_name`,
    `intensity`, `created`, `last_edit` → str/object
  - `program_length`, `time_per_workout`, `week`, `day`, `number_of_exercises`,
    `reps` → float64
  - `sets` → int64
  - `intensity` → float64 (0–10 numeric scale)
  - `reps`: positive = rep count, negative = timed set in seconds
  - `goal` and `level` → list-literal strings, e.g. `"['strength', 'hypertrophy']"`
    — parsed with `ast.literal_eval` into real Python lists
- **Confirmed unique equipment values** (from live data):
  - `Full Gym` — 462,441 rows — included (barbell access assumed)
  - `Garage Gym` — 96,648 rows — included (barbell-focused)
  - `At Home` — 29,741 rows — **excluded at load time** (no barbell)
  - `Dumbbell Only` — 12,889 rows — **excluded at load time** (no barbell)
  - Non-barbell rows are dropped in `load_gym()` before any query runs so they
    never surface even as fallback accessories
- **Join logic**: `program_summary.csv` joined onto exercises on `title`.
  Program-level `goal`/`equipment` wins on conflict — program author intent
  is more authoritative than per-exercise category tags.
- **Filtering before join**: drop programs where `goal == '[]'`,
  `equipment == '[]'` or null, `description` is null, `last_edit` is null.
- **Goal classification**: `STRENGTH_GOAL_KEYWORDS` seed keywords classify real
  dataset goal strings at load time. Not hardcoded filter values.
- **Caching**: merged file saved to `data/combined_600k_dataset.csv` on first run.

### Periodization design

**721 PPL single-individual log: rejected.** Single person, noise indistinguishable
from programming, units in pounds, not representative of OPL population.

**600K shape derivation: rejected.** Intensity distribution (71% high) produces a
near-flat curve — no accumulation → peak arc.

**OPL owns the AMPLITUDE.** Mean inter-meet total progression per training level.
Three constraints (all must be maintained):
1. Athletes classified by **peak career Dots** — not per-meet Dots
2. Only meets where athlete was within **95% of peak Dots** — consolidation only
3. Positive progressions only, **capped at 30%** — excludes bomb-out recoveries

Confirmed amplitudes from real OPL data:

| Level | Amplitude | Progressions used |
|---|---|---|
| novice | ~6.34% | 56,543 |
| intermediate | ~5.70% | 249,831 |
| advanced | ~5.53% | 222,903 |
| elite | ~5.39% | 68,840 |

**PERIODIZATION_SHAPE owns the SHAPE.** Literature-derived 12-week normalised
0→1 arc. 0 = floor, 1 = ceiling for that level.
Source: Prilepin (1974) intensity zones; Issurin (2008) block phase structure.

```python
PERIODIZATION_SHAPE = [
    0.00, 0.07, 0.14, 0.21,   # accumulation    weeks 1–4  (~7% steps)
    0.29, 0.37, 0.46, 0.55,   # intensification weeks 5–8  (~8% steps)
    0.65, 0.80, 1.00,          # realisation     weeks 9–11 (accelerating push)
    0.20,                      # deload          week 12
]
```

**Per-level floor and ceiling** (fraction of competition 1RM):
Source: Peterson et al. (2011); Haff & Triplett NSCA (2016); Helms et al. (2017);
Issurin (2008); Sheiko programs.

```python
LEVEL_SHAPE_PARAMS = {
    "novice":       {"floor": 0.65, "ceiling": 0.90},
    "intermediate": {"floor": 0.68, "ceiling": 0.93},
    "advanced":     {"floor": 0.72, "ceiling": 0.95},
    "elite":        {"floor": 0.75, "ceiling": 0.93},  # elite < advanced — Issurin 2008
}
```

**Week percentage formula:**
```
week_pct[w] = floor + PERIODIZATION_SHAPE[w] × (ceiling - floor)
```

**Block target and working weight:**
```
block_target_kg = competition_1rm × (1 + amplitude)
working_kg      = block_target_kg × week_pct[week - 1]
                  rounded to nearest 2.5kg
```

**RPE curve** per level. Floor differs by level — Helms et al. (2017): elite lifters
rate identical loads ~0.5–1.0 RPE higher than novices.
Source: Tuchscherer (2008); Helms et al. (2017).

```python
RPE_FLOOR_BY_LEVEL = {"novice": 6.5, "intermediate": 7.0, "advanced": 7.5, "elite": 8.0}
RPE_CEILING = 9.5   # competition is the true RPE 10 moment

rpe[w] = rpe_floor + PERIODIZATION_SHAPE[w] × (RPE_CEILING - rpe_floor)
```

**OHP estimate** — no powerlifting competition equivalent. Estimated from bench:
Source: Lehman (2005) J Strength Cond Res; Stastny et al. (2017) J Human Kinetics.

```python
OHP_BENCH_RATIO = {"novice": 0.61, "intermediate": 0.64, "advanced": 0.67, "elite": 0.70}
```

**Session structure — body keyword matching.** Each session day has `body_keywords`
matched against `exercise_name` to ensure anatomically appropriate accessories.
Without this, squat days returned tricep exercises (confirmed bug in earlier version).

```python
SESSION_FOCUS = {
    0: {"label": "Lower A",  "body_keywords": ["squat", "leg", "lunge", "hamstring", "glute", "hip", "calf", "quad", "bulgarian", "goblet"]},
    1: {"label": "Upper A",  "body_keywords": ["bench", "press", "chest", "shoulder", "tricep", "push", "dip", "fly", "pec", "incline", "decline"]},
    2: {"label": "Lower B",  "body_keywords": ["deadlift", "row", "pull", "back", "lat", "trap", "bicep", "curl", "rdl", "romanian", "rack"]},
    3: {"label": "Upper B",  "body_keywords": ["overhead", "ohp", "shoulder", "press", "lateral", "raise", "tricep", "dip", "push", "deltoid", "shrug"]},
}
```

### Output DataFrames

**`sessions.csv`** — one row per session (12 weeks × 4 days × n athletes)

| Column | Purpose |
|---|---|
| `block_phase` | accumulation/intensification/realisation/deload — tags RAG chunks |
| `main_lift_delta_kg` | week-over-week load change; NaN for week 1 |
| `main_lift_pct_of_peak` | working_kg / competition_1rm — may exceed 1.0 in peak week |
| `accessory_reps` | always positive absolute value |
| `accessory_reps_unit` | "reps" or "seconds" |
| `accessory_intensity` | raw 0–10 float from 600K dataset |

**`block_summary.csv`** — one row per athlete per lift (4 lifts × n athletes)

| Column | Purpose |
|---|---|
| `competition_1rm_kg` | OPL competition result |
| `week_1_kg` / `week_peak_kg` / `week_floor_kg` | absolute load range |
| `peak_week` / `floor_week` | graph annotation points |
| `total_gain_kg` | week_peak - week_1 |
| `peak_pct_of_1rm` | how close to competition best |
| `block_phase_at_peak` | must always be "realisation" — validation check |
| `weekly_kg_series` | pipe-separated 12-week series for line graphs |

---

## Non-negotiable constraints

1. All models must be free — no paid embedding, reranking, or generation APIs
2. Qdrant runs in local Docker — never Qdrant Cloud or Pinecone
3. Generation uses Gemini 2.5 Flash — only free multimodal generation option
4. Memory uses the custom `ConversationSummaryBufferMemory` from `app/memory.py`
5. RAGAS scores must appear in the README
6. Never commit raw CSVs or `.env` to git
7. The 721 PPL single-individual log is NOT used. Do not reintroduce it.
8. The 600K dataset is NOT used for shape derivation. `PERIODIZATION_SHAPE` is the
   shape source. Do not change without re-running intensity distribution analysis.
9. OPL amplitude must use: peak-career Dots, 95% consolidation filter, 30% cap.
   Do not relax without re-validating the four output amplitude values.
10. `BARBELL_EQUIPMENT_VALUES = {"Full Gym", "Garage Gym"}` — exact casing, confirmed
    from live data. Do not lowercase at query time.
11. `At Home` and `Dumbbell Only` rows must be dropped at load time in `load_gym()`,
    not at query time. They must never appear even as fallback accessories.
12. Accessory exercises must match session body part via `body_keywords` in
    `SESSION_FOCUS`. The 8-step filter chain in `query_accessories()` must keep
    body_mask in steps 0–3 before relaxing to non-body-filtered fallbacks.
13. Athlete IDs use 5-digit zero-padding: `athlete_00000` through `athlete_99999`.
    Supports up to 99,999 athletes without format changes.
14. OPL sampling is without replacement. `sample_athlete_persona()` accepts
    `used_indices: set` and walks forward on collision. `used_opl_indices` must
    be persisted in `checkpoint.json` alongside `completed` athlete IDs.
15. CSV output uses streaming appends (`mode="a"`, `header=False` after first batch).
    Never reload the full output CSV into memory. This is critical at 10K–20K scale.
16. The golden eval set (20 Q&A pairs) must be written manually — not generated.

---

## `config/settings.py` — complete specification

Claude Code must implement this file exactly. Every constant is cited.

```python
from pathlib import Path

# Paths
ROOT_DIR           = Path(__file__).resolve().parent.parent
DATA_DIR           = ROOT_DIR / "data"
RAW_DIR            = DATA_DIR / "raw"
OUT_DIR            = DATA_DIR / "output"
OPL_PATH           = RAW_DIR / "opl.csv"
GYM_EX_PATH        = RAW_DIR / "programs_detailed_boostcamp_kaggle.csv"
GYM_PROG_PATH      = RAW_DIR / "program_summary.csv"
OPL_CLEANED_PATH   = DATA_DIR / "opl_cleaned.csv"
GYM_COMBINED_PATH  = DATA_DIR / "combined_600k_dataset.csv"
SESSIONS_PATH      = OUT_DIR  / "sessions.csv"
BLOCK_SUMMARY_PATH = OUT_DIR  / "block_summary.csv"
CHECKPOINT_PATH    = DATA_DIR / "checkpoint.json"

# Generation
N_ATHLETES        = 200
CHECKPOINT_EVERY  = 50   # reduce to 20 for Colab
BLOCK_WEEKS       = 12
SESSIONS_PER_WEEK = 4
ACCESSORIES_PER_SESSION = 4

# OPL classification
DOTS_THRESHOLDS = {"elite": 400, "advanced": 300, "intermediate": 200}
MIN_MEETS_FOR_AMPLITUDE = 3
AMPLITUDE_DEFAULTS = {"novice": 0.068, "intermediate": 0.032,
                      "advanced": 0.018, "elite": 0.009}

# 600K level mapping — sets because dataset uses both "beginner" and "novice"
LEVEL_MAP = {
    "elite":        {"advanced"},
    "advanced":     {"advanced"},
    "intermediate": {"intermediate"},
    "novice":       {"beginner", "novice"},
}

# 600K goal classification seed keywords
STRENGTH_GOAL_KEYWORDS = {
    "strength", "muscle", "hypertrophy", "powerlifting",
    "bulk", "mass", "gain", "power", "weightlifting", "build",
}

# 600K equipment — exact casing, confirmed from live data
# Full Gym: 462,441 rows | Garage Gym: 96,648 rows
# At Home: 29,741 rows (excluded) | Dumbbell Only: 12,889 rows (excluded)
BARBELL_EQUIPMENT_VALUES = {"Full Gym", "Garage Gym"}

# Session structure with body_keywords for anatomically correct accessories
SESSION_FOCUS = {
    0: {"label": "Lower A",  "body_keywords": ["squat", "leg", "lunge", "hamstring",
        "glute", "hip", "calf", "quad", "bulgarian", "goblet"]},
    1: {"label": "Upper A",  "body_keywords": ["bench", "press", "chest", "shoulder",
        "tricep", "push", "dip", "fly", "pec", "incline", "decline"]},
    2: {"label": "Lower B",  "body_keywords": ["deadlift", "row", "pull", "back",
        "lat", "trap", "bicep", "curl", "rdl", "romanian", "rack"]},
    3: {"label": "Upper B",  "body_keywords": ["overhead", "ohp", "shoulder", "press",
        "lateral", "raise", "tricep", "dip", "push", "deltoid", "shrug"]},
}

PHASE_BOUNDARIES = {
    "accumulation":    range(1, 5),
    "intensification": range(5, 9),
    "realisation":     range(9, 12),
    "deload":          range(12, 13),
}

# Periodization shape — Prilepin (1974); Issurin (2008)
PERIODIZATION_SHAPE = [
    0.00, 0.07, 0.14, 0.21,
    0.29, 0.37, 0.46, 0.55,
    0.65, 0.80, 1.00, 0.20,
]

# Per-level floor/ceiling — Peterson et al. (2011); Haff & Triplett (2016);
# Helms et al. (2017); Issurin (2008); Sheiko programs
LEVEL_SHAPE_PARAMS = {
    "novice":       {"floor": 0.65, "ceiling": 0.90},
    "intermediate": {"floor": 0.68, "ceiling": 0.93},
    "advanced":     {"floor": 0.72, "ceiling": 0.95},
    "elite":        {"floor": 0.75, "ceiling": 0.93},
}

# RPE — Tuchscherer (2008); Helms et al. (2017)
RPE_FLOOR_BY_LEVEL = {"novice": 6.5, "intermediate": 7.0,
                      "advanced": 7.5, "elite": 8.0}
RPE_CEILING = 9.5

# OHP/bench ratio — Lehman (2005); Stastny et al. (2017)
OHP_BENCH_RATIO = {"novice": 0.61, "intermediate": 0.64,
                   "advanced": 0.67, "elite": 0.70}

# 600K intensity band mapping (used for accessory sorting only)
INTENSITY_BAND_MAP = {"default": 0.725, "low": 0.675, "medium": 0.775, "high": 0.875}
```

---

## `pipeline/dataset/models.py` — complete specification

```python
# Dataclasses only — no logic

@dataclass class PeriodizationTemplate:
    training_level: str
    week_pcts: list[float]   # 12 values, index 0 = week 1
    rpe_curve: list[float]
    amplitude: float
    amp_source: int          # number of OPL athletes contributing

@dataclass class AthletePersona:
    athlete_id, sex, age, bodyweight_kg, weight_class_kg
    squat_peak_kg, bench_peak_kg, deadlift_peak_kg, total_kg, dots
    training_level: str

@dataclass class Exercise:
    name, goal, equipment, intensity: float   # raw 0-10
    sets: int
    reps_value: float   # always positive abs value
    reps_unit: str      # "reps" or "seconds"
    level: str

@dataclass class SessionLog:
    week, day_index: int
    day_label, main_lift: str
    main_lift_kg, main_lift_rpe, volume_pct: float
    block_phase: str
    accessories: list[Exercise]

@dataclass class AthleteRecord:
    persona: AthletePersona
    sessions: list[SessionLog]

class GymData(NamedTuple):
    df: pd.DataFrame
    strength_goals: set
    barbell_equip: set
```

---

## `pipeline/dataset/opl_loader.py` — complete specification

### `load_opl(path) -> pd.DataFrame`
- Check `OPL_CLEANED_PATH` first. If exists, load directly + re-parse Date column.
- On cache miss: read raw CSV, select 11 columns, apply cleaning rules, save to
  `OPL_CLEANED_PATH`, return DataFrame.
- Print raw row count → retained row count on first run.
- Print "loading from cache" on subsequent runs.

### `classify_training_level(dots) -> str`
- ≥400 → elite, ≥300 → advanced, ≥200 → intermediate, else → novice

### `derive_opl_amplitude(opl_df) -> dict[str, float]`
1. Find athletes with ≥ `MIN_MEETS_FOR_AMPLITUDE` (3) competition entries
2. Classify each athlete by **peak career Dots** (not per-meet Dots)
3. Keep only meets where `Dots / peak_dots >= 0.95`
4. Sort by Name + Date, compute `progression = (total - prev_total) / prev_total`
5. Keep only `0 < progression <= 0.30`
6. Average per training level
7. Fall back to `AMPLITUDE_DEFAULTS` for any level with no data
8. Print per-level amplitude and progression count

---

## `pipeline/dataset/gym_loader.py` — complete specification

### `_parse_list_field(val) -> list[str]`
- `"['strength', 'hypertrophy']"` → `["strength", "hypertrophy"]`
- Plain string → `[string.lower()]`
- `"[]"` / NaN / None → `[]`
- Uses `ast.literal_eval` for bracket strings

### `_derive_strength_goals(goals: pd.Series) -> set`
- Flatten all goal lists → unique strings → keep those containing any
  `STRENGTH_GOAL_KEYWORDS` keyword
- Print count and sorted list of classified strings

### `load_gym(ex_path, prog_path) -> GymData`
- Check `GYM_COMBINED_PATH` first. If exists: load, re-parse goal/level lists,
  derive strength_goals, set barbell_equip, return GymData.
- On cache miss, run full pipeline:
  1. Read both CSVs
  2. Drop programs where `goal == '[]'`, `equipment == '[]'` or null,
     `description` null, `last_edit` null
  3. **Drop exercise rows where equipment not in `BARBELL_EQUIPMENT_VALUES`**
     (removes At Home + Dumbbell Only at source — they must never reach any query)
  4. Filter exercise rows to surviving program titles
  5. Join prog_slim on title; program-level goal/equipment wins
  6. Parse goal and level into Python lists in-place
  7. Dtype: reps → float64, intensity → float64 fillna(0.0)
  8. Save to `GYM_COMBINED_PATH`
- Print shape at each filtering step
- Set `barbell_equip = BARBELL_EQUIPMENT_VALUES` (exact casing, no classification)

### `query_accessories(gym_df, training_level, strength_goals, barbell_equip, day_index, n, seed) -> list[Exercise]`
- Build masks:
  - `level_targets = LEVEL_MAP[training_level]` (a set)
  - `level_mask`: `set(x) & level_targets` (intersection, handles both "beginner"/"novice")
  - `goal_mask`: `set(x) & strength_goals`
  - `equip_mask`: `.str.strip().isin(barbell_equip)` — no lowercasing
  - `body_mask`: any `SESSION_FOCUS[day_index]["body_keywords"]` kw in `exercise_name.lower()`
- 8-step filter chain (use first non-empty result):
  0. body + level + goal + equipment
  1. body + level + goal
  2. body + goal
  3. body only
  4. level + goal + equipment
  5. level + goal
  6. goal only
  7. full dataset
- Sort by intensity descending; replace 0 with -1 (lowest priority)
- Drop rows with null exercise_name or reps
- `head(n * 3).sample(frac=1, random_state=seed).head(n)`
- Split reps: `reps_value = abs(reps)`, `reps_unit = "seconds" if reps < 0 else "reps"`

---

## `pipeline/dataset/periodization.py` — complete specification

### `build_periodization_templates(opl_df, n_weeks=12) -> dict[str, PeriodizationTemplate]`
- Assert `len(PERIODIZATION_SHAPE) == n_weeks`
- Call `derive_opl_amplitude(opl_df)` to get amplitude per level
- For each level in `["novice", "intermediate", "advanced", "elite"]`:
  - `floor = LEVEL_SHAPE_PARAMS[level]["floor"]`
  - `ceiling = LEVEL_SHAPE_PARAMS[level]["ceiling"]`
  - `week_pcts[w] = round(floor + shape[w] * (ceiling - floor), 4)`
  - `rpe_curve[w] = round(rpe_floor + shape[w] * (RPE_CEILING - rpe_floor), 2)`
  - Print: floor, ceiling, range, RPE range, amplitude, and full week_pcts list

---

## `pipeline/dataset/athlete_generator.py` — complete specification

### `_week_to_phase(week) -> str`
- Lookup in `PHASE_BOUNDARIES`. Default "deload".

### `_main_lift_for_day(day_index, persona) -> tuple[str, float]`
- `ohp_peak = bench_peak_kg × OHP_BENCH_RATIO[training_level]`
- Return dict `{0: Squat, 1: Bench, 2: Deadlift, 3: OHP}[day_index % 4]`

### `sample_athlete_persona(opl_df, athlete_id, used_indices: set) -> tuple[AthletePersona, int]`
- Drop Name and Date from persona_df before sampling
- `seed = int.from_bytes(athlete_id.encode(), "big") % (2**31)`
- `idx = random.Random(seed).randint(0, n-1)`
- Walk forward while `idx in used_indices` (increment + modulo wrap)
- Raise `RuntimeError` if all rows exhausted
- Return `(AthletePersona, idx)`

### `build_training_block(persona, gym_df, template, strength_goals, barbell_equip, seed) -> list[SessionLog]`
- For each week 1–12, each day 0–3:
  - `block_target_kg = competition_1rm × (1 + template.amplitude)`
  - `working_kg = round(block_target_kg × week_pct / 2.5) × 2.5`
  - `block_phase = _week_to_phase(week)`
  - Call `query_accessories(..., day_index=day_idx, seed=seed + week*100 + day_idx)`

### `generate_one_athlete(..., used_indices: set) -> tuple[AthleteRecord, int]`
- Call `sample_athlete_persona(opl_df, athlete_id, used_indices)` → `(persona, opl_idx)`
- Build block, return `(AthleteRecord, opl_idx)`

---

## `pipeline/dataset/export.py` — complete specification

### `records_to_sessions_df(records) -> pd.DataFrame`
- One row per session. Base columns: athlete_id, sex, age, bodyweight_kg,
  weight_class_kg, squat/bench/deadlift/total peak, dots, training_level
- Per-session columns: week, day_index, day_label, block_phase, main_lift,
  main_lift_kg, main_lift_rpe, volume_pct
- `main_lift_pct_of_peak = main_lift_kg / peak_for_lift` (uses `OHP_BENCH_RATIO`)
- Pipe-separated accessory columns: accessories, accessory_sets, accessory_reps,
  accessory_reps_unit, accessory_intensity
- After building rows: sort by `[athlete_id, main_lift, week, day_index]`,
  compute `main_lift_delta_kg = diff()` per `[athlete_id, main_lift]` group,
  re-sort by `[athlete_id, week, day_index]`

### `records_to_block_summary_df(records) -> pd.DataFrame`
- One row per athlete per lift (4 rows per athlete)
- Representative day = `sessions[0].day_index` per lift
- Columns: athlete_id, sex, training_level, dots, lift, competition_1rm_kg,
  week_1_kg, week_peak_kg, week_floor_kg, peak_week, floor_week,
  total_gain_kg, peak_pct_of_1rm, block_phase_at_peak, weekly_kg_series

---

## `pipeline/dataset/dataset_main.py` — complete specification

Entry point. Located at `pipeline/dataset/dataset_main.py`.

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
```
This makes `config` and `pipeline` importable from any working directory.

### Checkpoint functions
```python
def load_checkpoint(path) -> tuple[set[str], set[int]]:
    # returns (completed_athlete_ids, used_opl_indices)
    # reads "completed" and "used_opl_indices" keys from checkpoint.json

def save_checkpoint(path, completed: set, used_indices: set) -> None:
    # atomic write via .tmp file
    # writes both "completed" and "used_opl_indices" to checkpoint.json

def clear_checkpoint(path) -> None:
    # delete checkpoint file if it exists
```

### `run()` function
```
1. Create OUT_DIR
2. Load OPL, 600K, build templates
3. Validate: n_athletes <= len(opl_df), else raise ValueError
4. Print OPL utilisation: n_athletes / len(opl_df)
5. Load checkpoint if --resume, else start fresh
6. Compute remaining = [ids not in completed]
7. Set sessions_write_header = not (resume and sessions_out exists)
   Set summary_write_header  = not (resume and summary_out exists)
8. Loop over remaining:
   a. generate_one_athlete(..., used_indices=used_indices)
   b. completed.add(aid), used_indices.add(opl_idx)
   c. Every CHECKPOINT_EVERY or at end of remaining:
      - records_to_sessions_df(batch) → append to sessions_out (mode="a")
      - records_to_block_summary_df(batch) → append to summary_out (mode="a")
      - Set write_header = False after first write
      - save_checkpoint(path, completed, used_indices)
      - Clear batch
9. Print final summary: athletes done, unique OPL rows, rows written
```

### CLI arguments
```
--n-athletes   int    default: N_ATHLETES
--resume       flag   default: True
--no-resume    flag   clears checkpoint, starts fresh
--out-sessions str
--out-summary  str
--opl          str
--gym-ex       str
--gym-prog     str
--checkpoint   str
```

### Run commands
```bash
# From gym_rag/ root:
python pipeline/dataset/dataset_main.py --n-athletes 200
python pipeline/dataset/dataset_main.py --n-athletes 20000 --resume
python pipeline/dataset/dataset_main.py --n-athletes 200 --no-resume
```

---

## GPU strategy

| Phase | Day | Colab T4 | Reason |
|---|---|---|---|
| Dataset generation | 1 | No | All CPU |
| Bulk image embedding | 2 | **Yes — critical** | 600+ PNGs on CPU = 30+ hrs. T4 < 1 hour. |
| Bulk text/table embedding | 2 | **Yes — high value** | 2000+ chunks in minutes |
| BM25 index build | 2 | No | Pure Python, seconds |
| Reranker development | 3 | **Yes** | Seconds on GPU vs minutes on CPU |
| Generation chain | 4 | No | Gemini is an API call |
| RAGAS + LangSmith | 5 | No | API calls |

Colab → Docker handoff after Day 2. Export from Colab:
- Qdrant collection snapshots (all 3) → Google Drive
- `data/bm25_index.pkl` → Google Drive
- `data/bm25_corpus.json` → Google Drive
- `data/checkpoint.json` → Google Drive

---

## Day 1 — Dataset generation

**Environment**: Local (no GPU)
**Goal**: `sessions.csv` and `block_summary.csv` for n athletes

### Pre-work
- [ ] Download `opl.csv` → `data/raw/opl.csv`
- [ ] Download `programs_detailed_boostcamp_kaggle.csv` → `data/raw/`
- [ ] Download `program_summary.csv` → `data/raw/`
- [ ] Get Gemini API key → `GEMINI_API_KEY` in `.env`
- [ ] `pip install -r requirements.txt`
- [ ] Create `data/raw/`, `data/output/` directories

### Implementation order
1. `config/settings.py` — all constants exactly as specified above
2. `config/__init__.py` — `from .settings import *`
3. `pipeline/dataset/models.py` — dataclasses
4. `pipeline/dataset/opl_loader.py` — `load_opl`, `classify_training_level`, `derive_opl_amplitude`
5. `pipeline/dataset/gym_loader.py` — `_parse_list_field`, `_derive_strength_goals`, `load_gym`, `query_accessories`
6. `pipeline/dataset/periodization.py` — `build_periodization_templates`
7. `pipeline/dataset/athlete_generator.py` — `_week_to_phase`, `_main_lift_for_day`, `sample_athlete_persona`, `build_training_block`, `generate_one_athlete`
8. `pipeline/dataset/export.py` — `records_to_sessions_df`, `records_to_block_summary_df`
9. `pipeline/dataset/dataset_main.py` — checkpoint helpers + `run()` + CLI

### Day 1 validation
- [ ] `python pipeline/dataset/dataset_main.py --n-athletes 5` completes without error
- [ ] `opl_cleaned.csv` exists after first run; second run prints "loading from cache"
- [ ] `combined_600k_dataset.csv` exists after first run; second run prints "loading from cache"
- [ ] `checkpoint.json` contains both `completed` and `used_opl_indices` keys
- [ ] No two athletes share the same OPL row index: `len(set(used_opl_indices)) == n_athletes`
- [ ] Amplitude ordering: novice > intermediate > advanced > elite
- [ ] `sessions.csv` has `n × 48` rows
- [ ] `block_summary.csv` has `n × 4` rows
- [ ] `block_phase` cycles: accumulation → intensification → realisation → deload
- [ ] `main_lift_delta_kg` is NaN for all week-1 rows
- [ ] `block_phase_at_peak` == "realisation" for all rows in block_summary
- [ ] Lower A sessions: no tricep/chest/shoulder accessories
- [ ] Lower B sessions: no leg/quad/glute accessories
- [ ] `accessory_reps_unit` contains "seconds" for at least some rows
- [ ] Resume: run with `--n-athletes 10`, interrupt at athlete 4, re-run with `--resume` — confirm continues from athlete 5 without duplicates
- [ ] Scale: run `--n-athletes 1000` — memory stays flat (no CSV reloads), streaming append confirmed

---

## Day 2 — Ingestion and embedding

**Environment**: Google Colab T4 GPU (embedding), local (Qdrant)
**Goal**: Three Qdrant collections + BM25 index on disk

- [ ] Start Qdrant: `docker run -d -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant`
- [ ] Create `notebooks/colab_ingestion.ipynb`
- [ ] Load `nvidia/llama-nemotron-embed-vl-1b-v2` float16 on T4
- [ ] `embed_image(image_path)` → 2048-dim list
- [ ] `embed_text(text)` → 2048-dim list (local), `embed_text_api(text)` → OpenRouter
- [ ] Create collections: `gym_images`, `gym_text`, `gym_tables` (2048-dim, COSINE)
- [ ] `ingest_images(athlete_dir, athlete_id, client)` — 3 PNGs per athlete
- [ ] `chunk_text(text, chunk_size=512, overlap=64)` — tiktoken cl100k_base
- [ ] `sessions_to_nl(sessions_df, athlete_id, week)` — include block_phase and delta_kg:
  `"Week 3 (intensification), Lower A: Squat 185.0kg (+5.0kg), RPE 7.8"`
- [ ] `ingest_text`, `ingest_tables` — embed and upsert with checkpoint recovery
- [ ] `build_bm25_index(chunks)` → `bm25_index.pkl` + `bm25_corpus.json`

### Day 2 validation
- [ ] `gym_images` vectors == n_athletes × 3
- [ ] BM25 search for "realisation" returns phase-tagged chunks
- [ ] BM25 search for "deadlift week 8" returns week 8 rows

---

## Day 3 — Retrieval layer

**Environment**: Local (Colab T4 for reranker)
**Goal**: End-to-end retrieval with offline metrics

- [ ] `dense_search`, `sparse_search`, `reciprocal_rank_fusion(k=60)`, `hybrid_search`
- [ ] `rerank(query, candidates, top_k=10)` — handles image and text candidates
- [ ] `assemble_context(ranked_results, max_tokens=4096)`
- [ ] `retrieve(query, query_image_path, bm25, corpus)`
- [ ] Write 20 golden Q&A pairs in `data/golden_eval_set.json` (manually, not generated)
- [ ] `run_retrieval_eval()` — MRR and NDCG@5

---

## Day 4 — Generation, API, UI

**Environment**: Local
**Goal**: Full system running locally

- [ ] `ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)`
- [ ] `retrieval_step` + `generation_step` with `@traceable`
- [ ] `RunnableWithMessageHistory` with `session_id`
- [ ] LangSmith: `LANGCHAIN_TRACING_V2=true`
- [ ] `POST /chat`, `GET /health`, `DELETE /chat/{session_id}`
- [ ] Streamlit UI: image upload, source expanders, session management

### Day 4 validation
- [ ] `/health` returns ok
- [ ] Text query cites athlete_id, week, block_phase
- [ ] Chart upload references visual trend
- [ ] Memory recalls first question after 3 turns
- [ ] LangSmith shows retrieval + rerank + generation spans

---

## Day 5 — Evaluation, deployment, README

- [ ] RAGAS eval: faithfulness, answer_relevancy, context_recall — record scores
- [ ] LangSmith groundedness evaluator
- [ ] Final MRR and NDCG@5
- [ ] Dockerfile + docker-compose (qdrant + app)
- [ ] Streamlit Community Cloud deploy — get public URL
- [ ] README: data strategy table, eval scores, architecture, demo link
- [ ] Resume: project entry, GitHub link, two metric bullets

---

## Critical path — what to cut if behind

| Day | Cuttable | Never cut |
|---|---|---|
| Day 1 | Reduce to 50 athletes | Data grounding, caching, checkpoint/resume, deduplication |
| Day 2 | Skip `gym_tables` | Image embedding, BM25 index, checkpointing |
| Day 3 | Skip reranker | Golden eval set, hybrid search |
| Day 4 | Skip image upload | LangSmith tracing, memory, /health |
| Day 5 | Skip groundedness evaluator | RAGAS scores, Docker, public URL, README |

---

## What done looks like

1. Public Streamlit URL loads for anyone with the link
2. Text query cites specific athlete IDs, week numbers, and block phases
3. Image upload references visual trends in the chart
4. README shows RAGAS faithfulness ≥ 0.70
5. LangSmith trace trees show all spans with latencies
6. Resume lists project with date, GitHub link, demo URL, two metric bullets
7. `docker-compose up` starts full system in under 2 minutes
8. `block_phase_at_peak` == "realisation" for all athletes in `block_summary.csv`
9. No two athletes share an OPL row index — confirmed by `len(set(used_opl_indices)) == n_athletes`
10. Second run of `dataset_main.py` prints cache hit messages for both OPL and 600K — no reprocessing

---

## Environment variables

```bash
GEMINI_API_KEY=           # Google AI Studio — free, 1K req/day
LANGSMITH_API_KEY=        # LangSmith — free, 5K traces/month
LANGSMITH_PROJECT=gym-rag
OPENROUTER_API_KEY=       # OpenRouter — nemotron embed at query time
QDRANT_HOST=localhost
QDRANT_PORT=6333
```