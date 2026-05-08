"""
Golden question bank — single source of truth for all eval harnesses.

50 questions covering all 5 intents (factual, trend, comparison, coaching, visual)
across all 4 training levels (elite, advanced, intermediate, novice).

Athletes are pseudo-randomly selected across training levels:
  elite       : athlete_00042, athlete_00089, athlete_00033
  advanced    : athlete_00117, athlete_00178
  intermediate: athlete_00250, athlete_00301
  novice      : athlete_00421, athlete_00567   (run sample_athletes.py to confirm)

All questions are phrased colloquially — as a coach or athlete would actually ask —
to stress-test the query rewrite and pronoun-resolution pipeline.

Run   eval/sample_athletes.py   first to validate that all athlete IDs used here
exist in your Qdrant instance before generating references.
"""
from __future__ import annotations
from typing import Any

GOLDEN_QUESTIONS: list[dict[str, Any]] = [

    # ── FACTUAL (15) ────────────────────────────────────────────────────────────
    # Single data-point look-ups for one athlete.

    {
        "id": "Q01",
        "query": "Hey, how heavy was athlete 42 going on squat in week 8, and what RPE did they hit?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["elite"],
    },
    {
        "id": "Q02",
        "query": "How many sets and reps was athlete 117 doing on bench during that intensification block?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "factual",
        "difficulty": "medium",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q03",
        "query": "What accessory exercises did athlete 250 do on their lower body days in week 3?",
        "gt_athlete_ids": ["athlete_00250"],
        "intent": "factual",
        "difficulty": "medium",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q04",
        "query": "What's athlete 89's DOTS score and does that make them elite, advanced, or what?",
        "gt_athlete_ids": ["athlete_00089"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["elite"],
    },
    {
        "id": "Q05",
        "query": "What did athlete 33 squat, bench, and pull in competition?",
        "gt_athlete_ids": ["athlete_00033"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["elite"],
    },
    {
        "id": "Q06",
        "query": "What program was athlete 178 running — did they have a secondary program on top of it?",
        "gt_athlete_ids": ["athlete_00178"],
        "intent": "factual",
        "difficulty": "medium",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q07",
        "query": "What's athlete 301's bodyweight and which IPF class does that put them in?",
        "gt_athlete_ids": ["athlete_00301"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q08",
        "query": "What RPE was athlete 42 working at on week 6 of their squat sessions?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "factual",
        "difficulty": "medium",
        "training_levels": ["elite"],
    },
    {
        "id": "Q09",
        "query": "How heavy was athlete 117 on deadlift in week 1 — like what was their starting load?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q10",
        "query": "What's athlete 89's bodyweight?",
        "gt_athlete_ids": ["athlete_00089"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["elite"],
    },
    {
        "id": "Q11",
        "query": "Did athlete 250 have an overhead press in their program and what was their peak?",
        "gt_athlete_ids": ["athlete_00250"],
        "intent": "factual",
        "difficulty": "medium",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q12",
        "query": "What program did athlete 33 follow?",
        "gt_athlete_ids": ["athlete_00033"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["elite"],
    },
    {
        "id": "Q13",
        "query": "How light did athlete 178 go during the deload in week 12?",
        "gt_athlete_ids": ["athlete_00178"],
        "intent": "factual",
        "difficulty": "medium",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q14",
        "query": "What was athlete 301's peak squat number?",
        "gt_athlete_ids": ["athlete_00301"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q15",
        "query": "What load was athlete 42 benching in week 4?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "factual",
        "difficulty": "easy",
        "training_levels": ["elite"],
    },

    # ── TREND (10) ──────────────────────────────────────────────────────────────
    # Week-by-week progression for a single athlete across the full block.

    {
        "id": "Q16",
        "query": "Walk me through how athlete 42's squat progressed — did it go up every week or were there dips?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "trend",
        "difficulty": "medium",
        "training_levels": ["elite"],
    },
    {
        "id": "Q17",
        "query": "What happened to athlete 42's squat load in week 10 — was that a deload or did they push hard?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "trend",
        "difficulty": "medium",
        "training_levels": ["elite"],
    },
    {
        "id": "Q18",
        "query": "How did athlete 117's deadlift numbers change from the start of the block all the way to week 12?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "trend",
        "difficulty": "hard",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q19",
        "query": "How did athlete 89's training volume shift between the accumulation and realisation phases?",
        "gt_athlete_ids": ["athlete_00089"],
        "intent": "trend",
        "difficulty": "hard",
        "training_levels": ["elite"],
    },
    {
        "id": "Q20",
        "query": "Did athlete 250's RPE keep climbing all the way to week 11 or did it level off?",
        "gt_athlete_ids": ["athlete_00250"],
        "intent": "trend",
        "difficulty": "medium",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q21",
        "query": "Show me how athlete 33's total training volume looked week by week across the whole block.",
        "gt_athlete_ids": ["athlete_00033"],
        "intent": "trend",
        "difficulty": "hard",
        "training_levels": ["elite"],
    },
    {
        "id": "Q22",
        "query": "Did athlete 178's main lifts go up consistently or were there any rough patches?",
        "gt_athlete_ids": ["athlete_00178"],
        "intent": "trend",
        "difficulty": "medium",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q23",
        "query": "How did athlete 301's squat load trend throughout the 12-week block?",
        "gt_athlete_ids": ["athlete_00301"],
        "intent": "trend",
        "difficulty": "medium",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q24",
        "query": "How different was athlete 42's deadlift intensity in realisation compared to the accumulation phase?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "trend",
        "difficulty": "hard",
        "training_levels": ["elite"],
    },
    {
        "id": "Q25",
        "query": "Did athlete 117's bench trend upward the whole block or did they plateau?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "trend",
        "difficulty": "medium",
        "training_levels": ["advanced"],
    },

    # ── COMPARISON (10) ─────────────────────────────────────────────────────────
    # Cross-athlete and cross-level comparisons.

    {
        "id": "Q26",
        "query": "Who was squatting more — athlete 42 or athlete 117?",
        "gt_athlete_ids": ["athlete_00042", "athlete_00117"],
        "intent": "comparison",
        "difficulty": "medium",
        "training_levels": ["elite", "advanced"],
    },
    {
        "id": "Q27",
        "query": "Compare athlete 89 and athlete 33 — who's got the better DOTS and what level is each of them?",
        "gt_athlete_ids": ["athlete_00089", "athlete_00033"],
        "intent": "comparison",
        "difficulty": "medium",
        "training_levels": ["elite"],
    },
    {
        "id": "Q28",
        "query": "Between athlete 250 and athlete 301, who was training at higher RPE in intensification?",
        "gt_athlete_ids": ["athlete_00250", "athlete_00301"],
        "intent": "comparison",
        "difficulty": "hard",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q29",
        "query": "Do elite athletes generally push higher RPE than novices across the full block?",
        "gt_athlete_ids": [],
        "intent": "comparison",
        "difficulty": "hard",
        "training_levels": ["elite", "novice"],
    },
    {
        "id": "Q30",
        "query": "How much heavier are advanced athletes squatting compared to intermediate ones on average?",
        "gt_athlete_ids": [],
        "intent": "comparison",
        "difficulty": "hard",
        "training_levels": ["advanced", "intermediate"],
    },
    {
        "id": "Q31",
        "query": "Who had the bigger DOTS — athlete 42 or athlete 89?",
        "gt_athlete_ids": ["athlete_00042", "athlete_00089"],
        "intent": "comparison",
        "difficulty": "easy",
        "training_levels": ["elite"],
    },
    {
        "id": "Q32",
        "query": "Comparing athlete 178 and athlete 301 — who ran more total volume across the block?",
        "gt_athlete_ids": ["athlete_00178", "athlete_00301"],
        "intent": "comparison",
        "difficulty": "medium",
        "training_levels": ["advanced", "intermediate"],
    },
    {
        "id": "Q33",
        "query": "How different does the realisation phase look between intermediate and advanced athletes?",
        "gt_athlete_ids": [],
        "intent": "comparison",
        "difficulty": "hard",
        "training_levels": ["intermediate", "advanced"],
    },
    {
        "id": "Q34",
        "query": "Between athlete 33 and athlete 117, who made more absolute progress on deadlift?",
        "gt_athlete_ids": ["athlete_00033", "athlete_00117"],
        "intent": "comparison",
        "difficulty": "hard",
        "training_levels": ["elite", "advanced"],
    },
    {
        "id": "Q35",
        "query": "How do elite and novice athletes differ in how they structure their deload weeks?",
        "gt_athlete_ids": [],
        "intent": "comparison",
        "difficulty": "hard",
        "training_levels": ["elite", "novice"],
    },

    # ── COACHING (10) ───────────────────────────────────────────────────────────
    # Open-ended advice grounded in retrieved data.

    {
        "id": "Q36",
        "query": "Looking at athlete 42's RPE trend, do you think they were ready to peak or did they push too hard?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "coaching",
        "difficulty": "hard",
        "training_levels": ["elite"],
    },
    {
        "id": "Q37",
        "query": "Based on athlete 117's data what should their next training block focus on?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "coaching",
        "difficulty": "hard",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q38",
        "query": "What's the weakest area in athlete 89's training based on their numbers and what would you fix?",
        "gt_athlete_ids": ["athlete_00089"],
        "intent": "coaching",
        "difficulty": "hard",
        "training_levels": ["elite"],
    },
    {
        "id": "Q39",
        "query": "Give athlete 250 some coaching advice for improving their bench press based on their block data.",
        "gt_athlete_ids": ["athlete_00250"],
        "intent": "coaching",
        "difficulty": "hard",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q40",
        "query": "What could athlete 33 change in their programming to bump that DOTS score?",
        "gt_athlete_ids": ["athlete_00033"],
        "intent": "coaching",
        "difficulty": "hard",
        "training_levels": ["elite"],
    },
    {
        "id": "Q41",
        "query": "Did the program athlete 178 chose actually suit their training level, or would something different work better?",
        "gt_athlete_ids": ["athlete_00178"],
        "intent": "coaching",
        "difficulty": "hard",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q42",
        "query": "Looking at athlete 301's week 12 loads, what adjustments would you make going into a meet?",
        "gt_athlete_ids": ["athlete_00301"],
        "intent": "coaching",
        "difficulty": "hard",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q43",
        "query": "What kind of program structure tends to work best for intermediate powerlifters based on the data?",
        "gt_athlete_ids": [],
        "intent": "coaching",
        "difficulty": "medium",
        "training_levels": ["intermediate"],
    },
    {
        "id": "Q44",
        "query": "Looking at both athlete 42 and athlete 117's data, which of the two is training more optimally?",
        "gt_athlete_ids": ["athlete_00042", "athlete_00117"],
        "intent": "coaching",
        "difficulty": "hard",
        "training_levels": ["elite", "advanced"],
    },
    {
        "id": "Q45",
        "query": "How should elite athletes handle deload weeks differently from how novices do it?",
        "gt_athlete_ids": [],
        "intent": "coaching",
        "difficulty": "medium",
        "training_levels": ["elite", "novice"],
    },

    # ── VISUAL (5) ──────────────────────────────────────────────────────────────
    # Questions about charts and PDF visualizations.

    {
        "id": "Q46",
        "query": "Can you describe what athlete 42's load progression chart looks like across the block?",
        "gt_athlete_ids": ["athlete_00042"],
        "intent": "visual",
        "difficulty": "medium",
        "training_levels": ["elite"],
    },
    {
        "id": "Q47",
        "query": "What does athlete 117's volume radar chart tell us about how they split upper and lower body work?",
        "gt_athlete_ids": ["athlete_00117"],
        "intent": "visual",
        "difficulty": "medium",
        "training_levels": ["advanced"],
    },
    {
        "id": "Q48",
        "query": "What does the RPE heatmap look like for athlete 89 — which weeks show the highest intensity?",
        "gt_athlete_ids": ["athlete_00089"],
        "intent": "visual",
        "difficulty": "medium",
        "training_levels": ["elite"],
    },
    {
        "id": "Q49",
        "query": "Describe what athlete 33's periodization chart looks like — do you see a clear peak-and-taper shape?",
        "gt_athlete_ids": ["athlete_00033"],
        "intent": "visual",
        "difficulty": "hard",
        "training_levels": ["elite"],
    },
    {
        "id": "Q50",
        "query": "What does the load progression chart show for athlete 250 — did their lifts actually go up the way they were supposed to?",
        "gt_athlete_ids": ["athlete_00250"],
        "intent": "visual",
        "difficulty": "hard",
        "training_levels": ["intermediate"],
    },
]

# ── Convenience accessors ─────────────────────────────────────────────────────

def by_intent(intent: str) -> list[dict]:
    return [q for q in GOLDEN_QUESTIONS if q["intent"] == intent]

def by_difficulty(difficulty: str) -> list[dict]:
    return [q for q in GOLDEN_QUESTIONS if q["difficulty"] == difficulty]

def by_id(qid: str) -> dict | None:
    return next((q for q in GOLDEN_QUESTIONS if q["id"] == qid), None)

# Intent distribution summary (printed on import in verbose mode)
_INTENT_COUNTS = {
    intent: sum(1 for q in GOLDEN_QUESTIONS if q["intent"] == intent)
    for intent in ("factual", "trend", "comparison", "coaching", "visual")
}

if __name__ == "__main__":
    print(f"Total questions: {len(GOLDEN_QUESTIONS)}")
    for intent, count in _INTENT_COUNTS.items():
        print(f"  {intent:<12} {count}")
