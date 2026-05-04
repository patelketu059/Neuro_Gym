from __future__ import annotations
import logging
import re
import json

from config.model_settings import GEMINI_AUX_MODEL
from config.rag_config import HYDE_INTENTS

ATHLETE_ID_RE = re.compile(r'athlete_\d{5}')
INTENT_LABELS = ("factual", "trend", "comparison", "coaching", "visual")

PRONOUNS = re.compile(
    r'\b(their|his|her|they|them|that athlete|this athlete|same athlete|the athlete)\b',
    re.IGNORECASE,
)


class EntityRegister:

    def __init__(self) -> None:
        self._ids: list[str] = []

    def update_from_history(self, history: list[dict]) -> None:
        seen = set(self._ids)
        for msg in history:
            for aid in ATHLETE_ID_RE.findall(msg.get('content', '')):
                if aid not in seen:
                    self._ids.append(aid)
                    seen.add(aid)

    def update_from_text(self, text: str) -> None:
        seen = set(self._ids)
        for aid in ATHLETE_ID_RE.findall(text):
            if aid not in seen:
                self._ids.append(aid)
                seen.add(aid)

    def most_recent(self) -> str | None:
        return self._ids[-1] if self._ids else None

    def all_ids(self) -> list[str]:
        return list(self._ids)
    
    def resolve_pronouns(self, query: str) -> str:
        recent = self.most_recent()
        if not recent: return query
        return PRONOUNS.sub(recent, query)



_LEVEL_VALUES = ('elite', 'advanced', 'intermediate', 'novice')

_COMBINED_PROMPT = """
You are a query optimizer for a powerlifting training database.
The database contains athlete session logs, coaching text, and PDF training reports.

Analyze the question and return ONLY a JSON object - no markdown, no explanation:
{{
    "intent": "<factual|trend|comparison|coaching|visual>",
    "rewritten_query": "<self-contained retrieval query, max 25 words>",
    "athlete_ids": ["<athlete_id>", ...],
    "sub_queries": ["<query1>", "<query2>"],
    "training_levels": ["<elite|advanced|intermediate|novice>", ...]
}}

Intent Rules:
- factual       : specific data point for one athlete (week N lift, RPE, program name, Dots)
- trend         : progression pattern across multiple weeks for one athlete
- comparison    : explicitly comparing two or more athletes OR two or more training levels
- coaching      : open-ended advice, recommendation, or coaching question
- visual        : asks about a chart, heatmap, radar, or PDF page

rewritten_query rules:
-   Resolve ALL pronouns (his/her/their/it/that athlete) using athlete IDs from HISTORY
-   Include: athlete IDs, lift names (squat/bench/deadlift/OHP), week numbers, RPE values
-   Remove conversational filler ("can you tell me", "I was wondering")
-   Max 25 words, no punctuation

athlete_ids rules:
-   List ALL athlete IDs relevant to this question (from query text + HISTORY)
-   Empty list [] for no specific athletes (e.g. general questions about a level group)

sub_queries rules (ONLY populate if intent == "comparison"):
-   For athlete comparisons: one self-contained query per athlete, e.g.
    ["athlete_00088 deadlift peak progression", "athlete_03985 deadlift peak progression"]
-   For level-group comparisons: one self-contained query per level mentioning the topic, e.g.
    ["advanced athletes RPE progression across block", "novice athletes RPE progression across block"]
-   Empty list [] for all other intents

training_levels rules:
-   List every training level explicitly mentioned in the query
-   Use ONLY these values: "elite", "advanced", "intermediate", "novice"
-   Single level query ("elite athletes only")   → ["elite"]
-   Cross-level comparison ("advanced vs novice") → ["advanced", "novice"]
-   No level mentioned, or question about a named athlete → [] (empty list)

CONVERSATION HISTORY:
{history}

QUESTION: {query}
"""


def _call_combined(query: str, history: list[dict], gemini) -> dict:

    summary_block = [m for m in history if m['content'].startswith("[Summary")]
    live_turns    = [m for m in history if not m["content"].startswith("[Summary")]
    
    context_msgs = summary_block + live_turns[-3:]

    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in context_msgs
    )
    prompt = _COMBINED_PROMPT.format(history=history_str, query=query)
    try:
        from google.genai import types
        response = gemini.models.generate_content(
            model    = GEMINI_AUX_MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                temperature       = 0.0,
                max_output_tokens = 200,
            ),
        )

        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        intent = data.get('intent', 'factual')
        if intent not in INTENT_LABELS:
            intent = 'factual'

        rewritten = data.get('rewritten_query', query).strip()
        if not rewritten or len(rewritten.split()) > 40:
            rewritten = query

        rewritten = re.sub(r'^["\']|["\']$', "", rewritten).strip() or query

        athlete_ids = [
            aid for aid in data.get('athlete_ids', [])
            if ATHLETE_ID_RE.fullmatch(aid)
        ]

        sub_queries = data.get('sub_queries', []) if intent == 'comparison' else []

        training_levels = [
            lvl for lvl in data.get('training_levels', [])
            if lvl in _LEVEL_VALUES
        ]

        return {
            'intent': intent,
            'rewritten_query': rewritten,
            'athlete_ids': athlete_ids,
            'sub_queries': sub_queries,
            'training_levels': training_levels,
        }

    except Exception as e:
        logging.warning("[augmentation] LLM query-analysis call failed: %s", e)
        return {
            'intent': 'factual',
            'rewritten_query': query,
            'athlete_ids': [],
            'sub_queries': [],
            'training_levels': [],
        }



_HYDE_PROMPT = """\
Generate a training record passage (4-6 sentences, approximately 100 words) that \
would perfectly answer this question.
Write as if it is a real coaching note or session log from the athlete database.
Match the length of an actual training record chunk — not a short summary.
Include: athlete ID if mentioned, week number, lift names (squat/bench/deadlift/OHP),
kg loads, RPE values, sets x reps, block phase name.
Do not use markdown, headers, or bullet points. Plain prose only.

QUESTION: {query}

TRAINING RECORD PASSAGE:\
"""

def _generate_hyde_document(query: str, gemini) -> str | None:

    try:
        from google.genai import types
        response = gemini.models.generate_content(
            model   = "gemini-2.0-flash",
            contents= _HYDE_PROMPT.format(query=query),
            config  = types.GenerateContentConfig(temperature=0.4, max_output_tokens=400),
        )
        text = response.text.strip()
        return text if text else None
    except Exception as e:
        logging.warning("[augmentation] HyDE generation failed: %s", e)
        return None
    

def augment(inputs: dict) -> dict:
    raw_query = inputs['query']
    memory = inputs['memory']
    gemini = inputs.get('gemini')
    history = memory.get_history()
    use_hyde = inputs.get('use_hyde', True)


    if not history or not gemini:
        return {
            **inputs,
            'retrieval_query': raw_query,
            'hyde_vector': [],
            'sub_queries': [],
            'intent': "factual",
            'query_rewritten': False,
            'original_query': raw_query,
            'training_levels': [],
        }
    

    register = EntityRegister()
    register.update_from_history(history)
    register.update_from_text(raw_query)
    pronoun_resolved = register.resolve_pronouns(raw_query)

    def _run_combined():
        return _call_combined(pronoun_resolved, history, gemini)

    def _run_hyde():
        return _generate_hyde_document(pronoun_resolved, gemini)

    # Always run combined analysis first (fast, temp=0.0)
    combined_result = _run_combined()
    intent_early = combined_result.get('intent', 'factual')

    # Only run HyDE for intents that benefit from it
    hyde_document: str | None = None
    if use_hyde and intent_early in HYDE_INTENTS and gemini:
        hyde_document = _run_hyde()

    intent = combined_result['intent']
    rewritten_query = combined_result['rewritten_query']
    athlete_ids = combined_result['athlete_ids']
    sub_queries = combined_result['sub_queries']

    if intent == 'comparison' and len(register.all_ids()) >= 2 and not sub_queries:
        topic = rewritten_query
        sub_queries = [
            f"{aid} {topic}" for aid in register.all_ids()[:4]
        ]


    hyde_vector: list[float] = []
    if use_hyde and hyde_document and intent != 'visual':
        try:
            from pipeline.ingestion.embedder import embed_query
            hyde_vector = embed_query(hyde_document, mode = 'passage')
        except Exception:
            hyde_vector = []


    return {
        **inputs,
        "retrieval_query":  rewritten_query,
        "original_query":   raw_query,
        "pronoun_resolved": pronoun_resolved,
        "query_rewritten":  rewritten_query != raw_query,
        "intent":           intent,
        "athlete_ids":      athlete_ids,
        "sub_queries":      sub_queries,
        "hyde_document":    hyde_document or "",
        "hyde_vector":      hyde_vector,
        "training_levels":  combined_result.get('training_levels', []),
    }