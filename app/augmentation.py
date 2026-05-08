from __future__ import annotations
import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config.rag_config import HYDE_INTENTS

# ── Regex patterns ─────────────────────────────────────────────────────────────

ATHLETE_ID_RE = re.compile(r'athlete_\d{5}')

# Matches: "athlete 89", "athlete_89", "athlete89", "athlete #89", "athlete # 89"
_INFORMAL_ATHLETE_RE = re.compile(r'\bathlete[\s_]*#?\s*(\d{1,5})\b', re.IGNORECASE)

PRONOUNS = re.compile(
    r'\b(their|his|her|they|them|that athlete|this athlete|same athlete|the athlete)\b',
    re.IGNORECASE,
)

INTENT_LABELS = ("factual", "trend", "comparison", "coaching", "visual")
_LEVEL_VALUES  = ("elite", "advanced", "intermediate", "novice")


# ── ID normalisation ───────────────────────────────────────────────────────────

def _normalize_athlete_refs(text: str) -> tuple[str, list[str]]:
    """
    Replace informal athlete references with zero-padded canonical IDs.
    'athlete 89' -> 'athlete_00089'.
    Returns (normalized_text, [athlete_00089, ...]).
    """
    found: list[str] = []

    def _replace(m: re.Match) -> str:
        canonical = f"athlete_{int(m.group(1)):05d}"
        found.append(canonical)
        return canonical

    normalized = _INFORMAL_ATHLETE_RE.sub(_replace, text)
    seen: set[str] = set()
    unique = [aid for aid in found if not (aid in seen or seen.add(aid))]  # type: ignore[func-returns-value]
    return normalized, unique


def _parse_any_athlete_ref(value: str) -> str | None:
    """
    Parse any athlete ID variant into canonical form.
    Accepts: 'athlete_00089', 'athlete_89', 'athlete 89', '89' (bare number).
    Returns canonical string or None if unparseable.
    """
    # Already canonical
    if ATHLETE_ID_RE.fullmatch(value):
        return value
    # Informal format with prefix
    m = _INFORMAL_ATHLETE_RE.match(value)
    if m:
        return f"athlete_{int(m.group(1)):05d}"
    # Bare number
    if value.isdigit() and len(value) <= 5:
        return f"athlete_{int(value):05d}"
    return None


# ── Pydantic schema for structured output ─────────────────────────────────────

class QueryAnalysis(BaseModel):
    """Structured analysis of a powerlifting coaching query."""

    intent: Literal["factual", "trend", "comparison", "coaching", "visual"] = Field(
        description="Query intent category."
    )
    rewritten_query: str = Field(
        description="Self-contained retrieval query, max 25 words, no conversational filler.",
        max_length=200,
    )
    athlete_ids: list[str] = Field(
        default_factory=list,
        description=(
            "All athlete IDs relevant to this question. "
            "Zero-pad to 5 digits: 'athlete 89' -> 'athlete_00089'. "
            "Format: athlete_NNNNN. Empty list if no specific athlete mentioned."
        ),
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="One self-contained sub-query per athlete/level, only for comparison intent.",
    )
    training_levels: list[str] = Field(
        default_factory=list,
        description="Explicitly mentioned levels: elite, advanced, intermediate, novice.",
    )

    @field_validator("athlete_ids", mode="before")
    @classmethod
    def normalize_ids(cls, values: list) -> list[str]:
        """Normalize any athlete ID variant to athlete_NNNNN format."""
        result: list[str] = []
        seen: set[str] = set()
        for raw in values:
            canonical = _parse_any_athlete_ref(str(raw).strip())
            if canonical and canonical not in seen:
                result.append(canonical)
                seen.add(canonical)
        return result

    @field_validator("training_levels", mode="before")
    @classmethod
    def filter_levels(cls, values: list) -> list[str]:
        return [v for v in values if v in _LEVEL_VALUES]

    @field_validator("intent", mode="before")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        return v if v in INTENT_LABELS else "factual"


# ── LangChain chain builder ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a query optimizer for a powerlifting training database.
The database contains athlete session logs, coaching text, and PDF training reports.

Intent rules:
- factual    : specific data point for one athlete (week N lift, RPE, program name, DOTS)
- trend      : progression pattern across multiple weeks for one athlete
- comparison : comparing two or more athletes OR two or more training levels
- coaching   : open-ended advice, recommendation, or coaching question
- visual     : asks about a chart, heatmap, radar, or PDF page

rewritten_query rules:
- Resolve ALL pronouns (his/her/their) using athlete IDs from HISTORY
- Include athlete IDs, lift names, week numbers, RPE values
- Remove conversational filler ("can you tell me", "I was wondering")
- Max 25 words, no punctuation

athlete_ids rules:
- List ALL athlete IDs relevant to this question
- ALWAYS zero-pad to 5 digits: "athlete 89" → "athlete_00089", "athlete_7" → "athlete_00007"
- Format MUST be athlete_NNNNN (underscore + exactly 5 digits)
- Empty list [] for general questions with no specific athlete

sub_queries rules:
- ONLY populate when intent == "comparison"
- One self-contained query per athlete, e.g. ["athlete_00088 deadlift progression", "athlete_03985 deadlift progression"]
- Empty list [] for all other intents

training_levels rules:
- List every level explicitly mentioned: elite, advanced, intermediate, novice
- Empty list [] if no level mentioned or question is about a named athlete
"""

_HUMAN_PROMPT = """\
CONVERSATION HISTORY:
{history}

QUESTION: {query}
"""

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_PROMPT),
])


def _build_chain(gemini_api_key: str):
    """Build the LangChain structured-output chain (cached per process)."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key,
        temperature=0.0,
        max_output_tokens=300,
    )
    return _PROMPT_TEMPLATE | llm.with_structured_output(QueryAnalysis)


_chain_cache: dict = {}


def _get_chain(gemini_api_key: str):
    if gemini_api_key not in _chain_cache:
        _chain_cache[gemini_api_key] = _build_chain(gemini_api_key)
    return _chain_cache[gemini_api_key]


# ── Combined analysis ──────────────────────────────────────────────────────────

def _call_combined(query: str, history: list[dict], gemini_api_key: str) -> QueryAnalysis:
    summary_block = [m for m in history if m["content"].startswith("[Summary")]
    live_turns    = [m for m in history if not m["content"].startswith("[Summary")]
    context_msgs  = summary_block + live_turns[-3:]

    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in context_msgs
    ) or "None"

    try:
        chain  = _get_chain(gemini_api_key)
        result = chain.invoke({"history": history_str, "query": query})
        return result  # already a QueryAnalysis instance
    except Exception as e:
        logging.warning("[augmentation] LangChain structured-output call failed: %s", e)
        return QueryAnalysis(
            intent="factual",
            rewritten_query=query,
            athlete_ids=[],
            sub_queries=[],
            training_levels=[],
        )


# ── HyDE ──────────────────────────────────────────────────────────────────────

_HYDE_PROMPT = """\
Generate a training record passage (4-6 sentences, approximately 100 words) that \
would perfectly answer this question.
Write as if it is a real coaching note or session log from the athlete database.
Include: athlete ID if mentioned, week number, lift names (squat/bench/deadlift/OHP),
kg loads, RPE values, sets x reps, block phase name.
Do not use markdown, headers, or bullet points. Plain prose only.

QUESTION: {query}

TRAINING RECORD PASSAGE:\
"""

_HYDE_TEMPLATE = ChatPromptTemplate.from_messages([("human", _HYDE_PROMPT)])


def _generate_hyde_document(query: str, gemini_api_key: str) -> str | None:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
            temperature=0.4,
            max_output_tokens=400,
        )
        chain    = _HYDE_TEMPLATE | llm
        response = chain.invoke({"query": query})
        text = response.content.strip()
        return text if text else None
    except Exception as e:
        logging.warning("[augmentation] HyDE generation failed: %s", e)
        return None


# ── Entity register (pronoun resolution) ─────────────────────────────────────

class EntityRegister:

    def __init__(self) -> None:
        self._ids: list[str] = []

    def update_from_history(self, history: list[dict]) -> None:
        seen = set(self._ids)
        for msg in history:
            for aid in ATHLETE_ID_RE.findall(msg.get("content", "")):
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
        if not recent:
            return query
        return PRONOUNS.sub(recent, query)


# ── Public entry point ────────────────────────────────────────────────────────

def augment(inputs: dict) -> dict:
    raw_query = inputs["query"]
    memory    = inputs["memory"]
    gemini    = inputs.get("gemini")
    history   = memory.get_history()
    use_hyde  = inputs.get("use_hyde", True)

    # Resolve API key from the Gemini client or environment
    gemini_api_key = (
        getattr(getattr(gemini, "_api_client", None), "api_key", None)
        or os.environ.get("GEMINI_API_KEY", "")
    )

    # Always normalize before the early-exit so first messages still get athlete_ids
    normalized_query, informal_ids = _normalize_athlete_refs(raw_query)

    if not history or not gemini_api_key:
        return {
            **inputs,
            "retrieval_query":  normalized_query,
            "hyde_vector":      [],
            "sub_queries":      [],
            "intent":           "factual",
            "query_rewritten":  normalized_query != raw_query,
            "original_query":   raw_query,
            "athlete_ids":      informal_ids,
            "training_levels":  [],
            "hyde_document":    "",
        }

    # ── Step 2: pronoun resolution using history + normalized query ───────────
    register = EntityRegister()
    register.update_from_history(history)
    register.update_from_text(normalized_query)
    pronoun_resolved = register.resolve_pronouns(normalized_query)

    # ── Step 3: LangChain structured analysis (intent + rewrite + IDs) ────────
    analysis = _call_combined(pronoun_resolved, history, gemini_api_key)

    # ── Step 4: merge regex IDs (ground truth) with LLM IDs ──────────────────
    seen_ids: set[str] = set()
    athlete_ids: list[str] = []
    for aid in informal_ids + analysis.athlete_ids:
        if aid not in seen_ids:
            athlete_ids.append(aid)
            seen_ids.add(aid)

    # ── Step 5: HyDE for intents that benefit from it ─────────────────────────
    hyde_document: str | None = None
    if use_hyde and analysis.intent in HYDE_INTENTS:
        hyde_document = _generate_hyde_document(pronoun_resolved, gemini_api_key)

    # ── Step 6: sub-query fallback for comparison intent ─────────────────────
    sub_queries = analysis.sub_queries
    if analysis.intent == "comparison" and len(register.all_ids()) >= 2 and not sub_queries:
        sub_queries = [
            f"{aid} {analysis.rewritten_query}" for aid in register.all_ids()[:4]
        ]

    # ── Step 7: embed HyDE document ───────────────────────────────────────────
    hyde_vector: list[float] = []
    if use_hyde and hyde_document and analysis.intent != "visual":
        try:
            from pipeline.ingestion.embedder import embed_query
            hyde_vector = embed_query(hyde_document, mode="passage")
        except Exception:
            hyde_vector = []

    return {
        **inputs,
        "retrieval_query":  analysis.rewritten_query,
        "original_query":   raw_query,
        "pronoun_resolved": pronoun_resolved,
        "query_rewritten":  analysis.rewritten_query != raw_query,
        "intent":           analysis.intent,
        "athlete_ids":      athlete_ids,
        "sub_queries":      sub_queries,
        "hyde_document":    hyde_document or "",
        "hyde_vector":      hyde_vector,
        "training_levels":  analysis.training_levels,
    }
