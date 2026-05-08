from __future__ import annotations
import logging
import os
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config.model_settings import GEMINI_AUX_MODEL
from config.rag_config import HYDE_INTENTS

# ── Regex patterns ─────────────────────────────────────────────────────────────

ATHLETE_ID_RE = re.compile(r'athlete_\d{5}')

PRONOUNS = re.compile(
    r'\b(their|his|her|they|them|that athlete|this athlete|same athlete|the athlete)\b',
    re.IGNORECASE,
)

INTENT_LABELS = ("factual", "trend", "comparison", "coaching", "visual")
_LEVEL_VALUES  = ("elite", "advanced", "intermediate", "novice")


# ── ID normalisation ───────────────────────────────────────────────────────────

def _parse_any_athlete_ref(value: str) -> str | None:
    """
    Parse any athlete ID variant into canonical form.
    Accepts: 'athlete_00089', 'athlete_89', 'athlete 89', '89' (bare number).
    Returns canonical string or None if unparseable.
    """
    v = value.strip().lower()
    # Already canonical
    if ATHLETE_ID_RE.fullmatch(v):
        return v
    # "athlete_89", "athlete 89", "athlete #89" etc.
    m = re.match(r'athlete[\s_#]*(\d{1,5})$', v)
    if m:
        return f"athlete_{int(m.group(1)):05d}"
    # Bare number
    if v.isdigit() and 1 <= len(v) <= 5:
        return f"athlete_{int(v):05d}"
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
            "All athlete IDs relevant to this question including bare numbers. "
            "Zero-pad to 5 digits: '42' -> 'athlete_00042', '2424' -> 'athlete_02424'. "
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
- Bare numbers in athlete context ARE athlete IDs — "compare 42 to 2424" means athlete_00042 vs athlete_02424
- Numbers that are clearly weights (kg/lbs), weeks, RPE, or set counts are NOT athlete IDs
- ALWAYS zero-pad to 5 digits: "42" → "athlete_00042", "2424" → "athlete_02424"
- Format MUST be athlete_NNNNN (underscore + exactly 5 digits)
- Empty list [] only for general questions with no specific athlete

sub_queries rules:
- ONLY populate when intent == "comparison"
- One self-contained query per athlete, e.g. ["athlete_00042 squat progression", "athlete_02424 squat progression"]
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
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_AUX_MODEL,
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


# ── Retry helpers ─────────────────────────────────────────────────────────────

_RETRY_DELAYS = (2.0, 8.0, 30.0)  # seconds between attempts 1→2, 2→3, 3→4

def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "ResourceExhausted" in type(exc).__name__
    )

def _suggested_delay(exc: Exception) -> float:
    """Extract Google's retryDelay value from the error string if present."""
    m = re.search(r'retryDelay["\s:\']+(\d+)s', str(exc))
    return float(m.group(1)) if m else 0.0


# ── Combined analysis ──────────────────────────────────────────────────────────

def _call_combined(query: str, history: list[dict], gemini_api_key: str) -> QueryAnalysis:
    summary_block = [m for m in history if m["content"].startswith("[Summary")]
    live_turns    = [m for m in history if not m["content"].startswith("[Summary")]
    context_msgs  = summary_block + live_turns[-3:]

    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in context_msgs
    ) or "None"

    last_exc: Exception | None = None
    for attempt, fallback_delay in enumerate([0.0, *_RETRY_DELAYS]):
        if fallback_delay:
            time.sleep(fallback_delay)
        try:
            chain  = _get_chain(gemini_api_key)
            result = chain.invoke({"history": history_str, "query": query})
            return result
        except Exception as exc:
            if _is_rate_limited(exc):
                wait = _suggested_delay(exc) or fallback_delay or _RETRY_DELAYS[0]
                logging.warning(
                    "[augmentation] Rate limited, waiting %.0fs (attempt %d/%d)",
                    wait, attempt + 1, len(_RETRY_DELAYS) + 1,
                )
                time.sleep(wait)
                last_exc = exc
            else:
                last_exc = exc
                break

    logging.warning("[augmentation] _call_combined failed after retries: %s", last_exc)
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
    last_exc: Exception | None = None
    for attempt, fallback_delay in enumerate([0.0, *_RETRY_DELAYS]):
        if fallback_delay:
            time.sleep(fallback_delay)
        try:
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_AUX_MODEL,
                google_api_key=gemini_api_key,
                temperature=0.4,
                max_output_tokens=400,
            )
            chain    = _HYDE_TEMPLATE | llm
            response = chain.invoke({"query": query})
            text = response.content.strip()
            return text if text else None
        except Exception as exc:
            if _is_rate_limited(exc):
                wait = _suggested_delay(exc) or fallback_delay or _RETRY_DELAYS[0]
                logging.warning(
                    "[augmentation] HyDE rate limited, waiting %.0fs (attempt %d/%d)",
                    wait, attempt + 1, len(_RETRY_DELAYS) + 1,
                )
                time.sleep(wait)
                last_exc = exc
            else:
                last_exc = exc
                break

    logging.warning("[augmentation] HyDE generation failed after retries: %s", last_exc)
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

    gemini_api_key = (
        getattr(getattr(gemini, "_api_client", None), "api_key", None)
        or os.environ.get("GEMINI_API_KEY", "")
    )

    # Only skip LLM analysis when there is genuinely no API key.
    # Empty history is fine — the LLM still extracts IDs and intent correctly;
    # pronoun resolution simply has nothing to resolve on the first turn.
    if not gemini_api_key:
        return {
            **inputs,
            "retrieval_query":  raw_query,
            "hyde_vector":      [],
            "sub_queries":      [],
            "intent":           "factual",
            "query_rewritten":  False,
            "original_query":   raw_query,
            "athlete_ids":      [],
            "training_levels":  [],
            "hyde_document":    "",
        }

    # ── Step 1: pronoun resolution ────────────────────────────────────────────
    register = EntityRegister()
    register.update_from_history(history)
    register.update_from_text(raw_query)
    pronoun_resolved = register.resolve_pronouns(raw_query)

    # ── Step 2: LangChain structured analysis ─────────────────────────────────
    # Handles intent, query rewrite, athlete ID extraction (including bare
    # numbers like "42"), and training level detection. Pydantic validators
    # in QueryAnalysis normalise every ID variant to athlete_NNNNN.
    analysis = _call_combined(pronoun_resolved, history, gemini_api_key)

    # ── Step 3: HyDE for intents that benefit from it ─────────────────────────
    hyde_document: str | None = None
    if use_hyde and analysis.intent in HYDE_INTENTS:
        hyde_document = _generate_hyde_document(pronoun_resolved, gemini_api_key)

    # ── Step 4: sub-query fallback for comparison intent ─────────────────────
    sub_queries = analysis.sub_queries
    if analysis.intent == "comparison" and len(register.all_ids()) >= 2 and not sub_queries:
        sub_queries = [
            f"{aid} {analysis.rewritten_query}" for aid in register.all_ids()[:4]
        ]

    # ── Step 5: embed HyDE document ───────────────────────────────────────────
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
        "athlete_ids":      analysis.athlete_ids,
        "sub_queries":      sub_queries,
        "hyde_document":    hyde_document or "",
        "hyde_vector":      hyde_vector,
        "training_levels":  analysis.training_levels,
    }
