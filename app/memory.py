from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Literal

from config.model_settings import GEMINI_AUX_MODEL
from config.rag_config import SESSION_TTL_SECONDS


@dataclass
class Message:
    role: Literal['user', 'assistant']
    content: str

    @property
    def tokens(self) -> int:
        return max(1, len(self.content) // 4)


class ConversationSummaryBufferMemory:

    def __init__(
            self,
            session_id: str,
            gemini=None,
            k: int = 8,
            max_token_budget: int = 2000,
            gemini_model: str = GEMINI_AUX_MODEL
    ):
        self.session_id       = session_id
        self.gemini           = gemini
        self.k                = k
        self.max_token_budget = max_token_budget
        self.gemini_model     = gemini_model
        self.buffer:  list[Message] = []
        self.summary: str           = ""

    # ── serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "k":                self.k,
            "max_token_budget": self.max_token_budget,
            "gemini_model":     self.gemini_model,
            "buffer":           [{"role": m.role, "content": m.content} for m in self.buffer],
            "summary":          self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict, gemini=None) -> "ConversationSummaryBufferMemory":
        obj = cls(
            session_id       = data["session_id"],
            gemini           = gemini,
            k                = data.get("k", 8),
            max_token_budget = data.get("max_token_budget", 2000),
            gemini_model     = data.get("gemini_model", GEMINI_AUX_MODEL),
        )
        obj.buffer  = [Message(role=m["role"], content=m["content"]) for m in data.get("buffer", [])]
        obj.summary = data.get("summary", "")
        return obj

    def _persist(self) -> None:
        """Write current state to session store (best-effort, never raises)."""
        try:
            from app.session_store import get_store
            get_store().set(self.session_id, self.to_dict())
        except Exception:
            pass

    # ── public API ───────────────────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        self.buffer.append(Message(role='user', content=content))
        self._maybe_summarise()
        self._persist()

    def add_ai_message(self, content: str) -> None:
        self.buffer.append(Message(role='assistant', content=content))
        self._maybe_summarise()
        self._persist()

    def get_history(self) -> list[dict]:
        msgs: list[dict] = []
        if self.summary:
            msgs.append({
                "role":    "user",
                "content": f"[Summary of earlier conversation]\n{self.summary}",
            })
        msgs.extend(
            {"role": m.role, "content": m.content}
            for m in self.buffer[-self.k:]
        )
        return msgs

    def clear(self) -> None:
        self.buffer  = []
        self.summary = ""

    def buffer_tokens(self) -> int:
        return sum(m.tokens for m in self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    # ── internal ─────────────────────────────────────────────────────────────

    def _maybe_summarise(self) -> None:
        if self.buffer_tokens() <= self.max_token_budget:
            return

        half             = max(1, len(self.buffer) // 2)
        to_compress      = self.buffer[:half]
        self.buffer      = self.buffer[half:]
        new_summary      = self._call_gemini(to_compress)
        self.summary     = f"{self.summary}\n\n{new_summary}" if self.summary else new_summary

    def _call_gemini(self, messages: list[Message]) -> str:
        import json as _json

        transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        prompt = (
            "Summarize this powerlifting coaching conversation as a JSON object. "
            "Preserve exact kg values, RPE values, week numbers, athlete IDs, and training phases.\n\n"
            "Return ONLY a JSON object — no markdown, no explanation:\n"
            '{"athletes": {"athlete_XXXXX": "key facts discussed about this athlete"}, '
            '"open_questions": ["any unresolved questions the user still has"], '
            '"theme": "one sentence on the overall conversation topic"}\n\n'
            f"{transcript}"
        )
        cfg = {"temperature": 0.1, "max_output_tokens": 400}

        def _parse(text: str) -> str:
            raw = text.strip()
            try:
                _json.loads(raw)  # validate
            except Exception:
                pass
            return raw

        if self.gemini is not None:
            try:
                from google.genai import types
                resp = self.gemini.models.generate_content(
                    model    = self.gemini_model,
                    contents = prompt,
                    config   = types.GenerateContentConfig(**cfg),
                )
                return _parse(resp.text.strip() if hasattr(resp, 'text') else str(resp))
            except Exception:
                pass

        try:
            from google import genai
            from google.genai import types
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return self._fallback_summary(messages)
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model    = self.gemini_model,
                contents = prompt,
                config   = types.GenerateContentConfig(**cfg),
            )
            return _parse(resp.text.strip() if hasattr(resp, 'text') else str(resp))
        except Exception:
            return self._fallback_summary(messages)

    @staticmethod
    def _fallback_summary(messages: list[Message]) -> str:
        import json as _json
        user_msgs = [m.content for m in messages if m.role == 'user']
        theme = f"Earlier the user asked about: {'; '.join(user_msgs[:3])}." if user_msgs else ""
        return _json.dumps({"athletes": {}, "open_questions": [], "theme": theme})


# ── Session registry ──────────────────────────────────────────────────────────

_sessions:    dict[str, ConversationSummaryBufferMemory] = {}
_last_access: dict[str, float]                           = {}


def _evict_stale() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    stale  = [sid for sid, ts in _last_access.items() if ts < cutoff]
    for sid in stale:
        _sessions.pop(sid, None)
        _last_access.pop(sid, None)


def get_or_create_memory(
    session_id: str,
    gemini=None,
    k: int = 8,
    max_token_budget: int = 2000,
) -> ConversationSummaryBufferMemory:

    _evict_stale()

    if session_id not in _sessions:
        # Try to restore from persistent store (Redis or in-memory fallback)
        try:
            from app.session_store import get_store
            data = get_store().get(session_id)
            if data:
                mem = ConversationSummaryBufferMemory.from_dict(data, gemini=gemini)
            else:
                mem = ConversationSummaryBufferMemory(
                    session_id=session_id, gemini=gemini,
                    k=k, max_token_budget=max_token_budget,
                )
        except Exception:
            mem = ConversationSummaryBufferMemory(
                session_id=session_id, gemini=gemini,
                k=k, max_token_budget=max_token_budget,
            )
        _sessions[session_id] = mem
    else:
        if gemini is not None:
            _sessions[session_id].gemini = gemini

    _last_access[session_id] = time.time()
    return _sessions[session_id]


def clear_memory(session_id: str) -> None:
    _sessions.pop(session_id, None)
    _last_access.pop(session_id, None)
    try:
        from app.session_store import get_store
        get_store().delete(session_id)
    except Exception:
        pass


def active_sessions() -> list[str]:
    return list(_sessions.keys())
