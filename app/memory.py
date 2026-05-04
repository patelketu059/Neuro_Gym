from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal

from config.model_settings import GEMINI_AUX_MODEL


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
            gemini = None,
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
        

    def add_user_message(self, content: str) -> None:
        self.buffer.append(Message(role = 'user', content = content))
        self._maybe_summarise()

    def add_ai_message(self, content: str) -> None:
        self.buffer.append(Message(role = 'assistant', content = content))
        self._maybe_summarise()

    def get_history(self) -> list[dict]:
        msgs: list[dict] = []
        if self.summary:
            msgs.append({
                "role": "user",
                "content": f"[Summary of earlier conversation]\n{self.summary}"
            })

        msgs.extend(
            {"role": m.role, "content": m.content}
            for m in self.buffer[-self.k:]
        )
        return msgs
    
    def clear(self) -> None:
        self.buffer = []
        self.summary = ""

    def buffer_tokens(self) -> int:
        return sum(m.tokens for m in self.buffer)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def _maybe_summarise(self) -> None:
        if self.buffer_tokens() <= self.max_token_budget:
            return
        
        half = max(1, len(self.buffer) // 2)
        to_compress = self.buffer[:half]
        self.buffer = self.buffer[half:]

        new_summary = self._call_gemini(to_compress)
        self.summary = (
            f"{self.summary}\n\n{new_summary}" if self.summary else new_summary
        )

    
    def _call_gemini(self, messages: list[Message]) -> str:
        transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        prompt = (
            "Summarise the following powerlifting coaching conversation in 2-3 sentences. "
            "Preserve specific athlete IDs, lift numbers (kg), week numbers, RPE values, "
            "and training phases. Try to preserve as much information as possible.\n\n"
            f"{transcript}"
        )

        cfg = {"temperature": 0.1, 'max_output_tokens': 200}

        if self.gemini is not None:
            try:
                from google.genai import types
                return self.gemini.models.generate_content(
                    model = self.gemini_model,
                    contents = prompt,
                    config = types.GenerateContentConfig(**cfg)).text.strip()
            
            except Exception:
                pass

        try:
            from google import genai
            from google.genai import types
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return self._fallback_summary(messages)
            client = genai.Client(api_key = api_key)
            
            return client.models.generate_content(
                model   = self.gemini_model,
                contents= prompt,
                config  = types.GenerateContentConfig(**cfg),
            ).text.strip()
        except Exception:
            return self._fallback_summary(messages)
        
    @staticmethod
    def _fallback_summary(messages: list[Message]) -> str:
        user_msgs = [m.content for m in messages if m.role == 'user']
        if not user_msgs: return ""
        return f"Earlier the user asked about: {'; '.join(user_msgs[:3])}."
    


_sessions: dict[str, ConversationSummaryBufferMemory] = {}

def get_or_create_memory(
    session_id: str,
    gemini = None,
    k: int = 8,
    max_token_budget: int = 2000,
) -> ConversationSummaryBufferMemory:

    if session_id not in _sessions:
        _sessions[session_id] = ConversationSummaryBufferMemory(
            session_id = session_id,
            gemini = gemini,
            k = k,
            max_token_budget = max_token_budget,
        )
    elif gemini is not None:
        _sessions[session_id].gemini = gemini
    return _sessions[session_id]


def clear_memory(session_id: str) -> None:
    _sessions.pop(session_id, None)


def active_sessions() -> list[str]:
    return list(_sessions.keys())