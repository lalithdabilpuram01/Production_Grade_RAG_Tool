"""Conversation memory for the chat UI.

Keeps a rolling window of recent turns verbatim and rolls anything older into a
short running summary, so the prompt stays bounded no matter how long the chat
runs. It also rewrites follow-up questions ("what about the rear one?") into
standalone retrieval queries, which is what makes multi-turn RAG work.

The transcript is built by appending, so its prefix is stable from turn to
turn. That is deliberate: it lets Groq reuse the cached prefix of the previous
request. See llm_cache.py.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq

from groq_models import MEMORY, build_chat_model
from llm_cache import cached_chat


MEMORY_WINDOW_TURNS = int(os.getenv("MEMORY_WINDOW_TURNS", "6"))
MEMORY_ANSWER_CHARS = int(os.getenv("MEMORY_ANSWER_CHARS", "700"))
MEMORY_CONDENSE_QUESTIONS = os.getenv("MEMORY_CONDENSE_QUESTIONS", "true").lower() == "true"

CONDENSE_SYSTEM_PROMPT = (
    "You rewrite the latest user question into a single standalone search "
    "query for a document retriever. Resolve pronouns and implicit references "
    "using the conversation. Keep the user's own wording and entities wherever "
    "possible, add nothing that was not asked, and never answer the question. "
    "Return only the rewritten query on one line."
)

SUMMARY_SYSTEM_PROMPT = (
    "You maintain a running summary of a research chat between a user and a "
    "document assistant. Merge the existing summary with the new turns into at "
    "most six short bullet points. Preserve entities, numbers, document names, "
    "and any constraints the user stated. Return only the bullet points."
)


@dataclass
class Turn:
    question: str
    answer: str
    sources: str = ""


@dataclass
class ConversationMemory:
    window_turns: int = MEMORY_WINDOW_TURNS
    turns: List[Turn] = field(default_factory=list)
    summary: str = ""

    def add_turn(self, question: str, answer: str, sources: str = "") -> None:
        self.turns.append(Turn(question=question, answer=answer, sources=sources))
        self._roll_over_old_turns()

    def clear(self) -> None:
        self.turns = []
        self.summary = ""

    def is_empty(self) -> bool:
        return not self.turns and not self.summary

    def transcript(self) -> str:
        """Summary of older turns plus the verbatim recent window."""
        blocks = []
        if self.summary:
            blocks.append(f"Summary of earlier conversation:\n{self.summary}")

        for turn in self.turns:
            answer = turn.answer.strip()
            if len(answer) > MEMORY_ANSWER_CHARS:
                answer = answer[:MEMORY_ANSWER_CHARS].rstrip() + " [...]"
            blocks.append(f"User: {turn.question.strip()}\nAssistant: {answer}")

        return "\n\n".join(blocks)

    def condense_question(self, question: str) -> Dict[str, Any]:
        """Rewrite a follow-up into a standalone retrieval query."""
        if self.is_empty() or not MEMORY_CONDENSE_QUESTIONS:
            return {"search_query": question, "rewritten": False, "reason": "no conversation history"}

        try:
            rewritten = cached_chat(
                _memory_llm(),
                CONDENSE_SYSTEM_PROMPT,
                f"Conversation so far:\n{self.transcript()}\n\nLatest user question: {question}\n\nStandalone query:",
                tag="condense",
            )
        except Exception as exc:
            return {"search_query": question, "rewritten": False, "reason": f"condense failed: {exc}"}

        cleaned = rewritten.strip().strip('"').splitlines()[0].strip() if rewritten.strip() else ""
        if not cleaned:
            return {"search_query": question, "rewritten": False, "reason": "empty rewrite"}

        return {
            "search_query": cleaned,
            "rewritten": cleaned.lower() != question.strip().lower(),
            "reason": "rewritten from conversation history",
        }

    def _roll_over_old_turns(self) -> None:
        overflow = len(self.turns) - self.window_turns
        if overflow <= 0:
            return

        aging_out = self.turns[:overflow]
        self.turns = self.turns[overflow:]
        new_material = "\n\n".join(
            f"User: {turn.question.strip()}\nAssistant: {turn.answer.strip()[:MEMORY_ANSWER_CHARS]}"
            for turn in aging_out
        )

        try:
            self.summary = cached_chat(
                _memory_llm(),
                SUMMARY_SYSTEM_PROMPT,
                f"Existing summary:\n{self.summary or '(none)'}\n\nNew turns to fold in:\n{new_material}\n\nUpdated summary:",
                tag="summary",
            ).strip()
        except Exception:
            # Losing the summary is better than losing the turn, so fall back to
            # a plain concatenation the next prompt can still read.
            self.summary = (self.summary + "\n" + new_material).strip()[: MEMORY_ANSWER_CHARS * 4]


_llm: Optional[ChatGroq] = None


def _memory_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = build_chat_model(MEMORY, temperature=0.0, max_tokens=400)
    return _llm


def reset_memory_client() -> None:
    global _llm
    _llm = None
