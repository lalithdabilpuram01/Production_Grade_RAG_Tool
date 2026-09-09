import json
import re
from typing import Any, Dict, List

from groq_models import GRADER, build_chat_model
from llm_cache import cached_chat


# One fixed instruction block per grader. Everything question-specific goes in
# the user message so both cache layers can match on the prefix.
IS_RETRIEVE_SYSTEM_PROMPT = (
    "Decide whether the question needs retrieval from the user's provided "
    "documents or can be answered without the corpus. Return only JSON: "
    '{"needs_retrieval": true, "confidence": 0.0, "reason": "..."}.'
)

IS_RELEVANT_SYSTEM_PROMPT = (
    "Grade whether the retrieved context is relevant to the question. Return "
    "only JSON: "
    '{"is_relevant": true, "confidence": 0.0, "reason": "...", "rewrite_query": "..."}.'
)

IS_SUPPORTIVE_SYSTEM_PROMPT = (
    "Grade whether the answer is fully supported by the retrieved context. "
    "Return only JSON: "
    '{"is_supported": true, "confidence": 0.0, "reason": "..."}.'
)


class SelfRAGGrader:
    # 500 tokens rather than 300: reasoning models spend part of the completion
    # budget before the JSON verdict, and a truncated verdict fails to parse.
    def __init__(self, max_tokens: int = 500):
        self.llm = build_chat_model(GRADER, temperature=0.0, max_tokens=max_tokens)

    def is_retrieve(self, query: str) -> Dict[str, Any]:
        return self._grade(
            IS_RETRIEVE_SYSTEM_PROMPT,
            f"Question: {query}",
            "is-retrieve",
            {"needs_retrieval": True, "confidence": 0.5, "reason": "default"},
        )

    def is_relevant(self, query: str, docs: List[Any]) -> Dict[str, Any]:
        context = _format_docs(docs, max_chars=3500)
        return self._grade(
            IS_RELEVANT_SYSTEM_PROMPT,
            f"Question: {query}\n\nRetrieved context:\n{context}",
            "is-relevant",
            {
                "is_relevant": bool(docs),
                "confidence": 0.5,
                "reason": "default",
                "rewrite_query": query,
            },
        )

    def is_supportive(self, query: str, answer: str, docs: List[Any]) -> Dict[str, Any]:
        context = _format_docs(docs, max_chars=5000)
        return self._grade(
            IS_SUPPORTIVE_SYSTEM_PROMPT,
            f"Question: {query}\n\nAnswer:\n{answer}\n\nRetrieved context:\n{context}",
            "is-supportive",
            {"is_supported": True, "confidence": 0.5, "reason": "default"},
        )

    def _grade(
        self,
        system_prompt: str,
        user_prompt: str,
        tag: str,
        default: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            raw = cached_chat(self.llm, system_prompt, user_prompt, tag=tag)
            parsed = _load_json_object(raw)
            return {**default, **parsed}
        except Exception as exc:
            # Default to permissive verdicts so a grader formatting issue does not break QA.
            return {**default, "reason": f"grader fallback: {exc}"}


def _format_docs(docs: List[Any], max_chars: int) -> str:
    blocks = []
    total = 0
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        text = doc.page_content.strip()
        block = f"[{idx}] Source: {source}\n{text}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _load_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Groq models can occasionally return JSON with a short preface.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
