import json
import os
import re
from typing import Any, Dict, List

from langchain_groq import ChatGroq


GRADER_MODEL = os.getenv("GROQ_GRADER_MODEL", "llama-3.1-8b-instant")


class SelfRAGGrader:
    def __init__(self, model: str = GRADER_MODEL):
        self.llm = ChatGroq(model=model, temperature=0.0, max_tokens=300)

    def is_retrieve(self, query: str) -> Dict[str, Any]:
        prompt = (
            "Decide whether this question needs retrieval from provided real "
            "documents or can be answered without the corpus. Return only JSON: "
            '{"needs_retrieval": true, "confidence": 0.0, "reason": "..."}.\n\n'
            f"Question: {query}"
        )
        return self._grade(prompt, {"needs_retrieval": True, "confidence": 0.5, "reason": "default"})

    def is_relevant(self, query: str, docs: List[Any]) -> Dict[str, Any]:
        context = _format_docs(docs, max_chars=3500)
        prompt = (
            "Grade whether the retrieved context is relevant to the question. "
            "Return only JSON: "
            '{"is_relevant": true, "confidence": 0.0, "reason": "...", "rewrite_query": "..."}.\n\n'
            f"Question: {query}\n\nRetrieved context:\n{context}"
        )
        return self._grade(
            prompt,
            {
                "is_relevant": bool(docs),
                "confidence": 0.5,
                "reason": "default",
                "rewrite_query": query,
            },
        )

    def is_supportive(self, query: str, answer: str, docs: List[Any]) -> Dict[str, Any]:
        context = _format_docs(docs, max_chars=5000)
        prompt = (
            "Grade whether the answer is fully supported by the retrieved "
            "context. Return only JSON: "
            '{"is_supported": true, "confidence": 0.0, "reason": "..."}.\n\n'
            f"Question: {query}\n\nAnswer:\n{answer}\n\nRetrieved context:\n{context}"
        )
        return self._grade(
            prompt,
            {"is_supported": True, "confidence": 0.5, "reason": "default"},
        )

    def _grade(self, prompt: str, default: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", str(response)).strip()
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
