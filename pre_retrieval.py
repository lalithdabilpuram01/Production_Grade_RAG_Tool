import json
import os
import re
from typing import Any, Dict, List

from langchain_groq import ChatGroq

from groq_models import HYDE, build_chat_model
from llm_cache import cached_chat


ASSISTANT_ROLE = os.getenv("ASSISTANT_ROLE", "domain-neutral research assistant")

# Fixed instruction blocks, kept out of the per-question text so both cache
# layers can match on them. See llm_cache.py.
HYDE_SYSTEM_PROMPT = (
    f"Write a concise, factual, ideal answer as a {ASSISTANT_ROLE}. Include "
    "likely terms, entities, dates, metrics, and domain-specific phrasing that "
    "would help retrieve relevant source text, but do not invent citations. "
    "Return only the hypothetical answer."
)

DECOMPOSE_SYSTEM_PROMPT = (
    "Split the user question into independent search sub-queries for a RAG "
    "retriever. Return only JSON in this shape: "
    '{"sub_queries": ["..."], "reason": "..."}.'
)

_hyde_llm = None
_decompose_llm = None


def _get_llm(temperature: float, max_tokens: int, slot: str) -> ChatGroq:
    # Reused across calls so the underlying HTTP client and its connection pool
    # survive Streamlit reruns.
    global _hyde_llm, _decompose_llm
    if slot == "hyde":
        if _hyde_llm is None:
            _hyde_llm = build_chat_model(HYDE, temperature=temperature, max_tokens=max_tokens)
        return _hyde_llm

    if _decompose_llm is None:
        _decompose_llm = build_chat_model(HYDE, temperature=temperature, max_tokens=max_tokens)
    return _decompose_llm


def reset_pre_retrieval_clients() -> None:
    global _hyde_llm, _decompose_llm
    _hyde_llm = None
    _decompose_llm = None


def generate_hypothetical_answer(query: str) -> Dict[str, Any]:
    hyde_text = cached_chat(
        _get_llm(0.2, 500, "hyde"),
        HYDE_SYSTEM_PROMPT,
        f"Question: {query}\n\nHypothetical answer:",
        tag="hyde",
    )
    return {"query_for_embedding": hyde_text or query, "hyde_answer": hyde_text}


def decompose_query(query: str) -> Dict[str, Any]:
    lowered = query.lower()
    has_multi_signal = any(
        signal in lowered
        for signal in [" compare ", " versus ", " vs ", " and ", " or ", ";", "?", "difference between"]
    )

    if not has_multi_signal:
        return {"sub_queries": [query], "used_llm": False, "reason": "single-part query"}

    try:
        raw = cached_chat(
            _get_llm(0.0, 550, "decompose"),
            DECOMPOSE_SYSTEM_PROMPT,
            f"Question: {query}",
            tag="decompose",
        )
        payload = _load_json_object(raw)
        sub_queries = [
            item.strip()
            for item in payload.get("sub_queries", [])
            if isinstance(item, str) and item.strip()
        ]
        if sub_queries:
            return {
                "sub_queries": _dedupe_preserve_order(sub_queries),
                "used_llm": True,
                "reason": payload.get("reason", "multi-part query"),
            }
    except Exception as exc:
        return {
            "sub_queries": _fallback_split(query),
            "used_llm": False,
            "reason": f"LLM decomposition failed: {exc}",
        }

    return {"sub_queries": _fallback_split(query), "used_llm": False, "reason": "fallback split"}


def _fallback_split(query: str) -> List[str]:
    parts = re.split(r"\s+(?:and|or|versus|vs\.?)\s+|;|\?", query, flags=re.IGNORECASE)
    cleaned = [part.strip(" ,.") for part in parts if part.strip(" ,.")]
    return _dedupe_preserve_order(cleaned) or [query]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _load_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Grader/decomposer prompts demand JSON, but small models may still wrap it in prose.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
