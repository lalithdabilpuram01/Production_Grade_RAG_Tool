import json
import os
import re
from typing import Any, Dict, List

from langchain_groq import ChatGroq


HYDE_MODEL = os.getenv("GROQ_HYDE_MODEL", "llama-3.1-8b-instant")
ASSISTANT_ROLE = os.getenv("ASSISTANT_ROLE", "domain-neutral research assistant")


def generate_hypothetical_answer(query: str) -> Dict[str, Any]:
    llm = ChatGroq(model=HYDE_MODEL, temperature=0.2, max_tokens=300)
    prompt = (
        f"Write a concise, factual, ideal answer as a {ASSISTANT_ROLE}. "
        "Include likely terms, entities, dates, metrics, and domain-specific "
        "phrasing that would help retrieve relevant source text, but do not "
        "invent citations.\n\n"
        f"Question: {query}\n\nHypothetical answer:"
    )
    response = llm.invoke(prompt)
    hyde_text = getattr(response, "content", str(response)).strip()
    return {"query_for_embedding": hyde_text or query, "hyde_answer": hyde_text}


def decompose_query(query: str) -> Dict[str, Any]:
    lowered = query.lower()
    has_multi_signal = any(
        signal in lowered
        for signal in [" compare ", " versus ", " vs ", " and ", " or ", ";", "?", "difference between"]
    )

    if not has_multi_signal:
        return {"sub_queries": [query], "used_llm": False, "reason": "single-part query"}

    llm = ChatGroq(model=HYDE_MODEL, temperature=0.0, max_tokens=350)
    prompt = (
        "Split the user question into independent search sub-queries for a RAG "
        "retriever. Return only JSON in this shape: "
        '{"sub_queries": ["..."], "reason": "..."}.\n\n'
        f"Question: {query}"
    )

    try:
        response = llm.invoke(prompt)
        raw = getattr(response, "content", str(response)).strip()
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
