"""Prompt caching for every Groq call in the pipeline.

Two layers work together:

1. Server-side prefix caching. Groq reuses the KV cache of a request whose
   prompt shares a prefix with an earlier one, so every helper here sends a
   fixed system message first and appends the volatile parts (context, current
   question) last. `cached_prompt_tokens` in the stats below is what Groq
   reports back as actually reused.
2. Local exact-match caching. Identical (model, system, user) triples skip the
   network entirely. Repeated questions, re-runs after a UI rerun, and the
   grader/HyDE calls in an evaluation sweep all hit this layer.
"""

import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage


PROMPT_CACHE_ENABLED = os.getenv("PROMPT_CACHE_ENABLED", "true").lower() == "true"
PROMPT_CACHE_MAX_ENTRIES = int(os.getenv("PROMPT_CACHE_MAX_ENTRIES", "256"))
PROMPT_CACHE_TTL_SECONDS = int(os.getenv("PROMPT_CACHE_TTL_SECONDS", "3600"))

_lock = threading.Lock()
_entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_stats: Dict[str, int] = {
    "local_hits": 0,
    "local_misses": 0,
    "llm_calls": 0,
    "prompt_tokens": 0,
    "cached_prompt_tokens": 0,
    "completion_tokens": 0,
}


def cached_chat(llm, system_prompt: str, user_prompt: str, tag: str = "") -> str:
    """Invoke a chat model with a cache-friendly message layout.

    `system_prompt` must stay byte-identical across calls of the same kind so
    both cache layers can match on it.
    """
    key = _cache_key(llm, system_prompt, user_prompt, tag)
    cached = _read(key)
    if cached is not None:
        return cached

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    text = getattr(response, "content", str(response)).strip()
    _record_usage(response)
    _write(key, text)
    return text


def cache_snapshot() -> Dict[str, Any]:
    with _lock:
        stats = dict(_stats)
        stats["entries"] = len(_entries)

    lookups = stats["local_hits"] + stats["local_misses"]
    stats["local_hit_rate"] = round(stats["local_hits"] / lookups, 3) if lookups else 0.0
    prompt_tokens = stats["prompt_tokens"]
    stats["prefix_hit_rate"] = (
        round(stats["cached_prompt_tokens"] / prompt_tokens, 3) if prompt_tokens else 0.0
    )
    stats["enabled"] = PROMPT_CACHE_ENABLED
    return stats


def clear_cache() -> None:
    with _lock:
        _entries.clear()
        for name in _stats:
            _stats[name] = 0


def _cache_key(llm, system_prompt: str, user_prompt: str, tag: str) -> str:
    model = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    temperature = getattr(llm, "temperature", "")
    payload = "\x1f".join([str(model), str(temperature), tag, system_prompt, user_prompt])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(key: str) -> Optional[str]:
    if not PROMPT_CACHE_ENABLED:
        return None

    with _lock:
        entry = _entries.get(key)
        if entry and time.time() - entry["stored_at"] <= PROMPT_CACHE_TTL_SECONDS:
            _entries.move_to_end(key)
            _stats["local_hits"] += 1
            return entry["text"]
        if entry:
            del _entries[key]
        _stats["local_misses"] += 1
        return None


def _write(key: str, text: str) -> None:
    if not PROMPT_CACHE_ENABLED:
        return

    with _lock:
        _entries[key] = {"text": text, "stored_at": time.time()}
        _entries.move_to_end(key)
        while len(_entries) > PROMPT_CACHE_MAX_ENTRIES:
            _entries.popitem(last=False)


def _record_usage(response) -> None:
    metadata = getattr(response, "response_metadata", None) or {}
    usage = metadata.get("token_usage") or getattr(response, "usage_metadata", None) or {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    details = usage.get("prompt_tokens_details") or {}
    if hasattr(details, "get"):
        cached_tokens = details.get("cached_tokens") or 0
    else:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

    with _lock:
        _stats["llm_calls"] += 1
        _stats["prompt_tokens"] += int(prompt_tokens or 0)
        _stats["completion_tokens"] += int(completion_tokens or 0)
        _stats["cached_prompt_tokens"] += int(cached_tokens or 0)
