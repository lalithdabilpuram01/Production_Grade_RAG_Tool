"""Central resolution of Groq model ids and construction of chat clients.

Groq retires hosted model ids on a rolling basis, and a retired id fails at
request time with a 404 that surfaces to the user as a broken answer. Every
module therefore asks this registry for its model instead of hardcoding one:
the registry checks the id against the live catalogue for the active API key
and, when the configured id is gone, falls back to the best still-available
model for that role.

The catalogue lookup is fail-soft. If it cannot reach Groq (no key yet, no
network, a self-hosted gateway) the configured id is returned unvalidated, so
this module never becomes a new point of failure.
"""

import hashlib
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq
from langchain_groq import ChatGroq


logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when no usable chat model exists for the active API key."""


GENERATION = "generation"
HYDE = "hyde"
GRADER = "grader"
MEMORY = "memory"
ROUTER = "router"
JUDGE = "judge"

# Env var read for each role, and the id used when the var is unset.
ROLE_SETTINGS: Dict[str, Tuple[str, str]] = {
    GENERATION: ("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b"),
    HYDE: ("GROQ_HYDE_MODEL", "openai/gpt-oss-20b"),
    GRADER: ("GROQ_GRADER_MODEL", "openai/gpt-oss-20b"),
    MEMORY: ("GROQ_MEMORY_MODEL", "openai/gpt-oss-20b"),
    ROUTER: ("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b"),
    JUDGE: ("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b"),
}

# Ordered by preference. The first id present in the live catalogue wins, so
# retired ids can stay listed for accounts that still have them.
_LARGE_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
_FAST_MODELS = [
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]

ROLE_PREFERENCES: Dict[str, List[str]] = {
    GENERATION: _LARGE_MODELS,
    JUDGE: _LARGE_MODELS,
    HYDE: _FAST_MODELS,
    GRADER: _FAST_MODELS,
    MEMORY: _FAST_MODELS,
    ROUTER: _FAST_MODELS,
}

# Substrings identifying models that cannot serve a chat completion, or that
# are classifiers rather than assistants. Used only for the last-resort scan.
_NON_CHAT_MARKERS = (
    "whisper",
    "tts",
    "orpheus",
    "prompt-guard",
    "guard",
    "embed",
    "playai",
    "moderation",
)

# Reasoning models spend part of the completion budget on hidden reasoning
# tokens, which truncates short JSON replies from the graders. Only the
# gpt-oss family documents the low/medium/high scale, so the effort hint is
# restricted to it.
_REASONING_EFFORT_MARKERS = ("gpt-oss",)

CATALOGUE_TTL_SECONDS = int(os.getenv("GROQ_MODEL_CATALOGUE_TTL_SECONDS", "900"))
CATALOGUE_TIMEOUT_SECONDS = float(os.getenv("GROQ_MODEL_CATALOGUE_TIMEOUT", "10"))
GENERATION_REASONING_EFFORT = os.getenv("GROQ_GENERATION_REASONING_EFFORT", "medium")
HELPER_REASONING_EFFORT = os.getenv("GROQ_HELPER_REASONING_EFFORT", "low")

_ROLE_REASONING_EFFORT: Dict[str, str] = {
    GENERATION: GENERATION_REASONING_EFFORT,
    JUDGE: HELPER_REASONING_EFFORT,
    HYDE: HELPER_REASONING_EFFORT,
    GRADER: HELPER_REASONING_EFFORT,
    MEMORY: HELPER_REASONING_EFFORT,
    ROUTER: HELPER_REASONING_EFFORT,
}

_lock = threading.Lock()
_catalogue: Optional[Dict[str, Dict[str, Any]]] = None
_catalogue_key_fingerprint: Optional[str] = None
_catalogue_fetched_at: float = 0.0
_resolved: Dict[str, str] = {}
_resolution_notes: Dict[str, str] = {}


def configured_model(role: str) -> str:
    """The id requested for a role, before any availability check."""
    env_var, default = ROLE_SETTINGS[role]
    return (os.getenv(env_var) or default).strip()


def resolve_model(role: str) -> str:
    """Return a model id for `role` that the active API key can actually call.

    Falls back through `ROLE_PREFERENCES` when the configured id is missing
    from the catalogue, and returns the configured id unchanged when the
    catalogue cannot be read.
    """
    if role not in ROLE_SETTINGS:
        raise KeyError(f"Unknown model role: {role}")

    with _lock:
        cached = _resolved.get(role)
    if cached:
        return cached

    requested = configured_model(role)
    catalogue = _load_catalogue()

    if catalogue is None:
        # Offline or key-less: trust the configuration rather than block.
        return requested

    chosen, note = _select_from_catalogue(role, requested, catalogue)
    with _lock:
        _resolved[role] = chosen
        _resolution_notes[role] = note
    if chosen != requested:
        logger.warning("Groq model '%s' unavailable for role '%s'; using '%s'.", requested, role, chosen)
    return chosen


def build_chat_model(role: str, temperature: float, max_tokens: int, **kwargs: Any) -> ChatGroq:
    """Construct a ChatGroq client for `role` with role-appropriate defaults."""
    model = resolve_model(role)
    params: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    effort = _ROLE_REASONING_EFFORT.get(role)
    if effort and supports_reasoning_effort(model):
        params["reasoning_effort"] = effort
    params.update(kwargs)
    return ChatGroq(**params)


def supports_reasoning_effort(model: str) -> bool:
    return any(marker in model for marker in _REASONING_EFFORT_MARKERS)


def available_chat_models() -> List[str]:
    """Ids of chat-capable models for the active key, best context first."""
    catalogue = _load_catalogue()
    if catalogue is None:
        return []
    return [model_id for model_id, _ in _rank_chat_models(catalogue)]


def resolution_report() -> Dict[str, Dict[str, str]]:
    """Per-role view of what was requested and what will actually be called."""
    report = {}
    for role in ROLE_SETTINGS:
        requested = configured_model(role)
        with _lock:
            resolved = _resolved.get(role)
            note = _resolution_notes.get(role, "")
        report[role] = {
            "requested": requested,
            "resolved": resolved or requested,
            "status": note or ("unresolved" if resolved is None else ""),
        }
    return report


def reset_model_registry() -> None:
    """Forget the catalogue and every resolution so a new API key is re-checked."""
    global _catalogue, _catalogue_fetched_at, _catalogue_key_fingerprint
    with _lock:
        _catalogue = None
        _catalogue_fetched_at = 0.0
        _catalogue_key_fingerprint = None
        _resolved.clear()
        _resolution_notes.clear()


def _select_from_catalogue(
    role: str,
    requested: str,
    catalogue: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    if requested in catalogue:
        return requested, "configured"

    for candidate in ROLE_PREFERENCES.get(role, []):
        if candidate in catalogue:
            return candidate, f"'{requested}' unavailable; fell back to preferred model"

    ranked = _rank_chat_models(catalogue)
    if ranked:
        return ranked[0][0], f"'{requested}' unavailable; fell back to largest available chat model"

    raise ModelUnavailableError(
        f"No chat-capable Groq model is available for this API key (role '{role}', "
        f"requested '{requested}'). Models visible to the key: "
        f"{', '.join(sorted(catalogue)) or 'none'}."
    )


def _rank_chat_models(catalogue: Dict[str, Dict[str, Any]]) -> List[Tuple[str, int]]:
    chat_models = [
        (model_id, int(meta.get("context_window") or 0))
        for model_id, meta in catalogue.items()
        if not any(marker in model_id.lower() for marker in _NON_CHAT_MARKERS)
    ]
    return sorted(chat_models, key=lambda item: (-item[1], item[0]))


def _load_catalogue() -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch and cache the active key's model catalogue; None if unreadable."""
    global _catalogue, _catalogue_fetched_at, _catalogue_key_fingerprint

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    now = time.time()

    with _lock:
        fresh = (
            _catalogue is not None
            and _catalogue_key_fingerprint == fingerprint
            and now - _catalogue_fetched_at <= CATALOGUE_TTL_SECONDS
        )
        if fresh:
            return _catalogue

    try:
        listing = Groq(api_key=api_key, timeout=CATALOGUE_TIMEOUT_SECONDS).models.list()
    except Exception as exc:
        logger.warning("Could not read the Groq model catalogue (%s); using configured ids as-is.", exc)
        return None

    catalogue = {
        entry.id: {"context_window": getattr(entry, "context_window", 0)}
        for entry in getattr(listing, "data", []) or []
        if getattr(entry, "id", None) and getattr(entry, "active", True)
    }
    if not catalogue:
        return None

    with _lock:
        _catalogue = catalogue
        _catalogue_fetched_at = now
        if _catalogue_key_fingerprint != fingerprint:
            # A different key can expose a different catalogue, so drop prior picks.
            _resolved.clear()
            _resolution_notes.clear()
        _catalogue_key_fingerprint = fingerprint
        return catalogue
