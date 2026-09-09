"""Semantic router that keeps conversational turns out of the RAG pipeline.

"hi", "thanks, that helped", and "what can you do?" have no answer in the
user's documents. Sending them through retrieval wastes an embedding search
and several LLM calls, and worse, it makes the model answer social messages
out of unrelated document chunks.

Routing is done by embedding similarity, not by an LLM call: the incoming
message is embedded once with the same model that powers retrieval and
compared against labelled exemplar utterances. `document_qa` is a route like
any other, so a message is only treated as conversational when it beats the
document-question exemplars by a margin, rather than merely clearing a fixed
threshold.
"""

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm_cache import cached_chat


logger = logging.getLogger(__name__)

ROUTER_ENABLED = os.getenv("SEMANTIC_ROUTER_ENABLED", "true").lower() == "true"
# Tuned on a held-out set of 30 conversational and 30 document questions with
# all-MiniLM-L6-v2: both are the mid-point of the band that misclassified
# nothing in either direction, so neither sits on an accuracy cliff.
ROUTER_THRESHOLD = float(os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.40"))
ROUTER_MARGIN = float(os.getenv("SEMANTIC_ROUTER_MARGIN", "0.15"))
# Weight of the single best exemplar against the mean of the top matches.
ROUTER_TOP_WEIGHT = float(os.getenv("SEMANTIC_ROUTER_TOP_WEIGHT", "0.6"))
# Social messages are short. A long message that merely opens with a greeting
# is a document question and must not be short-circuited.
ROUTER_MAX_WORDS = int(os.getenv("SEMANTIC_ROUTER_MAX_WORDS", "14"))
ROUTER_TOP_EXEMPLARS = int(os.getenv("SEMANTIC_ROUTER_TOP_EXEMPLARS", "2"))

DOCUMENT_QA = "document_qa"


@dataclass(frozen=True)
class Route:
    name: str
    description: str
    skips_retrieval: bool
    exemplars: Sequence[str]
    # Fixed per route so both prompt-cache layers can match on the prefix.
    reply_instruction: str = ""
    fallback_reply: str = ""


@dataclass
class RouteDecision:
    route: str
    skips_retrieval: bool
    score: float
    margin: float
    scores: Dict[str, float] = field(default_factory=dict)
    method: str = "embedding"
    reason: str = ""

    def as_trace(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "skips_retrieval": self.skips_retrieval,
            "score": round(self.score, 3),
            "margin": round(self.margin, 3),
            "method": self.method,
            "reason": self.reason,
            "scores": {name: round(value, 3) for name, value in sorted(self.scores.items())},
        }


ROUTES: List[Route] = [
    Route(
        name="greeting",
        description="Opening pleasantries with no question attached.",
        skips_retrieval=True,
        exemplars=[
            "hi",
            "hello",
            "hey there",
            "good morning",
            "good evening",
            "hi, how's it going",
            "hello again",
            "yo",
            "greetings",
            "hey, are you there",
        ],
        reply_instruction=(
            "Greet the user in one short friendly sentence and invite them to "
            "ask about their loaded documents. Do not mention retrieval, "
            "context, or that you skipped a search."
        ),
        fallback_reply="Hello. Ask me anything about the documents you've loaded.",
    ),
    Route(
        name="gratitude",
        description="Thanks or praise for a previous answer.",
        skips_retrieval=True,
        exemplars=[
            "thanks",
            "thank you",
            "thanks a lot, that helps",
            "thank you so much",
            "perfect, thanks",
            "great, that's what I needed",
            "nice work",
            "awesome, appreciate it",
            "that was helpful",
            "cheers",
        ],
        reply_instruction=(
            "Acknowledge the thanks in one short sentence and offer to keep "
            "going. Do not restate the previous answer."
        ),
        fallback_reply="Glad that helped. Let me know what else you'd like to look up.",
    ),
    Route(
        name="farewell",
        description="Ending the conversation.",
        skips_retrieval=True,
        exemplars=[
            "bye",
            "goodbye",
            "see you later",
            "that's all for now",
            "I'm done, thanks",
            "talk to you tomorrow",
            "catch you later",
            "we can stop here",
            "no more questions",
            "that will be all",
        ],
        reply_instruction="Say goodbye warmly in one short sentence.",
        fallback_reply="Goodbye. Come back whenever you need more from your documents.",
    ),
    Route(
        name="smalltalk",
        description="Social chatter directed at the assistant.",
        skips_retrieval=True,
        exemplars=[
            "how are you",
            "how's your day going",
            "what's up",
            "are you a real person",
            "do you ever get tired",
            "you're pretty smart",
            "haha that's funny",
            "just testing you",
            "are you still there",
            "tell me a joke",
        ],
        reply_instruction=(
            "Reply to the small talk in one or two friendly sentences, stay "
            "professional, and steer back to the user's documents."
        ),
        fallback_reply="I'm doing well, thanks. What would you like to know from your documents?",
    ),
    Route(
        name="capability",
        description="Questions about the assistant itself rather than the documents.",
        skips_retrieval=True,
        exemplars=[
            "what can you do",
            "who are you",
            "what are you",
            "how do you work",
            "what is this tool",
            "what are your capabilities",
            "can you help me",
            "how should I use you",
            "what kind of questions can I ask",
            "what documents do you have loaded",
            "which sources are indexed",
            "what files can you see",
        ],
        reply_instruction=(
            "Describe what this assistant can do, using the assistant profile "
            "and the indexed sources listed in the message. Keep it under five "
            "short sentences and do not invent features or documents."
        ),
        fallback_reply=(
            "I answer questions from the documents you load, with citations back "
            "to the source page. Load a PDF or URL in the sidebar and ask away."
        ),
    ),
    Route(
        name=DOCUMENT_QA,
        description="A question to be answered from the indexed documents.",
        skips_retrieval=False,
        exemplars=[
            "what is the recommended tire pressure",
            "summarize the maintenance schedule",
            "how do I replace the cabin air filter",
            "what does the warranty cover",
            "list the safety warnings in the document",
            "compare the two configurations described in the manual",
            "what torque setting is specified for the wheel bolts",
            "when is the first service due",
            "explain the section about the cooling system",
            "what are the key findings of the report",
            "how many pages mention the battery",
            "which chapter covers troubleshooting",
            "what did the authors conclude",
            "give me the specification table",
            "hi, can you tell me what the service interval is",
            "thanks, and what about the rear axle",
        ],
    ),
]

_ROUTES_BY_NAME = {route.name: route for route in ROUTES}

# Highest-frequency single utterances, matched exactly so one-word messages
# never depend on an embedding threshold.
_EXACT_ROUTES: Dict[str, str] = {
    "hi": "greeting",
    "hii": "greeting",
    "hey": "greeting",
    "hello": "greeting",
    "yo": "greeting",
    "hi there": "greeting",
    "good morning": "greeting",
    "good afternoon": "greeting",
    "good evening": "greeting",
    "thanks": "gratitude",
    "thank you": "gratitude",
    "thx": "gratitude",
    "ty": "gratitude",
    "cheers": "gratitude",
    "bye": "farewell",
    "goodbye": "farewell",
    "see ya": "farewell",
    "ok bye": "farewell",
}

CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are the conversational voice of a document research assistant. The "
    "user's message is small talk or a question about the assistant itself, so "
    "no document was retrieved. Reply briefly and naturally. Never invent "
    "facts about the user's documents, never cite sources, and never mention "
    "retrieval, routing, or context."
)

_lock = threading.Lock()
_embedder: Optional[Any] = None
_exemplar_vectors: Optional[Any] = None
_exemplar_routes: List[str] = []


def configure_router(embedder: Any) -> None:
    """Attach the embedding model used for routing.

    Pass the same instance the retriever uses; re-encoding the exemplars is
    only needed when the model itself changes.
    """
    global _embedder, _exemplar_vectors, _exemplar_routes
    with _lock:
        if _embedder is embedder:
            return
        _embedder = embedder
        _exemplar_vectors = None
        _exemplar_routes = []


def reset_router() -> None:
    global _embedder, _exemplar_vectors, _exemplar_routes
    with _lock:
        _embedder = None
        _exemplar_vectors = None
        _exemplar_routes = []


def route_query(query: str) -> RouteDecision:
    """Classify a message as conversational or as a document question."""
    text = (query or "").strip()
    if not text:
        return RouteDecision(DOCUMENT_QA, False, 0.0, 0.0, reason="empty message", method="guard")

    if not ROUTER_ENABLED:
        return RouteDecision(DOCUMENT_QA, False, 0.0, 0.0, reason="router disabled", method="guard")

    normalized = _normalize(text)
    exact_route = _EXACT_ROUTES.get(normalized)
    if exact_route:
        return RouteDecision(
            exact_route, True, 1.0, 1.0, reason="exact conversational phrase", method="exact"
        )

    word_count = len(normalized.split())
    if word_count > ROUTER_MAX_WORDS:
        return RouteDecision(
            DOCUMENT_QA,
            False,
            0.0,
            0.0,
            reason=f"message is {word_count} words, above the conversational limit",
            method="guard",
        )

    try:
        scores = _score_routes(text)
    except Exception as exc:
        # Routing is an optimisation. If it fails, answer from the documents.
        logger.warning("Semantic routing failed (%s); defaulting to document retrieval.", exc)
        return RouteDecision(DOCUMENT_QA, False, 0.0, 0.0, reason=f"routing failed: {exc}", method="fallback")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ranked[0]
    document_score = scores.get(DOCUMENT_QA, 0.0)
    margin = best_score - document_score

    if best_name == DOCUMENT_QA:
        return RouteDecision(
            DOCUMENT_QA, False, best_score, 0.0, scores, reason="closest to document questions"
        )
    if best_score < ROUTER_THRESHOLD:
        return RouteDecision(
            DOCUMENT_QA,
            False,
            best_score,
            margin,
            scores,
            reason=f"best conversational score {best_score:.2f} below threshold {ROUTER_THRESHOLD:.2f}",
        )
    if margin < ROUTER_MARGIN:
        return RouteDecision(
            DOCUMENT_QA,
            False,
            best_score,
            margin,
            scores,
            reason=f"only {margin:.2f} above the document-question score",
        )

    return RouteDecision(
        best_name,
        True,
        best_score,
        margin,
        scores,
        reason=f"matched '{best_name}' exemplars",
    )


def answer_conversationally(
    llm: Any,
    query: str,
    decision: RouteDecision,
    history: str = "",
    profile: str = "",
) -> str:
    """Produce a reply for a conversational route without touching retrieval."""
    route = _ROUTES_BY_NAME.get(decision.route)
    if route is None or not route.skips_retrieval:
        raise ValueError(f"Route '{decision.route}' is not a conversational route")

    sections = [f"Route: {route.name}", f"Instruction: {route.reply_instruction}"]
    if profile:
        sections.append(f"Assistant profile:\n{profile}")
    if history:
        sections.append(f"Conversation so far:\n{history}")
    sections.append(f"User message: {query.strip()}")
    sections.append("Reply:")

    try:
        reply = cached_chat(
            llm,
            CONVERSATIONAL_SYSTEM_PROMPT,
            "\n\n".join(sections),
            tag=f"router-{route.name}",
        )
    except Exception as exc:
        logger.warning("Conversational reply failed (%s); using the static reply.", exc)
        return route.fallback_reply

    return reply.strip() or route.fallback_reply


def route_names() -> List[str]:
    return [route.name for route in ROUTES]


def _score_routes(text: str) -> Dict[str, float]:
    import numpy as np

    vectors, route_labels = _exemplar_matrix()
    query_vector = np.asarray(_require_embedder().embed_query(text), dtype="float32")
    query_vector /= (np.linalg.norm(query_vector) or 1.0)

    similarities = vectors @ query_vector
    scores: Dict[str, float] = {}
    for route in ROUTES:
        mask = route_labels == route.name
        if not mask.any():
            continue
        route_similarities = np.sort(similarities[mask])[::-1]
        # Blend the best match with the mean of the top few, so a route needs
        # more than one lucky exemplar but is not penalised for having many.
        top = route_similarities[: max(1, ROUTER_TOP_EXEMPLARS)]
        scores[route.name] = float(
            ROUTER_TOP_WEIGHT * route_similarities[0] + (1.0 - ROUTER_TOP_WEIGHT) * top.mean()
        )
    return scores


def _exemplar_matrix():
    import numpy as np

    global _exemplar_vectors, _exemplar_routes
    with _lock:
        if _exemplar_vectors is not None:
            return _exemplar_vectors, np.asarray(_exemplar_routes)

    texts: List[str] = []
    labels: List[str] = []
    for route in ROUTES:
        for exemplar in route.exemplars:
            texts.append(exemplar)
            labels.append(route.name)

    raw = np.asarray(_require_embedder().embed_documents(texts), dtype="float32")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = raw / norms

    with _lock:
        _exemplar_vectors = normalized
        _exemplar_routes = labels
    return normalized, np.asarray(labels)


def _require_embedder():
    with _lock:
        embedder = _embedder
    if embedder is None:
        raise RuntimeError("Semantic router has no embedding model; call configure_router() first.")
    return embedder


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s']+", " ", text.lower()).strip()
