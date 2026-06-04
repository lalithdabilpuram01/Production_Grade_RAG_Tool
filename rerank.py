import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple


RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def _load_cross_encoder():
    # Cross-encoders are slow to load, so reuse the model across Streamlit reruns.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def rerank_documents(query: str, docs: List[Any], top_n: int = 5) -> Tuple[List[Any], List[Dict[str, Any]]]:
    if not docs:
        return [], []

    model = _load_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    scored_docs = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
    selected = scored_docs[: max(1, top_n)]

    trace = []
    for rank, (doc, score) in enumerate(selected, start=1):
        trace.append(
            {
                "rank": rank,
                "score": float(score),
                "source": doc.metadata.get("source", "unknown"),
                "preview": doc.page_content[:240].replace("\n", " "),
            }
        )

    return [doc for doc, _ in selected], trace
