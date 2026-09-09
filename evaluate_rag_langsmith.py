import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Union
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langsmith import Client

from groq_models import JUDGE, build_chat_model, resolve_model
from rag import AdvancedRAGConfig, generate_answer, process_sources


DEFAULT_DATASET_NAME = "model-3-owners-manual-rag-testset"
DEFAULT_EXPERIMENT_PREFIX = "owners-manual-rag"
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


load_dotenv()
judge_llm = None


def load_examples(testset_path: Path, dataset_name: str) -> List[Dict[str, Any]]:
    with testset_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    examples = []
    for row in rows:
        question = row.get("question")
        answer = row.get("answer")
        if not question or answer is None:
            raise ValueError(f"Invalid testset row; expected question and answer: {row}")

        examples.append(
            {
                "id": str(uuid5(NAMESPACE_URL, f"{dataset_name}:{question}")),
                "inputs": {"question": question},
                "outputs": {"answer": answer, "page": row.get("page")},
                "metadata": {"page": row.get("page")},
            }
        )

    return examples


def ensure_dataset(client: Client, dataset_name: str, examples: List[Dict[str, Any]]) -> str:
    """Create or update a LangSmith dataset with the local examples.

    Stable example IDs let repeated runs update the same rows instead of creating
    duplicate examples.
    """
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="RAG evaluation set for Owners_Manual.pdf.",
        )

    try:
        client.create_examples(dataset_id=dataset.id, examples=examples)
    except Exception as exc:
        if "Conflict" not in exc.__class__.__name__ and "already exists" not in str(exc):
            raise
        updates = [
            {
                "id": example["id"],
                "inputs": example["inputs"],
                "outputs": example["outputs"],
                "metadata": example["metadata"],
            }
            for example in examples
        ]
        client.update_examples(dataset_id=dataset.id, updates=updates)
    return dataset_name


def build_rag_config(args: argparse.Namespace) -> AdvancedRAGConfig:
    return AdvancedRAGConfig(
        use_hyde=args.hyde,
        use_decomposition=args.decomposition,
        use_hybrid=args.hybrid,
        use_parent_child=not args.no_parent_child,
        use_rerank=args.rerank,
        use_self_rag=args.self_rag,
        top_k_retrieve=args.top_k,
        top_n_rerank=args.top_n,
        dense_weight=args.dense_weight,
        sparse_weight=round(1.0 - args.dense_weight, 2),
    )


def make_target(config: AdvancedRAGConfig):
    def rag_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        answer, sources, trace = generate_answer(
            inputs["question"],
            config=config,
            return_trace=True,
        )
        latency_seconds = time.perf_counter() - started_at
        cited_pages = extract_cited_pages(sources)
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_context": trace.get("retrieved_context", []),
            "cited_pages": cited_pages,
            "citation_count": len(cited_pages),
            "source_count": count_sources(sources),
            "answer_word_count": len(answer.split()),
            "answer_char_count": len(answer),
            "latency_seconds": latency_seconds,
            "trace_steps": trace.get("steps", []),
            "retrieved_count": trace.get("retrieved_count"),
            "mode": trace.get("mode"),
        }

    return rag_target


EvaluatorResult = Union[Dict[str, Any], List[Dict[str, Any]]]


def make_rag_evaluator(include_llm_judge: bool = True):
    def rag_eval_metrics(inputs: dict, outputs: dict, reference_outputs: dict) -> EvaluatorResult:
        expected = reference_outputs["answer"]
        metrics = operational_metrics(outputs, reference_outputs)
        if expected == INSUFFICIENT_CONTEXT:
            quality_metrics = [insufficient_context_agreement(outputs, reference_outputs)]
        else:
            quality_metrics = [
                answer_similarity(outputs, reference_outputs),
                keyword_recall(outputs, reference_outputs),
                citation_presence(outputs, reference_outputs),
                expected_page_citation_match(outputs, reference_outputs),
            ]

        if include_llm_judge:
            quality_metrics.extend(llm_judge_metrics(inputs, outputs, reference_outputs))

        return [metric for metric in [*quality_metrics, *metrics] if metric]

    return rag_eval_metrics


def rag_eval_metrics(inputs: dict, outputs: dict, reference_outputs: dict) -> EvaluatorResult:
    return make_rag_evaluator(include_llm_judge=True)(inputs, outputs, reference_outputs)


def deterministic_rag_eval_metrics(outputs: dict, reference_outputs: dict) -> EvaluatorResult:
    expected = reference_outputs["answer"]
    metrics = operational_metrics(outputs, reference_outputs)
    if expected == INSUFFICIENT_CONTEXT:
        return [insufficient_context_agreement(outputs, reference_outputs), *metrics]

    quality_metrics = [
        answer_similarity(outputs, reference_outputs),
        keyword_recall(outputs, reference_outputs),
        citation_presence(outputs, reference_outputs),
        expected_page_citation_match(outputs, reference_outputs),
    ]
    return [metric for metric in [*quality_metrics, *metrics] if metric]


def llm_judge_metrics(inputs: dict, outputs: dict, reference_outputs: dict) -> List[Dict[str, Any]]:
    question = inputs["question"]
    answer = outputs.get("answer", "")
    expected = reference_outputs.get("answer", "")
    context = format_retrieved_context(outputs.get("retrieved_context") or [])

    prompt = (
        "You are evaluating a retrieval-augmented generation answer. "
        "Return only JSON with numeric scores from 0.0 to 1.0 and short reasons. "
        "Use this schema exactly: "
        '{"groundedness": {"score": 0.0, "reason": "..."}, '
        '"faithfulness": {"score": 0.0, "reason": "..."}, '
        '"context_relevance": {"score": 0.0, "reason": "..."}, '
        '"answer_relevance": {"score": 0.0, "reason": "..."}}.\n\n'
        "Definitions:\n"
        "- groundedness: factual claims in the answer are supported by the retrieved context.\n"
        "- faithfulness: the answer does not add, distort, or contradict retrieved context.\n"
        "- context_relevance: retrieved context contains information useful for answering the question.\n"
        "- answer_relevance: answer directly addresses the question; if the reference answer is "
        "INSUFFICIENT_CONTEXT, a clear refusal due to missing context is relevant.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{expected}\n\n"
        f"RAG answer:\n{answer}\n\n"
        f"Retrieved context:\n{context}"
    )

    default = {
        "groundedness": {"score": 0.0, "reason": "judge failed"},
        "faithfulness": {"score": 0.0, "reason": "judge failed"},
        "context_relevance": {"score": 0.0, "reason": "judge failed"},
        "answer_relevance": {"score": 0.0, "reason": "judge failed"},
    }
    try:
        verdict = {**default, **load_json_object(invoke_judge(prompt))}
    except Exception as exc:
        verdict = {
            key: {"score": 0.0, "reason": f"judge fallback: {exc}"}
            for key in default
        }

    return [
        {
            "key": key,
            "score": clamp_score(value.get("score", 0.0)),
            "comment": value.get("reason", ""),
        }
        for key, value in verdict.items()
    ]


def operational_metrics(outputs: dict, reference_outputs: dict) -> List[Dict[str, Any]]:
    metrics = [
        {
            "key": "latency_seconds",
            "score": float(outputs.get("latency_seconds") or 0.0),
            "comment": "Wall-clock time for this RAG answer generation call.",
        },
        {
            "key": "answer_word_count",
            "score": int(outputs.get("answer_word_count") or 0),
            "comment": "Number of whitespace-delimited words in the answer.",
        },
        {
            "key": "answer_char_count",
            "score": int(outputs.get("answer_char_count") or 0),
            "comment": "Number of characters in the answer.",
        },
        {
            "key": "source_count",
            "score": int(outputs.get("source_count") or 0),
            "comment": "Number of returned source citation lines.",
        },
        {
            "key": "citation_count",
            "score": int(outputs.get("citation_count") or 0),
            "comment": "Number of page citations parsed from returned sources.",
        },
        {
            "key": "trace_step_count",
            "score": len(outputs.get("trace_steps") or []),
            "comment": "Number of advanced RAG trace steps reported.",
        },
    ]

    retrieved_count = outputs.get("retrieved_count")
    if retrieved_count is not None:
        metrics.append(
            {
                "key": "retrieved_count",
                "score": int(retrieved_count),
                "comment": "Number of retrieved documents reported by the RAG trace.",
            }
        )

    return metrics


def insufficient_context_agreement(outputs: dict, reference_outputs: dict) -> Dict[str, Any]:
    expected = reference_outputs["answer"]
    actual = outputs["answer"]
    if expected != INSUFFICIENT_CONTEXT:
        raise ValueError("insufficient_context_agreement only supports insufficient-context examples.")

    actual_normalized = normalize(actual)
    agreed = any(
        phrase in actual_normalized
        for phrase in [
            "insufficient context",
            "not enough information",
            "do not have enough information",
            "does not contain",
            "cannot answer",
        ]
    )
    return {
        "key": "insufficient_context_agreement",
        "score": int(agreed),
        "comment": "Expected the system to refuse because the manual lacks this answer.",
    }


def answer_similarity(outputs: dict, reference_outputs: dict) -> Dict[str, Any]:
    expected = reference_outputs["answer"]
    actual = outputs["answer"]
    if expected == INSUFFICIENT_CONTEXT:
        raise ValueError("answer_similarity only supports answerable examples.")

    score = SequenceMatcher(None, normalize(expected), normalize(actual)).ratio()
    return {
        "key": "answer_similarity",
        "score": score,
        "comment": "Normalized SequenceMatcher ratio against the reference answer.",
    }


def keyword_recall(outputs: dict, reference_outputs: dict) -> Dict[str, Any]:
    expected = reference_outputs["answer"]
    actual = outputs["answer"]
    if expected == INSUFFICIENT_CONTEXT:
        raise ValueError("keyword_recall only supports answerable examples.")

    expected_keywords = extract_keywords(expected)
    if not expected_keywords:
        return {
            "key": "keyword_recall",
            "score": 0.0,
            "comment": "No reference keywords were available after normalization.",
        }

    actual_words = set(extract_keywords(actual))
    matched = [word for word in expected_keywords if word in actual_words]
    return {
        "key": "keyword_recall",
        "score": len(matched) / len(expected_keywords),
        "comment": f"Matched {len(matched)}/{len(expected_keywords)} reference keywords.",
    }


def citation_presence(outputs: dict, reference_outputs: dict) -> Dict[str, Any]:
    has_citation = bool(outputs.get("sources", "").strip())
    return {
        "key": "citation_presence",
        "score": int(has_citation),
        "comment": "Checks whether the answer returned at least one source citation.",
    }


def expected_page_citation_match(outputs: dict, reference_outputs: dict) -> Dict[str, Any]:
    expected_page = reference_outputs.get("page")
    cited_pages = set(outputs.get("cited_pages") or [])
    matched = expected_page is not None and int(expected_page) in cited_pages
    return {
        "key": "expected_page_citation_match",
        "score": int(matched),
        "comment": f"Expected page {expected_page}; cited pages: {sorted(cited_pages)}.",
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def extract_cited_pages(sources: str) -> List[int]:
    pages = []
    for match in re.finditer(r"\bpage\s+(\d+)\b", sources, flags=re.IGNORECASE):
        pages.append(int(match.group(1)))
    return pages


def count_sources(sources: str) -> int:
    return len([line for line in sources.splitlines() if line.strip()])


def format_retrieved_context(context_docs: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    blocks = []
    total = 0
    for idx, doc in enumerate(context_docs, start=1):
        block = (
            f"[{idx}] Source: {doc.get('source', 'unknown')}\n"
            f"{doc.get('content', '').strip()}"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def invoke_judge(prompt: str) -> str:
    global judge_llm
    if judge_llm is None:
        judge_llm = build_chat_model(JUDGE, temperature=0.0, max_tokens=700)

    response = judge_llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def load_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))


def extract_keywords(text: str) -> List[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "when",
        "with",
        "you",
    }
    return [
        token
        for token in normalize(text).split()
        if len(token) > 2 and token not in stopwords
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate this RAG system on Owners_Manual.pdf with LangSmith."
    )
    parser.add_argument("--pdf", type=Path, default=Path("Owners_Manual.pdf"))
    parser.add_argument("--testset", type=Path, default=Path("testsets/testset.json"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--experiment-prefix", default=DEFAULT_EXPERIMENT_PREFIX)
    parser.add_argument(
        "--upload-dataset",
        action="store_true",
        help="Deprecated: dataset upload is now always used for SDK compatibility.",
    )
    parser.add_argument("--disable-pdf-ocr", action="store_true")
    parser.add_argument("--hyde", action="store_true")
    parser.add_argument("--decomposition", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--self-rag", action="store_true")
    parser.add_argument("--no-parent-child", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--dense-weight", type=float, default=0.65)
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip groundedness, faithfulness, context relevance, and answer relevance judge metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    testset_path = args.testset.expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not testset_path.exists():
        raise FileNotFoundError(f"Testset not found: {testset_path}")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY before running the RAG evaluation.")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("Set LANGSMITH_API_KEY before running LangSmith evaluation.")

    examples = load_examples(testset_path, args.dataset_name)
    config = build_rag_config(args)

    print(f"Indexing {pdf_path.name}...")
    indexing_started_at = time.perf_counter()
    for status in process_sources(
        pdf_paths=[pdf_path],
        enable_pdf_ocr=not args.disable_pdf_ocr,
    ):
        print(status)
    indexing_seconds = time.perf_counter() - indexing_started_at

    client = Client()
    data: Any = ensure_dataset(client, args.dataset_name, examples)

    print(f"Running {len(examples)} examples with LangSmith...")
    results = client.evaluate(
        make_target(config),
        data=data,
        evaluators=[make_rag_evaluator(include_llm_judge=not args.skip_llm_judge)],
        experiment_prefix=args.experiment_prefix,
        metadata={
            "pdf": str(pdf_path),
            "pdf_size_bytes": pdf_path.stat().st_size,
            "testset": str(testset_path),
            "example_count": len(examples),
            "indexing_seconds": indexing_seconds,
            "llm_judge_enabled": not args.skip_llm_judge,
            "llm_judge_model": resolve_model(JUDGE) if not args.skip_llm_judge else None,
            "rag_config": config.__dict__,
        },
        max_concurrency=1,
    )
    print(results)


if __name__ == "__main__":
    main()
