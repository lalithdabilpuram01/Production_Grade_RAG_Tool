import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pre_retrieval import decompose_query, generate_hypothetical_answer
from rerank import rerank_documents
from self_rag import SelfRAGGrader

try:
    from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
    from langchain.storage import InMemoryStore
except Exception:  # pragma: no cover - import location differs by LangChain version.
    EnsembleRetriever = None
    ParentDocumentRetriever = None
    InMemoryStore = None


warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()


PARENT_CHUNK_SIZE = int(os.getenv("RAG_PARENT_CHUNK_SIZE", "1800"))
CHILD_CHUNK_SIZE = int(os.getenv("RAG_CHILD_CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "multi_domain")
VECTORSTORE_DIR = Path(__file__).parent / "resources_RAG/vectorstore"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_GENERATION_MODEL = os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile")
ASSISTANT_ROLE = os.getenv("ASSISTANT_ROLE", "domain-neutral research assistant")
PDF_LOAD_BATCH_SIZE = int(os.getenv("PDF_LOAD_BATCH_SIZE", "5"))
VECTOR_ADD_BATCH_SIZE = int(os.getenv("VECTOR_ADD_BATCH_SIZE", "500"))
ENABLE_PDF_OCR = os.getenv("ENABLE_PDF_OCR", "true").lower() == "true"
OCR_MIN_PAGE_CHARS = int(os.getenv("OCR_MIN_PAGE_CHARS", "80"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))


@dataclass
class AdvancedRAGConfig:
    use_hyde: bool = False
    use_decomposition: bool = False
    use_hybrid: bool = False
    use_parent_child: bool = True
    use_rerank: bool = False
    use_self_rag: bool = False
    top_k_retrieve: int = 8
    top_n_rerank: int = 5
    dense_weight: float = 0.65
    sparse_weight: float = 0.35
    max_self_rag_retries: int = 2


llm = None
embedding_function = None
vector_store = None
bm25_retriever = None
parent_retriever = None
parent_docstore = None
parent_docs_by_id: Dict[str, Document] = {}
indexed_parent_docs: List[Document] = []


def initialize_components():
    global llm, embedding_function, vector_store

    if llm is None:
        llm = ChatGroq(model=GROQ_GENERATION_MODEL, temperature=0.2, max_tokens=700)

    if embedding_function is None:
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )

    if vector_store is None:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=str(VECTORSTORE_DIR),
        )


def process_urls(urls):
    yield from process_sources(urls=urls)


def process_sources(urls=None, pdf_paths=None, enable_pdf_ocr: Optional[bool] = None):
    global bm25_retriever, parent_retriever, parent_docstore, parent_docs_by_id, indexed_parent_docs

    try:
        urls = urls or []
        pdf_paths = pdf_paths or []
        enable_pdf_ocr = ENABLE_PDF_OCR if enable_pdf_ocr is None else enable_pdf_ocr

        yield "Initializing components"
        initialize_components()

        yield "Resetting vector store"
        vector_store.reset_collection()
        parent_docs_by_id = {}
        indexed_parent_docs = []

        data = []

        if urls:
            yield "Scraping and loading data from URLs"
            loader = WebBaseLoader(
                web_path=urls,
                header_template={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                },
                continue_on_failure=True,
            )
            data.extend(loader.load())

        if pdf_paths:
            total_pdfs = len(pdf_paths)
            for index, pdf_path in enumerate(pdf_paths, start=1):
                yield f"Loading PDF {index}/{total_pdfs}: {_display_source_name(pdf_path)}"
                pdf_docs, used_ocr, note = _load_pdf_documents(pdf_path, enable_pdf_ocr)
                if note:
                    yield note
                if used_ocr:
                    yield f"OCR extracted text from {_display_source_name(pdf_path)}"
                data.extend(pdf_docs)
                if index % PDF_LOAD_BATCH_SIZE == 0:
                    yield f"Loaded {index}/{total_pdfs} PDFs"

        if not data:
            yield "Error: Could not retrieve any content from the provided sources."
            return

        yield "Splitting parent sections"
        parent_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=80,
        )

        parent_docs = parent_splitter.split_documents(data)
        if not parent_docs:
            yield "Error: No text chunks could be extracted from the documents."
            return

        # Parent chunks give the LLM enough context; child chunks keep vector search precise.
        child_docs = []
        for parent_doc in parent_docs:
            parent_id = str(uuid4())
            parent_doc.metadata = {**parent_doc.metadata, "parent_id": parent_id}
            parent_docs_by_id[parent_id] = parent_doc
            child_chunks = child_splitter.split_documents([parent_doc])
            for child in child_chunks:
                child.metadata = {**child.metadata, "parent_id": parent_id}
                child_docs.append(child)

        indexed_parent_docs = parent_docs

        yield f"Adding {len(child_docs)} child chunks to vector database"
        for start in range(0, len(child_docs), VECTOR_ADD_BATCH_SIZE):
            batch = child_docs[start : start + VECTOR_ADD_BATCH_SIZE]
            vector_store.add_documents(batch, ids=[str(uuid4()) for _ in batch])
            yield f"Indexed {min(start + len(batch), len(child_docs))}/{len(child_docs)} child chunks"

        yield "Building sparse BM25 index"
        bm25_retriever = BM25Retriever.from_documents(parent_docs)
        bm25_retriever.k = 8

        if ParentDocumentRetriever and InMemoryStore:
            yield "Preparing parent-child retriever"
            parent_docstore = InMemoryStore()
            parent_retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=parent_docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                id_key="parent_id",
            )
            # We already inserted child vectors above, so only hydrate the parent docstore here.
            parent_docstore.mset([(doc.metadata["parent_id"], doc) for doc in parent_docs])
        else:
            parent_retriever = None

        yield "Success: Sources processed successfully!"

    except Exception as e:
        yield f"Error processing sources: {str(e)}"


def generate_answer(query, config: Optional[AdvancedRAGConfig] = None, return_trace: bool = False):
    if not vector_store:
        raise RuntimeError("VectorDB is not initialized")

    config = config or AdvancedRAGConfig()
    if not _uses_advanced_features(config):
        answer, sources = _generate_basic_answer(query)
        if return_trace:
            return answer, sources, {"mode": "basic", "steps": []}
        return answer, sources

    result = _generate_advanced_answer(query, config)
    if return_trace:
        return result["answer"], result["sources"], result["trace"]
    return result["answer"], result["sources"]


def _generate_basic_answer(query: str) -> Tuple[str, str]:
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    )
    result = chain.invoke({"question": query}, return_only_outputs=True)
    return result["answer"], result.get("sources", "")


def _generate_advanced_answer(query: str, config: AdvancedRAGConfig) -> Dict[str, Any]:
    trace: Dict[str, Any] = {"mode": "advanced", "steps": []}
    grader = SelfRAGGrader() if config.use_self_rag else None

    if grader:
        retrieve_verdict = grader.is_retrieve(query)
        trace["is_retrieve"] = retrieve_verdict
        trace["steps"].append("Self-RAG IsRetrieve")
        if not retrieve_verdict.get("needs_retrieval", True):
            answer = _answer_without_retrieval(query)
            return {"answer": answer, "sources": "", "trace": trace}

    sub_queries = [query]
    if config.use_decomposition:
        decomposition = decompose_query(query)
        sub_queries = decomposition["sub_queries"]
        trace["decomposition"] = decomposition
        trace["steps"].append("Query decomposition")

    retrieved_docs = []
    retrieval_queries = []
    for sub_query in sub_queries:
        retrieval_query = sub_query
        if config.use_hyde:
            hyde = generate_hypothetical_answer(sub_query)
            retrieval_query = hyde["query_for_embedding"]
            trace.setdefault("hyde", []).append({"sub_query": sub_query, **hyde})
            trace["steps"].append("HyDE")

        retrieval_queries.append({"user_query": sub_query, "retrieval_query": retrieval_query})
        retrieved_docs.extend(_retrieve_documents(retrieval_query, config))

    docs = _dedupe_documents(retrieved_docs)[: config.top_k_retrieve]
    trace["retrieval_queries"] = retrieval_queries
    trace["retrieved_count"] = len(docs)
    trace["steps"].append("Hybrid retrieval" if config.use_hybrid else "Dense retrieval")

    if grader:
        docs, relevant_trace = _ensure_relevant_docs(query, docs, config, grader)
        trace["is_relevant"] = relevant_trace
        trace["steps"].append("Self-RAG IsRelevant")

    if config.use_rerank:
        docs, rerank_trace = rerank_documents(query, docs, config.top_n_rerank)
        trace["rerank"] = rerank_trace
        trace["steps"].append("Cross-encoder reranking")

    answer = _generate_grounded_answer(query, docs)
    sources = _extract_sources(docs)

    if grader:
        support_trace = grader.is_supportive(query, answer, docs)
        trace["is_supportive"] = support_trace
        trace["steps"].append("Self-RAG IsSupportive")
        if not support_trace.get("is_supported", True):
            regenerated = _generate_grounded_answer(query, docs, low_confidence=True)
            second_trace = grader.is_supportive(query, regenerated, docs)
            trace["support_retry"] = second_trace
            answer = regenerated
            if not second_trace.get("is_supported", True):
                answer = "Low confidence: the retrieved sources may not fully support this answer.\n\n" + answer

    return {"answer": answer, "sources": sources, "trace": trace}


def _retrieve_documents(query: str, config: AdvancedRAGConfig) -> List[Document]:
    if config.use_hybrid and bm25_retriever:
        docs = _hybrid_retrieve(query, config)
    elif config.use_parent_child and parent_retriever:
        docs = parent_retriever.invoke(query)
    else:
        docs = vector_store.similarity_search(query, k=config.top_k_retrieve)

    if config.use_parent_child:
        docs = _map_children_to_parents(docs)

    return docs


def _hybrid_retrieve(query: str, config: AdvancedRAGConfig) -> List[Document]:
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": config.top_k_retrieve})
    bm25_retriever.k = config.top_k_retrieve

    if EnsembleRetriever:
        ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[config.dense_weight, config.sparse_weight],
        )
        return ensemble.invoke(query)

    # Keep hybrid search available across LangChain versions that lack EnsembleRetriever.
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = bm25_retriever.invoke(query)
    scores: Dict[str, Tuple[Document, float]] = {}
    for weight, docs in [(config.dense_weight, dense_docs), (config.sparse_weight, sparse_docs)]:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            current = scores.get(key, (doc, 0.0))[1]
            scores[key] = (doc, current + weight / (rank + 60))
    return [doc for doc, _ in sorted(scores.values(), key=lambda item: item[1], reverse=True)]


def _ensure_relevant_docs(
    query: str,
    docs: List[Document],
    config: AdvancedRAGConfig,
    grader: SelfRAGGrader,
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    verdicts = []
    active_query = query
    active_docs = docs

    for attempt in range(config.max_self_rag_retries + 1):
        verdict = grader.is_relevant(active_query, active_docs)
        verdict["attempt"] = attempt + 1
        verdicts.append(verdict)
        if verdict.get("is_relevant", bool(active_docs)):
            return active_docs, verdicts

        # The grader can suggest a rewrite; otherwise broaden toward source-grounded terms.
        active_query = verdict.get("rewrite_query") or f"{query} relevant facts from the provided documents"
        active_docs = _retrieve_documents(active_query, config)

    return active_docs, verdicts


def _generate_grounded_answer(query: str, docs: List[Document], low_confidence: bool = False) -> str:
    context = _format_docs_for_prompt(docs)
    caution = "If support is weak, clearly say what is missing. " if low_confidence else ""
    prompt = (
        f"You are a {ASSISTANT_ROLE}. Answer strictly from the retrieved "
        "context. Preserve source citations by referencing "
        "the source URL or label when making factual claims. "
        f"{caution}If the context does not contain the answer, say you do not "
        "have enough information.\n\n"
        f"Question: {query}\n\nRetrieved context:\n{context}\n\nAnswer:"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _answer_without_retrieval(query: str) -> str:
    prompt = (
        "Answer this general question concisely. Do not cite local retrieved "
        "sources because retrieval was intentionally skipped.\n\n"
        f"Question: {query}"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _map_children_to_parents(docs: Iterable[Document]) -> List[Document]:
    mapped = []
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        mapped.append(parent_docs_by_id.get(parent_id, doc))
    return _dedupe_documents(mapped)


def _dedupe_documents(docs: Iterable[Document]) -> List[Document]:
    seen = set()
    output = []
    for doc in docs:
        key = _doc_key(doc)
        if key not in seen:
            seen.add(key)
            output.append(doc)
    return output


def _doc_key(doc: Document) -> str:
    return doc.metadata.get("parent_id") or f"{doc.metadata.get('source', '')}:{hash(doc.page_content)}"


def _format_docs_for_prompt(docs: List[Document]) -> str:
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{idx}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _extract_sources(docs: List[Document]) -> str:
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
    return "\n".join(sources)


def _load_pdf_documents(pdf_path, enable_pdf_ocr: bool) -> Tuple[List[Document], bool, str]:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["file_path"] = str(pdf_path)
        doc.metadata["source"] = _display_source_name(pdf_path)
        doc.metadata["ocr"] = False

    if not enable_pdf_ocr or not _needs_ocr(docs):
        return docs, False, ""

    try:
        ocr_docs = _ocr_pdf_documents(pdf_path)
        if ocr_docs:
            return ocr_docs, True, ""
        return docs, False, f"OCR found no text in {_display_source_name(pdf_path)}; using PDF text extraction."
    except Exception as exc:
        return docs, False, f"OCR fallback skipped for {_display_source_name(pdf_path)}: {exc}"


def _needs_ocr(docs: List[Document]) -> bool:
    if not docs:
        return True
    pages_with_text = sum(1 for doc in docs if len(doc.page_content.strip()) >= OCR_MIN_PAGE_CHARS)
    return pages_with_text < max(1, len(docs) // 2)


def _ocr_pdf_documents(pdf_path) -> List[Document]:
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(str(pdf_path), dpi=OCR_DPI)
    docs = []
    for page_number, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image).strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": _display_source_name(pdf_path),
                    "file_path": str(pdf_path),
                    "page": page_number - 1,
                    "ocr": True,
                },
            )
        )
    return docs


def _display_source_name(path) -> str:
    name = Path(path).name
    return name.split("--", 1)[1] if "--" in name else name


def _uses_advanced_features(config: AdvancedRAGConfig) -> bool:
    return any(
        [
            config.use_hyde,
            config.use_decomposition,
            config.use_hybrid,
            config.use_rerank,
            config.use_self_rag,
        ]
    )


if __name__ == "__main__":
    initialize_components()
