import os
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st

from rag import AdvancedRAGConfig, generate_answer, process_sources


APP_NAME = os.getenv("APP_NAME", "Multi-Domain RAG Research Tool")
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
UPLOAD_DIR = Path(__file__).parent / "resources_RAG/uploads"
MAX_PDF_UPLOADS = int(os.getenv("MAX_PDF_UPLOADS", "50"))

if "api_key_input" not in st.session_state:
    st.session_state["api_key_input"] = ""

api_key = st.sidebar.text_input(
    "Enter your groq api key here",
    type="password",
    value=st.session_state.get("api_key_input", ""),
)


def render_trace(trace):
    if not trace:
        return

    with st.expander("Reasoning trace", expanded=False):
        st.write("Mode:", trace.get("mode", "unknown"))
        if trace.get("steps"):
            st.write("Steps ran:", " -> ".join(trace["steps"]))
        if trace.get("decomposition"):
            st.write("Decomposed sub-queries:", trace["decomposition"].get("sub_queries", []))
        if trace.get("hyde"):
            st.write("HyDE retrieval queries:")
            for item in trace["hyde"]:
                st.write({"sub_query": item["sub_query"], "hyde_answer": item["hyde_answer"]})
        if trace.get("retrieval_queries"):
            st.write("Retrieval queries:", trace["retrieval_queries"])
        if "retrieved_count" in trace:
            st.write("Retrieved document count:", trace["retrieved_count"])
        if trace.get("rerank"):
            st.write("Reranked documents:")
            st.dataframe(trace["rerank"], use_container_width=True)
        if trace.get("is_retrieve"):
            st.write("Self-RAG IsRetrieve:", trace["is_retrieve"])
        if trace.get("is_relevant"):
            st.write("Self-RAG IsRelevant:", trace["is_relevant"])
        if trace.get("is_supportive"):
            st.write("Self-RAG IsSupportive:", trace["is_supportive"])
        if trace.get("support_retry"):
            st.write("Self-RAG support retry:", trace["support_retry"])


def save_uploaded_pdfs(uploaded_files):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files or []:
        safe_name = Path(uploaded_file.name).name
        target_path = UPLOAD_DIR / f"{uuid4()}--{safe_name}"
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)
    return saved_paths


if api_key:
    cleaned_key = api_key.strip().strip("'\"")
    st.session_state["api_key_input"] = cleaned_key
    os.environ["GROQ_API_KEY"] = cleaned_key
    st.sidebar.success("API Key Active")

    st.sidebar.subheader("URLs")
    url1 = st.sidebar.text_input("URL 1")
    url2 = st.sidebar.text_input("URL 2")
    url3 = st.sidebar.text_input("URL 3")
    uploaded_pdfs = st.sidebar.file_uploader(
        f"PDF documents, up to {MAX_PDF_UPLOADS}",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_pdfs:
        st.sidebar.caption(f"{len(uploaded_pdfs)} PDF(s) selected")
    enable_pdf_ocr = st.sidebar.toggle("OCR fallback for scanned PDFs", value=True)

    st.sidebar.subheader("Advanced RAG")
    use_hyde = st.sidebar.toggle("HyDE", value=False)
    use_decomposition = st.sidebar.toggle("Query decomposition", value=False)
    use_hybrid = st.sidebar.toggle("Hybrid dense + BM25", value=False)
    use_rerank = st.sidebar.toggle("Cross-encoder reranking", value=False)
    use_self_rag = st.sidebar.toggle("Self-RAG reflection", value=False)

    top_k_retrieve = st.sidebar.slider("Top-K retrieve", min_value=2, max_value=20, value=8)
    top_n_rerank = st.sidebar.slider("Top-N rerank", min_value=1, max_value=10, value=5)
    dense_weight = st.sidebar.slider("Dense weight", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    sparse_weight = round(1.0 - dense_weight, 2)
    st.sidebar.caption(f"Sparse BM25 weight: {sparse_weight}")

    placeholder = st.empty()
    process_url_button = st.sidebar.button("Process Sources")

    if process_url_button:
        urls = [url for url in (url1, url2, url3) if url]
        if uploaded_pdfs and len(uploaded_pdfs) > MAX_PDF_UPLOADS:
            placeholder.error(f"Please upload {MAX_PDF_UPLOADS} PDFs or fewer at one time.")
        elif len(urls) == 0 and not uploaded_pdfs:
            placeholder.text("You must provide at least one URL or PDF")
        else:
            pdf_paths = save_uploaded_pdfs(uploaded_pdfs)
            for status in process_sources(urls=urls, pdf_paths=pdf_paths, enable_pdf_ocr=enable_pdf_ocr):
                placeholder.text(status)

    query = placeholder.text_input("Question")

    if query:
        try:
            config = AdvancedRAGConfig(
                use_hyde=use_hyde,
                use_decomposition=use_decomposition,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                use_self_rag=use_self_rag,
                top_k_retrieve=top_k_retrieve,
                top_n_rerank=top_n_rerank,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            answer, sources, trace = generate_answer(query, config=config, return_trace=True)

            st.header("Answer:")
            st.write(answer)

            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)

            render_trace(trace)

        except RuntimeError:
            placeholder.text("You must process urls first")
        except Exception as exc:
            placeholder.error(f"Error generating answer: {exc}")

    clear_api_key_button = st.sidebar.button("Clear API Key")
    if clear_api_key_button:
        os.environ.pop("GROQ_API_KEY", None)
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()

        success_msg = st.success("cleared api key")
        if success_msg:
            time.sleep(2)
            success_msg.empty()
            st.rerun()
