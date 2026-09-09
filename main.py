import os
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st

from llm_cache import cache_snapshot, clear_cache
from memory import ConversationMemory
from rag import (
    AdvancedRAGConfig,
    active_models,
    generate_answer,
    process_sources,
    reset_llm_clients,
)


APP_NAME = os.getenv("APP_NAME", "Multi-Domain RAG Research Tool")
st.set_page_config(page_title=APP_NAME, page_icon="💬", layout="wide")
UPLOAD_DIR = Path(__file__).parent / "resources_RAG/uploads"
MAX_PDF_UPLOADS = int(os.getenv("MAX_PDF_UPLOADS", "50"))

DEFAULT_STATE = {
    "api_key_input": "",
    "messages": [],
    "memory": ConversationMemory(),
    "sources_ready": False,
    "indexed_sources": [],
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def render_trace(trace):
    if not trace:
        return

    with st.expander("Reasoning trace", expanded=False):
        st.write("Mode:", trace.get("mode", "unknown"))
        if trace.get("steps"):
            st.write("Steps ran:", " -> ".join(trace["steps"]))

        routing = trace.get("routing")
        if routing:
            st.write(
                f"Semantic route: {routing['route']} "
                f"({'retrieval skipped' if routing['skips_retrieval'] else 'retrieval used'})"
            )
            st.caption(
                f"score {routing['score']}, margin over document questions "
                f"{routing['margin']}, decided by {routing['method']}: {routing['reason']}"
            )
            if routing.get("scores"):
                st.write("Route scores:", routing["scores"])

        memory_trace = trace.get("memory") or {}
        if memory_trace.get("turns_in_window") or memory_trace.get("has_summary"):
            st.write("Conversation turns in window:", memory_trace.get("turns_in_window", 0))
            if memory_trace.get("rewritten"):
                st.write("Question rewritten for retrieval:", memory_trace.get("search_query"))
            if memory_trace.get("summary"):
                st.write("Running summary of older turns:")
                st.caption(memory_trace["summary"])

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
        if trace.get("cache"):
            st.write("Prompt cache after this turn:", trace["cache"])


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources", expanded=False):
                for source in message["sources"].split("\n"):
                    st.markdown(f"- {source}")
        render_trace(message.get("trace"))


def save_uploaded_pdfs(uploaded_files):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files or []:
        safe_name = Path(uploaded_file.name).name
        target_path = UPLOAD_DIR / f"{uuid4()}--{safe_name}"
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)
    return saved_paths


def reset_conversation():
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationMemory()


st.title(APP_NAME)

api_key = st.sidebar.text_input(
    "Enter your groq api key here",
    type="password",
    value=st.session_state.get("api_key_input", ""),
)

if not api_key:
    st.info("Add your Groq API key in the sidebar, then load some sources to start chatting.")
    st.stop()

cleaned_key = api_key.strip().strip("'\"")
if cleaned_key != st.session_state.get("api_key_input"):
    reset_llm_clients()
st.session_state["api_key_input"] = cleaned_key
os.environ["GROQ_API_KEY"] = cleaned_key
st.sidebar.success("API Key Active")

with st.sidebar.expander("Active Groq models", expanded=False):
    try:
        for role, info in active_models().items():
            if info["resolved"] != info["requested"]:
                st.warning(
                    f"{role}: {info['requested']} is unavailable, using {info['resolved']}",
                    icon="⚠️",
                )
            else:
                st.caption(f"{role}: {info['resolved']}")
    except Exception as exc:
        st.caption(f"Could not read the model catalogue: {exc}")

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
process_url_button = st.sidebar.button("Process Sources", type="primary")

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

st.sidebar.subheader("Conversation memory")
memory = st.session_state["memory"]
memory_window = st.sidebar.slider(
    "Turns kept verbatim",
    min_value=1,
    max_value=12,
    value=memory.window_turns,
    help="Older turns are folded into a running summary instead of being dropped.",
)
memory.window_turns = memory_window
st.sidebar.caption(
    f"{len(memory.turns)} turn(s) in window"
    + (", running summary active" if memory.summary else "")
)
if st.sidebar.button("Clear conversation"):
    reset_conversation()
    st.rerun()

st.sidebar.subheader("Prompt cache")
cache_stats = cache_snapshot()
st.sidebar.caption(
    f"Local hits: {cache_stats['local_hits']}/{cache_stats['local_hits'] + cache_stats['local_misses']} "
    f"({cache_stats['local_hit_rate']:.0%})"
)
st.sidebar.caption(
    f"Groq prefix reuse: {cache_stats['cached_prompt_tokens']}/{cache_stats['prompt_tokens']} prompt tokens "
    f"({cache_stats['prefix_hit_rate']:.0%})"
)
if st.sidebar.button("Clear prompt cache"):
    clear_cache()
    st.rerun()

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url]
    if uploaded_pdfs and len(uploaded_pdfs) > MAX_PDF_UPLOADS:
        st.sidebar.error(f"Please upload {MAX_PDF_UPLOADS} PDFs or fewer at one time.")
    elif not urls and not uploaded_pdfs:
        st.sidebar.warning("You must provide at least one URL or PDF")
    else:
        pdf_paths = save_uploaded_pdfs(uploaded_pdfs)
        failed = False
        with st.status("Processing sources", expanded=True) as status:
            for update in process_sources(urls=urls, pdf_paths=pdf_paths, enable_pdf_ocr=enable_pdf_ocr):
                st.write(update)
                if update.startswith("Error"):
                    failed = True
            status.update(
                label="Sources processed" if not failed else "Source processing failed",
                state="complete" if not failed else "error",
            )
        if not failed:
            st.session_state["sources_ready"] = True
            st.session_state["indexed_sources"] = urls + [
                Path(path).name.split("--", 1)[-1] for path in pdf_paths
            ]
            # A new corpus makes earlier answers stale, so start a fresh thread.
            reset_conversation()

if st.session_state["indexed_sources"]:
    st.caption("Indexed: " + ", ".join(st.session_state["indexed_sources"]))

for message in st.session_state["messages"]:
    render_message(message)

placeholder_text = (
    "Ask a question about your sources"
    if st.session_state["sources_ready"]
    else "Say hello, or process a source to ask document questions"
)
# Not gated on sources: the semantic router answers conversational messages
# without retrieval, and document questions raise below with a clear message.
query = st.chat_input(placeholder_text)

if query:
    user_message = {"role": "user", "content": query}
    st.session_state["messages"].append(user_message)
    render_message(user_message)

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

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating"):
            try:
                answer, sources, trace = generate_answer(
                    query,
                    config=config,
                    return_trace=True,
                    memory=memory,
                )
            except RuntimeError:
                answer, sources, trace = (
                    "That looks like a question about your documents, but no "
                    "sources are loaded yet. Add a URL or PDF in the sidebar "
                    "and press Process Sources.",
                    "",
                    None,
                )
                st.session_state["sources_ready"] = False
            except Exception as exc:
                answer, sources, trace = f"Error generating answer: {exc}", "", None

    if trace is not None:
        memory.add_turn(query, answer, sources)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "trace": trace,
    }
    st.session_state["messages"].append(assistant_message)
    st.rerun()

if st.sidebar.button("Clear API Key"):
    os.environ.pop("GROQ_API_KEY", None)
    reset_llm_clients()
    st.session_state.clear()
    st.cache_data.clear()
    st.cache_resource.clear()

    success_msg = st.sidebar.success("cleared api key")
    time.sleep(2)
    success_msg.empty()
    st.rerun()
