import re
import uuid
from pathlib import Path
from typing import List

import streamlit as st

try:
    from .chunker import split_documents
    from .config import (
        DEFAULT_STREAMLIT_UPLOAD_ROOT,
        DEFAULT_STREAMLIT_VECTOR_ROOT,
    )
    from .embedding_manager import EmbeddingManager
    from .ingest import (
        load_documents_from_uploaded_files,
        load_wikipedia_topics,
    )
    from .llm import get_llm
    from .pipeline import AdvancedRagPipeline
    from .retriever import RAGRetriever
    from .vector_store import VectorStore
except ImportError:
    from chunker import split_documents
    from config import (
        DEFAULT_STREAMLIT_UPLOAD_ROOT,
        DEFAULT_STREAMLIT_VECTOR_ROOT,
    )
    from embedding_manager import EmbeddingManager
    from ingest import (
        load_documents_from_uploaded_files,
        load_wikipedia_topics,
    )
    from llm import get_llm
    from pipeline import AdvancedRagPipeline
    from retriever import RAGRetriever
    from vector_store import VectorStore


def _init_session() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = None
    if "llm_model_name" not in st.session_state:
        st.session_state.llm_model_name = None
    if "llm_config_signature" not in st.session_state:
        st.session_state.llm_config_signature = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:10]
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = f"streamlit_{st.session_state.session_id}"
    if "vector_dir" not in st.session_state:
        st.session_state.vector_dir = (
            DEFAULT_STREAMLIT_VECTOR_ROOT / st.session_state.collection_name
        )
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = (
            DEFAULT_STREAMLIT_UPLOAD_ROOT / st.session_state.collection_name
        )


def _parse_topics(raw_topics: str) -> List[str]:
    return [t.strip() for t in re.split(r"[,\n]", raw_topics) if t.strip()]


def _build_retriever(
    uploaded_files: List[object],
    wiki_topics: List[str],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    wiki_docs_per_topic: int,
) -> tuple[RAGRetriever, int]:
    all_documents = []

    if uploaded_files:
        all_documents.extend(
            load_documents_from_uploaded_files(
                uploaded_files=uploaded_files,
                temp_dir=st.session_state.upload_dir,
            )
        )

    if wiki_topics:
        wiki_docs = load_wikipedia_topics(
            topics=wiki_topics,
            max_docs_per_topic=wiki_docs_per_topic,
        )
        all_documents.extend(wiki_docs)

    if not all_documents:
        raise ValueError("No documents found from uploads or Wikipedia topics")

    chunks = split_documents(
        all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = [doc.page_content for doc in chunks]

    embedding_manager = EmbeddingManager(model_name=embedding_model)
    embeddings = embedding_manager.generate_embeddings(texts)

    vector_dir = Path(st.session_state.vector_dir)
    vector_dir.mkdir(parents=True, exist_ok=True)

    vector_store = VectorStore(
        collection_name=st.session_state.collection_name,
        persist_directory=str(vector_dir),
    )
    vector_store.clear()
    vector_store.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)
    return retriever, len(chunks)


def main() -> None:
    st.set_page_config(page_title="Context-Aware RAG Chatbot", page_icon=":books:")
    _init_session()

    st.title("Context-Aware RAG Chatbot")
    st.caption("Upload PDF/TXT files, add Wikipedia topics, then chat with memory.")
    with st.expander("How to create and use a Gemini API key", expanded=False):
        st.markdown(
            "1. Open Google AI Studio at https://aistudio.google.com/app/apikey\n"
            "2. Sign in with your Google account.\n"
            "3. Click `Create API key` and copy the key.\n"
            "4. Paste the key in the sidebar field, or save it in `.env`.\n"
            "5. If using `.env`, add this line:\n"
            "   `GEMINI_API_KEY=\"your_api_key_here\"`\n"
            "6. Save `.env` and restart Streamlit (if edited).\n"
            "7. Process your sources and start chatting."
        )

    with st.sidebar:
        st.subheader("Data Sources")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        raw_topics = st.text_area(
            "Wikipedia topics (comma or new line separated)",
            value="",
            height=110,
        )
        wiki_topics = _parse_topics(raw_topics)
        wiki_docs_per_topic = st.slider("Max Wikipedia docs/topic", min_value=1, max_value=5, value=2)

        st.subheader("Model Settings")
        provider = "gemini"
        st.caption("LLM provider: Gemini")
        model_name = st.text_input("Optional model name override", value="")
        api_key = st.text_input(
            "Gemini API key (optional; overrides .env)",
            value="",
            type="password",
        )
        temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        max_tokens = st.number_input("Max response tokens", min_value=256, max_value=8192, value=2048, step=128)
        embedding_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")

        st.subheader("Retrieval Settings")
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=10)
        top_k = st.slider("Top K", min_value=1, max_value=10, value=4)
        min_score = st.slider("Min similarity score", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        history_window = st.slider("Conversation memory turns", min_value=1, max_value=10, value=4)

        process_clicked = st.button("Process Sources", type="primary", use_container_width=True)
        clear_chat_clicked = st.button("Clear Chat History", use_container_width=True)

    if clear_chat_clicked:
        st.session_state.chat_messages = []
        if st.session_state.pipeline is not None:
            st.session_state.pipeline.history = []
        st.success("Chat history cleared")

    if process_clicked:
        try:
            with st.spinner("Processing documents and building vector store..."):
                retriever, chunk_count = _build_retriever(
                    uploaded_files=uploaded_files or [],
                    wiki_topics=wiki_topics,
                    embedding_model=embedding_model,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    wiki_docs_per_topic=int(wiki_docs_per_topic),
                )
                st.session_state.retriever = retriever
                st.session_state.pipeline = None
                st.session_state.llm_provider = None
                st.session_state.llm_model_name = None
                st.session_state.llm_config_signature = None
                st.session_state.chat_messages = []
            st.success(f"Ready. Indexed {chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Failed to process sources: {exc}")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask a question about uploaded files or Wikipedia topics")
    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        if st.session_state.retriever is None:
            assistant_text = "Process at least one source first (upload or Wikipedia topic)."
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.markdown(assistant_text)
            return

        current_model_name = model_name.strip() or None
        api_key_fingerprint = hash(api_key) if api_key else None
        llm_signature = (
            provider,
            current_model_name,
            api_key_fingerprint,
            float(temperature),
            int(max_tokens),
        )
        needs_new_pipeline = (
            st.session_state.pipeline is None
            or st.session_state.llm_provider != provider
            or st.session_state.llm_model_name != current_model_name
            or st.session_state.llm_config_signature != llm_signature
        )
        if needs_new_pipeline:
            try:
                llm = get_llm(
                    provider=provider,
                    model_name=current_model_name,
                    api_key=api_key or None,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                st.session_state.pipeline = AdvancedRagPipeline(
                    retriever=st.session_state.retriever,
                    llm=llm,
                )
                st.session_state.llm_provider = provider
                st.session_state.llm_model_name = current_model_name
                st.session_state.llm_config_signature = llm_signature
            except Exception as exc:
                assistant_text = f"LLM initialization failed: {exc}"
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})
                with st.chat_message("assistant"):
                    st.markdown(assistant_text)
                return

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.pipeline.query(
                    user_prompt,
                    top_k=int(top_k),
                    min_score=float(min_score),
                    summarize=False,
                    stream=False,
                    use_history=True,
                    history_window=int(history_window),
                )
                assistant_text = result["answer"]
                st.markdown(assistant_text)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": assistant_text}
        )


if __name__ == "__main__":
    main()
