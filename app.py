from pathlib import Path
import re
from typing import Any, Dict, List

import streamlit as st

from src.chunker import TextChunker
from src.config import config
from src.database.repository import DatabaseRepository
from src.exact_search import search_exact_match
from src.search import TFIDFSearchEngine


def initialize_session_state() -> None:
    """Initializes session state stores for documents and search engines."""
    if "repository" not in st.session_state:
        st.session_state.repository = DatabaseRepository(config.DB_PATH)

    if "search_engine" not in st.session_state:
        st.session_state.search_engine = TFIDFSearchEngine(config.DB_PATH)

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []


def highlight_keyword(text: str, keyword: str) -> str:
    """Highlights keyword matches in snippet using HTML mark tags."""
    if not keyword or not keyword.strip():
        return text
    pattern = re.compile(rf"({re.escape(keyword.strip())})", re.IGNORECASE)
    return pattern.sub(r"**\1**", text)


def main() -> None:
    st.set_page_config(
        page_title="ML/DL Knowledge Base",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()
    repo: DatabaseRepository = st.session_state.repository
    searcher: TFIDFSearchEngine = st.session_state.search_engine

    # --- SIDEBAR: Upload & System Status ---
    with st.sidebar:
        st.title("⚙️ System Control")
        st.write("Intelligent Knowledge Base for ML & Deep Learning")

        doc_count = repo.count_documents()
        chunk_count = repo.count_chunks()

        col1, col2 = st.columns(2)
        col1.metric("Documents", doc_count)
        col2.metric("Chunks", chunk_count)

        st.divider()
        st.subheader("📤 Manual Document Ingestion")
        uploaded_file = st.file_uploader(
            "Upload `.txt` document to knowledge base",
            type=["txt"],
            help="File will be cleaned, chunked with 50-word overlap, and indexed.",
        )

        if uploaded_file is not None:
            file_name = uploaded_file.name
            raw_text = uploaded_file.read().decode("utf-8")
            title = Path(file_name).stem.replace("_", " ").title()

            if st.button("Process & Index Document", use_container_width=True):
                with st.spinner("Chunking and saving to database..."):
                    chunker = TextChunker(
                        chunk_size=config.CHUNK_SIZE_WORDS,
                        overlap=config.CHUNK_OVERLAP_WORDS,
                    )
                    cleaned = chunker.clean_text(raw_text)
                    chunks = chunker.chunk_text(cleaned)

                    # Persist to SQLite
                    doc_id = repo.ingest_document_with_chunks(
                        title=title,
                        raw_content=raw_text,
                        chunks=chunks,
                    )

                    # Add to session state
                    st.session_state.uploaded_docs.append({
                        "id": doc_id,
                        "title": title,
                        "chunk_count": len(chunks),
                    })

                    # Reload search engine index
                    searcher.reload()
                    st.success(f"Ingested '{title}' (ID: {doc_id}) into {len(chunks)} chunks!")
                    st.rerun()

        st.caption(f"Storage path: `{config.DB_PATH.name}`")

    # --- MAIN VIEW: Tabs ---
    st.title("🧠 ML/DL Knowledge Base")
    st.markdown(
        "Explore and search machine learning knowledge across English Wikipedia articles "
        "using overlapping text chunking (300w / 50w overlap)."
    )

    tab_docs, tab_search = st.tabs(["📚 Documents", "🔍 Search"])

    # --- TAB 1: Documents Viewer ---
    with tab_docs:
        st.subheader("Stored Document Resources")

        all_chunks = repo.get_all_chunks_with_parent_info()
        if not all_chunks:
            st.info("No documents found in database. Ingest documents via the sidebar or run `populate_database.py`.")
        else:
            # Group chunks by document
            docs_summary: Dict[str, Dict[str, Any]] = {}
            for item in all_chunks:
                doc_title = item["document_title"]
                if doc_title not in docs_summary:
                    docs_summary[doc_title] = {
                        "Document ID": item["document_id"],
                        "Title": doc_title,
                        "Chunks Count": 0,
                        "First Chunk Preview": item["chunk_text"][:140] + "...",
                    }
                docs_summary[doc_title]["Chunks Count"] += 1

            summary_list = list(docs_summary.values())
            st.dataframe(
                summary_list,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()
            st.subheader("📖 Document Chunk Inspector")
            selected_title = st.selectbox(
                "Select a document to inspect individual overlapping chunks:",
                options=list(docs_summary.keys()),
            )

            if selected_title:
                selected_chunks = [c for c in all_chunks if c["document_title"] == selected_title]
                st.write(f"Showing **{len(selected_chunks)} chunks** for: `{selected_title}`")

                for ch in selected_chunks:
                    with st.expander(f"Chunk #{ch['chunk_index']} (ID: {ch['chunk_id']})"):
                        st.markdown(ch["chunk_text"])
                        words = ch["chunk_text"].split()
                        st.caption(f"Word count: {len(words)} words")

    # --- TAB 2: Search ---
    with tab_search:
        st.subheader("Search Engine")

        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            query = st.text_input(
                "Search query:",
                value="neural",
                placeholder="Enter search phrase (e.g. 'neural', 'deep learning')...",
            )
        with search_col2:
            search_mode = st.selectbox(
                "Search Algorithm:",
                options=["Exact Match (Keyword)", "TF-IDF (Cosine Similarity)"],
            )

        top_k = st.slider("Max results (Top-K):", min_value=1, max_value=20, value=5)

        if query:
            chunks_to_search = repo.get_all_chunks_with_parent_info()

            if search_mode == "Exact Match (Keyword)":
                st.markdown(f"##### Results for exact phrase: `'{query}'`")
                results = search_exact_match(query, chunks_to_search)[:top_k]

                if not results:
                    st.warning(f"No exact matches found for keyword: `{query}`.")
                else:
                    st.success(f"Found {len(results)} matching chunks containing `{query}`.")
                    for rank, res in enumerate(results, start=1):
                        with st.container(border=True):
                            col_head, col_badge = st.columns([4, 1])
                            col_head.markdown(f"### #{rank} 📄 {res['document_title']}")
                            col_badge.caption(f"Occurrences: **{res['match_count']}**")

                            highlighted = highlight_keyword(res["chunk_text"], query)
                            st.markdown(highlighted)
                            st.caption(
                                f"Chunk Index: {res['chunk_index']} | "
                                f"Parent Doc ID: {res['document_id']} | "
                                f"Chunk ID: {res['chunk_id']}"
                            )

            elif search_mode == "TF-IDF (Cosine Similarity)":
                st.markdown(f"##### Ranked TF-IDF results for: `'{query}'`")
                tfidf_results = searcher.search_tfidf(query, top_k=top_k)

                if not tfidf_results:
                    st.warning(f"No relevant chunks found for query: `{query}`.")
                else:
                    for rank, res in enumerate(tfidf_results, start=1):
                        with st.container(border=True):
                            col_head, col_score = st.columns([4, 1])
                            col_head.markdown(f"### #{rank} 📄 {res['document_title']}")
                            col_score.metric("TF-IDF Score", f"{res['score']:.4f}")

                            st.markdown(res["chunk_text"])
                            st.caption(
                                f"Chunk Index: {res['chunk_index']} | "
                                f"Parent Doc ID: {res['document_id']} | "
                                f"Chunk ID: {res['chunk_id']}"
                            )


if __name__ == "__main__":
    main()
