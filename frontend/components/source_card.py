"""
Structured Evidence Source Card Component.

Renders clinical source citations with document name, page, relevance score,
and quoted excerpt, cleanly separated from AI-generated response text.
Seamlessly adapts to Light and Dark themes via CSS variables.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
from frontend.utils.helpers import format_similarity, clean_filename


def render_source_cards(sources: Optional[List[Dict[str, Any]]]):
    """
    Renders structured evidence source cards for RAG retrieved passages.
    Strictly uses 'Retrieval relevance' to avoid misleading clinical confidence implications.
    """
    if not sources:
        st.info("ℹ️ No relevant healthcare sources were retrieved for this response.")
        return

    st.markdown(
        """
        <div style="margin-top: 16px; margin-bottom: 12px;">
            <div style="font-size: 0.84rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; gap: 6px;">
                <span>📚</span>
                <span>Grounded Evidence Sources</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    for idx, src in enumerate(sources, start=1):
        source_idx = src.get("source_index", idx)
        doc_name = clean_filename(src.get("filename") or src.get("document_id") or "Healthcare Document")
        page = src.get("page", src.get("page_number", "1"))
        chunk_id = src.get("chunk_id", f"CHUNK_{idx}")
        score = src.get("similarity_score")
        relevance_str = format_similarity(score)
        excerpt = src.get("excerpt") or src.get("preview_text") or src.get("chunk_text") or ""

        with st.container():
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="
                                font-size: 0.74rem;
                                font-weight: 800;
                                color: var(--teal-primary);
                                background-color: var(--teal-light);
                                border: 1px solid var(--teal-border);
                                padding: 2px 8px;
                                border-radius: var(--radius-sm);
                                letter-spacing: 0.04em;
                            ">
                                SOURCE {source_idx}
                            </span>
                            <span style="font-weight: 700; color: var(--text-primary); font-size: 0.95rem;">
                                {doc_name}
                            </span>
                        </div>
                        <span class="relevance-pill">
                            Page {page} • {relevance_str}
                        </span>
                    </div>
                    <div style="
                        font-size: 0.86rem;
                        color: var(--text-secondary);
                        line-height: 1.55;
                        background-color: var(--bg-surface-subtle);
                        padding: 10px 14px;
                        border-radius: var(--radius-sm);
                        border: 1px solid var(--border-subtle);
                        font-style: italic;
                    ">
                        "{excerpt.strip()}"
                    </div>
                    <div style="margin-top: 6px; font-size: 0.72rem; color: var(--text-muted);">
                        Indexed Knowledge Chunk: <code>{chunk_id}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
