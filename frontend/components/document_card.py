"""
Clinical Document Card Component.
Renders an ingested document record with file metadata, status badge, and inspection triggers.
Supports both Light and Dark modes seamlessly using CSS variables.
"""

from typing import Dict, Any, Callable, Optional
import streamlit as st
from frontend.utils.helpers import format_timestamp, format_bytes, clean_filename
from frontend.components.status import render_status_badge


def render_document_card(
    doc: Dict[str, Any],
    on_select: Optional[Callable[[int], None]] = None,
    is_selected: bool = False
):
    """
    Renders an interactive card for an ingested medical document.
    """
    doc_id = doc.get("id", 0)
    filename = clean_filename(doc.get("filename") or "Clinical Document")
    pages = doc.get("num_pages") or doc.get("pages") or 1
    chunks = doc.get("num_chunks", 0)
    file_size = format_bytes(doc.get("file_size_bytes"))
    created_at = format_timestamp(doc.get("created_at"))
    status = (doc.get("status") or "completed").upper()

    border_color = "var(--teal-primary)" if is_selected else "var(--border-card)"
    bg_color = "var(--teal-light)" if is_selected else "var(--bg-card)"

    with st.container():
        st.markdown(
            f"""
            <div class="health-card" style="
                border: 1.5px solid {border_color};
                background-color: {bg_color};
                padding: 18px 22px;
                margin-bottom: 12px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div>
                        <div style="font-weight: 700; color: var(--text-primary); font-size: 1.05rem; display: flex; align-items: center; gap: 8px;">
                            <span>📄</span>
                            <span>{filename}</span>
                        </div>
                        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px;">
                            Ingested on {created_at} • Size: {file_size}
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 20px; align-items: center; margin-top: 10px; font-size: 0.86rem; color: var(--text-secondary);">
                    <span><strong>Pages:</strong> {pages}</span>
                    <span>•</span>
                    <span><strong>Chunks:</strong> {chunks}</span>
                    <span>•</span>
                    <span><strong>Status:</strong> <span style="font-weight: 700; color: var(--teal-primary);">{status}</span></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_badge, col_btn = st.columns([3, 1])
        with col_badge:
            render_status_badge(status)
        with col_btn:
            if on_select and st.button("Inspect Details →", key=f"btn_inspect_doc_{doc_id}", type="primary" if is_selected else "secondary", use_container_width=True):
                on_select(doc_id)
