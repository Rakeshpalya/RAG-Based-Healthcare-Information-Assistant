"""
Clinical Document Management & Ingestion Page.

Allows authenticated users to upload healthcare research or personal documents
to build an isolated knowledge base, indexed into FAISS and PostgreSQL.
Includes real-time processing feedback and interactive document inspection.
"""

from typing import Callable, Optional
import streamlit as st
from frontend.api_client import api_client
from frontend.components.document_card import render_document_card
from frontend.components.status import render_status_badge
from frontend.utils.helpers import format_bytes, format_timestamp

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def render_documents_page(set_page_fn: Optional[Callable[[str], None]] = None):
    """
    Renders the clinical document ingestion and repository inspection interface.
    """
    # 1. Header
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">
                Healthcare Knowledge Base
            </div>
            <h1 style="color: var(--text-primary); font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                Your Healthcare Documents
            </h1>
            <p style="color: var(--text-secondary); font-size: 1rem; margin-top: 6px; margin-bottom: 0; line-height: 1.5;">
                Upload healthcare research or clinical records to build your private, isolated knowledge base.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_upload, tab_repo = st.tabs(["📤 Upload Healthcare Document", "📁 Document Repository & Knowledge"])

    # --------------------------------------------------------------------------
    # TAB 1: UPLOAD HEALTHCARE DOCUMENT
    # --------------------------------------------------------------------------
    with tab_upload:
        st.markdown(
            """
            <div class="health-card" style="margin-top: 14px; margin-bottom: 16px; text-align: center; padding: 2rem;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📄</div>
                <h3 style="font-size: 1.35rem; font-weight: 800; color: var(--text-primary); margin: 0 0 6px 0;">
                    Upload Healthcare Document
                </h3>
                <p style="font-size: 0.92rem; color: var(--text-secondary); margin: 0 0 12px 0;">
                    Add clinical trials, guidelines, or medical notes to your private knowledge base.
                </p>
                <div style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; font-weight: 700; color: var(--teal-primary); background-color: var(--teal-light); border: 1px solid var(--teal-border); padding: 4px 12px; border-radius: var(--radius-full);">
                    <span>🔒 Private & User-Isolated • Maximum File Size: 10 MB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a healthcare PDF document",
            type=["pdf"],
            help="Select a clinical trial paper, discharge summary, or medical guideline PDF.",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_size = len(file_bytes)
            filename = uploaded_file.name

            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                st.markdown(f"**Filename:** `{filename}`")
            with col_meta2:
                st.markdown(f"**File Size:** `{format_bytes(file_size)}`")

            # Validate size
            if file_size > MAX_FILE_SIZE_BYTES:
                st.error(f"File size ({format_bytes(file_size)}) exceeds the maximum allowed limit of 10 MB.")
            elif file_size == 0:
                st.error("The selected file is empty (0 bytes).")
            else:
                col_up_btn, col_clear_btn = st.columns([2, 1])
                with col_up_btn:
                    do_upload = st.button("Upload & Ingest Document", type="primary", use_container_width=True)
                with col_clear_btn:
                    if st.button("Cancel / Clear", type="secondary", use_container_width=True):
                        st.rerun()

                if do_upload:
                    prog_bar = st.progress(20, text="Step 1/3: Validating PDF structure and transmitting file...")
                    res = api_client.upload_document(uploaded_file)

                    if res.get("success"):
                        prog_bar.progress(70, text="Step 2/3: Extracting text, generating 384-d embeddings & indexing FAISS...")
                        data = res.get("data", {})
                        doc_id = data.get("document_id")
                        doc_filename = data.get("filename", filename)
                        num_pages = data.get("pages") or data.get("number_of_pages", 1)
                        num_chunks = data.get("num_chunks", 0)
                        extracted_text = data.get("extracted_text", "")

                        if doc_id:
                            st.session_state.selected_document_id = doc_id

                        prog_bar.progress(100, text="Step 3/3: Document ingestion and indexing complete!")
                        st.success("✓ Ingestion Completed: Document successfully indexed into your private healthcare knowledge base!")

                        # Structured Ingestion Summary Card
                        st.markdown(
                            f"""
                            <div class="health-card" style="
                                border: 1.5px solid var(--emerald-success);
                                background-color: var(--emerald-bg);
                                padding: 20px 24px;
                                margin-top: 14px;
                                margin-bottom: 14px;
                            ">
                                <h4 style="margin: 0 0 10px 0; color: var(--emerald-text); font-size: 1.15rem; font-weight: 800;">
                                    ✓ Ingestion Summary
                                </h4>
                                <div style="font-size: 0.92rem; color: var(--text-primary); line-height: 1.8;">
                                    • <strong>Document ID:</strong> <code>{doc_id}</code><br/>
                                    • <strong>Filename:</strong> {doc_filename}<br/>
                                    • <strong>Status:</strong> <span style="font-weight: 700; color: var(--emerald-success);">Completed</span><br/>
                                    • <strong>Pages:</strong> {num_pages}<br/>
                                    • <strong>Knowledge Chunks:</strong> {num_chunks}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        col_act_chat, _ = st.columns([2, 2])
                        with col_act_chat:
                            if st.button("💬 Ask Questions About This Document →", type="primary", use_container_width=True):
                                if set_page_fn:
                                    set_page_fn("Research Chat")
                                else:
                                    st.session_state.current_page = "Research Chat"
                                st.rerun()

                        if extracted_text:
                            with st.expander("📄 View Extracted Text Preview", expanded=False):
                                st.text_area(
                                    "Extracted Content",
                                    value=extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""),
                                    height=200,
                                    disabled=True
                                )
                    else:
                        err_msg = res.get("error", "Unable to process this document. Please verify that it is a valid PDF.")
                        status_placeholder.empty()
                        st.error(f"Processing failed: {err_msg}")

    # --------------------------------------------------------------------------
    # TAB 2: DOCUMENT REPOSITORY & KNOWLEDGE
    # --------------------------------------------------------------------------
    with tab_repo:
        documents = api_client.list_documents(limit=100)
        if not documents:
            st.info("No documents currently stored in your repository. Use the Upload tab to add healthcare references.")
        else:
            col_list, col_details = st.columns([1.5, 2])

            with col_list:
                st.markdown(
                    f"""
                    <div style="font-size: 0.84rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">
                        Stored Documents ({len(documents)})
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                selected_id = st.session_state.get("selected_document_id") or documents[0].get("id")

                for doc in documents:
                    d_id = doc.get("id")
                    is_active = d_id == selected_id

                    def _handle_select(doc_id=d_id):
                        st.session_state.selected_document_id = doc_id
                        st.rerun()

                    render_document_card(doc, on_select=_handle_select, is_selected=is_active)

            with col_details:
                selected_id = st.session_state.get("selected_document_id") or documents[0].get("id")
                active_doc = next((d for d in documents if d.get("id") == selected_id), documents[0])

                if active_doc:
                    d_id = active_doc.get("id")
                    fname = active_doc.get("filename", "Clinical Document")
                    pages = active_doc.get("num_pages") or active_doc.get("pages") or 1
                    chunks = active_doc.get("num_chunks", 0)
                    created_at = format_timestamp(active_doc.get("created_at"))
                    size_bytes = format_bytes(active_doc.get("file_size_bytes"))

                    doc_status = active_doc.get("status", "processed").capitalize()
                    st.markdown(
                        f"""
                        <div class="health-card" style="margin-bottom: 16px;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em;">
                                Document Inspector
                            </div>
                            <h2 style="margin: 4px 0 8px 0; font-size: 1.35rem; font-weight: 800; color: var(--text-primary);">
                                {fname}
                            </h2>
                            <div style="font-size: 0.86rem; color: var(--text-secondary); line-height: 1.8;">
                                • <strong>Document ID:</strong> <code>{d_id}</code><br/>
                                • <strong>Status:</strong> <span style="color: var(--emerald-success); font-weight: 700;">● {doc_status}</span><br/>
                                • <strong>Uploaded:</strong> {created_at}<br/>
                                • <strong>File Size:</strong> {size_bytes}<br/>
                                • <strong>Pages:</strong> {pages} • <strong>Indexed Chunks:</strong> {chunks}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    col_doc_act1, col_doc_act2 = st.columns([3, 1])
                    with col_doc_act1:
                        if st.button("💬 Research This Document in Chat →", key="btn_chat_with_doc", type="primary", use_container_width=True):
                            if set_page_fn:
                                set_page_fn("Research Chat")
                            else:
                                st.session_state.current_page = "Research Chat"
                            st.rerun()

                    with col_doc_act2:
                        if st.button("🗑️ Delete", key=f"btn_del_doc_{d_id}", type="secondary", use_container_width=True, help="Delete this document"):
                            st.session_state[f"confirm_delete_doc_{d_id}"] = True
                            st.rerun()

                    if st.session_state.get(f"confirm_delete_doc_{d_id}", False):
                        st.markdown(
                            f"""
                            <div class="history-danger-box">
                                <strong>⚠️ Confirm Deletion:</strong> Are you sure you want to permanently delete <code>{fname}</code>?<br/>
                                This will remove its metadata and knowledge chunks from your private repository.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        c_yes, c_no = st.columns(2)
                        with c_yes:
                            if st.button("Yes, Delete Document", key=f"btn_confirm_del_{d_id}", type="primary", use_container_width=True):
                                del_res = api_client.delete_document(d_id)
                                st.session_state[f"confirm_delete_doc_{d_id}"] = False
                                if del_res.get("success"):
                                    st.session_state.selected_document_id = None
                                    st.success(f"Document '{fname}' deleted successfully.")
                                else:
                                    st.error(del_res.get("error", "Failed to delete document."))
                                st.rerun()
                        with c_no:
                            if st.button("Cancel", key=f"btn_cancel_del_{d_id}", type="secondary", use_container_width=True):
                                st.session_state[f"confirm_delete_doc_{d_id}"] = False
                                st.rerun()

                    # Fetch document details including chunks
                    doc_details = api_client.get_document(d_id)
                    chunks_data = doc_details.get("chunks", [])

                    if chunks_data:
                        st.markdown(
                            f"""
                            <div style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin: 16px 0 8px 0;">
                                Indexed Chunks ({len(chunks_data)})
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        for c in chunks_data[:5]:
                            c_idx = c.get("chunk_index", 0)
                            c_text = c.get("text", "")
                            with st.expander(f"Chunk #{c_idx} (Chars: {len(c_text)})", expanded=False):
                                st.write(c_text)
