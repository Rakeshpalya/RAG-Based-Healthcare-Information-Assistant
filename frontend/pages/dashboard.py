"""
Modern Clinical Assistant Dashboard Page.

Features:
- Personalized Welcome Hero with direct primary actions
- Live statistics cards connected to backend data
- 4 Quick Action cards with verified working routes (Zero Dead Buttons)
- Recent Ingested Documents feed with instant inspection trigger
- Recent Consultations feed with instant session resumption
- Clear Medical Safety Notice
- Complete Light & Dark mode support
"""

from typing import Callable, Optional
from datetime import datetime
import streamlit as st
from frontend.api_client import api_client
from frontend.components.document_card import render_document_card
from frontend.utils.helpers import format_timestamp


def _get_time_greeting() -> str:
    """Returns a time-sensitive greeting."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


def render_dashboard_page(set_page_fn: Optional[Callable[[str], None]] = None):
    """
    Renders the redesigned healthcare research assistant dashboard.
    """
    auth_user = st.session_state.get("authenticated_user") or {}
    user_email = auth_user.get("email") if isinstance(auth_user, dict) else ""
    user_display = user_email.split("@")[0].capitalize() if user_email and "@" in user_email else "Researcher"
    greeting = _get_time_greeting()

    # 1. Fetch live data from backend
    health = api_client.check_backend_health(timeout=2.0)
    is_connected = health.get("connected", False)

    documents = api_client.list_documents(limit=10)
    conversations = api_client.list_conversations(limit=10)

    total_docs = len(documents)
    total_conversations = len(conversations)
    total_chunks = sum(int(doc.get("num_chunks") or 0) for doc in documents)

    # 2. HERO / WELCOME SECTION
    with st.container():
        st.markdown(
            f"""
            <div class="hero-welcome-card">
                <div class="hero-tag">
                    <span>🩺</span>
                    <span>{greeting}, {user_display} 👋</span>
                </div>
                <h1 class="hero-heading">
                    Welcome to HealthAI Assistant
                </h1>
                <p class="hero-description">
                    Research healthcare documents, understand medical information, and ask grounded questions using your private knowledge base.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_hero_act1, col_hero_act2, _ = st.columns([1.5, 1.5, 3])
        with col_hero_act1:
            if st.button("📄 Upload Document", key="hero_btn_upload", type="primary", use_container_width=True):
                if set_page_fn:
                    set_page_fn("Documents")
                else:
                    st.session_state.current_page = "Documents"
                st.rerun()

        with col_hero_act2:
            if st.button("💬 Start Research", key="hero_btn_chat", type="secondary", use_container_width=True):
                if set_page_fn:
                    set_page_fn("Research Chat")
                else:
                    st.session_state.current_page = "Research Chat"
                st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3. LIVE STATISTICS CARDS
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Documents</div>
                <div class="stat-value">{total_docs}</div>
                <div class="stat-subtext">📄 Ingested Reference PDFs</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Conversations</div>
                <div class="stat-value">{total_conversations}</div>
                <div class="stat-subtext">💬 Research Sessions</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Knowledge Chunks</div>
                <div class="stat-value">{total_chunks}</div>
                <div class="stat-subtext">📚 Indexed Segments</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        status_label = "Healthy" if is_connected else "Offline"
        dot_class = "connected" if is_connected else "disconnected"
        status_color = "var(--emerald-success)" if is_connected else "var(--rose-danger)"
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">System Status</div>
                <div class="stat-value" style="font-size: 1.5rem; display: flex; align-items: center; gap: 8px;">
                    <span class="status-dot {dot_class}"></span>
                    <span>Backend</span>
                </div>
                <div class="stat-subtext" style="color: {status_color};">
                    ● {status_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 4. QUICK AI QUESTION (5.2 Requirement)
    st.markdown(
        """
        <div class="health-card" style="margin-bottom: 24px; padding: 1.5rem 1.8rem;">
            <div style="font-size: 0.78rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                Instant Knowledge Base Query
            </div>
            <h3 style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary); margin: 0 0 6px 0;">
                💡 Quick Clinical AI Question
            </h3>
            <p style="font-size: 0.88rem; color: var(--text-secondary); margin: 0 0 14px 0;">
                Ask any clinical inquiry grounded directly in your uploaded reference documents and clinical guidelines.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("form_quick_question", clear_on_submit=True):
        col_q_in, col_q_btn = st.columns([5, 1])
        with col_q_in:
            quick_q = st.text_input(
                "Clinical Question",
                placeholder="e.g., What are the primary diagnostic criteria for hypertension?",
                label_visibility="collapsed",
                key="dash_quick_question_input"
            )
        with col_q_btn:
            submitted_q = st.form_submit_button("Ask AI →", type="primary", use_container_width=True)

        if submitted_q and quick_q.strip():
            st.session_state.pending_quick_query = quick_q.strip()
            if set_page_fn:
                set_page_fn("Research Chat")
            else:
                st.session_state.current_page = "Research Chat"
            st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 5. QUICK ACTIONS (4 Cards, All Functional)
    st.markdown(
        """
        <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin-bottom: 12px;">
            ⚡ Quick Actions
        </div>
        """,
        unsafe_allow_html=True
    )

    qa1, qa2, qa3, qa4 = st.columns(4)

    with qa1:
        st.markdown(
            """
            <div class="action-card">
                <div>
                    <div class="action-icon-title">
                        <span style="font-size: 1.3rem;">📄</span>
                        <h4 class="action-title">Upload Document</h4>
                    </div>
                    <p class="action-desc">
                        Add a PDF to your private healthcare knowledge base.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Upload Document →", key="qa_upload_btn", type="primary", use_container_width=True):
            if set_page_fn:
                set_page_fn("Documents")
            else:
                st.session_state.current_page = "Documents"
            st.rerun()

    with qa2:
        st.markdown(
            """
            <div class="action-card">
                <div>
                    <div class="action-icon-title">
                        <span style="font-size: 1.3rem;">💬</span>
                        <h4 class="action-title">Research Documents</h4>
                    </div>
                    <p class="action-desc">
                        Ask questions and receive answers grounded in your sources.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Start Research →", key="qa_research_btn", type="secondary", use_container_width=True):
            if set_page_fn:
                set_page_fn("Research Chat")
            else:
                st.session_state.current_page = "Research Chat"
            st.rerun()

    with qa3:
        st.markdown(
            """
            <div class="action-card">
                <div>
                    <div class="action-icon-title">
                        <span style="font-size: 1.3rem;">📚</span>
                        <h4 class="action-title">Explore Knowledge</h4>
                    </div>
                    <p class="action-desc">
                        Review and inspect your uploaded healthcare references.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View Documents →", key="qa_explore_btn", type="secondary", use_container_width=True):
            if set_page_fn:
                set_page_fn("Documents")
            else:
                st.session_state.current_page = "Documents"
            st.rerun()

    with qa4:
        st.markdown(
            """
            <div class="action-card">
                <div>
                    <div class="action-icon-title">
                        <span style="font-size: 1.3rem;">🕘</span>
                        <h4 class="action-title">Consultation History</h4>
                    </div>
                    <p class="action-desc">
                        Continue or review your previous research sessions.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View History →", key="qa_history_btn", type="secondary", use_container_width=True):
            if set_page_fn:
                set_page_fn("History")
            else:
                st.session_state.current_page = "History"
            st.rerun()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 5. RECENT ACTIVITY SPLIT: Documents & Research Consultations
    col_docs, col_convs = st.columns([3, 2])

    with col_docs:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: var(--text-primary);">
                    Recent Ingested Documents
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not documents:
            st.info("No documents currently ingested. Click 'Upload Document' to add your first healthcare PDF.")
        else:
            for doc in documents[:3]:
                def _handle_select(doc_id=doc.get("id")):
                    st.session_state.selected_document_id = doc_id
                    if set_page_fn:
                        set_page_fn("Documents")
                    else:
                        st.session_state.current_page = "Documents"
                    st.rerun()

                render_document_card(doc, on_select=_handle_select)

    with col_convs:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: var(--text-primary);">
                    Recent Consultations
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not conversations:
            st.info("No prior research sessions recorded. Head to Research Chat to begin your first inquiry.")
        else:
            for conv in conversations[:4]:
                conv_id = conv.get("id")
                title = conv.get("title", "Clinical Consultation")
                updated_at = format_timestamp(conv.get("updated_at") or conv.get("created_at"))

                with st.container():
                    st.markdown(
                        f"""
                        <div class="health-card" style="padding: 12px 16px; margin-bottom: 10px;">
                            <div style="font-weight: 700; font-size: 0.88rem; color: var(--text-primary);">
                                💬 {title}
                            </div>
                            <div style="font-size: 0.76rem; color: var(--text-muted); margin-top: 3px;">
                                {updated_at}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button("Resume Session →", key=f"btn_dash_resume_{conv_id}", use_container_width=True):
                        # Load messages and switch to Chat
                        hist_messages = api_client.get_conversation_messages(conv_id)
                        st.session_state.chat_messages = [
                            {
                                "sender": m.get("sender", "user"),
                                "text": m.get("text", ""),
                                "agent_type": m.get("agent_type"),
                                "sources": m.get("citations") if isinstance(m.get("citations"), list) else [],
                                "retrieval_status": "success",
                            }
                            for m in hist_messages
                        ]
                        st.session_state.active_conversation_id = conv_id
                        if set_page_fn:
                            set_page_fn("Research Chat")
                        else:
                            st.session_state.current_page = "Research Chat"
                        st.rerun()

    # 6. MEDICAL SAFETY NOTICE
    st.markdown(
        """
        <div class="medical-safety-banner">
            ⚠️ <strong>Medical Notice:</strong> Answers are synthesized from your uploaded reference documents.
            This tool provides research and informational assistance only. It does not provide medical diagnoses or prescribe medications.
            For emergencies, seek immediate local emergency medical assistance.
        </div>
        """,
        unsafe_allow_html=True
    )
