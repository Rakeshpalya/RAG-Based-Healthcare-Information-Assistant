"""
Clinical User Settings and System Observability Page.

Provides:
- User Profile & Authentication Account Details
- Clinical Preferences (Theme, Streaming SSE, Evidence retrieval thresholds)
- Privacy, Security & Compliance Information (HIPAA / Non-training disclaimers)
- Admin & Observability Telemetry (Request volume, error rate, cache hit rate,
  safety interceptions, LLM failure rate, and p50/p95/p99 latencies)
Strict Security: NEVER exposes secrets, bearer tokens, API keys, PHI, or raw queries.
"""

from typing import Callable, Optional
import streamlit as st
from frontend.api_client import api_client


def render_settings_page(set_page_fn: Optional[Callable[[str], None]] = None):
    """
    Renders the User Settings & System Telemetry view.
    """
    # 1. Header
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">
                Configuration & System Health
            </div>
            <h1 style="color: var(--text-primary); font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                User Settings & Telemetry
            </h1>
            <p style="color: var(--text-secondary); font-size: 1rem; margin-top: 6px; margin-bottom: 0; line-height: 1.5;">
                Manage your clinical account preferences, security boundaries, and monitor real-time system performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_profile, tab_pref, tab_telemetry, tab_privacy = st.tabs([
        "👤 Account & Identity",
        "⚙️ Clinical Preferences",
        "📊 Observability & Telemetry",
        "🔒 Privacy & Compliance",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: ACCOUNT & IDENTITY
    # --------------------------------------------------------------------------
    with tab_profile:
        auth_user = st.session_state.get("authenticated_user") or {}
        user_email = auth_user.get("email", "researcher@healthai.local")
        user_id = auth_user.get("id", "N/A")
        is_active = auth_user.get("is_active", True)
        is_verified = auth_user.get("is_verified", True)

        st.markdown(
            f"""
            <div class="health-card" style="margin-top: 14px; margin-bottom: 20px;">
                <h3 style="margin: 0 0 12px 0; font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">
                    Identity & Authentication Profile
                </h3>
                <div style="font-size: 0.92rem; color: var(--text-secondary); line-height: 1.9;">
                    • <strong>Account Email:</strong> <code>{user_email}</code><br/>
                    • <strong>User ID:</strong> <code>{user_id}</code><br/>
                    • <strong>Account Status:</strong> <span style="color: var(--emerald-success); font-weight: 700;">{'Active' if is_active else 'Suspended'}</span><br/>
                    • <strong>Verification Status:</strong> <span style="color: var(--emerald-success); font-weight: 700;">{'Verified' if is_verified else 'Pending'}</span><br/>
                    • <strong>Session State:</strong> <span style="color: var(--teal-primary); font-weight: 700;">Authenticated (JWT Active)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Session Controls")
        col_out1, col_out2 = st.columns([1.5, 3])
        with col_out1:
            if st.button("⇥ Terminate Session (Sign Out)", key="settings_logout_btn", type="primary", use_container_width=True):
                api_client.logout()
                st.session_state.is_authenticated = False
                st.session_state.access_token = None
                st.session_state.authenticated_user = None
                st.session_state.active_conversation_id = None
                st.session_state.selected_document_id = None
                st.session_state.selected_history_conv_id = None
                if "chat_messages" in st.session_state:
                    del st.session_state["chat_messages"]
                st.rerun()

    # --------------------------------------------------------------------------
    # TAB 2: CLINICAL PREFERENCES
    # --------------------------------------------------------------------------
    with tab_pref:
        st.markdown(
            """
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin-top: 14px; margin-bottom: 12px;">
                Application Appearance & Execution Modes
            </div>
            """,
            unsafe_allow_html=True
        )

        # 1. Theme toggle
        is_dark = st.session_state.get("theme") == "dark"
        theme_choice = st.radio(
            "Interface Color Theme",
            options=["Light Theme (Clean Clinical White)", "Dark Theme (Low-Glare Slate)"],
            index=1 if is_dark else 0,
            key="radio_theme_pref"
        )
        new_theme = "dark" if "Dark" in theme_choice else "light"
        if new_theme != st.session_state.get("theme"):
            st.session_state.theme = new_theme
            st.rerun()

        st.markdown("---")

        # 2. Streaming responses toggle
        if "use_streaming" not in st.session_state:
            st.session_state.use_streaming = True

        st.session_state.use_streaming = st.checkbox(
            "Enable Real-Time SSE Token Streaming",
            value=st.session_state.use_streaming,
            help="Streams answer tokens incrementally using Server-Sent Events after full safety validation.",
            key="chk_use_streaming"
        )

        st.markdown("---")

        # 3. Default retrieval thresholds
        st.markdown("**Default RAG Retrieval Parameters:**")
        pref_col1, pref_col2 = st.columns(2)
        with pref_col1:
            st.slider(
                "Default Top-K Passages",
                min_value=1,
                max_value=10,
                value=5,
                key="pref_default_top_k",
                help="Default number of clinical chunks retrieved for query synthesis."
            )
        with pref_col2:
            st.slider(
                "Default Relevance Cutoff",
                min_value=0.10,
                max_value=0.90,
                value=0.25,
                step=0.05,
                key="pref_default_threshold",
                help="Default cosine similarity threshold required for document evidence."
            )

    # --------------------------------------------------------------------------
    # TAB 3: OBSERVABILITY & SYSTEM TELEMETRY (ADMIN / METRICS)
    # --------------------------------------------------------------------------
    with tab_telemetry:
        metrics = api_client.get_observability_metrics()

        st.markdown(
            """
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin-top: 14px; margin-bottom: 6px;">
                Production Telemetry & Reliability Counters
            </div>
            <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 16px;">
                Aggregated system health and latency distribution extracted safely from Prometheus instrumentation.
                No Personal Health Information (PHI), private tokens, or raw medical queries are ever stored or displayed.
            </p>
            """,
            unsafe_allow_html=True
        )

        # Row 1: System Status & Volume
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            b_status = metrics.get("backend_status", "offline")
            b_color = "var(--emerald-success)" if b_status == "healthy" else "var(--rose-danger)"
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">System Health</div>
                    <div class="stat-value" style="font-size: 1.5rem; color: {b_color};">
                        ● {b_status.capitalize()}
                    </div>
                    <div class="stat-subtext">Backend Service Status</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Total Requests</div>
                    <div class="stat-value">{metrics.get('request_count', 0)}</div>
                    <div class="stat-subtext">Cumulative Inquiries</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Cache Hit Ratio</div>
                    <div class="stat-value">{metrics.get('cache_hit_rate', 0.0)}%</div>
                    <div class="stat-subtext">{metrics.get('cache_hits', 0)} hits / {metrics.get('cache_misses', 0)} misses</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Safety Interceptions</div>
                    <div class="stat-value" style="color: var(--amber-warning);">{metrics.get('safety_interceptions', 0)}</div>
                    <div class="stat-subtext">Emergency/Policy Blocks</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Row 2: Latency & Reliability
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">p50 Latency</div>
                    <div class="stat-value">{metrics.get('p50_latency_ms', 0.0)} ms</div>
                    <div class="stat-subtext">Median Response Time</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with l2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">p95 Latency</div>
                    <div class="stat-value">{metrics.get('p95_latency_ms', 0.0)} ms</div>
                    <div class="stat-subtext">95th Percentile</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with l3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">p99 Latency</div>
                    <div class="stat-value">{metrics.get('p99_latency_ms', 0.0)} ms</div>
                    <div class="stat-subtext">Tail Latency</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with l4:
            cb_state = metrics.get("circuit_breaker_state", "closed")
            cb_color = "var(--emerald-success)" if cb_state == "closed" else "var(--rose-danger)"
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Circuit Breaker</div>
                    <div class="stat-value" style="font-size: 1.5rem; color: {cb_color};">
                        ● {cb_state.upper()}
                    </div>
                    <div class="stat-subtext">Cascading Failure Guard</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Refresh telemetry button
        if st.button("🔄 Refresh Telemetry Metrics", key="btn_refresh_metrics", type="secondary"):
            st.rerun()

    # --------------------------------------------------------------------------
    # TAB 4: PRIVACY & COMPLIANCE
    # --------------------------------------------------------------------------
    with tab_privacy:
        st.markdown(
            """
            <div class="health-card" style="margin-top: 14px; margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0; font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">
                    🔒 Privacy, Data Isolation & Clinical Boundaries
                </h3>
                <div style="font-size: 0.92rem; color: var(--text-secondary); line-height: 1.8;">
                    <p>
                        <strong>1. Strict Multi-Tenant Isolation:</strong>
                        All uploaded medical documents, FAISS vector embeddings, conversation sessions,
                        and cache keys are cryptographically and relationally bound to your authenticated user identity.
                        No other user can access, search, or view your uploaded materials.
                    </p>
                    <p>
                        <strong>2. Zero Model Training Guarantee:</strong>
                        Your clinical documents, queries, and conversational transcripts are never utilized to fine-tune
                        or train foundation models. Inquiries are processed transiently through zero-data-retention APIs.
                    </p>
                    <p>
                        <strong>3. Observability Redaction:</strong>
                        Prometheus metrics and system telemetry completely redact Personal Health Information (PHI),
                        user identities, and query text. Only aggregated latency counters and low-cardinality status codes
                        are collected.
                    </p>
                    <p>
                        <strong>4. Clinical Disclaimer & Scope:</strong>
                        HealthAI Assistant is an educational and clinical research tool designed to retrieve evidence
                        strictly from reference documentation. It does not possess medical licensure, does not provide
                        clinical diagnosis, and does not prescribe pharmaceuticals. For all medical conditions, consult
                        a licensed healthcare practitioner.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
