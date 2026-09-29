"""
Sleek clinical console sidebar navigation, live system health indicator,
authenticated user profile card, and medical safety panel.
"""

from typing import Dict, Any
import streamlit as st
from frontend.api_client import api_client
from frontend.components.status import render_backend_indicator

NAV_DASHBOARD = "📊 Dashboard"
NAV_DOCUMENTS = "📄 Documents"
NAV_CHAT = "💬 Chat"
NAV_HISTORY = "🕘 History"


def render_sidebar() -> str:
    """
    Renders the clinical console sidebar with modern navigation, live health status,
    active user information, and international medical safety notices.
    Returns the selected page identifier.
    """
    with st.sidebar:
        # 1. Header Branding
        st.markdown(
            """
            <div style="margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="
                        background: linear-gradient(135deg, #0F766E, #0D9488);
                        width: 36px;
                        height: 36px;
                        border-radius: 8px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-size: 1.25rem;
                        font-weight: 800;
                        box-shadow: 0 2px 4px rgba(15, 118, 110, 0.25);
                    ">
                        +
                    </div>
                    <div>
                        <div style="font-size: 1.12rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em; line-height: 1.2;">
                            HealthAI Assistant
                        </div>
                        <div style="font-size: 0.72rem; color: #0F766E; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">
                            Research & Patient Assistance
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        # 2. Navigation Options
        is_auth = st.session_state.get("is_authenticated", False)
        raw_page = st.session_state.get("current_page", "Dashboard")
        # Normalize
        if "Doc" in raw_page:
            current_index = 1
        elif "Chat" in raw_page:
            current_index = 2
        elif "Hist" in raw_page:
            current_index = 3
        else:
            current_index = 0

        nav_options = [NAV_DASHBOARD, NAV_DOCUMENTS, NAV_CHAT, NAV_HISTORY]

        st.markdown(
            """
            <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                Navigation
            </div>
            """,
            unsafe_allow_html=True
        )

        selected_page = st.radio(
            "Application Navigation",
            options=nav_options,
            index=current_index,
            label_visibility="collapsed",
            key="sidebar_navigation_radio"
        )

        st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        # 3. Live System Status
        st.markdown(
            """
            <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                System Status
            </div>
            """,
            unsafe_allow_html=True
        )

        health = api_client.check_backend_health(timeout=2.0)
        is_connected = health.get("connected", False)
        render_backend_indicator(is_connected, version="0.5.0", url=api_client.base_url)

        if is_auth:
            st.markdown(
                """
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 8px 14px;
                    background-color: #ECFDF5;
                    border: 1px solid #A7F3D0;
                    border-radius: 8px;
                    color: #065F46;
                    font-size: 0.83rem;
                    font-weight: 600;
                    margin-top: 6px;
                ">
                    <span class="status-dot connected"></span>
                    <span>Authentication Active</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        # 4. Medical Safety & Scope
        st.markdown(
            """
            <div style="
                background-color: #FFFBEB;
                border: 1px solid #FDE68A;
                border-radius: 8px;
                padding: 12px 14px;
                margin-bottom: 16px;
            ">
                <div style="font-size: 0.76rem; font-weight: 700; color: #92400E; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;">
                    🛡️ Medical Safety & Scope
                </div>
                <div style="font-size: 0.77rem; color: #78350F; line-height: 1.45;">
                    This assistant provides <strong>informational and research assistance only</strong>.
                    <br/><br/>
                    • Does not provide medical diagnoses.<br/>
                    • Does not prescribe or alter medications.<br/>
                    • For emergencies, seek immediate local emergency medical assistance.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 5. Authenticated User Profile & Logout
        if is_auth:
            auth_user = st.session_state.get("authenticated_user") or {}
            user_email = auth_user.get("email") if isinstance(auth_user, dict) else "Active User"

            st.markdown(
                f"""
                <div style="
                    background-color: #F8FAFC;
                    border: 1px solid #E2E8F0;
                    border-radius: 8px;
                    padding: 10px 14px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 0.72rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">
                        Authenticated User
                    </div>
                    <div style="font-size: 0.84rem; font-weight: 700; color: #0F172A; word-break: break-all; margin-top: 2px;">
                        👤 {user_email}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("Sign Out", key="sidebar_sign_out_btn", use_container_width=True):
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

    return selected_page
