"""
Modern Sticky Top Navigation Bar Component for HealthAI Assistant.

Matches the reference design:
┌──────────────────────────────────────────────────────────────────────────────┐
│ 🩺 HealthAI Assistant │ 🏠 Dashboard  📄 Documents  💬 Research Chat  🕘 History │ ☀️ Light │
│    RESEARCH & PATIENT  │                                            👤 ram11234 ▾ │
│    ASSISTANCE          │                                            ⇥ Sign Out   │
└──────────────────────────────────────────────────────────────────────────────┘

Architecture:
- Single floating rounded card container (height: 76px, radius: 24px)
- EXACTLY ONE SINGLE HORIZONTAL ROW on desktop (flex-wrap: nowrap)
- 3 logical sections:
  1. Brand (Logo badge + title + tagline + divider)
  2. Center Navigation (Dashboard, Documents, Research Chat, History)
  3. Right Actions (Divider + Theme + User + Sign Out)
- Subtle individual color accents and micro-interaction glows:
  - Dashboard: Teal glow
  - Documents: Blue glow
  - Research Chat: Purple glow
  - History: Orange glow
  - Theme: Amber pill (Light) / Cyan-blue pill (Dark)
  - User: Lavender pill with caret
  - Sign Out: Rose/red outline pill
- Mobile responsive (< 768px): Hamburger menu with accessible dropdown drawer
"""

from typing import Callable, Optional
import streamlit as st
from frontend.api_client import api_client

NAV_DASHBOARD = "Dashboard"
NAV_DOCUMENTS = "Documents"
NAV_CHAT = "Research Chat"
NAV_HISTORY = "History"

NAV_ITEMS = [NAV_DASHBOARD, NAV_DOCUMENTS, NAV_CHAT, NAV_HISTORY]


def normalize_page_name(page_name: Optional[str]) -> str:
    """Normalizes emoji-prefixed legacy names to standard page names."""
    if not page_name:
        return NAV_DASHBOARD
    clean = (
        page_name.replace("📊", "")
        .replace("📄", "")
        .replace("💬", "")
        .replace("🕘", "")
        .replace("🏠", "")
        .strip()
    )
    if "Chat" in clean:
        return NAV_CHAT
    if "Doc" in clean:
        return NAV_DOCUMENTS
    if "Hist" in clean:
        return NAV_HISTORY
    return NAV_DASHBOARD


def render_navbar(set_page_fn: Optional[Callable[[str], None]] = None) -> str:
    """
    Renders the modern sticky top navigation bar in ONE SINGLE HORIZONTAL ROW.
    Returns the currently selected page name.
    """
    # 1. State initialization
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "mobile_nav_open" not in st.session_state:
        st.session_state.mobile_nav_open = False

    current_page = normalize_page_name(st.session_state.get("current_page", NAV_DASHBOARD))
    is_dark = st.session_state.theme == "dark"

    auth_user = st.session_state.get("authenticated_user") or {}
    user_email = auth_user.get("email", "") if isinstance(auth_user, dict) else ""
    user_short = user_email.split("@")[0] if "@" in user_email else (user_email or "ram11234")

    def handle_navigate(dest_page: str):
        st.session_state.mobile_nav_open = False
        if set_page_fn:
            set_page_fn(dest_page)
        else:
            st.session_state.current_page = dest_page
        st.rerun()

    def handle_logout():
        api_client.logout()
        st.session_state.is_authenticated = False
        st.session_state.access_token = None
        st.session_state.authenticated_user = None
        st.session_state.active_conversation_id = None
        st.session_state.selected_document_id = None
        st.session_state.selected_history_conv_id = None
        st.session_state.mobile_nav_open = False
        if "chat_messages" in st.session_state:
            del st.session_state["chat_messages"]
        st.rerun()

    # 2. Render Single Horizontal Container with 3 flex groups
    col_brand, col_nav, col_actions = st.columns([3.2, 5.2, 3.8])

    # SECTION 1: Brand Logo, Title, Subtitle, and Divider
    with col_brand:
        st.markdown(
            f"""
            <div id="healthai-navbar-root" class="navbar-brand-wrapper">
                <div class="brand-section">
                    <div class="brand-icon-badge">🩺</div>
                    <div class="brand-title-group">
                        <span class="brand-name">HealthAI Assistant</span>
                        <span class="brand-subtitle">RESEARCH & PATIENT ASSISTANCE</span>
                    </div>
                </div>
                <div class="navbar-divider"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # SECTION 2: Center Navigation Pills (Dashboard, Documents, Research Chat, History)
    with col_nav:
        st.markdown('<div id="navbar-nav-group"></div>', unsafe_allow_html=True)

        is_dash_active = current_page == NAV_DASHBOARD
        if st.button(
            "🏠 Dashboard",
            key="top_nav_dashboard",
            type="primary" if is_dash_active else "secondary",
            help="Overview metrics, quick actions, and recent activity",
        ):
            handle_navigate(NAV_DASHBOARD)

        is_docs_active = current_page == NAV_DOCUMENTS
        if st.button(
            "📄 Documents",
            key="top_nav_documents",
            type="primary" if is_docs_active else "secondary",
            help="Upload and manage healthcare reference documents",
        ):
            handle_navigate(NAV_DOCUMENTS)

        is_chat_active = current_page == NAV_CHAT
        if st.button(
            "💬 Research Chat",
            key="top_nav_chat",
            type="primary" if is_chat_active else "secondary",
            help="Ask questions grounded in your private documents",
        ):
            handle_navigate(NAV_CHAT)

        is_hist_active = current_page == NAV_HISTORY
        if st.button(
            "🕘 History",
            key="top_nav_history",
            type="primary" if is_hist_active else "secondary",
            help="Review previous research consultations",
        ):
            handle_navigate(NAV_HISTORY)

    # SECTION 3: Right Actions Group (Divider, Theme Toggle, User Profile, Sign Out)
    with col_actions:
        st.markdown(
            """
            <div id="navbar-actions-group" class="navbar-actions-anchor">
                <div class="navbar-divider"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        theme_icon = "☀️ Light" if not is_dark else "🌙 Dark"
        theme_help = "Switch to Dark Mode" if not is_dark else "Switch to Light Mode"
        if st.button(theme_icon, key="top_nav_theme_toggle", help=theme_help):
            st.session_state.theme = "dark" if not is_dark else "light"
            st.rerun()

        st.markdown(
            f"""
            <div class="nav-user-pill" title="{user_email}">
                <span class="user-avatar-icon">👤</span>
                <span class="user-name-text">{user_short}</span>
                <span class="user-caret">▾</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("⇥ Sign Out", key="top_nav_logout_btn", help="Sign out of your session"):
            handle_logout()

    # 3. Mobile Navigation Drawer (conditionally expanded on mobile viewports)
    if st.session_state.get("mobile_nav_open", False):
        with st.container():
            st.markdown(
                """
                <div class="mobile-nav-drawer">
                    <div style="font-size: 0.78rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                        Navigation Menu
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if st.button("🏠 Dashboard", key="m_nav_dash", use_container_width=True):
                    handle_navigate(NAV_DASHBOARD)
                if st.button("📄 Documents", key="m_nav_docs", use_container_width=True):
                    handle_navigate(NAV_DOCUMENTS)
            with m_col2:
                if st.button("💬 Research Chat", key="m_nav_chat", use_container_width=True):
                    handle_navigate(NAV_CHAT)
                if st.button("🕘 History", key="m_nav_hist", use_container_width=True):
                    handle_navigate(NAV_HISTORY)
            if st.button("⇥ Sign Out", key="m_nav_logout", use_container_width=True):
                handle_logout()

    return current_page
