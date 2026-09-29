import os
import sys
from pathlib import Path

# Ensure root workspace directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv

# Ensure environment variables are loaded
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()

import streamlit as st
from frontend.api_client import api_client
from frontend.components.navbar import (
    render_navbar,
    normalize_page_name,
    NAV_DASHBOARD,
    NAV_DOCUMENTS,
    NAV_CHAT,
    NAV_HISTORY,
)
from frontend.pages.auth import render_auth_page
from frontend.pages.dashboard import render_dashboard_page
from frontend.pages.documents import render_documents_page
from frontend.pages.history import render_history_page
from frontend.pages.chat import render_chat_page

# Configure Streamlit App Layout & Metadata (Clean SaaS full-width layout)
st.set_page_config(
    page_title="HealthAI Assistant | Clinical Research & Patient Assistance",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)

# Session State Initialization
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if "current_page" not in st.session_state:
    st.session_state.current_page = NAV_DASHBOARD

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None

if "selected_document_id" not in st.session_state:
    st.session_state.selected_document_id = None

if "selected_history_conv_id" not in st.session_state:
    st.session_state.selected_history_conv_id = None

# Synchronize APIClient bearer token from Streamlit session state
api_client.set_token(st.session_state.get("access_token"))

# Load and inject Clinical SaaS CSS Design System
styles_path = Path(__file__).resolve().parent / "assets" / "styles.css"
if styles_path.exists():
    with open(styles_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Dynamic Theme Override for Dark Mode
is_dark = st.session_state.get("theme") == "dark"
if is_dark:
    st.markdown(
        """
        <style>
        :root {
            --bg-app: #0B1120 !important;
            --bg-surface: #1E293B !important;
            --bg-surface-subtle: #0F172A !important;
            --bg-card: #1E293B !important;
            --bg-card-hover: #243248 !important;
            --text-primary: #F8FAFC !important;
            --text-secondary: #CBD5E1 !important;
            --text-muted: #94A3B8 !important;
            --border-subtle: #334155 !important;
            --border-card: #334155 !important;
            --border-hover: #14B8A6 !important;
            --teal-primary: #14B8A6 !important;
            --teal-hover: #2DD4BF !important;
            --teal-light: rgba(20, 184, 166, 0.16) !important;
            --teal-border: rgba(45, 212, 191, 0.4) !important;
            --nav-bg: rgba(15, 23, 42, 0.94) !important;
            --nav-border: #334155 !important;
            --nav-pill-active: rgba(20, 184, 166, 0.22) !important;
            --nav-pill-active-text: #2DD4BF !important;
            --nav-pill-border: rgba(45, 212, 191, 0.5) !important;
        }
        .stApp {
            background-color: #0B1120 !important;
            color: #F8FAFC !important;
        }
        div.stButton > button[kind="secondary"],
        button[data-testid="baseButton-secondary"] {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border-color: #334155 !important;
        }
        div.stButton > button[kind="secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #243248 !important;
            border-color: #14B8A6 !important;
            color: #14B8A6 !important;
        }
        div[data-testid="stMetric"] {
            background-color: #1E293B !important;
            border-color: #334155 !important;
        }
        div[data-testid="stExpander"] {
            background-color: #1E293B !important;
            border-color: #334155 !important;
        }
        div.st-key-top_nav_theme_toggle button {
            background: rgba(56, 189, 248, 0.15) !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            color: #38BDF8 !important;
        }
        .nav-user-pill {
            background-color: rgba(139, 92, 246, 0.18) !important;
            border-color: rgba(139, 92, 246, 0.45) !important;
            color: #DDD6FE !important;
        }
        div.st-key-top_nav_logout_btn button {
            background: rgba(239, 68, 68, 0.14) !important;
            border-color: rgba(239, 68, 68, 0.4) !important;
            color: #FDA4AF !important;
        }
        div[data-testid="stColumn"]:has(#navbar-nav-group) button[kind="secondary"],
        div[data-testid="column"]:has(#navbar-nav-group) button[kind="secondary"],
        .stColumn:has(#navbar-nav-group) button[kind="secondary"] {
            background-color: #0F172A !important;
            border-color: #334155 !important;
            color: #CBD5E1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def set_page(page_name: str):
    """Helper callback to switch current page programmatically."""
    st.session_state.current_page = normalize_page_name(page_name)


# Render Independent Floating Medical Chatbot (Bottom-Right)
from frontend.components.medical_chatbot import render_medical_chatbot

# Gate protected application behind authentication
if not st.session_state.get("is_authenticated", False):
    render_auth_page()
    render_medical_chatbot()
    st.stop()

# Render Sticky Top Navigation Bar
render_navbar(set_page_fn=set_page)

# Active Page Routing
current = normalize_page_name(st.session_state.current_page)

if current == NAV_DASHBOARD:
    render_dashboard_page(set_page_fn=set_page)
elif current == NAV_DOCUMENTS:
    render_documents_page(set_page_fn=set_page)
elif current == NAV_CHAT:
    render_chat_page()
elif current == NAV_HISTORY:
    render_history_page(set_page_fn=set_page)
else:
    render_dashboard_page(set_page_fn=set_page)

# Render Independent Floating Medical Chatbot (Bottom-Right)
render_medical_chatbot()

