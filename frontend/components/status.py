"""
Clinical status badge and live system connectivity indicator components.
"""

import streamlit as st


def render_status_badge(status_text: str):
    """
    Renders a clinical status pill indicator.
    Supports UPLOADED, PROCESSING, PROCESSED, FAILED, INDEXED, etc.
    """
    status_clean = (status_text or "UNKNOWN").upper().strip()

    status_styles = {
        "PROCESSED": ("#0D9488", "#CCFBF1", "✓ PROCESSED"),
        "INDEXED": ("#0D9488", "#CCFBF1", "✓ INDEXED"),
        "COMPLETED": ("#0D9488", "#CCFBF1", "✓ COMPLETED"),
        "UPLOADED": ("#2563EB", "#DBEAFE", "↑ UPLOADED"),
        "PROCESSING": ("#D97706", "#FEF3C7", "⟳ PROCESSING"),
        "FAILED": ("#DC2626", "#FEE2E2", "✕ FAILED"),
        "CONNECTED": ("#16A34A", "#DCFCE7", "● CONNECTED"),
        "DISCONNECTED": ("#DC2626", "#FEE2E2", "○ DISCONNECTED"),
    }

    color, bg_color, label = status_styles.get(
        status_clean,
        ("#4B5563", "#F3F4F6", f"• {status_clean}")
    )

    st.markdown(
        f"""
        <span style="
            display: inline-block;
            padding: 3px 10px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            color: {color};
            background-color: {bg_color};
            border-radius: 9999px;
            border: 1px solid {color}33;
        ">{label}</span>
        """,
        unsafe_allow_html=True
    )


def render_backend_indicator(connected: bool, version: str = "0.5.0", url: str = "http://127.0.0.1:8000"):
    """
    Renders live backend connection health status in the sidebar.
    Genuinely displays ● Backend Connected or ● Backend Disconnected.
    """
    if connected:
        st.markdown(
            f"""
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
                margin-top: 4px;
                margin-bottom: 6px;
            ">
                <span class="status-dot connected"></span>
                <span>Backend Connected <small style="color: #047857; font-weight: 500;">(v{version})</small></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 8px 14px;
                background-color: #FEF2F2;
                border: 1px solid #FECACA;
                border-radius: 8px;
                color: #991B1B;
                font-size: 0.83rem;
                font-weight: 600;
                margin-top: 4px;
                margin-bottom: 6px;
            ">
                <span class="status-dot disconnected"></span>
                <span>Backend Disconnected</span>
            </div>
            """,
            unsafe_allow_html=True
        )
