"""
Clinical Authentication Page (Part 5 Redesign).

Provides a premium centered-card authentication experience for Sign In and
Account Creation, integrated with Supabase Auth and FastAPI session verification.
"""

import re
import streamlit as st
from frontend.api_client import api_client


def _is_valid_email(email: str) -> bool:
    """Basic regex check for realistic email syntax."""
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return bool(re.match(pattern, email.strip()))


def render_auth_page():
    """
    Renders the clinical centered card authentication layout.
    """
    # Centered layout container
    _, center_col, _ = st.columns([1, 2.2, 1])

    with center_col:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 28px; margin-bottom: 24px;">
                <div style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 52px;
                    height: 52px;
                    background: linear-gradient(135deg, #0F766E, #14B8A6);
                    color: #FFFFFF;
                    border-radius: 14px;
                    font-size: 1.75rem;
                    box-shadow: 0 4px 12px rgba(15, 118, 110, 0.25);
                    margin-bottom: 12px;
                ">
                    🩺
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; line-height: 1.2;">
                    HealthAI Assistant
                </div>
                <div style="font-size: 0.84rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">
                    Research & Patient Assistance
                </div>
                <p style="font-size: 0.9rem; color: #64748B; max-width: 480px; margin: 10px auto 0 auto; line-height: 1.5;">
                    Explore and research your healthcare documents with grounded AI assistance,
                    strict source evidence, and multi-tenant patient privacy.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Create Account"])

        # ======================================================================
        # TAB 1: SIGN IN
        # ======================================================================
        with tab_login:
            st.markdown(
                """
                <div style="margin-top: 14px; margin-bottom: 12px;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">
                        Sign In to Your Account
                    </div>
                    <div style="font-size: 0.82rem; color: #64748B;">
                        Enter your registered clinical credentials to access your private documents.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.form("form_login_modern", clear_on_submit=False):
                email = st.text_input(
                    "Email Address",
                    placeholder="physician@clinic.org",
                    key="login_email_input",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="••••••••••••",
                    key="login_password_input",
                )
                submit_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submit_login:
                clean_email = email.strip().lower()
                if not clean_email or not _is_valid_email(clean_email):
                    st.error("Please enter a valid email address.")
                elif not password:
                    st.error("Please enter your password.")
                else:
                    with st.spinner("Verifying credentials..."):
                        res = api_client.login(clean_email, password)

                    if res.get("success"):
                        token = res.get("token")
                        try:
                            me_res = api_client.get_current_user()
                            user = (me_res.get("user") if isinstance(me_res, dict) else None) or res.get("user") or {"email": clean_email}
                        except Exception:
                            user = res.get("user") or {"email": clean_email}

                        st.session_state.is_authenticated = True
                        st.session_state.access_token = token
                        st.session_state.authenticated_user = user
                        st.session_state.current_page = "Dashboard"
                        st.session_state.mobile_nav_open = False
                        st.success("Authentication successful! Entering workspace...")
                        st.rerun()
                    else:
                        err_msg = res.get("error") or "Authentication failed. Please verify your email and password."
                        if "timed out" in err_msg.lower() or "connection" in err_msg.lower():
                            st.warning(f"⚠️ {err_msg}")
                        else:
                            st.error(f"Sign in failed: {err_msg}")

        # ======================================================================
        # TAB 2: CREATE ACCOUNT
        # ======================================================================
        with tab_register:
            st.markdown(
                """
                <div style="margin-top: 14px; margin-bottom: 12px;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">
                        Create Your Research Account
                    </div>
                    <div style="font-size: 0.82rem; color: #64748B;">
                        Get started with isolated document storage and grounded AI consultations.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.form("form_register_modern", clear_on_submit=False):
                reg_email = st.text_input(
                    "Email Address",
                    placeholder="researcher@healthcare.org",
                    key="reg_email_input",
                )
                reg_password = st.text_input(
                    "Password (minimum 6 characters)",
                    type="password",
                    placeholder="Choose a strong password",
                    key="reg_password_input",
                )
                reg_confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter password",
                    key="reg_confirm_input",
                )
                submit_register = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submit_register:
                clean_reg_email = reg_email.strip().lower()
                if not clean_reg_email or not _is_valid_email(clean_reg_email):
                    st.error("Please provide a valid email address.")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match. Please re-type your password confirmation.")
                else:
                    with st.spinner("Creating secure account..."):
                        res = api_client.signup(clean_reg_email, reg_password)

                    if res.get("success"):
                        if res.get("needs_email_confirmation"):
                            st.info(
                                "Account created successfully! Please check your email inbox to confirm your registration before signing in."
                            )
                        else:
                            st.success("Account created successfully! Signing you in...")
                            login_res = api_client.login(clean_reg_email, reg_password)
                            if login_res.get("success"):
                                token = login_res.get("token")
                                me_res = api_client.get_current_user()
                                user = me_res.get("user") or login_res.get("user") or {"email": clean_reg_email}
                                st.session_state.is_authenticated = True
                                st.session_state.access_token = token
                                st.session_state.authenticated_user = user
                                st.rerun()
                            else:
                                st.info("Registration complete. You may now switch to the 'Sign In' tab.")
                    else:
                        err_msg = res.get("error") or "Registration failed. Please try again."
                        st.error(f"Registration error: {err_msg}")

        # Security & Compliance Footer
        st.markdown(
            """
            <div style="
                margin-top: 24px;
                padding-top: 14px;
                border-top: 1px solid #E2E8F0;
                text-align: center;
                font-size: 0.74rem;
                color: #64748B;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 16px;
                flex-wrap: wrap;
            ">
                <span>🔒 Encrypted Session</span>
                <span>•</span>
                <span>🛡️ Private Knowledge Base</span>
                <span>•</span>
                <span>📚 Grounded RAG Evidence</span>
            </div>
            """,
            unsafe_allow_html=True
        )
