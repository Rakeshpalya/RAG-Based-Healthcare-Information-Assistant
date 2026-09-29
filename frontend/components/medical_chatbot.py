"""
Floating Medical Chatbot Widget for Streamlit.

Renders an independent, compact floating HealthAI Medical Assistant at the
bottom-right corner of the application.
Communicates strictly with POST /api/medical-chat.
Completely decoupled from the existing RAG retrieval interface.
"""

import html
import re
import uuid
import streamlit as st
from frontend.api_client import api_client

MEDICAL_CHATBOT_CSS = """
<style>
/* =============================================================================
   HEALTHAI FLOATING MEDICAL CHATBOT STYLES
   ============================================================================= */

/* Container resets so floating elements take ZERO page layout space */
div[data-testid="stElementContainer"]:has(> div.st-key-healthai_floating_launcher),
.element-container:has(> div.st-key-healthai_floating_launcher),
div[data-testid="stElementContainer"]:has(> div.st-key-healthai_floating_popup),
.element-container:has(> div.st-key-healthai_floating_popup) {
    position: fixed !important;
    bottom: 24px !important;
    right: 24px !important;
    z-index: 999999 !important;
    width: auto !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}

/* 1. CLOSED STATE: Small Rounded Floating Launcher Button with Pulse Glow */
div.st-key-healthai_floating_launcher {
    position: fixed !important;
    bottom: 24px !important;
    right: 24px !important;
    z-index: 999999 !important;
    width: auto !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 !important;
}

div.st-key-healthai_floating_launcher button {
    background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(45, 212, 191, 0.5) !important;
    border-radius: 9999px !important;
    padding: 10px 22px !important;
    font-size: 14.5px !important;
    font-weight: 700 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    box-shadow: 0 8px 24px rgba(13, 148, 136, 0.4) !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    white-space: nowrap !important;
    animation: healthaiPulseGlow 2.5s infinite !important;
}

div.st-key-healthai_floating_launcher button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 12px 28px rgba(13, 148, 136, 0.6) !important;
    border-color: #2DD4BF !important;
    color: #FFFFFF !important;
}

@keyframes healthaiPulseGlow {
    0% {
        box-shadow: 0 0 0 0 rgba(13, 148, 136, 0.55), 0 8px 24px rgba(13, 148, 136, 0.35);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(13, 148, 136, 0), 0 8px 24px rgba(13, 148, 136, 0.35);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(13, 148, 136, 0), 0 8px 24px rgba(13, 148, 136, 0.35);
    }
}

/* 2. OPEN STATE: Compact Floating Popup Window Frame */
div.st-key-healthai_floating_popup {
    position: fixed !important;
    bottom: 24px !important;
    right: 24px !important;
    width: 380px !important;
    max-width: calc(100vw - 32px) !important;
    height: 540px !important;
    max-height: calc(100vh - 48px) !important;
    background: #090E17 !important;
    border: 1px solid #1E293B !important;
    border-radius: 18px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.75), 0 0 0 1px rgba(255, 255, 255, 0.08) !important;
    z-index: 999999 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    animation: healthaiModalPop 0.26s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-sizing: border-box !important;
    padding: 0 !important;
}

@keyframes healthaiModalPop {
    from {
        opacity: 0;
        transform: translateY(18px) scale(0.96);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

/* Header Container */
div.st-key-healthai_header_box {
    background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%) !important;
    padding: 10px 14px !important;
    margin: 0 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-sizing: border-box !important;
    flex-shrink: 0 !important;
    border-radius: 18px 18px 0 0 !important;
}

div.st-key-healthai_header_box [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

.healthai-header-title-box {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}

.healthai-header-title {
    font-size: 14.5px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    margin: 0 !important;
    line-height: 1.25 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.healthai-header-subtitle {
    font-size: 11px !important;
    color: #CCFBF1 !important;
    margin: 0 !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
}

/* Close Button (✕) */
div.st-key-healthai_close_btn {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

div.st-key-healthai_close_btn button {
    background: rgba(0, 0, 0, 0.2) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 0 !important;
    width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    max-height: 28px !important;
    border-radius: 50% !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    box-shadow: none !important;
}

div.st-key-healthai_close_btn button:hover {
    background: rgba(239, 68, 68, 0.85) !important;
    color: #FFFFFF !important;
    transform: scale(1.1) !important;
}

/* Chat Messages Scroll Container */
div.st-key-healthai_chat_scroll_box {
    background: #090E17 !important;
    padding: 10px 12px !important;
    overflow-y: auto !important;
    border: none !important;
    border-radius: 0 !important;
    box-sizing: border-box !important;
}

div.st-key-healthai_chat_scroll_box::-webkit-scrollbar {
    width: 4px !important;
}

div.st-key-healthai_chat_scroll_box::-webkit-scrollbar-thumb {
    background: #0D9488 !important;
    border-radius: 4px !important;
}

.healthai-messages-flow {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 100%;
}

/* Message Bubbles */
.healthai-bubble {
    padding: 9px 13px;
    border-radius: 14px;
    font-size: 12.5px;
    line-height: 1.45;
    word-break: break-word;
    box-sizing: border-box;
    animation: healthaiBubbleFade 0.2s ease-out;
}

@keyframes healthaiBubbleFade {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.healthai-bubble-assistant {
    background: #131C2E;
    color: #E2E8F0;
    border: 1px solid #1E293B;
    align-self: flex-start;
    max-width: 86%;
    border-bottom-left-radius: 3px;
}

.healthai-bubble-assistant .bubble-author {
    font-size: 11px;
    font-weight: 700;
    color: #2DD4BF;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.healthai-bubble-user {
    background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%);
    color: #FFFFFF;
    align-self: flex-end;
    max-width: 78%;
    border-bottom-right-radius: 3px;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(13, 148, 136, 0.3);
}

/* Animated Typing Indicator */
.healthai-typing-bubble {
    background: #131C2E !important;
    border: 1px solid #1E293B !important;
    padding: 10px 14px !important;
    align-self: flex-start !important;
    border-bottom-left-radius: 3px !important;
}

.typing-indicator-row {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-top: 2px;
}

.typing-dot {
    width: 6px;
    height: 6px;
    background-color: #2DD4BF;
    border-radius: 50%;
    display: inline-block;
    animation: healthaiTypingBounce 1.4s infinite ease-in-out both;
}

.typing-dot:nth-child(1) { animation-delay: -0.32s; }
.typing-dot:nth-child(2) { animation-delay: -0.16s; }
.typing-dot:nth-child(3) { animation-delay: 0s; }

@keyframes healthaiTypingBounce {
    0%, 80%, 100% {
        transform: scale(0.3);
        opacity: 0.35;
    }
    40% {
        transform: scale(1.0);
        opacity: 1;
    }
}

.typing-label {
    font-size: 11px;
    color: #94A3B8;
    margin-left: 6px;
    font-style: italic;
}

/* Suggestion Chips Section */
.healthai-chips-label {
    font-size: 10px;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 4px 12px 2px 12px;
    background: #090E17;
}

div.st-key-healthai_chips_box {
    background: #090E17 !important;
    padding: 2px 10px 6px 10px !important;
    margin: 0 !important;
}

div[class*="st-key-healthai_chip_"] button {
    background: rgba(15, 118, 110, 0.16) !important;
    border: 1px solid rgba(45, 212, 191, 0.35) !important;
    color: #2DD4BF !important;
    border-radius: 9999px !important;
    padding: 3px 10px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    min-height: 26px !important;
    height: 26px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    width: 100% !important;
}

div[class*="st-key-healthai_chip_"] button:hover {
    background: rgba(15, 118, 110, 0.35) !important;
    border-color: #2DD4BF !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
}

/* =============================================================================
   INPUT ROW (CRITICAL FIX: FORCE DARK STYLING ON ALL SELECTORS)
   ============================================================================= */
div.st-key-healthai_input_row {
    background: #090E17 !important;
    padding: 8px 12px 8px 12px !important;
    border-top: 1px solid #1E293B !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    flex-shrink: 0 !important;
}

div.st-key-healthai_input_row form {
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

div.st-key-healthai_input_row [data-testid="stHorizontalBlock"] {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* OVERRIDE STREAMLIT LIGHT THEME WHITE INPUT BACKGROUND ON ALL LEVELS */
div.st-key-healthai_input_row div[data-testid="stTextInput"],
div.st-key-healthai_input_row div[data-baseweb="input"],
div.st-key-healthai_input_row div[data-baseweb="base-input"],
div.st-key-healthai_input_row input[data-testid="stTextInputRootElement"],
div.st-key-healthai_input_row [data-testid="stTextInput"] input {
    background-color: #1E293B !important;
    background: #1E293B !important;
    color: #F8FAFC !important;
}

div.st-key-healthai_input_row div[data-baseweb="input"] {
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    padding: 0 10px !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.2s ease !important;
}

div.st-key-healthai_input_row div[data-baseweb="input"]:focus-within {
    border-color: #14B8A6 !important;
    box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.3), inset 0 1px 3px rgba(0, 0, 0, 0.3) !important;
}

div.st-key-healthai_input_row input {
    color: #F8FAFC !important;
    font-size: 13px !important;
    height: 38px !important;
    line-height: 38px !important;
    padding: 0 !important;
    border: none !important;
}

div.st-key-healthai_input_row input::placeholder {
    color: #94A3B8 !important;
    opacity: 1 !important;
}

/* Send Button (➤) - Sized perfectly, centered, vibrant teal */
div.st-key-healthai_input_row div[data-testid="stFormSubmitButton"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 40px !important;
    margin: 0 !important;
    padding: 0 !important;
}

div.st-key-healthai_input_row div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(45, 212, 191, 0.4) !important;
    border-radius: 12px !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    width: 40px !important;
    min-width: 40px !important;
    max-width: 40px !important;
    padding: 0 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(13, 148, 136, 0.4) !important;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div.st-key-healthai_input_row div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #115E59 0%, #0F766E 100%) !important;
    transform: scale(1.06) !important;
    box-shadow: 0 4px 14px rgba(13, 148, 136, 0.6) !important;
    border-color: #2DD4BF !important;
}

/* Safety Disclaimer Footer */
.healthai-popup-disclaimer {
    font-size: 9.5px !important;
    color: #64748B !important;
    text-align: center !important;
    padding: 4px 12px 6px 12px !important;
    background: #090E17 !important;
    line-height: 1.35 !important;
    flex-shrink: 0 !important;
}

/* Responsive Overrides */
@media (max-width: 768px) {
    div.st-key-healthai_floating_popup {
        bottom: 12px !important;
        right: 12px !important;
        width: calc(100vw - 24px) !important;
        max-width: calc(100vw - 24px) !important;
        height: 72vh !important;
        max-height: 72vh !important;
    }
    div.st-key-healthai_floating_launcher {
        bottom: 16px !important;
        right: 16px !important;
    }
}
</style>
"""


def init_medical_chatbot_state():
    """Initializes session state keys for the independent medical chatbot."""
    if "medical_chat_open" not in st.session_state:
        st.session_state.medical_chat_open = False

    if "medical_chat_session_id" not in st.session_state:
        st.session_state.medical_chat_session_id = f"session_{uuid.uuid4().hex[:8]}"

    if "medical_chat_messages" not in st.session_state:
        st.session_state.medical_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am the HealthAI Medical Assistant. Ask me general medical questions such as:\n"
                    "• What is hypertension?\n"
                    "• What are the symptoms of diabetes?\n"
                    "• What is asthma?"
                )
            }
        ]

    if "medical_chat_pending_prompt" not in st.session_state:
        st.session_state.medical_chat_pending_prompt = None

    if "medical_chat_input_counter" not in st.session_state:
        st.session_state.medical_chat_input_counter = 0


def _format_content_to_html(content: str) -> str:
    """Safely escapes HTML and formats markdown-like elements for bubble display."""
    escaped = html.escape(content)
    # Convert bold **text** to <strong>text</strong>
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escaped)
    # Convert italic *text* to <em>text</em>
    escaped = re.sub(r'\*(.*?)\*', r'<em>\1</em>', escaped)
    # Convert newlines to <br>
    escaped = escaped.replace('\n', '<br>')
    return escaped


def _handle_form_submit():
    """Form submit callback for text input and send button."""
    current_key = f"healthai_input_{st.session_state.medical_chat_input_counter}"
    submitted_text = st.session_state.get(current_key, "").strip()
    if submitted_text:
        st.session_state.medical_chat_input_counter += 1
        st.session_state.medical_chat_messages.append({
            "role": "user",
            "content": submitted_text
        })
        st.session_state.medical_chat_pending_prompt = submitted_text


def _handle_quick_question(question: str):
    """Handles quick suggestion clicks: clears input, appends user message, and queues request."""
    clean_q = question.strip()
    if not clean_q:
        return
    st.session_state.medical_chat_input_counter += 1
    st.session_state.medical_chat_messages.append({
        "role": "user",
        "content": clean_q
    })
    st.session_state.medical_chat_pending_prompt = clean_q
    st.rerun()


def render_medical_chatbot():
    """
    Renders the floating HealthAI Medical Chatbot in the bottom-right corner.
    Overlays existing application content without displacing page layout.
    """
    init_medical_chatbot_state()

    # Inject floating styling
    st.markdown(MEDICAL_CHATBOT_CSS, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # CLOSED STATE: Small Floating Launcher Button (Bottom-Right)
    # -------------------------------------------------------------------------
    if not st.session_state.medical_chat_open:
        if st.button(
            "🩺 HealthAI",
            key="healthai_floating_launcher",
            help="Open HealthAI Medical Assistant for General Medical Q&A"
        ):
            st.session_state.medical_chat_open = True
            st.rerun()
        return

    # -------------------------------------------------------------------------
    # OPEN STATE: Compact Floating Chat Window (Bottom-Right)
    # -------------------------------------------------------------------------
    with st.container(key="healthai_floating_popup"):
        # 1. Header Bar with Title, Subtitle, and Close '✕' Button
        with st.container(key="healthai_header_box"):
            col_hdr, col_close = st.columns([0.84, 0.16])
            with col_hdr:
                st.markdown(
                    '<div class="healthai-header-title-box">'
                    '<div class="healthai-header-title">🩺 HealthAI Medical Assistant</div>'
                    '<div class="healthai-header-subtitle">General Medical Q&A</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            with col_close:
                if st.button("✕", key="healthai_close_btn", help="Close chatbot"):
                    st.session_state.medical_chat_open = False
                    st.rerun()

        # 2. Scrollable Messages History Area (Compact 230px, no huge empty void)
        with st.container(height=230, key="healthai_chat_scroll_box"):
            html_chunks = ['<div class="healthai-messages-flow" id="healthai-messages-flow">']
            for msg in st.session_state.medical_chat_messages:
                role = msg.get("role", "assistant")
                formatted_text = _format_content_to_html(msg.get("content", ""))
                if role == "assistant":
                    html_chunks.append(
                        f'<div class="healthai-bubble healthai-bubble-assistant">'
                        f'<div class="bubble-author">🩺 HealthAI</div>'
                        f'<div>{formatted_text}</div>'
                        f'</div>'
                    )
                else:
                    html_chunks.append(
                        f'<div class="healthai-bubble healthai-bubble-user">'
                        f'<div>{formatted_text}</div>'
                        f'</div>'
                    )

            # Show active typing indicator immediately if query is in progress
            if st.session_state.get("medical_chat_pending_prompt"):
                html_chunks.append(
                    '<div class="healthai-bubble healthai-bubble-assistant healthai-typing-bubble">'
                    '<div class="bubble-author">🩺 HealthAI</div>'
                    '<div class="typing-indicator-row">'
                    '<span class="typing-dot"></span>'
                    '<span class="typing-dot"></span>'
                    '<span class="typing-dot"></span>'
                    '<span class="typing-label">HealthAI is thinking...</span>'
                    '</div>'
                    '</div>'
                )

            html_chunks.append('</div>')
            st.markdown("".join(html_chunks), unsafe_allow_html=True)

        # 3. Suggested Questions Chips (2x2 grid of neat pills)
        st.markdown('<div class="healthai-chips-label">💡 Suggested Questions:</div>', unsafe_allow_html=True)
        with st.container(key="healthai_chips_box"):
            chip_col1, chip_col2 = st.columns(2)
            with chip_col1:
                if st.button("What is hypertension?", key="healthai_chip_1", help="Ask about hypertension"):
                    _handle_quick_question("What is hypertension?")
                if st.button("What is asthma?", key="healthai_chip_3", help="Ask about asthma"):
                    _handle_quick_question("What is asthma?")
            with chip_col2:
                if st.button("Diabetes symptoms", key="healthai_chip_2", help="Ask about diabetes symptoms"):
                    _handle_quick_question("What are the symptoms of diabetes?")
                if st.button("What is high BP?", key="healthai_chip_4", help="Ask about high blood pressure"):
                    _handle_quick_question("What causes high blood pressure?")

        # 4. Input Row (Dark styled text input with send button)
        with st.container(key="healthai_input_row"):
            current_input_key = f"healthai_input_{st.session_state.medical_chat_input_counter}"
            with st.form("healthai_chat_input_form", clear_on_submit=True, border=False):
                col_inp, col_send = st.columns([0.80, 0.20])
                with col_inp:
                    user_msg = st.text_input(
                        "Ask about your health...",
                        key=current_input_key,
                        placeholder="Ask about your health...",
                        label_visibility="collapsed"
                    )
                with col_send:
                    send_clicked = st.form_submit_button(
                        "➤",
                        help="Send medical question",
                        on_click=_handle_form_submit
                    )

                # Direct submit fallback for immediate event handling
                if send_clicked and user_msg and user_msg.strip():
                    if not st.session_state.get("medical_chat_pending_prompt"):
                        st.session_state.medical_chat_input_counter += 1
                        st.session_state.medical_chat_messages.append({
                            "role": "user",
                            "content": user_msg.strip()
                        })
                        st.session_state.medical_chat_pending_prompt = user_msg.strip()
                        st.rerun()

        # 5. Safety Disclaimer Footer
        st.markdown(
            '<div class="healthai-popup-disclaimer">'
            'General educational information only. Does not diagnose or prescribe.'
            '</div>',
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # Process Pending Query (Two-phase flow: renders typing state first, then queries)
    # -------------------------------------------------------------------------
    if st.session_state.get("medical_chat_pending_prompt"):
        prompt_to_send = st.session_state.medical_chat_pending_prompt
        st.session_state.medical_chat_pending_prompt = None  # Clear flag immediately to prevent loop

        res = api_client.route_query(
            message=prompt_to_send,
            session_id=st.session_state.medical_chat_session_id,
            mode="auto",
            timeout=90
        )

        if res.get("success") and res.get("data"):
            data = res["data"]
            answer = data.get("answer", "No answer provided.")
            if data.get("session_id"):
                st.session_state.medical_chat_session_id = data["session_id"]
        else:
            answer = res.get("error") or "Sorry, I couldn't process that request right now. Please try again."

        st.session_state.medical_chat_messages.append({
            "role": "assistant",
            "content": answer
        })
        st.rerun()
