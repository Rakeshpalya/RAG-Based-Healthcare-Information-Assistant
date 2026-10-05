"""
Clinical Document Chat & RAG Page (Production UX).

Connects authenticated users to the backend RAG pipeline with Server-Sent Events (SSE) streaming,
grounded evidence citations, distinct medical safety warnings, and comprehensive error handling.
Strictly preserves multi-tenant isolation, validated citations, and clinical safety.
"""

import re
from typing import List, Dict, Any, Optional
import streamlit as st
from frontend.api_client import api_client
from frontend.components.source_card import render_source_cards
from frontend.utils.helpers import clean_ai_markdown


def render_safety_card(text: str, category: Optional[str] = None):
    """
    Renders distinct visual treatment for:
    - acute emergency warnings (911)
    - crisis / self-harm responses (988 Lifeline)
    - medication safety warnings
    - insufficient reference documentation
    Never hides safety warnings and ensures they are distinctly differentiated from system errors.
    """
    lower = text.lower()
    cat_lower = (category or "").lower()

    if "emergency" in cat_lower or "911" in lower or "emergency room" in lower or "acute emergency" in lower or "immediate medical attention" in lower:
        st.markdown(
            f"""
            <div class="safety-card-emergency">
                <h4>🚨 Acute Medical Emergency Notice</h4>
                <div style="font-size: 0.95rem; line-height: 1.6; font-weight: 500;">{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif "crisis" in cat_lower or "self_harm" in cat_lower or "988" in lower or "suicide" in lower or "lifeline" in lower or "crisis line" in lower:
        st.markdown(
            f"""
            <div class="safety-card-crisis">
                <h4>💜 Crisis & Mental Health Support (988 Lifeline)</h4>
                <div style="font-size: 0.95rem; line-height: 1.6; font-weight: 500;">{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif "medication" in cat_lower or "prescription" in lower or "physician" in lower or "dosage" in lower or "prescribe" in lower:
        st.markdown(
            f"""
            <div class="safety-card-medication">
                <h4>⚠️ Clinical Safety & Prescribing Boundary</h4>
                <div style="font-size: 0.95rem; line-height: 1.6; font-weight: 500;">{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif "insufficient" in cat_lower or "no relevant context" in lower or "does not contain" in lower:
        st.markdown(
            f"""
            <div class="safety-card-insufficient">
                <h4>ℹ️ Insufficient Reference Documentation</h4>
                <div style="font-size: 0.95rem; line-height: 1.6;">{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(text)


def render_chat_page():
    """
    Renders the authenticated healthcare document research chat interface.
    """
    # 1. Header
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">
                Grounded Clinical Intelligence
            </div>
            <h1 style="color: var(--text-primary); font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                Research Your Healthcare Documents
            </h1>
            <p style="color: var(--text-secondary); font-size: 1rem; margin-top: 6px; margin-bottom: 0; line-height: 1.5;">
                Ask questions and receive answers grounded strictly in your private uploaded sources.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Initialize chat state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = True

    # 3. Action Toolbar: Medical Notice & New Conversation
    col_tools1, col_tools2 = st.columns([3, 1])
    with col_tools1:
        st.markdown(
            """
            <div class="medical-safety-banner" style="margin: 0; padding: 10px 14px; font-size: 0.82rem;">
                ⚠️ <strong>Clinical Notice:</strong> Answers are synthesized strictly from your uploaded reference documents.
                This assistant provides informational research only. It does not provide medical diagnoses or prescribe medications.
                For emergencies, call <strong>911</strong> or seek immediate local emergency medical care.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_tools2:
        if st.button("➕ New Inquiry", key="btn_new_chat_session", type="primary", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.active_conversation_id = None
            st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 4. Optional Advanced Retrieval Settings Drawer
    with st.expander("⚙️ Advanced / Knowledge Retrieval Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider(
                "Evidence Chunks (Top-K)",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of relevant document passages retrieved for answer grounding."
            )
        with col2:
            similarity_threshold = st.slider(
                "Source Relevance Cutoff",
                min_value=0.10,
                max_value=0.90,
                value=0.25,
                step=0.05,
                help="Minimum cosine similarity cutoff required for evidence validation."
            )
        with col3:
            st.session_state.use_streaming = st.checkbox(
                "Real-Time SSE Streaming",
                value=st.session_state.use_streaming,
                help="Stream tokens incrementally using Server-Sent Events after pipeline safety validation."
            )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 5. Empty State with Starter Prompts (if chat is empty)
    if not st.session_state.chat_messages:
        st.markdown(
            """
            <div class="chat-empty-state">
                <div style="font-size: 2.4rem; margin-bottom: 10px;">🩺</div>
                <h3 style="margin: 0 0 8px 0; font-size: 1.3rem; font-weight: 800; color: var(--text-primary);">
                    How can I assist your clinical research today?
                </h3>
                <p style="font-size: 0.92rem; color: var(--text-secondary); max-width: 600px; margin: 0 auto 18px auto; line-height: 1.5;">
                    Ask any question about your uploaded clinical guidelines, medical notes, or trial documentation.
                    Or choose one of the suggested clinical queries below:
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        starter_queries = [
            "What are the diagnostic criteria for Type 2 Diabetes?",
            "What are the recommended first-line interventions for hypertension?",
            "What contraindications exist for metformin in renal impairment?",
            "Summarize the patient care protocols from my uploaded documents.",
        ]

        sq1, sq2 = st.columns(2)
        starter_clicked = None
        for idx, sq in enumerate(starter_queries):
            target_col = sq1 if idx % 2 == 0 else sq2
            with target_col:
                if st.button(f"💡 {sq}", key=f"btn_starter_{idx}", use_container_width=True):
                    starter_clicked = sq

        if starter_clicked:
            st.session_state.pending_quick_query = starter_clicked
            st.rerun()

    # 6. Render Existing Dialogue History
    for msg_idx, msg in enumerate(st.session_state.chat_messages):
        sender = msg.get("sender", "assistant")
        with st.chat_message(sender):
            msg_status = msg.get("retrieval_status")
            raw_sources = msg.get("sources") or []
            msg_text = msg.get("text", "")
            safety_cat = msg.get("safety_category")

            if sender == "user":
                st.markdown(msg_text)
            else:
                # Render safety card if intercepted or insufficient context
                if msg_status in ("safety_intercepted", "no_relevant_context"):
                    render_safety_card(msg_text, category=safety_cat or msg_status)
                elif msg_status == "error":
                    st.error(msg_text)
                    if st.button("🔄 Retry Inquiry", key=f"btn_retry_msg_{msg_idx}"):
                        # Find prior user query
                        prior_user_q = None
                        for p in reversed(st.session_state.chat_messages[:msg_idx]):
                            if p.get("sender") == "user":
                                prior_user_q = p.get("text")
                                break
                        if prior_user_q:
                            st.session_state.pending_quick_query = prior_user_q
                            st.rerun()
                else:
                    # Clean inline citations [Source N] -> [N]
                    st.markdown(clean_ai_markdown(msg_text, sources=raw_sources))

                    # Render verified evidence sources
                    if raw_sources and msg_status in ("success", "grounded_boundary"):
                        render_source_cards(raw_sources)

                    # General medical disclaimer box
                    st.markdown(
                        f"""
                        <div class="safety-card-disclaimer">
                            ⚕️ <strong>Disclaimer:</strong> {msg.get('disclaimer', 'This healthcare assistant provides educational information based on reference documentation. Always consult a qualified medical professional for diagnosis and treatment.')}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Provenance Drawer for assistant responses
                timings = msg.get("timings")
                if sender == "assistant" and timings:
                    with st.expander("🔍 Response Provenance & Latency Breakdown", expanded=False):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f"**Status:** `{msg_status or 'success'}`")
                        with c2:
                            tot_ms = timings.get("total_time_ms", 0.0)
                            st.markdown(f"**Total Latency:** `{tot_ms:.1f} ms`")
                        with c3:
                            st.markdown(f"**Sources Verified:** `{len(raw_sources)}`")

    # 7. Check for Pending Query (from Dashboard or Starter Prompts)
    pending_query = st.session_state.get("pending_quick_query")
    if pending_query:
        user_query = pending_query
        st.session_state.pending_quick_query = None
    else:
        is_generating = st.session_state.get("is_generating", False)
        user_query = st.chat_input(
            "Ask a question about your uploaded healthcare documents...",
            disabled=is_generating
        )

    # 8. Query Execution
    if user_query:
        st.session_state.is_generating = True

        # Append user message
        st.session_state.chat_messages.append({
            "sender": "user",
            "text": user_query,
        })

        # Track or Create active conversation session in DB
        active_conv_id = st.session_state.get("active_conversation_id")
        if not active_conv_id:
            conv_res = api_client.create_conversation(title=user_query[:50])
            if conv_res.get("success"):
                active_conv_id = conv_res["data"].get("id")
                st.session_state.active_conversation_id = active_conv_id

        # Persist user message to DB
        if active_conv_id:
            try:
                api_client.add_message_to_conversation(
                    conversation_id=active_conv_id,
                    sender="user",
                    text=user_query,
                )
            except Exception:
                pass

        with st.chat_message("user"):
            st.markdown(user_query)

        # Build recent conversation history for follow-up resolution
        recent_history = []
        for m in st.session_state.chat_messages:
            sender_role = m.get("sender")
            msg_txt = m.get("text", "")
            if sender_role in ("user", "assistant") and msg_txt and msg_txt != user_query:
                recent_history.append({"role": sender_role, "content": msg_txt})
        recent_history = recent_history[-4:]

        top_k_val = top_k if "top_k" in locals() else 5
        sim_val = similarity_threshold if "similarity_threshold" in locals() else 0.25
        use_streaming = st.session_state.get("use_streaming", True)

        with st.chat_message("assistant"):
            if use_streaming:
                # SSE STREAMING EXECUTION (5.5)
                status_box = st.empty()
                status_box.info("Searching healthcare reference documents...")
                text_box = st.empty()
                accumulated_text = ""
                final_data = None
                error_msg = None
                safety_cat = None

                for event_type, payload in api_client.stream_rag_query(
                    question=user_query,
                    top_k=top_k_val,
                    similarity_threshold=sim_val,
                    conversation_history=recent_history,
                ):
                    if event_type == "status":
                        step_msg = payload.get("message", "Processing...")
                        status_box.caption(f"⚡ {step_msg}")
                    elif event_type == "token":
                        tok = payload.get("token") or payload.get("text", "")
                        accumulated_text += tok
                        text_box.markdown(accumulated_text + " ▌")
                    elif event_type == "complete":
                        final_data = payload
                    elif event_type == "error":
                        error_msg = payload.get("error", "An error occurred during streaming.")
                        safety_cat = payload.get("category")

                status_box.empty()

                if error_msg and not final_data:
                    text_box.empty()
                    st.error(f"Inquiry could not be completed: {error_msg}")
                    st.session_state.chat_messages.append({
                        "sender": "assistant",
                        "text": f"⚠️ {error_msg}",
                        "sources": [],
                        "retrieval_status": "error",
                    })
                elif final_data:
                    raw_answer = final_data.get("answer", accumulated_text)
                    sources = final_data.get("sources", []) or []
                    retrieval_status = final_data.get("retrieval_status", "success")
                    timings = final_data.get("timings", {})
                    disclaimer = final_data.get("disclaimer", "")

                    text_box.empty()

                    if retrieval_status in ("safety_intercepted", "no_relevant_context"):
                        sources = []
                        render_safety_card(raw_answer, category=safety_cat or retrieval_status)
                        clean_answer = raw_answer
                    else:
                        clean_answer = clean_ai_markdown(raw_answer, sources=sources)
                        st.markdown(clean_answer)
                        if sources:
                            render_source_cards(sources)
                        st.markdown(
                            f"""
                            <div class="safety-card-disclaimer">
                                ⚕️ <strong>Disclaimer:</strong> {disclaimer or 'This assistant provides research information grounded in reference documents. Consult a licensed physician for clinical decisions.'}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.session_state.chat_messages.append({
                        "sender": "assistant",
                        "text": clean_answer,
                        "sources": sources,
                        "retrieval_status": retrieval_status,
                        "timings": timings,
                        "disclaimer": disclaimer,
                        "safety_category": safety_cat,
                    })

                    if active_conv_id:
                        try:
                            api_client.add_message_to_conversation(
                                conversation_id=active_conv_id,
                                sender="assistant",
                                text=clean_answer,
                                citations=sources,
                                agent_type="rag_agent",
                            )
                        except Exception:
                            pass
            else:
                # SYNCHRONOUS RAG EXECUTION (5.4 Fallback)
                with st.spinner("Searching your healthcare knowledge base..."):
                    rag_res = api_client.query_rag(
                        question=user_query,
                        top_k=top_k_val,
                        similarity_threshold=sim_val,
                        conversation_history=recent_history,
                    )

                if rag_res.get("success"):
                    data = rag_res.get("data", {})
                    raw_answer = data.get("answer", "")
                    sources = data.get("sources", []) or []
                    retrieval_status = data.get("retrieval_status", "success")
                    timings = data.get("timings", {})
                    disclaimer = data.get("disclaimer", "")

                    if retrieval_status in ("safety_intercepted", "no_relevant_context"):
                        sources = []
                        render_safety_card(raw_answer, category=retrieval_status)
                        clean_answer = raw_answer
                    else:
                        clean_answer = clean_ai_markdown(raw_answer, sources=sources)
                        st.markdown(clean_answer)
                        if sources:
                            render_source_cards(sources)
                        st.markdown(
                            f"""
                            <div class="safety-card-disclaimer">
                                ⚕️ <strong>Disclaimer:</strong> {disclaimer or 'This assistant provides research information grounded in reference documents. Consult a licensed physician for clinical decisions.'}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.session_state.chat_messages.append({
                        "sender": "assistant",
                        "text": clean_answer,
                        "sources": sources,
                        "retrieval_status": retrieval_status,
                        "timings": timings,
                        "disclaimer": disclaimer,
                    })

                    if active_conv_id:
                        try:
                            api_client.add_message_to_conversation(
                                conversation_id=active_conv_id,
                                sender="assistant",
                                text=clean_answer,
                                citations=sources,
                                agent_type="rag_agent",
                            )
                        except Exception:
                            pass
                else:
                    err = rag_res.get("error") or "Unable to process clinical inquiry."
                    st.error(f"Inquiry could not be completed: {err}")
                    st.session_state.chat_messages.append({
                        "sender": "assistant",
                        "text": f"⚠️ {err}",
                        "sources": [],
                        "retrieval_status": "error",
                    })

        st.session_state.is_generating = False
        st.rerun()
