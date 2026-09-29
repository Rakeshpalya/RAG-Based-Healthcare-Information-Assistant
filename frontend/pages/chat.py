"""
Clinical Document Chat & RAG Page.

Connects the authenticated Streamlit user to the backend RAG pipeline (POST /rag/query).
Ensures user-ownership isolation in FAISS retrieval, displays grounded evidence sources,
and provides seamless session persistence and clean clinical UX.
"""

import re
from typing import List, Dict, Any, Optional
import streamlit as st
from frontend.api_client import api_client
from frontend.components.source_card import render_source_cards
from frontend.utils.helpers import clean_ai_markdown


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

    # 2. Initialize chat history in session_state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "sender": "assistant",
                "text": "Hello! I am your HealthAI Assistant. "
                        "Ask any question about your uploaded healthcare documents, clinical trial papers, or medical guidelines.",
                "sources": [],
                "retrieval_status": "initial",
                "disclaimer": "This assistant provides educational and research assistance based strictly on your uploaded reference documents.",
            }
        ]

    # 3. Action Toolbar: New Conversation & Optional Settings
    col_tools1, col_tools2 = st.columns([3, 1])
    with col_tools1:
        st.markdown(
            """
            <div class="medical-safety-banner" style="margin: 0; padding: 10px 14px; font-size: 0.82rem;">
                ⚠️ <strong>Medical Notice:</strong> Answers are synthesized from your uploaded reference documents.
                This tool provides research and informational assistance only. It does not provide medical diagnoses or prescribe medications.
                For emergencies, seek immediate local emergency medical assistance.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_tools2:
        if st.button("➕ New Inquiry", key="btn_new_chat_session", type="primary", use_container_width=True):
            st.session_state.chat_messages = [
                {
                    "sender": "assistant",
                    "text": "Hello! I am your HealthAI Assistant. "
                            "Ask any question about your uploaded healthcare documents, clinical trial papers, or medical guidelines.",
                    "sources": [],
                    "retrieval_status": "initial",
                    "disclaimer": "This assistant provides educational and research assistance based strictly on your uploaded reference documents.",
                }
            ]
            st.session_state.active_conversation_id = None
            st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 4. Optional Advanced Retrieval Settings Drawer
    with st.expander("⚙️ Advanced / Knowledge Retrieval", expanded=False):
        col1, col2 = st.columns(2)
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
                help="Minimum relevance score required for a passage to be considered relevant evidence."
            )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 5. Render Dialogue History
    for msg in st.session_state.chat_messages:
        sender = msg.get("sender", "assistant")
        with st.chat_message(sender):
            msg_status = msg.get("retrieval_status")
            raw_sources = msg.get("sources") or []

            # If retrieval_status is not success or grounded_boundary, strictly enforce zero sources and no [Source N] tags
            if msg_status not in ("success", "grounded_boundary"):
                raw_sources = []
                msg["sources"] = []
                clean_text = re.sub(r'\[Source\s*(?:#|:)?\s*\d+\]', '', msg.get("text", ""), flags=re.IGNORECASE).strip()
            else:
                clean_text = msg.get("text", "")

            st.markdown(clean_ai_markdown(clean_text))

            # Render Grounded Sources strictly when retrieval_status is in ("success", "grounded_boundary") AND sources is not empty
            show_evidence = (msg_status in ("success", "grounded_boundary") and bool(raw_sources))
            if show_evidence:
                render_source_cards(raw_sources)

            # Response Details & Provenance Expander (Power User Drawer)
            timings = msg.get("timings")
            retrieval_status = msg_status
            if sender == "assistant" and (timings or show_evidence):
                with st.expander("🔍 Response Provenance & Latency Breakdown", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Retrieval Status:** `{retrieval_status or 'N/A'}`")
                    with c2:
                        total_ms = timings.get("total_time_ms", 0.0) if timings else 0.0
                        st.markdown(f"**Total Latency:** `{total_ms:.1f} ms`")
                    with c3:
                        st.markdown(f"**Unique Sources:** `{len(raw_sources) if show_evidence else 0}`")

                    if timings:
                        st.markdown("---")
                        st.markdown("**Latency Breakdown:**")
                        t1, t2, t3, t4 = st.columns(4)
                        with t1:
                            emb = timings.get("embedding_time_ms", timings.get("query_embedding_time_ms", 0.0))
                            st.caption(f"⚡ **Embedding:** {emb:.1f} ms")
                        with t2:
                            faiss_t = timings.get("faiss_retrieval_time_ms", timings.get("vector_search_time_ms", 0.0))
                            st.caption(f"🔎 **FAISS Search:** {faiss_t:.1f} ms")
                        with t3:
                            ctx_t = timings.get("context_construction_time_ms", 0.0)
                            st.caption(f"📄 **Context Prep:** {ctx_t:.1f} ms")
                        with t4:
                            llm_t = timings.get("llm_generation_time_ms", timings.get("generation_time_ms", 0.0))
                            st.caption(f"🤖 **Gemini Gen:** {llm_t:.1f} ms")

                        calls = timings.get("gemini_calls_count")
                        in_tok = timings.get("input_tokens")
                        out_tok = timings.get("output_tokens")
                        model_name = timings.get("model_used")
                        if calls is not None or in_tok is not None or model_name is not None:
                            d1, d2, d3 = st.columns(3)
                            with d1:
                                st.caption(f"🎯 **Model:** `{model_name or 'N/A'}`")
                            with d2:
                                st.caption(f"🔢 **Calls:** `{calls or 1}` request")
                            with d3:
                                st.caption(f"📊 **Tokens:** `{in_tok or 'N/A'}` in / `{out_tok or 'N/A'}` out")

    # 6. Chat Input Box (Visible at bottom with Send button)
    is_generating = st.session_state.get("is_generating", False)
    user_query = st.chat_input(
        "Ask a question about your uploaded healthcare documents...",
        disabled=is_generating
    )
    if user_query and not is_generating:
        st.session_state.is_generating = True

        # Append user message to UI state if not already the latest message
        if not st.session_state.chat_messages or st.session_state.chat_messages[-1].get("text") != user_query:
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

        # Record user message in DB if session is active
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

        # Extract immediate prior messages for conversation follow-up resolution
        recent_history = []
        for m in st.session_state.chat_messages:
            sender_role = m.get("sender")
            msg_txt = m.get("text", "")
            if sender_role in ("user", "assistant") and msg_txt and msg_txt != user_query:
                recent_history.append({"role": sender_role, "content": msg_txt})
        recent_history = recent_history[-4:]

        # Call RAG query API
        with st.chat_message("assistant"):
            with st.spinner("Searching your healthcare knowledge base..."):
                rag_res = api_client.query_rag(
                    question=user_query,
                    top_k=top_k if "top_k" in locals() else 5,
                    similarity_threshold=similarity_threshold if "similarity_threshold" in locals() else 0.25,
                    conversation_history=recent_history,
                )

            if rag_res.get("success"):
                data = rag_res.get("data", {})
                raw_answer = data.get("answer", "")
                sources = data.get("sources", []) or []
                retrieval_status = data.get("retrieval_status", "success")
                timings = data.get("timings", {})
                disclaimer = data.get("disclaimer", "")

                # Strict grounding enforcement:
                # If retrieval_status == 'no_relevant_context', force empty sources, hide cards, and strip citations
                if retrieval_status == "no_relevant_context":
                    sources = []
                    clean_answer = re.sub(r'\[Source\s*(?:#|:)?\s*\d+\]', '', raw_answer, flags=re.IGNORECASE).strip()
                    clean_answer = clean_ai_markdown(clean_answer)
                    show_evidence = False
                else:
                    clean_answer = clean_ai_markdown(raw_answer)
                    show_evidence = (retrieval_status in ("success", "grounded_boundary") and bool(sources))

                st.markdown(clean_answer)

                if show_evidence:
                    render_source_cards(sources)

                # Append assistant response to UI state
                st.session_state.chat_messages.append({
                    "sender": "assistant",
                    "text": clean_answer,
                    "sources": sources if show_evidence else [],
                    "retrieval_status": retrieval_status,
                    "timings": timings,
                    "disclaimer": disclaimer,
                })

                # Persist assistant message in DB
                if active_conv_id:
                    try:
                        api_client.add_message_to_conversation(
                            conversation_id=active_conv_id,
                            sender="assistant",
                            text=clean_answer,
                            citations=sources if show_evidence else [],
                            agent_type="rag_agent",
                        )
                    except Exception:
                        pass
            else:
                err = rag_res.get("error") or "An error occurred during medical query processing."
                st.error(f"Inquiry could not be completed: {err}")
                st.session_state.chat_messages.append({
                    "sender": "assistant",
                    "text": f"⚠️ {err}",
                    "sources": [],
                    "retrieval_status": "error",
                })

        st.session_state.is_generating = False
        st.rerun()


