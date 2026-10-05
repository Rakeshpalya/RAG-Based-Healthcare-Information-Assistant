"""
Clinical Conversation History Page.

Renders past research consultations, chronological message turns,
and verified source citations with instant session resumption capabilities.
Provides search filtering, per-conversation deletion, and clear history with confirmation.
"""

from typing import Callable, Optional
import streamlit as st
from frontend.api_client import api_client
from frontend.components.source_card import render_source_cards
from frontend.utils.helpers import format_timestamp, clean_ai_markdown


def render_history_page(set_page_fn: Optional[Callable[[str], None]] = None):
    """
    Renders the clinical conversation history interface.
    """
    # 1. Header & Top Toolbar
    col_hdr, col_clear = st.columns([3, 1])
    with col_hdr:
        st.markdown(
            """
            <div style="margin-bottom: 20px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--teal-primary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">
                    Consultation Archives
                </div>
                <h1 style="color: var(--text-primary); font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                    Research History
                </h1>
                <p style="color: var(--text-secondary); font-size: 1rem; margin-top: 6px; margin-bottom: 0; line-height: 1.5;">
                    Review past patient dialogues, research inquiries, and verified clinical citations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. Fetch conversations
    conversations = api_client.list_conversations(limit=100)

    # Clear History Button in Top Toolbar (if conversations exist)
    with col_clear:
        if conversations:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear History", key="btn_trigger_clear_history", type="secondary", use_container_width=True):
                st.session_state.show_clear_confirm = True
                st.rerun()

    # 3. Clear History Confirmation Box
    if st.session_state.get("show_clear_confirm", False):
        st.markdown(
            """
            <div class="history-danger-box">
                <strong>⚠️ Confirm Clear History:</strong> Are you sure you want to permanently delete all your conversation history?
                This action only deletes your own conversations and cannot be undone.
            </div>
            """,
            unsafe_allow_html=True
        )
        c_yes, c_no = st.columns([1, 1])
        with c_yes:
            if st.button("Yes, Clear All History", key="btn_confirm_clear_history", type="primary", use_container_width=True):
                res = api_client.clear_conversations()
                st.session_state.show_clear_confirm = False
                st.session_state.selected_history_conv_id = None
                st.session_state.active_conversation_id = None
                if res.get("success"):
                    st.success(f"History cleared ({res.get('deleted_count', 0)} sessions removed).")
                else:
                    st.error(res.get("error", "Failed to clear history."))
                st.rerun()
        with c_no:
            if st.button("Cancel", key="btn_cancel_clear_history", type="secondary", use_container_width=True):
                st.session_state.show_clear_confirm = False
                st.rerun()
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    if not conversations:
        st.info("No prior research sessions recorded. Head to Research Chat to begin your first inquiry!")
        return

    # 4. Search Filter
    col_conv_list, col_conv_view = st.columns([1, 2])

    with col_conv_list:
        search_query = st.text_input(
            "Search conversations",
            placeholder="🔍 Search conversations...",
            key="input_search_history",
            label_visibility="collapsed",
        )

        filtered_conversations = conversations
        if search_query.strip():
            q = search_query.strip().lower()
            filtered_conversations = [c for c in conversations if q in c.get("title", "").lower()]

        st.markdown(
            f"""
            <div style="font-size: 0.84rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin: 12px 0 10px 0;">
                Saved Sessions ({len(filtered_conversations)})
            </div>
            """,
            unsafe_allow_html=True
        )

        if not filtered_conversations:
            st.caption("No conversations match your search.")
        else:
            for conv in filtered_conversations:
                c_id = conv.get("id")
                title = conv.get("title", "Consultation")
                created_at = format_timestamp(conv.get("created_at"))
                is_active = st.session_state.get("selected_history_conv_id") == c_id

                short_title = (title[:22] + "...") if len(title) > 22 else title
                btn_label = f"💬 {short_title}"

                c_item_btn, c_item_del = st.columns([5, 1])
                with c_item_btn:
                    if st.button(
                        btn_label,
                        key=f"btn_conv_{c_id}",
                        help=f"{title}\nCreated: {created_at}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.selected_history_conv_id = c_id
                        st.rerun()
                with c_item_del:
                    if st.button(
                        "🗑️",
                        key=f"btn_del_item_{c_id}",
                        help=f"Delete '{title}'",
                        type="secondary",
                        use_container_width=True,
                    ):
                        del_res = api_client.delete_conversation(c_id)
                        if del_res.get("success"):
                            if st.session_state.get("selected_history_conv_id") == c_id:
                                st.session_state.selected_history_conv_id = None
                            if st.session_state.get("active_conversation_id") == c_id:
                                st.session_state.active_conversation_id = None
                            st.rerun()
                        else:
                            st.error(del_res.get("error", "Could not delete conversation."))

    with col_conv_view:
        active_id = st.session_state.get("selected_history_conv_id")
        # Default active selection to first matching conversation if none selected
        if (not active_id or not any(c.get("id") == active_id for c in filtered_conversations)) and filtered_conversations:
            active_id = filtered_conversations[0].get("id")
            st.session_state.selected_history_conv_id = active_id

        if active_id:
            active_conv = next((c for c in conversations if c.get("id") == active_id), None)
            if not active_conv and filtered_conversations:
                active_conv = filtered_conversations[0]
                active_id = active_conv.get("id")

        if active_id and active_conv:
            conv_title = active_conv.get("title", "Research Consultation Thread")
            created = format_timestamp(active_conv.get("created_at"))

            st.markdown(
                f"""
                <div class="health-card" style="margin-bottom: 16px;">
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: var(--text-primary);">{conv_title}</h2>
                    <div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">
                        Session Started: {created} • Conversation ID: <code>{active_id}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Action Buttons: Resume Session, Rename, and Delete Session
            col_act_resume, col_act_rename, col_act_del = st.columns([3, 1.2, 1])
            with col_act_resume:
                if st.button("💬 Resume Session in Chat →", key=f"btn_resume_chat_{active_id}", type="primary", use_container_width=True):
                    hist_messages = api_client.get_conversation_messages(active_id)
                    st.session_state.chat_messages = [
                        {
                            "sender": m.get("sender", "user"),
                            "text": clean_ai_markdown(m.get("text", "")),
                            "agent_type": m.get("agent_type"),
                            "sources": m.get("citations") if isinstance(m.get("citations"), list) else [],
                            "retrieval_status": "success",
                        }
                        for m in hist_messages
                    ]
                    st.session_state.active_conversation_id = active_id
                    if set_page_fn:
                        set_page_fn("Research Chat")
                    else:
                        st.session_state.current_page = "Research Chat"
                    st.rerun()

            with col_act_rename:
                if st.button("✏️ Rename", key=f"btn_rename_{active_id}", type="secondary", use_container_width=True, help="Rename conversation"):
                    st.session_state[f"show_rename_{active_id}"] = not st.session_state.get(f"show_rename_{active_id}", False)
                    st.rerun()

            with col_act_del:
                if st.button("🗑️ Delete", key=f"btn_delete_active_{active_id}", type="secondary", use_container_width=True, help="Delete this conversation"):
                    del_res = api_client.delete_conversation(active_id)
                    if del_res.get("success"):
                        st.session_state.selected_history_conv_id = None
                        if st.session_state.get("active_conversation_id") == active_id:
                            st.session_state.active_conversation_id = None
                        st.rerun()
                    else:
                        st.error(del_res.get("error", "Failed to delete conversation."))

            # Inline Rename Form
            if st.session_state.get(f"show_rename_{active_id}", False):
                with st.form(f"form_rename_{active_id}", clear_on_submit=False):
                    new_t = st.text_input("New Conversation Title", value=conv_title, key=f"input_rename_{active_id}")
                    c_save, c_cancel = st.columns(2)
                    with c_save:
                        if st.form_submit_button("Save Title", type="primary", use_container_width=True):
                            if new_t.strip():
                                r_res = api_client.rename_conversation(active_id, new_t.strip())
                                st.session_state[f"show_rename_{active_id}"] = False
                                if r_res.get("success"):
                                    st.success("Conversation renamed.")
                                else:
                                    st.error(r_res.get("error", "Failed to rename."))
                                st.rerun()
                    with c_cancel:
                        if st.form_submit_button("Cancel", use_container_width=True):
                            st.session_state[f"show_rename_{active_id}"] = False
                            st.rerun()

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Fetch chronological message turns
            messages = api_client.get_conversation_messages(active_id)
            if not messages:
                st.info("This session has no recorded dialogue turns.")
            else:
                for msg in messages:
                    sender = msg.get("sender", "user")
                    text = clean_ai_markdown(msg.get("text", ""))
                    citations = msg.get("citations")

                    with st.chat_message(sender):
                        st.markdown(text)
                        if citations and isinstance(citations, list):
                            render_source_cards(citations)

