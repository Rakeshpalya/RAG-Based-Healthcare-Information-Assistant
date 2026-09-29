from frontend.pages.chat import render_chat_page


def render_chat_interface():
    """
    Renders the clinical chat interface connected directly to authenticated RAG.
    Maintains backward compatibility with Phase 11.
    """
    return render_chat_page()

