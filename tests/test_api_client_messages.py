"""
Regression test for APIClient conversation message persistence.
Ensures that 'add_message_to_conversation' is implemented on APIClient
and cannot raise AttributeError.
"""

from unittest.mock import patch, MagicMock
import pytest
from frontend.api_client import APIClient, api_client


def test_api_client_has_add_message_to_conversation():
    """Verifies that APIClient exposes add_message_to_conversation."""
    client = APIClient()
    assert hasattr(client, "add_message_to_conversation")
    assert callable(client.add_message_to_conversation)
    assert hasattr(api_client, "add_message_to_conversation")
    assert callable(api_client.add_message_to_conversation)


def test_add_message_to_conversation_forwards_to_append():
    """Verifies add_message_to_conversation delegates to append_conversation_message."""
    client = APIClient(base_url="http://127.0.0.1:8000")
    client.set_token("test-jwt-token")

    with patch.object(client, "append_conversation_message") as mock_append:
        mock_append.return_value = {"success": True, "data": {"id": 42}, "error": None}

        res = client.add_message_to_conversation(
            conversation_id=10,
            sender="user",
            text="What is hypertension?",
            agent_type=None,
            citations=None,
        )

        assert res["success"] is True
        mock_append.assert_called_once_with(
            conversation_id=10,
            sender="user",
            text="What is hypertension?",
            agent_type=None,
            citations=None,
        )


def test_add_message_to_conversation_posts_correct_payload():
    """Verifies actual HTTP request payload and headers constructed."""
    client = APIClient(base_url="http://127.0.0.1:8000")
    client.set_token("test-jwt-token")

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "id": 1,
            "conversation_id": 10,
            "sender": "assistant",
            "text": "Hypertension is high blood pressure.",
            "agent_type": "rag_agent",
            "citations": [{"source_id": "doc4"}],
            "created_at": "2026-09-14T12:00:00Z"
        }
        mock_post.return_value = mock_resp

        res = client.add_message_to_conversation(
            conversation_id=10,
            sender="assistant",
            text="Hypertension is high blood pressure.",
            agent_type="rag_agent",
            citations=[{"source_id": "doc4"}],
        )

        assert res["success"] is True
        assert res["data"]["id"] == 1
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "http://127.0.0.1:8000/conversations/10/messages"
        assert call_kwargs["json"]["sender"] == "assistant"
        assert call_kwargs["json"]["text"] == "Hypertension is high blood pressure."
        assert call_kwargs["json"]["agent_type"] == "rag_agent"
        assert call_kwargs["json"]["citations"] == [{"source_id": "doc4"}]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-jwt-token"
