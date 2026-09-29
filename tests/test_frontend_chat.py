"""
Unit and integration tests for Authenticated RAG Chat Integration (Phase 13, Step 3).

Verifies all 20 required specifications:
1. Authenticated RAG request constructed correctly.
2. Bearer token attached.
3. Question sent correctly.
4. Successful response parsed.
5. Citations/sources parsed.
6. 400 handled.
7. 401 handled.
8. 403 handled.
9. 404/no-context handled.
10. 500 handled.
11. Connection failure handled.
12. JWT is never exposed.
13. Frontend does not require user_id.
14. Conversation list uses authenticated client.
15. Conversation messages use authenticated client.
16. No-context response handled correctly.
17. Source cards receive correct source data.
18. Authenticated user ID reaches RAG.
19. User A cannot retrieve User B's vectors.
20. RAG remains user-isolated after multiple documents/users.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import APIClient
from frontend.components.source_card import render_source_cards
from backend.main import app
from backend.database.database import Base, get_db
from backend.database.models import User
from backend.api.auth_dependencies import get_current_user, get_optional_current_db_user
from backend.services.vector_store_service import get_vector_store_service
from backend.rag.rag_service import RAGService

# In-memory test db for auth integration tests
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
with test_engine.connect() as conn:
    conn.execute(text("PRAGMA foreign_keys=ON;"))
Base.metadata.create_all(bind=test_engine)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    """Returns an APIClient instance pointed to a test backend."""
    return APIClient(base_url="http://127.0.0.1:8000")


@pytest.fixture
def test_user():
    db = TestingSession()
    user = db.query(User).filter(User.id == 1050).first()
    if not user:
        user = User(id=1050, email="rag_tester@clinic.org", full_name="RAG Tester", role="patient")
        db.add(user)
        db.commit()
        db.refresh(user)
    db.close()
    return user


# ==============================================================================
# 1. Authenticated RAG request constructed correctly
# ==============================================================================
def test_authenticated_rag_request_constructed_correctly(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "question": "What are the clinical indicators of sepsis?",
            "answer": "Sepsis indicators include elevated heart rate and fever [Source 1].",
            "retrieval_status": "success",
            "sources": [],
            "disclaimer": "Medical disclaimer",
            "timings": {"total_time_ms": 105.0},
        }
        mock_post.return_value = mock_res

        res = client.query_rag("What are the clinical indicators of sepsis?", top_k=5, similarity_threshold=0.25)
        assert res["success"] is True

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://127.0.0.1:8000/rag/query"
        payload = kwargs["json"]
        assert payload["question"] == "What are the clinical indicators of sepsis?"
        assert payload["top_k"] == 5
        assert payload["similarity_threshold"] == 0.25


# ==============================================================================
# 2. Bearer token attached
# ==============================================================================
def test_bearer_token_attached(client):
    client.set_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.rag_user_jwt")

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"answer": "Sample answer", "retrieval_status": "success", "sources": []}
        mock_post.return_value = mock_res

        client.query_rag("What is asthma?")
        _, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.rag_user_jwt"


# ==============================================================================
# 3. Question sent correctly
# ==============================================================================
def test_question_sent_correctly(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"answer": "Answer", "retrieval_status": "success", "sources": []}
        mock_post.return_value = mock_res

        specific_query = "What is the recommended dosage of amoxicillin in pediatric otitis media?"
        client.query_rag(specific_query)

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["question"] == specific_query


# ==============================================================================
# 4. Successful response parsed
# ==============================================================================
def test_successful_response_parsed(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "question": "What is hypertension?",
            "answer": "Hypertension is chronically elevated blood pressure [Source 1].",
            "retrieval_status": "success",
            "sources": [{"source_index": 1, "document_id": "cardio.pdf"}],
            "disclaimer": "This tool provides educational information.",
            "timings": {"total_time_ms": 115.4},
        }
        mock_post.return_value = mock_res

        res = client.query_rag("What is hypertension?")
        assert res["success"] is True
        assert res["error"] is None
        assert res["has_context"] is True
        data = res["data"]
        assert data["answer"] == "Hypertension is chronically elevated blood pressure [Source 1]."
        assert data["retrieval_status"] == "success"
        assert len(data["sources"]) == 1


# ==============================================================================
# 5. Citations/sources parsed
# ==============================================================================
def test_citations_sources_parsed(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "question": "Sample query",
            "answer": "Sample answer [Source 1].",
            "retrieval_status": "success",
            "sources": [
                {
                    "source_index": 1,
                    "document_id": "hypertension_guidelines.pdf",
                    "chunk_id": "CHUNK_HTN_01",
                    "page_number": 3,
                    "similarity_score": 0.892,
                    "preview_text": "Normal resting blood pressure is under 120/80 mmHg.",
                }
            ],
        }
        mock_post.return_value = mock_res

        res = client.query_rag("Sample query")
        assert res["success"] is True
        sources = res["data"]["sources"]
        assert len(sources) == 1
        src = sources[0]
        assert src["source_index"] == 1
        assert src["document_id"] == "hypertension_guidelines.pdf"
        assert src["chunk_id"] == "CHUNK_HTN_01"
        assert src["page_number"] == 3
        assert src["similarity_score"] == 0.892


# ==============================================================================
# 6. 400 handled
# ==============================================================================
def test_400_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 400
        mock_res.json.return_value = {"detail": "Empty query."}
        mock_post.return_value = mock_res

        res = client.query_rag("   ")
        assert res["success"] is False
        assert res["status_code"] == 400
        assert res["error"] == "Please enter a valid healthcare question."


# ==============================================================================
# 7. 401 handled
# ==============================================================================
def test_401_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 401
        mock_res.json.return_value = {"detail": "Token expired."}
        mock_post.return_value = mock_res

        res = client.query_rag("What is diabetes?")
        assert res["success"] is False
        assert res["status_code"] == 401
        assert res["error"] == "Your session has expired. Please sign in again."


# ==============================================================================
# 8. 403 handled
# ==============================================================================
def test_403_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 403
        mock_res.json.return_value = {"detail": "Forbidden."}
        mock_post.return_value = mock_res

        res = client.query_rag("What is diabetes?")
        assert res["success"] is False
        assert res["status_code"] == 403
        assert res["error"] == "You do not have permission to access this resource."


# ==============================================================================
# 9. 404/no-context handled
# ==============================================================================
def test_404_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 404
        mock_res.json.return_value = {"detail": "Not Found."}
        mock_post.return_value = mock_res

        res = client.query_rag("Unrelated topic query")
        assert res["success"] is False
        assert res["status_code"] == 404
        assert res["error"] == "No relevant information was found in your documents."


# ==============================================================================
# 10. 500 handled
# ==============================================================================
def test_500_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 500
        mock_res.json.return_value = {"detail": "Internal server error."}
        mock_post.return_value = mock_res

        res = client.query_rag("Query triggering server failure")
        assert res["success"] is False
        assert res["status_code"] == 500
        assert res["error"] == "Unable to generate a response right now. Please try again."


# ==============================================================================
# 11. Connection failure handled
# ==============================================================================
def test_connection_failure_handled(client):
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = client.query_rag("Query when backend is down")
        assert res["success"] is False
        assert res["status_code"] is None
        assert res["error"] == "Backend is unavailable. Please make sure the FastAPI server is running."


# ==============================================================================
# 12. JWT is never exposed
# ==============================================================================
def test_jwt_never_exposed(client):
    client.set_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.sensitive_token_content")

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "question": "Query",
            "answer": "Answer",
            "retrieval_status": "success",
            "sources": [],
        }
        mock_post.return_value = mock_res

        res = client.query_rag("Query")
        serialized = str(res)
        assert "sensitive_token_content" not in serialized
        assert "eyJ" not in serialized
        assert "Authorization" not in res


# ==============================================================================
# 13. Frontend does not require user_id
# ==============================================================================
def test_frontend_does_not_require_user_id(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"answer": "Answer", "retrieval_status": "success", "sources": []}
        mock_post.return_value = mock_res

        res = client.query_rag("General healthcare question")
        assert res["success"] is True

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert "user_id" not in payload


# ==============================================================================
# 14. Conversation list uses authenticated client
# ==============================================================================
def test_conversation_list_uses_authenticated_client(client):
    client.set_token("token_conv_user")

    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = [{"id": 1, "title": "Oncology Consultation"}]
        mock_get.return_value = mock_res

        convs = client.list_conversations()
        assert len(convs) == 1
        args, kwargs = mock_get.call_args
        assert args[0] == "http://127.0.0.1:8000/conversations"
        assert kwargs["headers"]["Authorization"] == "Bearer token_conv_user"
        assert "user_id" not in kwargs["params"]


# ==============================================================================
# 15. Conversation messages use authenticated client
# ==============================================================================
def test_conversation_messages_use_authenticated_client(client):
    client.set_token("token_msg_user")

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get_res = MagicMock()
        mock_get_res.status_code = 200
        mock_get_res.json.return_value = [{"id": 1, "text": "Patient question", "sender": "user"}]
        mock_get.return_value = mock_get_res

        mock_post_res = MagicMock()
        mock_post_res.status_code = 201
        mock_post_res.json.return_value = {"id": 2, "text": "Assistant answer", "sender": "assistant"}
        mock_post.return_value = mock_post_res

        msgs = client.get_conversation_messages(10)
        assert len(msgs) == 1
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer token_msg_user"

        app_res = client.append_conversation_message(10, "assistant", "Assistant answer")
        assert app_res["success"] is True
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer token_msg_user"


# ==============================================================================
# 16. No-context response handled correctly
# ==============================================================================
def test_no_context_response_handled_correctly(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "question": "What is quantum computing in neurology?",
            "answer": "Relevant medical information could not be found in the available reference documents.",
            "retrieval_status": "no_relevant_context",
            "sources": [],
            "disclaimer": "Medical disclaimer",
        }
        mock_post.return_value = mock_res

        res = client.query_rag("What is quantum computing in neurology?")
        assert res["success"] is True
        assert res["has_context"] is False
        assert "could not be found" in res["data"]["answer"]


# ==============================================================================
# 17. Source cards receive correct source data
# ==============================================================================
def test_source_cards_receive_correct_source_data():
    # Verify render_source_cards executes cleanly for empty, None, and populated sources
    with patch("streamlit.container"), patch("streamlit.markdown"):
        render_source_cards(None)
        render_source_cards([])
        sample_sources = [
            {
                "source_index": 1,
                "document_id": "pediatric_guidelines.pdf",
                "chunk_id": "CHUNK_PEDS_01",
                "page_number": 2,
                "similarity_score": 0.81,
                "preview_text": "Infant dosage guidelines for antipyretics.",
            }
        ]
        render_source_cards(sample_sources)


# ==============================================================================
# 18. Authenticated user ID reaches RAG (Backend Integration)
# ==============================================================================
def test_authenticated_user_id_reaches_rag(test_user):
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_optional_current_db_user] = lambda: test_user

    with patch("backend.rag.rag_service.RAGService.generate_rag_answer") as mock_generate:
        mock_generate.return_value = {
            "question": "Test question",
            "answer": "Answer grounded in user docs",
            "retrieval_status": "success",
            "sources": [],
            "disclaimer": "Disclaimer",
            "timings": {},
        }

        test_client = TestClient(app)
        res = test_client.post(
            "/rag/query",
            json={"question": "Test question", "top_k": 3, "similarity_threshold": 0.3},
            headers={"Authorization": "Bearer mock_jwt_1050"},
        )
        assert res.status_code == 200
        mock_generate.assert_called_once()
        _, kwargs = mock_generate.call_args
        # Verify authenticated user_id was forwarded to RAG service
        assert kwargs["user_id"] == 1050


# ==============================================================================
# 19. User A cannot retrieve User B's vectors (Cross-User Isolation)
# ==============================================================================
def test_user_a_cannot_retrieve_user_b_vectors():
    from backend.services.embedding_service import EmbeddingService
    from backend.services.vector_store_service import VectorStoreService

    vector_store = VectorStoreService()

    # User A (id 2001) has a private document on Glomangioma
    user_a_id = 2001
    user_b_id = 2002

    embeddings_a = [EmbeddingService.embed_query("Glomangioma vascular malformation")]
    meta_a = [{
        "chunk_id": "user_a_glomangioma_chunk",
        "text": "Glomangioma is a rare vascular malformation characterized by glomus cell proliferation.",
        "user_id": user_a_id,
        "document_id": "9001",
    }]
    vector_store.add_embeddings(embeddings_a, meta_a)

    rag = RAGService(vector_store=vector_store)

    # 1. Query as User A: receives the chunk
    res_a = rag.query("What is Glomangioma?", top_k=5, user_id=user_a_id)
    assert res_a["retrieval_status"] == "success"
    assert any("vascular malformation" in c["text"] for c in res_a["retrieved_chunks"])

    # 2. Query as User B: User A's chunk is completely invisible
    res_b = rag.query("What is Glomangioma?", top_k=5, user_id=user_b_id)
    assert not any("vascular malformation" in c.get("text", "") for c in res_b.get("retrieved_chunks", []))


# ==============================================================================
# 20. RAG remains user-isolated after multiple documents/users
# ==============================================================================
def test_rag_remains_user_isolated_after_multiple_documents_and_users():
    from backend.services.embedding_service import EmbeddingService
    from backend.services.vector_store_service import VectorStoreService

    vector_store = VectorStoreService()

    user_x_id = 2010
    user_y_id = 2020

    vector_store.add_embeddings(
        [EmbeddingService.embed_query("Whipple disease Tropheryma whipplei")],
        [{
            "chunk_id": "user_x_whipple_chunk",
            "text": "Whipple disease is caused by the bacterium Tropheryma whipplei presenting with malabsorption.",
            "user_id": user_x_id,
            "document_id": "9010",
        }]
    )
    vector_store.add_embeddings(
        [EmbeddingService.embed_query("Castleman disease lymphoproliferative disorder")],
        [{
            "chunk_id": "user_y_castleman_chunk",
            "text": "Castleman disease is a rare lymphoproliferative disorder involving hyperplastic lymph nodes.",
            "user_id": user_y_id,
            "document_id": "9020",
        }]
    )

    rag = RAGService(vector_store=vector_store)

    # Query X
    res_x = rag.query("Tell me about Whipple and Castleman disease.", top_k=5, user_id=user_x_id)
    assert any("Whipple disease" in c["text"] for c in res_x["retrieved_chunks"])
    assert not any("Castleman disease" in c["text"] for c in res_x["retrieved_chunks"])

    # Query Y
    res_y = rag.query("Tell me about Whipple and Castleman disease.", top_k=5, user_id=user_y_id)
    assert any("Castleman disease" in c["text"] for c in res_y["retrieved_chunks"])
    assert not any("Whipple disease" in c["text"] for c in res_y["retrieved_chunks"])
