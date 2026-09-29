"""
Centralized HTTP Client for the AI-Healthcare-Agent Streamlit frontend.

Handles communication with the FastAPI backend, authentication token lifecycle,
structured error handling, timeouts, and resource isolation headers.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

# Ensure environment variables are loaded from root .env file
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


def resolve_backend_url(base_url: Optional[str] = None) -> str:
    """
    Resolves the FastAPI backend base URL following strict priority:
    1. Explicit base_url argument
    2. API_BASE_URL environment variable
    3. BACKEND_URL environment variable
    4. FASTAPI_URL environment variable
    5. http://{HOST}:{PORT} if both exist in environment
    6. Fallback: http://127.0.0.1:8000
    Normalizes 'localhost' to '127.0.0.1' for Windows IPv4 connection stability.
    """
    url = (
        base_url
        or os.getenv("API_BASE_URL")
        or os.getenv("BACKEND_URL")
        or os.getenv("FASTAPI_URL")
    )
    if not url:
        host = os.getenv("HOST", "127.0.0.1")
        port = os.getenv("PORT", "8000")
        url = f"http://{host}:{port}"

    url = url.strip().rstrip("/")
    if url.startswith("http://localhost:"):
        url = url.replace("http://localhost:", "http://127.0.0.1:", 1)
    elif url.startswith("https://localhost:"):
        url = url.replace("https://localhost:", "https://127.0.0.1:", 1)
    elif url == "http://localhost":
        url = "http://127.0.0.1"
    elif url == "https://localhost":
        url = "https://127.0.0.1"
    return url


class APIClient:
    """
    Centralized HTTP client for communicating with the AI-Healthcare-Agent FastAPI backend.
    Enforces standardized error handling, timeouts, bearer authentication, and structured response objects.
    """

    DEFAULT_TIMEOUT_FAST = 15.0
    DEFAULT_TIMEOUT_HEALTH = 3.0
    DEFAULT_TIMEOUT_LONG = 60.0

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = resolve_backend_url(base_url)
        self.token: Optional[str] = None

    def set_token(self, token: Optional[str]) -> None:
        """Sets or clears the active JWT access token for authenticated requests."""
        self.token = token.strip() if token else None

    def get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Constructs standard HTTP request headers, injecting Authorization Bearer
        token when authenticated.
        """
        headers: Dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if additional_headers:
            headers.update(additional_headers)
        return headers

    # ==========================================================================
    # Authentication Methods (Phase 12 Step 7)
    # ==========================================================================

    def signup(self, email: str, password: str) -> Dict[str, Any]:
        """
        Registers a new user via POST /auth/signup.
        Detects whether email confirmation is required.
        """
        url = f"{self.base_url}/auth/signup"
        payload = {"email": email, "password": password}
        try:
            response = requests.post(url, json=payload, timeout=self.DEFAULT_TIMEOUT_FAST)
            if response.status_code == 201:
                data = response.json()
                session = data.get("session")
                return {
                    "success": True,
                    "data": data,
                    "needs_email_confirmation": session is None,
                    "error": None,
                }
            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            return {"success": False, "data": None, "error": err_msg}
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "data": None,
                "error": "Cannot connect to backend server. Verify FastAPI is running.",
            }
        except requests.exceptions.Timeout:
            return {"success": False, "data": None, "error": "Registration request timed out."}
        except Exception as e:
            return {"success": False, "data": None, "error": f"Registration failed: {str(e)}"}

    def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticates a user via POST /auth/login.
        On success, stores the returned JWT access token on the client.
        """
        url = f"{self.base_url}/auth/login"
        payload = {"email": email, "password": password}
        try:
            response = requests.post(url, json=payload, timeout=self.DEFAULT_TIMEOUT_FAST)
            if response.status_code == 200:
                data = response.json()
                session = data.get("session") or {}
                token = session.get("access_token")
                user = data.get("user")
                if token:
                    self.set_token(token)
                return {
                    "success": True,
                    "token": token,
                    "user": user,
                    "data": data,
                    "error": None,
                }
            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            return {"success": False, "token": None, "user": None, "error": err_msg, "status_code": response.status_code}
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "token": None,
                "user": None,
                "error": "Cannot connect to backend server. Verify FastAPI is running.",
            }
        except requests.exceptions.Timeout:
            return {"success": False, "token": None, "user": None, "error": "Login request timed out."}
        except Exception as e:
            return {"success": False, "token": None, "user": None, "error": f"Login failed: {str(e)}"}

    def logout(self) -> Dict[str, Any]:
        """
        Terminates the user session via POST /auth/logout and clears active token.
        """
        url = f"{self.base_url}/auth/logout"
        try:
            requests.post(url, headers=self.get_headers(), timeout=self.DEFAULT_TIMEOUT_FAST)
        except Exception:
            pass
        finally:
            self.set_token(None)
        return {"success": True, "error": None}

    def get_current_user(self) -> Dict[str, Any]:
        """
        Fetches current authenticated user profile via GET /auth/me.
        """
        url = f"{self.base_url}/auth/me"
        try:
            response = requests.get(url, headers=self.get_headers(), timeout=self.DEFAULT_TIMEOUT_FAST)
            if response.status_code == 200:
                data = response.json()
                return {"success": True, "user": data.get("user"), "error": None}
            if response.status_code == 401:
                self.set_token(None)
                return {
                    "success": False,
                    "user": None,
                    "error": "Not authenticated or session expired.",
                    "status_code": 401,
                }
            if response.status_code == 403:
                return {
                    "success": False,
                    "user": None,
                    "error": "Access forbidden.",
                    "status_code": 403,
                }
            return {
                "success": False,
                "user": None,
                "error": f"Failed to retrieve user: {response.text}",
                "status_code": response.status_code,
            }
        except Exception as e:
            return {"success": False, "user": None, "error": str(e)}

    # ==========================================================================
    # System & Document Operations
    # ==========================================================================

    def check_backend_health(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Pings the real FastAPI /health endpoint to verify connectivity and operational status.
        Uses a short, responsive timeout to prevent UI freezing. Never leaks secrets or raw tracebacks.
        """
        req_timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT_HEALTH
        url = f"{self.base_url}/health"
        try:
            response = requests.get(url, timeout=req_timeout)
            if response.status_code == 200:
                data = response.json()
                return {
                    "connected": True,
                    "status": data.get("status", "healthy"),
                    "service": data.get("service", "AI-Healthcare-Agent"),
                    "environment": data.get("environment", "development"),
                    "url": self.base_url,
                    "error": None,
                }
            return {
                "connected": False,
                "status": "error",
                "service": "AI-Healthcare-Agent",
                "environment": "unknown",
                "url": self.base_url,
                "error": f"Backend returned status {response.status_code}.",
            }
        except requests.exceptions.ConnectionError:
            return {
                "connected": False,
                "status": "disconnected",
                "service": "AI-Healthcare-Agent",
                "environment": "unknown",
                "url": self.base_url,
                "error": "Unable to connect to the backend server at " + self.base_url + ". Ensure FastAPI is running.",
            }
        except requests.exceptions.Timeout:
            timeout_int = int(req_timeout)
            return {
                "connected": False,
                "status": "timeout",
                "service": "AI-Healthcare-Agent",
                "environment": "unknown",
                "url": self.base_url,
                "error": f"Connection to the backend timed out after {timeout_int} seconds.",
            }
        except Exception as e:
            return {
                "connected": False,
                "status": "error",
                "service": "AI-Healthcare-Agent",
                "environment": "unknown",
                "url": self.base_url,
                "error": f"Unexpected health check error: {str(e)}",
            }

    def health_check(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Pings the backend health endpoint to check connection and operational status.
        Defaults to DEFAULT_TIMEOUT_FAST (5.0s) for backward-compatible test contracts.
        """
        req_timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT_FAST
        return self.check_backend_health(timeout=req_timeout)

    def upload_document(self, file: Any, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Uploads and ingests a medical PDF via authenticated POST /documents/upload.
        Extracts text, generates chunks/embeddings, indexes into FAISS, and persists metadata.

        Accepts either raw bytes or a file-like object (e.g. Streamlit UploadedFile).
        Automatically attaches active Bearer token and handles HTTP errors with user-friendly messages.
        """
        if hasattr(file, "getvalue"):
            file_bytes = file.getvalue()
            file_name = filename or getattr(file, "name", "document.pdf")
        elif hasattr(file, "read"):
            file_bytes = file.read()
            file_name = filename or getattr(file, "name", "document.pdf")
        elif isinstance(file, bytes):
            file_bytes = file
            file_name = filename or "document.pdf"
        else:
            file_bytes = bytes(file)
            file_name = filename or "document.pdf"

        url = f"{self.base_url}/documents/upload"
        files = {"file": (file_name, file_bytes, "application/pdf")}
        headers = self.get_headers()

        try:
            response = requests.post(url, files=files, headers=headers, timeout=self.DEFAULT_TIMEOUT_LONG)
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "error": None,
                    "status_code": 200,
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "data": None,
                    "error": "Your session has expired. Please sign in again.",
                    "status_code": 401,
                }
            elif response.status_code == 403:
                return {
                    "success": False,
                    "data": None,
                    "error": "You do not have permission to upload this document.",
                    "status_code": 403,
                }
            elif response.status_code == 413:
                return {
                    "success": False,
                    "data": None,
                    "error": "The PDF is too large. Maximum size is 10 MB.",
                    "status_code": 413,
                }
            elif response.status_code == 500:
                return {
                    "success": False,
                    "data": None,
                    "error": "Document processing failed. Please try again.",
                    "status_code": 500,
                }
            elif response.status_code == 400:
                detail = None
                try:
                    detail = response.json().get("detail")
                except Exception:
                    pass
                err_msg = detail if detail else "Unable to process this PDF. Please check that it contains readable text."
                return {
                    "success": False,
                    "data": None,
                    "error": err_msg,
                    "status_code": 400,
                }
            else:
                detail = None
                try:
                    detail = response.json().get("detail")
                except Exception:
                    pass
                err_msg = detail if detail else f"Document processing failed with status {response.status_code}."
                return {
                    "success": False,
                    "data": None,
                    "error": err_msg,
                    "status_code": response.status_code,
                }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "data": None,
                "error": "Backend is unavailable. Please make sure the FastAPI server is running.",
                "status_code": None,
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "data": None,
                "error": "Document upload and processing timed out.",
                "status_code": None,
            }
        except Exception:
            return {
                "success": False,
                "data": None,
                "error": "Document processing failed. Please try again.",
                "status_code": None,
            }

    def register_document_metadata(
        self,
        filename: str,
        file_path: str,
        file_size_bytes: Optional[int] = None,
        num_pages: Optional[int] = None,
        num_chunks: int = 0,
        status: str = "processed",
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Registers ingested document metadata in the relational database via POST /documents/metadata.
        Includes Bearer token for user-isolated ownership assignment.
        """
        url = f"{self.base_url}/documents/metadata"
        payload = {
            "filename": filename,
            "file_path": file_path,
            "file_size_bytes": file_size_bytes,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "status": status,
        }
        if user_id is not None:
            payload["user_id"] = user_id

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code == 201:
                return {"success": True, "data": response.json(), "error": None}
            if response.status_code == 401:
                return {"success": False, "data": None, "error": "Session expired or unauthenticated.", "status_code": 401}
            if response.status_code == 403:
                return {"success": False, "data": None, "error": "Access denied.", "status_code": 403}

            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            return {"success": False, "data": None, "error": err_msg}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def list_documents(self, user_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetches list of documents from GET /documents strictly isolated to authenticated user.
        """
        url = f"{self.base_url}/documents"
        params = {"skip": skip, "limit": limit}
        if user_id is not None:
            params["user_id"] = user_id
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def get_document(self, document_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves specific document details from GET /documents/{document_id}.
        """
        url = f"{self.base_url}/documents/{document_id}"
        try:
            response = requests.get(
                url,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def send_agent_query(
        self,
        question: str,
        explanation_level: str = "simple",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Sends an inquiry to the Agentic AI Orchestrator via POST /agent/query.
        """
        url = f"{self.base_url}/agent/query"
        payload = {
            "question": question,
            "explanation_level": explanation_level,
            "top_k": top_k,
        }
        try:
            response = requests.post(url, json=payload, headers=self.get_headers(), timeout=self.DEFAULT_TIMEOUT_LONG)
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}

            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            return {"success": False, "data": None, "error": err_msg}
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "data": None,
                "error": "Failed to connect to agent service. Ensure FastAPI backend is active.",
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "data": None,
                "error": "Agent query processing timed out. Please try a simpler question.",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": f"Query execution failed: {str(e)}"}

    def list_conversations(self, user_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetches conversation sessions from GET /conversations isolated to authenticated user.
        """
        url = f"{self.base_url}/conversations"
        params = {"skip": skip, "limit": limit}
        if user_id is not None:
            params["user_id"] = user_id
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def create_conversation(
        self,
        title: str = "Healthcare Consultation",
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Creates a new conversation session via POST /conversations.
        """
        url = f"{self.base_url}/conversations"
        payload: Dict[str, Any] = {"title": title}
        if user_id is not None:
            payload["user_id"] = user_id

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code == 201:
                return {"success": True, "data": response.json(), "error": None}
            if response.status_code == 401:
                return {"success": False, "data": None, "error": "Session expired or unauthenticated.", "status_code": 401}
            if response.status_code == 403:
                return {"success": False, "data": None, "error": "Access denied.", "status_code": 403}

            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            return {"success": False, "data": None, "error": err_msg}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def get_conversation_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        """
        Fetches chronological messages for a conversation via GET /conversations/{id}/messages.
        """
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        try:
            response = requests.get(
                url,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def append_conversation_message(
        self,
        conversation_id: int,
        sender: str,
        text: str,
        agent_type: Optional[str] = None,
        citations: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Appends a message turn via POST /conversations/{id}/messages.
        """
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {
            "sender": sender,
            "text": text,
            "agent_type": agent_type,
            "citations": citations,
        }
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code == 201:
                return {"success": True, "data": response.json(), "error": None}
            if response.status_code == 401:
                return {"success": False, "data": None, "error": "Session expired or unauthenticated.", "status_code": 401}
            if response.status_code == 403:
                return {"success": False, "data": None, "error": "Access denied.", "status_code": 403}

            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            return {"success": False, "data": None, "error": err_msg}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def add_message_to_conversation(
        self,
        conversation_id: int,
        sender: str,
        text: str,
        agent_type: Optional[str] = None,
        citations: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Appends a new message turn to a conversation session via POST /conversations/{id}/messages.
        First-class method wrapper for append_conversation_message.
        """
        return self.append_conversation_message(
            conversation_id=conversation_id,
            sender=sender,
            text=text,
            agent_type=agent_type,
            citations=citations,
        )

    def delete_conversation(self, conversation_id: int) -> Dict[str, Any]:
        """
        Deletes a conversation session via DELETE /conversations/{id}.
        Strictly isolated to authenticated user.
        """
        url = f"{self.base_url}/conversations/{conversation_id}"
        try:
            response = requests.delete(
                url,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code in (200, 204):
                return {"success": True, "error": None}
            if response.status_code == 401:
                return {"success": False, "error": "Session expired or unauthenticated."}
            if response.status_code == 403:
                return {"success": False, "error": "Access denied: Cannot delete another user's conversation."}
            if response.status_code == 404:
                return {"success": False, "error": "Conversation not found."}
            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            return {"success": False, "error": err_msg}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def clear_conversations(self) -> Dict[str, Any]:
        """
        Deletes all conversations for the authenticated user via DELETE /conversations.
        Strictly isolated to authenticated user.
        """
        url = f"{self.base_url}/conversations"
        try:
            response = requests.delete(
                url,
                headers=self.get_headers(),
                timeout=self.DEFAULT_TIMEOUT_FAST,
            )
            if response.status_code in (200, 204):
                data = response.json() if response.content else {}
                return {"success": True, "deleted_count": data.get("deleted_count", 0), "error": None}
            if response.status_code == 401:
                return {"success": False, "error": "Session expired or unauthenticated."}
            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            return {"success": False, "error": err_msg}
        except Exception as e:
            return {"success": False, "error": str(e)}


    def query_rag(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.25,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Sends an authenticated RAG query to POST /rag/query.
        Attaches the active Bearer token for user ownership isolation in FAISS.
        """
        url = f"{self.base_url}/rag/query"
        payload = {
            "question": question,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "conversation_history": conversation_history,
        }
        headers = self.get_headers()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.DEFAULT_TIMEOUT_LONG)
            if response.status_code == 200:
                data = response.json()
                retrieval_status = data.get("retrieval_status")
                is_no_context = retrieval_status == "no_relevant_context"
                return {
                    "success": True,
                    "data": data,
                    "has_context": not is_no_context,
                    "error": None,
                    "status_code": 200,
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "data": None,
                    "error": "Your session has expired. Please sign in again.",
                    "status_code": 401,
                }
            elif response.status_code == 403:
                return {
                    "success": False,
                    "data": None,
                    "error": "You do not have permission to access this resource.",
                    "status_code": 403,
                }
            elif response.status_code == 400:
                return {
                    "success": False,
                    "data": None,
                    "error": "Please enter a valid healthcare question.",
                    "status_code": 400,
                }
            elif response.status_code == 404:
                return {
                    "success": False,
                    "data": None,
                    "error": "No relevant information was found in your documents.",
                    "status_code": 404,
                }
            elif response.status_code == 500:
                return {
                    "success": False,
                    "data": None,
                    "error": "Unable to generate a response right now. Please try again.",
                    "status_code": 500,
                }
            else:
                err_detail = None
                try:
                    err_json = response.json()
                    err_detail = err_json.get("detail")
                except Exception:
                    pass
                return {
                    "success": False,
                    "data": None,
                    "error": err_detail or "Unable to generate a response right now. Please try again.",
                    "status_code": response.status_code,
                }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "data": None,
                "error": "Backend is unavailable. Please make sure the FastAPI server is running.",
                "status_code": None,
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "data": None,
                "error": "Unable to generate a response right now. Please try again.",
                "status_code": None,
            }
        except Exception:
            return {
                "success": False,
                "data": None,
                "error": "Unable to generate a response right now. Please try again.",
                "status_code": None,
            }

    def send_medical_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        timeout: int = 90
    ) -> Dict[str, Any]:
        """
        Sends a message to the HealthAI Medical Chatbot endpoint (POST /api/medical-chat).
        Completely separate from the RAG retrieval pipeline.

        Args:
            message: User's medical/health question.
            session_id: Optional session ID for conversation tracking.
            timeout: Request timeout in seconds.

        Returns:
            Dict containing success flag, data (answer, session_id, agent), and error message.
        """
        endpoint = f"{self.base_url}/api/medical-chat"
        payload = {
            "message": message,
            "session_id": session_id
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=self.get_headers(),
                timeout=timeout
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "data": data,
                    "error": None,
                    "status_code": 200
                }
            else:
                err_detail = "Failed to obtain a response from HealthAI Assistant."
                try:
                    err_json = response.json()
                    err_detail = err_json.get("detail", err_detail)
                except Exception:
                    pass
                return {
                    "success": False,
                    "data": None,
                    "error": err_detail,
                    "status_code": response.status_code
                }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "data": None,
                "error": "Backend server is unavailable. Please verify FastAPI is running.",
                "status_code": None
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "data": None,
                "error": "Request timed out waiting for the Medical Assistant.",
                "status_code": None
            }
        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "error": f"An unexpected error occurred: {str(exc)}",
                "status_code": None
            }

    def route_query(
        self,
        message: str,
        mode: str = "auto",
        conversation_mode: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.25,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Submits an inquiry to the Intelligent Query Router (POST /api/query).
        Dynamically classifies and dispatches to either Document RAG or Agno Medical Agent.

        Args:
            message: User query string.
            mode: 'auto' (default), 'rag', or 'agno'.
            conversation_mode: Optional context ('rag' or 'agno').
            session_id: Optional session identifier for Agno history tracking.
            conversation_history: Prior dialogue turns for follow-up resolution.
            top_k: Number of chunks for RAG.
            similarity_threshold: Relevance cutoff for RAG.
            timeout: HTTP request timeout in seconds.

        Returns:
            Dict containing success flag, data (route, answer, sources, etc.), and error message.
        """
        endpoint = f"{self.base_url}/api/query"
        payload = {
            "message": message,
            "mode": mode,
            "conversation_mode": conversation_mode,
            "session_id": session_id,
            "conversation_history": conversation_history,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=self.get_headers(),
                timeout=timeout
            )
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "error": None,
                    "status_code": 200
                }
            else:
                err_detail = "Failed to process routed query."
                try:
                    err_detail = response.json().get("detail", err_detail)
                except Exception:
                    pass
                return {
                    "success": False,
                    "data": None,
                    "error": err_detail,
                    "status_code": response.status_code
                }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "data": None,
                "error": "Backend server is unavailable. Please verify FastAPI is running.",
                "status_code": None
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "data": None,
                "error": "Request timed out waiting for query routing response.",
                "status_code": None
            }
        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "error": f"An unexpected error occurred: {str(exc)}",
                "status_code": None
            }


# Singleton client instance
api_client = APIClient()

