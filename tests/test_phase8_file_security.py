import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.security import sanitize_filename
from backend.api.auth_dependencies import get_current_user
from backend.services.vector_store_service import VectorStoreService


@pytest.fixture(autouse=True)
def setup_auth_and_vector_isolation(tmp_path, monkeypatch):
    """
    Mocks authenticated user and isolates vector store to temporary directory
    to strictly preserve production vector store invariant.
    """
    import backend.services.vector_store_service as vss_module
    test_store = VectorStoreService(storage_dir=tmp_path)
    monkeypatch.setattr(vss_module, "_shared_vector_store", test_store)
    monkeypatch.setattr(vss_module, "get_vector_store_service", lambda *a, **k: test_store)

    user = {"id": 999, "email": "security_auditor@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user

    yield

    app.dependency_overrides.clear()
    vss_module._shared_vector_store = None


client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer mock_valid_jwt_token"}


def test_sanitize_filename_prevents_directory_escape():
    """Verify that sanitize_filename neutralizes Unix and Windows traversal sequences."""
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename(r"..\..\Windows\System32\cmd.exe.pdf") == "cmd.exe.pdf"
    assert sanitize_filename(".../...//hidden.pdf") == "hidden.pdf"
    assert sanitize_filename("file\x00name.pdf") == "filename.pdf"
    assert sanitize_filename("") == "uploaded_document.pdf"
    assert sanitize_filename("...") == "sanitized_document.pdf"


def test_path_traversal_forward_slashes_rejected():
    """Verify that uploading a file with '../../malicious.pdf' is blocked with 400."""
    file_content = b"%PDF-1.4 test document content"
    files = {"file": ("../../malicious.pdf", io.BytesIO(file_content), "application/pdf")}
    res = client.post("/documents/upload", files=files, headers=AUTH_HEADERS)
    assert res.status_code == 400
    assert "Path traversal sequences are not permitted" in res.json()["detail"]


def test_path_traversal_backslashes_rejected():
    """Verify that uploading a file with Windows backslashes '..\\..\\malicious.pdf' is blocked."""
    file_content = b"%PDF-1.4 test document content"
    files = {"file": (r"..\..\malicious.pdf", io.BytesIO(file_content), "application/pdf")}
    res = client.post("/documents/upload", files=files, headers=AUTH_HEADERS)
    assert res.status_code == 400
    assert "Path traversal sequences are not permitted" in res.json()["detail"]


def test_non_pdf_extension_rejected():
    """Verify that executable or script files are rejected with 400."""
    file_content = b"echo 'malicious shell script'"
    files = {"file": ("script.sh", io.BytesIO(file_content), "text/x-shellscript")}
    res = client.post("/documents/upload", files=files, headers=AUTH_HEADERS)
    assert res.status_code == 400
    assert "Only PDF files (.pdf) are supported" in res.json()["detail"]


def test_invalid_mime_type_rejected():
    """Verify that a PDF extension with an executable MIME type is rejected."""
    file_content = b"%PDF-1.4 test"
    files = {"file": ("exploit.pdf", io.BytesIO(file_content), "application/x-dosexec")}
    res = client.post("/documents/upload", files=files, headers=AUTH_HEADERS)
    assert res.status_code == 400
    assert "Invalid MIME type" in res.json()["detail"]


def test_empty_file_rejected():
    """Verify that an empty (0 bytes) upload is rejected."""
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    res = client.post("/documents/upload", files=files, headers=AUTH_HEADERS)
    assert res.status_code == 400


def test_corrupted_pdf_rejected_safely():
    """Verify that a corrupted, non-PDF file claiming to be a PDF fails safely."""
    garbage_bytes = b"NOT_A_REAL_PDF_HEADER_JUST_GARBAGE_BYTES_12345"
    files = {"file": ("corrupt.pdf", io.BytesIO(garbage_bytes), "application/pdf")}
    res = client.post("/documents/upload", files=files, headers=AUTH_HEADERS)
    # Must reject with 400 without crashing the server
    assert res.status_code in (400, 422)


def test_oversized_file_rejected():
    """Verify that an upload exceeding 10MB is rejected."""
    oversized_bytes = b"%PDF-1.4 " + b"0" * (11 * 1024 * 1024)
    files = {"file": ("giant.pdf", io.BytesIO(oversized_bytes), "application/pdf")}
    res = client.post("/documents/upload", files=files, headers=AUTH_HEADERS)
    assert res.status_code in (400, 413)
