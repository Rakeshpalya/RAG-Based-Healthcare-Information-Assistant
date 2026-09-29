import io
import sys
import asyncio
from pathlib import Path
from fastapi import UploadFile, HTTPException

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.api.document_router import upload_document
from backend.services.document_service import DocumentService, DocumentProcessingError


def create_minimal_pdf_bytes(text: str = "Medical Report: Patient John Doe diagnosed with mild hypertension.") -> bytes:
    """Generates a valid minimal raw PDF byte stream containing extractable text."""
    escaped_text = text.replace("(", "\\(").replace(")", "\\)")
    content_stream = f"BT /F1 12 Tf 50 750 Td ({escaped_text}) Tj ET"
    stream_len = len(content_stream)
    
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >> endobj\n"
        f"4 0 obj << /Length {stream_len} >> stream\n"
        f"{content_stream}\n"
        "endstream\n"
        "endobj\n"
        "xref\n"
        "0 5\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000115 00000 n \n"
        "0000000272 00000 n \n"
        "trailer << /Size 5 /Root 1 0 R >>\n"
        "startxref\n"
        "350\n"
        "%%EOF\n"
    )
    return pdf_content.encode("latin-1")


def test_clean_text_utility():
    """Test whitespace cleaning and normalization."""
    raw_text = "  Patient Name:   John Doe  \n\n\n\n  Diagnosis:   Hypertension   \n\n  "
    cleaned = DocumentService.clean_text(raw_text)
    assert cleaned == "Patient Name: John Doe\n\nDiagnosis: Hypertension"
    print("[PASS] clean_text utility test passed.")


def test_valid_pdf_extraction():
    """Test PDF text extraction using DocumentService."""
    sample_text = "Medical Report: Patient Jane Smith status normal."
    pdf_bytes = create_minimal_pdf_bytes(sample_text)
    
    result = DocumentService.extract_text_from_pdf(pdf_bytes, "clinical_report.pdf")
    assert result["filename"] == "clinical_report.pdf"
    assert result["number_of_pages"] == 1
    assert result["extracted_text_length"] > 0
    assert "Patient Jane Smith" in result["extracted_text"]
    print("[PASS] test_valid_pdf_extraction test passed.")


def test_empty_pdf_raises_error():
    """Test that empty or textless PDFs raise DocumentProcessingError."""
    empty_pdf_bytes = b"%PDF-1.4\n1 0 obj << >> endobj\ntrailer << >>\n%%EOF"
    raised = False
    try:
        DocumentService.extract_text_from_pdf(empty_pdf_bytes, "test_empty.pdf")
    except DocumentProcessingError as exc:
        raised = True
        assert "No extractable text found in PDF" in str(exc) or "Failed to parse PDF" in str(exc)
    assert raised, "DocumentProcessingError was not raised for empty PDF"
    print("[PASS] empty_pdf_raises_error test passed.")


def test_api_upload_valid_pdf_direct():
    """Test successful PDF upload and text extraction directly through upload_document handler."""
    pdf_bytes = create_minimal_pdf_bytes("Patient report: Blood pressure 120/80 mmHg.")
    upload_file = UploadFile(
        file=io.BytesIO(pdf_bytes),
        filename="report.pdf",
        headers={"content-type": "application/pdf"}
    )
    
    result = asyncio.run(upload_document(upload_file))
    assert result["filename"] == "report.pdf"
    assert result["number_of_pages"] == 1
    assert result["extracted_text_length"] > 0
    assert "Blood pressure 120/80" in result["extracted_text"]
    print("[PASS] test_api_upload_valid_pdf_direct test passed.")


def test_api_upload_non_pdf_direct():
    """Test that uploading a non-PDF file triggers a 400 Bad Request HTTPException."""
    upload_file = UploadFile(
        file=io.BytesIO(b"Plain text content"),
        filename="test.txt",
        headers={"content-type": "text/plain"}
    )
    raised = False
    try:
        asyncio.run(upload_document(upload_file))
    except HTTPException as exc:
        raised = True
        assert exc.status_code == 400
        assert "Invalid file type" in exc.detail
    assert raised, "HTTPException was not raised for non-PDF file"
    print("[PASS] test_api_upload_non_pdf_direct test passed.")


def test_api_upload_oversized_pdf_direct():
    """Test that uploading a file larger than 10MB triggers a 400 Bad Request error."""
    oversized_bytes = b"0" * (11 * 1024 * 1024)  # 11 MB
    upload_file = UploadFile(
        file=io.BytesIO(oversized_bytes),
        filename="large.pdf",
        headers={"content-type": "application/pdf"}
    )
    raised = False
    try:
        asyncio.run(upload_document(upload_file))
    except HTTPException as exc:
        raised = True
        assert exc.status_code == 400
        assert "exceeds the maximum allowed limit" in exc.detail
    assert raised, "HTTPException was not raised for oversized file"
    print("[PASS] test_api_upload_oversized_pdf_direct test passed.")


if __name__ == "__main__":
    test_clean_text_utility()
    test_valid_pdf_extraction()
    test_empty_pdf_raises_error()
    test_api_upload_valid_pdf_direct()
    test_api_upload_non_pdf_direct()
    test_api_upload_oversized_pdf_direct()
    print("[SUCCESS] All DocumentService & Upload API unit tests passed successfully!")
