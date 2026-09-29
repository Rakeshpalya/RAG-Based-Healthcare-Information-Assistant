import io
import re
from typing import Dict, Any
from pypdf import PdfReader


class DocumentProcessingError(Exception):
    """Custom exception raised when PDF text extraction fails or yields empty text."""
    pass


class DocumentService:
    """Service class for processing medical documents and extracting text."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted text by normalizing whitespace.
        - Replaces multiple consecutive spaces with a single space.
        - Replaces more than two consecutive newlines with two newlines.
        - Strips leading and trailing whitespace.
        """
        if not text:
            return ""
        
        # Replace non-breaking spaces and weird whitespace characters
        text = text.replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace multiple horizontal spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace 3 or more consecutive newlines with 2 newlines (preserve paragraph boundaries)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        cleaned_text = '\n'.join(lines).strip()
        return cleaned_text

    @classmethod
    def extract_text_from_pdf(cls, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Extracts cleaned text from an uploaded PDF file stream.

        Args:
            pdf_bytes: Raw bytes of the PDF file.
            filename: Name of the uploaded file.

        Returns:
            Dict containing filename, number_of_pages, extracted_text_length, and extracted_text.

        Raises:
            DocumentProcessingError: If the file is invalid or contains no extractable text.
        """
        if not pdf_bytes:
            raise DocumentProcessingError(f"The uploaded PDF file '{filename}' is empty (0 bytes).")

        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_stream)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse PDF file '{filename}': {str(e)}")

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise DocumentProcessingError(f"The PDF file '{filename}' contains no pages.")

        page_texts = []
        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                cleaned_page = cls.clean_text(text)
                if cleaned_page:
                    page_texts.append(cleaned_page)
            except Exception as e:
                # Log error per page if needed and continue
                continue

        full_extracted_text = "\n\n".join(page_texts).strip()

        if not full_extracted_text:
            raise DocumentProcessingError(
                f"No extractable text found in PDF '{filename}'. "
                "The document may contain only scanned images or unreadable vector paths."
            )

        return {
            "filename": filename,
            "number_of_pages": num_pages,
            "extracted_text_length": len(full_extracted_text),
            "extracted_text": full_extracted_text
        }
