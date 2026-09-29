import os
import re
from typing import Optional
from fastapi import Request, HTTPException, status


def sanitize_filename(raw_filename: Optional[str]) -> str:
    r"""
    Sanitizes uploaded filenames to prevent path traversal and arbitrary filesystem overwrite.
    
    Protections:
    1. Normalizes Windows backslashes (`\`) to forward slashes (`/`).
    2. Extracts only the basename component.
    3. Strips null bytes (`\x00`).
    4. Strips traversal tokens (`..`).
    5. Restricts characters to alphanumeric, underscores, hyphens, and single dots.
    6. Ensures non-empty fallback filename.
    """
    if not raw_filename or not str(raw_filename).strip():
        return "uploaded_document.pdf"

    # Step 1: Normalize separators
    normalized = str(raw_filename).replace("\\", "/")

    # Step 2: Extract base filename
    base_name = os.path.basename(normalized)

    # Step 3: Strip null bytes and control chars
    base_name = re.sub(r'[\x00-\x1f\x7f]', '', base_name)

    # Step 4: Remove path traversal sequences
    base_name = re.sub(r'\.{2,}', '.', base_name)

    # Step 5: Replace any non-safe character with underscore
    # Safe chars: alphanumeric, '.', '-', '_', ' '
    safe_name = re.sub(r'[^a-zA-Z0-9.\-_ ]', '_', base_name).strip()

    # Step 6: Guard against empty or dot-only filenames
    if not safe_name or safe_name in ('.', '..', '.pdf'):
        return "sanitized_document.pdf"

    return safe_name


async def verify_request_content_length(request: Request, max_bytes: int = 10 * 1024 * 1024) -> None:
    """
    Verifies that the incoming HTTP request does not exceed the allowed content length.
    Rejects oversized requests early before reading entire body into memory (DoS protection).
    """
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length = int(content_length)
            if length > max_bytes:
                max_mb = max_bytes // (1024 * 1024)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request payload ({length} bytes) exceeds maximum allowed limit of {max_mb} MB."
                )
        except ValueError:
            pass
