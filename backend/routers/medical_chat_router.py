"""
Re-export for backend/routers/medical_chat_router.py convention.
"""
from backend.api.medical_chat_router import (
    router,
    MedicalChatRequest,
    MedicalChatResponse,
)

__all__ = ["router", "MedicalChatRequest", "MedicalChatResponse"]
