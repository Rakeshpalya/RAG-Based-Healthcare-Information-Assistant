from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# User Schemas
# ==============================================================================

class UserBase(BaseModel):
    email: str = Field(..., description="Unique email address for user identification.")
    role: str = Field(default="patient", description="Role: patient, clinician, or admin.")
    full_name: Optional[str] = Field(default=None, description="Full legal name or pseudonym.")


class UserCreate(UserBase):
    pass


class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# Document & Chunk Schemas
# ==============================================================================

class DocumentBase(BaseModel):
    filename: str = Field(..., description="Original filename of the ingested document.")
    file_path: str = Field(..., description="Local or cloud storage path for the file.")
    file_size_bytes: Optional[int] = Field(default=None, description="File size in bytes.")
    num_pages: Optional[int] = Field(default=None, description="Total number of pages.")
    num_chunks: Optional[int] = Field(default=0, description="Total number of chunks created.")
    status: Optional[str] = Field(default="processed", description="Status: uploaded, chunked, indexed, failed.")
    user_id: Optional[int] = Field(default=None, description="Optional ID of the owning user.")


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: int
    num_chunks: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentChunkCreate(BaseModel):
    chunk_id: str = Field(..., description="Unique deterministic chunk ID (e.g., MED_CHUNK_0).")
    chunk_index: int = Field(..., description="Zero-based sequence index.")
    text: str = Field(..., description="Extracted chunk text.")
    page_number: Optional[int] = Field(default=None, description="1-based page number.")
    token_count: Optional[int] = Field(default=None, description="Estimated token count.")
    embedding_index: Optional[int] = Field(default=None, description="Index in FAISS IndexFlatIP.")


class DocumentChunkResponse(DocumentChunkCreate):
    id: int
    document_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# Conversation & Message Schemas
# ==============================================================================

class ConversationBase(BaseModel):
    user_id: Optional[int] = Field(default=None, description="Optional ID of the user owning this chat.")
    title: Optional[str] = Field(default="Healthcare Consultation", description="Topic or consultation summary.")


class ConversationCreate(ConversationBase):
    pass


class ConversationResponse(ConversationBase):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageCreate(BaseModel):
    sender: str = Field(..., description="Sender type: user, assistant, or agent.")
    text: str = Field(..., description="Dialogue message content.")
    agent_type: Optional[str] = Field(default=None, description="Agent provenance (e.g. OrchestratorAgent, ResearchAgent).")
    citations: Optional[Any] = Field(default=None, description="Citations or reference metadata list.")


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    sender: str
    text: str
    agent_type: Optional[str] = None
    citations: Optional[Any] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
