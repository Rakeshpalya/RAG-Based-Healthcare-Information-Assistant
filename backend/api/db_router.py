from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import User
from backend.database.schemas import (
    UserCreate,
    UserResponse,
    DocumentCreate,
    DocumentResponse,
    ConversationCreate,
    ConversationResponse,
    MessageCreate,
    MessageResponse,
)
from backend.database.repositories import (
    UserRepository,
    DocumentRepository,
    ConversationRepository,
    MessageRepository,
)
from backend.api.auth_dependencies import get_current_db_user

router = APIRouter(tags=["Database & Persistence"])


# ==============================================================================
# User Endpoints
# ==============================================================================

@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """Creates a new user profile (patient, clinician, or admin)."""
    existing_user = UserRepository.get_by_email(db, user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{user_in.email}' already exists.",
        )
    return UserRepository.create(db, user_in)


@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Get User Profile",
)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Retrieves a user profile by unique user ID."""
    user = UserRepository.get_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
    return user


# ==============================================================================
# Document Endpoints (Protected with User Isolation)
# ==============================================================================

@router.post(
    "/documents/metadata",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Document Metadata",
)
def create_document(
    doc_in: DocumentCreate,
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Registers document metadata in the relational database assigned to the authenticated user."""
    # Prevent client from spoofing another user ID
    if doc_in.user_id is not None and doc_in.user_id != current_user.id:
        target_user = UserRepository.get_by_id(db, doc_in.user_id)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {doc_in.user_id} not found.",
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot assign document to another user.",
        )

    # Enforce authoritative ownership from verified identity
    doc_in.user_id = current_user.id
    return DocumentRepository.create(db, doc_in)


@router.get(
    "/documents",
    response_model=List[DocumentResponse],
    summary="List Document Metadata",
)
def list_documents(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Lists ingested document records strictly isolated to the authenticated user."""
    # Always enforce the authenticated user's ID as the owner filter
    return DocumentRepository.list_all(db, user_id=current_user.id, skip=skip, limit=limit)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document Metadata",
)
def get_document(
    document_id: int,
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Retrieves document metadata and chunk stats by document ID, enforcing ownership isolation."""
    doc = DocumentRepository.get_by_id(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found.",
        )
    if doc.user_id is not None and doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to access this document.",
        )
    return doc


# ==============================================================================
# Conversation & Message Endpoints (Protected with User Isolation)
# ==============================================================================

@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize Conversation Session",
)
def create_conversation(
    conv_in: ConversationCreate,
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Initializes a new dialogue session assigned to the authenticated user."""
    # Prevent client from spoofing another user ID
    if conv_in.user_id is not None and conv_in.user_id != current_user.id:
        target_user = UserRepository.get_by_id(db, conv_in.user_id)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {conv_in.user_id} not found.",
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot create conversation for another user.",
        )

    # Enforce authoritative ownership from verified identity
    conv_in.user_id = current_user.id
    return ConversationRepository.create(db, conv_in)


@router.get(
    "/conversations",
    response_model=List[ConversationResponse],
    summary="List Conversations",
)
def list_conversations(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Lists conversation sessions strictly isolated to the authenticated user."""
    return ConversationRepository.list_by_user(db, user_id=current_user.id, skip=skip, limit=limit)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[MessageResponse],
    summary="Get Conversation Messages",
)
def get_conversation_messages(
    conversation_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Fetches chronological dialogue messages for a conversation owned by the authenticated user."""
    conv = ConversationRepository.get_by_id(db, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found.",
        )
    if conv.user_id is not None and conv.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to access this conversation.",
        )
    return MessageRepository.list_by_conversation(db, conversation_id, skip=skip, limit=limit)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Append Conversation Message",
)
def create_conversation_message(
    conversation_id: int,
    msg_in: MessageCreate,
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Appends a new message turn to a conversation owned by the authenticated user."""
    conv = ConversationRepository.get_by_id(db, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found.",
        )
    if conv.user_id is not None and conv.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot add messages to another user's conversation.",
        )
    return MessageRepository.create(db, conversation_id, msg_in)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Conversation Session",
)
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Deletes a conversation session strictly owned by the authenticated user."""
    conv = ConversationRepository.get_by_id(db, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found.",
        )
    if conv.user_id is not None and conv.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot delete another user's conversation.",
        )
    success = ConversationRepository.delete(db, conversation_id)
    return {"success": success, "message": "Conversation deleted successfully."}


@router.delete(
    "/conversations",
    status_code=status.HTTP_200_OK,
    summary="Clear User Conversation History",
)
def clear_conversations(
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """Deletes all conversation sessions strictly belonging to the authenticated user."""
    deleted_count = ConversationRepository.delete_by_user(db, current_user.id)
    return {
        "success": True,
        "deleted_count": deleted_count,
        "message": f"Successfully cleared {deleted_count} conversations.",
    }

