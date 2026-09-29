from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.database.models import User, Document, DocumentChunk, Conversation, Message
from backend.database.schemas import (
    UserCreate,
    DocumentCreate,
    DocumentChunkCreate,
    ConversationCreate,
    MessageCreate,
)


class UserRepository:
    """Repository handling database operations for User entities."""

    @staticmethod
    def create(db: Session, user_data: UserCreate) -> User:
        user = User(
            email=user_data.email,
            role=user_data.role,
            full_name=user_data.full_name,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def list_all(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        return db.query(User).offset(skip).limit(limit).all()

    @staticmethod
    def delete(db: Session, user_id: int) -> bool:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        db.delete(user)
        db.commit()
        return True


class DocumentRepository:
    """Repository handling database operations for Document entities."""

    @staticmethod
    def create(db: Session, doc_data: DocumentCreate) -> Document:
        doc = Document(
            user_id=doc_data.user_id,
            filename=doc_data.filename,
            file_path=doc_data.file_path,
            file_size_bytes=doc_data.file_size_bytes,
            num_pages=doc_data.num_pages,
            num_chunks=doc_data.num_chunks or 0,
            status=doc_data.status or "processed",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc

    @staticmethod
    def get_by_id(db: Session, document_id: int) -> Optional[Document]:
        return db.query(Document).filter(Document.id == document_id).first()

    @staticmethod
    def list_all(db: Session, user_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[Document]:
        query = db.query(Document)
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def update_status(
        db: Session,
        document_id: int,
        status: str,
        num_chunks: Optional[int] = None
    ) -> Optional[Document]:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return None
        doc.status = status
        if num_chunks is not None:
            doc.num_chunks = num_chunks
        db.commit()
        db.refresh(doc)
        return doc

    @staticmethod
    def delete(db: Session, document_id: int) -> bool:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return False
        db.delete(doc)
        db.commit()
        return True


class ChunkRepository:
    """Repository handling database operations for DocumentChunk entities."""

    @staticmethod
    def bulk_create(
        db: Session,
        document_id: int,
        chunks_data: List[DocumentChunkCreate]
    ) -> List[DocumentChunk]:
        """Bulk inserts chunks for an ingested document and updates document.num_chunks."""
        chunk_models = [
            DocumentChunk(
                document_id=document_id,
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                text=c.text,
                page_number=c.page_number,
                token_count=c.token_count,
                embedding_index=c.embedding_index,
            )
            for c in chunks_data
        ]
        db.add_all(chunk_models)
        
        # Update parent document chunk count
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.num_chunks = len(chunk_models)

        db.commit()
        for chunk in chunk_models:
            db.refresh(chunk)
        return chunk_models

    @staticmethod
    def get_by_chunk_id(db: Session, chunk_id: str) -> Optional[DocumentChunk]:
        return db.query(DocumentChunk).filter(DocumentChunk.chunk_id == chunk_id).first()

    @staticmethod
    def get_by_embedding_index(db: Session, embedding_index: int) -> Optional[DocumentChunk]:
        return db.query(DocumentChunk).filter(DocumentChunk.embedding_index == embedding_index).first()

    @staticmethod
    def list_by_document(db: Session, document_id: int) -> List[DocumentChunk]:
        return (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )


class ConversationRepository:
    """Repository handling database operations for Conversation chat sessions."""

    @staticmethod
    def create(db: Session, conv_data: ConversationCreate) -> Conversation:
        conv = Conversation(
            user_id=conv_data.user_id,
            title=conv_data.title or "Healthcare Consultation",
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    @staticmethod
    def get_by_id(db: Session, conversation_id: int) -> Optional[Conversation]:
        return db.query(Conversation).filter(Conversation.id == conversation_id).first()

    @staticmethod
    def list_by_user(
        db: Session,
        user_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        query = db.query(Conversation)
        if user_id is not None:
            query = query.filter(Conversation.user_id == user_id)
        return query.order_by(Conversation.updated_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def delete(db: Session, conversation_id: int) -> bool:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            return False
        db.delete(conv)
        db.commit()
        return True

    @staticmethod
    def delete_by_user(db: Session, user_id: int) -> int:
        """Deletes all conversations belonging to a specific user, cascading to their messages."""
        convs = db.query(Conversation).filter(Conversation.user_id == user_id).all()
        count = len(convs)
        for conv in convs:
            db.delete(conv)
        db.commit()
        return count


class MessageRepository:
    """Repository handling database operations for Message entities within conversations."""

    @staticmethod
    def create(db: Session, conversation_id: int, msg_data: MessageCreate) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            sender=msg_data.sender,
            text=msg_data.text,
            agent_type=msg_data.agent_type,
            citations=msg_data.citations,
        )
        db.add(msg)
        
        # Touch conversation updated_at
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            from sqlalchemy.sql import func
            conv.updated_at = func.now()

        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def list_by_conversation(
        db: Session,
        conversation_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Message]:
        return (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .offset(skip)
            .limit(limit)
            .all()
        )
