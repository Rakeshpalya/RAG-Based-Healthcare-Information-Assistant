"""
Authentication Dependencies for FastAPI.

Provides reusable FastAPI security dependencies for extracting and cryptographically
validating Supabase JWT access tokens from the HTTP Authorization Bearer header,
and mapping the authenticated identity to application database user records.
"""

import logging
from typing import Optional, Any
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.services.auth_service import auth_service
from backend.database.database import get_db
from backend.database.models import User
from backend.database.repositories import UserRepository
from backend.database.schemas import UserCreate

logger = logging.getLogger(__name__)

# Security scheme for OpenAPI Swagger documentation
http_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> Any:
    """
    FastAPI dependency to extract and cryptographically verify Supabase JWT access tokens.

    Workflow:
    1. Inspects the HTTP Authorization header.
    2. Enforces the 'Bearer' authentication scheme.
    3. Extracts the non-empty access token.
    4. Validates the token against Supabase Auth (cryptographic verification).
    5. Returns the authenticated Supabase user profile.

    Args:
        request: Raw FastAPI Request object.
        credentials: Optional extracted HTTPAuthorizationCredentials from HTTPBearer.

    Returns:
        The authenticated Supabase user object or dictionary.

    Raises:
        HTTPException: HTTP 401 Unauthorized for missing, malformed, expired,
                       or invalid tokens, accompanied by a WWW-Authenticate header.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = auth_header.strip().split()
    if len(parts) == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme = parts[0]
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if len(parts) < 2 or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1].strip()

    try:
        user = auth_service.get_current_user(jwt=token)
    except Exception as exc:
        # Never log or expose raw JWT tokens or sensitive exception traces
        logger.warning("Token verification failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_current_db_user(
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency to resolve the authenticated Supabase user to an application database User record.

    Workflow:
    1. If current_user is already a User model instance, return it directly.
    2. If current_user contains an integer ID, attempt to lookup by ID.
    3. Extract user email from Supabase Auth payload.
    4. Query UserRepository for an existing user record.
    5. If found, return the existing User.
    6. If not found, provision a new User record in the application database using verified identity.

    Args:
        current_user: The authenticated Supabase user profile from get_current_user.
        db: SQLAlchemy database session.

    Returns:
        User: SQLAlchemy User model representing the authenticated user.

    Raises:
        HTTPException: HTTP 401 Unauthorized if user identity cannot be resolved.
    """
    if isinstance(current_user, User):
        return current_user

    # If mock user or dict with existing integer ID
    if isinstance(current_user, dict) and isinstance(current_user.get("id"), int):
        user_by_id = UserRepository.get_by_id(db, current_user["id"])
        if user_by_id:
            return user_by_id

    email: Optional[str] = None
    if isinstance(current_user, dict):
        email = current_user.get("email")
    else:
        email = getattr(current_user, "email", None)

    if not email or not str(email).strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user has no valid email address.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    clean_email = str(email).strip().lower()

    # Check if user already exists in application database
    db_user = UserRepository.get_by_email(db, clean_email)
    if db_user:
        return db_user

    # Extract metadata for initial creation if present
    full_name: Optional[str] = None
    role: str = "patient"
    if isinstance(current_user, dict):
        metadata = current_user.get("user_metadata") or {}
        if isinstance(metadata, dict):
            full_name = metadata.get("full_name") or metadata.get("name")
            role = metadata.get("role", "patient")
    elif hasattr(current_user, "user_metadata"):
        metadata = getattr(current_user, "user_metadata", {})
        if isinstance(metadata, dict):
            full_name = metadata.get("full_name") or metadata.get("name")
            role = metadata.get("role", "patient")

    if not isinstance(full_name, str):
        full_name = None
    if not isinstance(role, str) or not role:
        role = "patient"

    new_user = UserRepository.create(
        db,
        UserCreate(
            email=clean_email,
            full_name=full_name or clean_email.split("@")[0],
            role=role,
        ),
    )
    return new_user


def get_optional_current_db_user(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Optional FastAPI dependency to resolve an authenticated application user if a Bearer token is provided.
    Returns None if unauthenticated without header.
    Raises HTTP 401 if an Authorization header is provided but invalid or expired.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.strip():
        return None

    parts = auth_header.strip().split()
    if len(parts) < 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication header format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1].strip()
    try:
        current_user = auth_service.get_current_user(jwt=token)
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return get_current_db_user(current_user=current_user, db=db)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


