"""
Authentication API Router for AI-Healthcare-Agent.

Provides REST endpoints for user signup, login, logout, and retrieving current user profile
backed by Supabase Auth service.
"""

import re
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.services.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ==============================================================================
# Request & Response Schemas
# ==============================================================================

class SignupRequest(BaseModel):
    """User registration request payload."""
    email: str = Field(..., description="Valid email address for registration")
    password: str = Field(
        ...,
        min_length=6,
        description="User password (minimum 6 characters)",
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format")
        return clean


class LoginRequest(BaseModel):
    """User login request payload."""
    email: str = Field(..., description="Registered user email address")
    password: str = Field(
        ...,
        min_length=1,
        description="User account password",
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format")
        return clean


# ==============================================================================
# Serialization Helpers
# ==============================================================================

def _safe_user_dict(user: Any) -> Optional[Dict[str, Any]]:
    """
    Extract safe, JSON-serializable user attributes.
    Excludes sensitive internal tokens and secrets.
    """
    if user is None:
        return None

    if isinstance(user, dict):
        return {
            "id": str(user.get("id", "")),
            "email": user.get("email"),
            "created_at": str(user.get("created_at", "")) if user.get("created_at") else None,
            "user_metadata": user.get("user_metadata") or {},
        }

    if hasattr(user, "model_dump"):
        dumped = user.model_dump()
        return {
            "id": str(dumped.get("id", "")),
            "email": dumped.get("email"),
            "created_at": str(dumped.get("created_at", "")) if dumped.get("created_at") else None,
            "user_metadata": dumped.get("user_metadata") or {},
        }

    return {
        "id": str(getattr(user, "id", "")),
        "email": getattr(user, "email", None),
        "created_at": str(getattr(user, "created_at", "")) if getattr(user, "created_at", None) else None,
        "user_metadata": getattr(user, "user_metadata", {}) or {},
    }


def _safe_session_dict(session: Any) -> Optional[Dict[str, Any]]:
    """
    Extract safe, JSON-serializable session attributes required by client.
    Excludes refresh tokens and secret keys.
    """
    if session is None:
        return None

    if isinstance(session, dict):
        return {
            "access_token": session.get("access_token"),
            "token_type": session.get("token_type", "bearer"),
            "expires_in": session.get("expires_in"),
            "expires_at": session.get("expires_at"),
        }

    if hasattr(session, "model_dump"):
        dumped = session.model_dump()
        return {
            "access_token": dumped.get("access_token"),
            "token_type": dumped.get("token_type", "bearer"),
            "expires_in": dumped.get("expires_in"),
            "expires_at": dumped.get("expires_at"),
        }

    return {
        "access_token": getattr(session, "access_token", None),
        "token_type": getattr(session, "token_type", "bearer"),
        "expires_in": getattr(session, "expires_in", None),
        "expires_at": getattr(session, "expires_at", None),
    }


# ==============================================================================
# Authentication Endpoints
# ==============================================================================

@router.post(
    "/signup",
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
    description="Registers a new user with Supabase Auth using email and password.",
)
def signup(payload: SignupRequest) -> Dict[str, Any]:
    """
    Register a new user account.
    """
    sanitized_email = payload.email.strip().lower()
    try:
        result = auth_service.signup(sanitized_email, payload.password)
    except Exception as exc:
        err_msg = str(exc)
        logger.warning("Signup failed: %s", err_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {err_msg}",
        )

    user = result.get("user")
    session = result.get("session")

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed. Please check user details and try again.",
        )

    return {
        "message": "User registered successfully",
        "user": _safe_user_dict(user),
        "session": _safe_session_dict(session),
    }


@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticates a user using email and password, returning session token.",
)
def login(payload: LoginRequest) -> Dict[str, Any]:
    """
    Authenticate an existing user account.
    """
    sanitized_email = payload.email.strip().lower()
    logger.info("Login request received for email: %s", sanitized_email)
    try:
        result = auth_service.login(sanitized_email, payload.password)
    except Exception as exc:
        err_name = type(exc).__name__
        err_msg = str(exc)
        logger.warning(
            "Login authentication failed for %s: error_type=%s, detail=%s",
            sanitized_email,
            err_name,
            err_msg,
        )
        is_network_timeout = any(
            t in err_name.lower() or t in err_msg.lower()
            for t in ["timeout", "connect", "handshake", "remotedisconnected", "ssl"]
        )
        if is_network_timeout:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Authentication service connection timed out. Please click Sign In again.",
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user = result.get("user")
    session = result.get("session")

    if not user or not session:
        logger.warning(
            "Login returned incomplete payload for %s (user=%s, session=%s)",
            sanitized_email,
            bool(user),
            bool(session),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    safe_user = _safe_user_dict(user)
    user_id = safe_user.get("id") if safe_user else "unknown"
    logger.info("Login authentication successful for user_id=%s (email=%s)", user_id, sanitized_email)

    return {
        "message": "Login successful",
        "user": safe_user,
        "session": _safe_session_dict(session),
    }


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="User Logout",
    description="Terminates the current user session.",
)
def logout() -> Dict[str, Any]:
    """
    Log out the currently active user session.
    """
    try:
        auth_service.logout()
    except Exception as exc:
        logger.warning("Logout operation error: %s", exc)
        # Continue to return clean success to avoid leaving client in an uncertain state
    return {
        "status": "success",
        "message": "User signed out successfully",
    }


@router.get(
    "/me",
    status_code=status.HTTP_200_OK,
    summary="Get Current User Profile",
    description="Retrieves the profile of the currently authenticated user.",
)
def get_current_user() -> Dict[str, Any]:
    """
    Retrieve profile details for the currently authenticated user.
    """
    try:
        user = auth_service.get_current_user()
    except Exception as exc:
        logger.warning("Error fetching current user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    return {
        "user": _safe_user_dict(user),
    }
