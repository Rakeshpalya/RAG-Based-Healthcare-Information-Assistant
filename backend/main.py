import os
import time
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.api.document_router import router as document_router
from backend.api.rag_router import router as rag_router
from backend.api.agent_router import router as agent_router
from backend.api.db_router import router as db_router
from backend.api.auth_router import router as auth_router
from backend.api.medical_chat_router import router as medical_chat_router
from backend.api.query_api import router as query_router
from backend.evaluation.observability import (
    generate_request_id,
    HTTPRequestLogEvent,
    StructuredRAGLogger
)
from backend.database.database import check_db_connection, dispose_engine

# Load environment variables from .env file
load_dotenv()

APP_NAME = os.getenv("APP_NAME", "AI Healthcare Research & Patient Assistance Agent")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Initialize FastAPI Application with OpenAPI Swagger metadata
app = FastAPI(
    title=APP_NAME,
    description="Backend API for AI Healthcare Research & Patient Assistance Agent",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Security Headers & Request Size Limit Middleware (10 MB payload limit)
MAX_REQUEST_SIZE_BYTES = 10 * 1024 * 1024


@app.on_event("shutdown")
def on_shutdown():
    """Cleanly disposes database connection pools and resources on app termination."""
    dispose_engine()


@app.middleware("http")
async def security_and_size_middleware(request: Request, call_next):
    request_id = (
        request.headers.get("X-Request-ID")
        or request.headers.get("request_id")
        or generate_request_id()
    )
    request.state.request_id = request_id

    content_length = request.headers.get("content-length")
    # Document upload endpoint handles its own 10MB file validation with 400 Bad Request
    if content_length and request.url.path != "/documents/upload":
        try:
            if int(content_length) > MAX_REQUEST_SIZE_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={
                        "detail": f"Request payload ({content_length} bytes) exceeds maximum allowed limit of {MAX_REQUEST_SIZE_BYTES // (1024 * 1024)} MB."
                    }
                )
        except ValueError:
            pass

    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Emit structured HTTP request audit log
    try:
        client_host = request.client.host if request.client else None
        http_event = HTTPRequestLogEvent(
            request_id=request_id,
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            client_ip=client_host
        )
        StructuredRAGLogger.emit_http_log(http_event, use_json=True)
    except Exception:
        pass

    return response


# Enable CORS for local frontend development and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(auth_router)
app.include_router(document_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(db_router)
app.include_router(medical_chat_router)
app.include_router(query_router)

# Mount production React frontend static bundle if dist directory exists
dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dist")
if os.path.exists(dist_dir):
    from fastapi.staticfiles import StaticFiles
    app.mount("/app", StaticFiles(directory=dist_dir, html=True), name="frontend")


@app.get("/", summary="Root Endpoint", tags=["System"])
async def root():
    """
    Root endpoint for the AI Healthcare Agent API.
    Provides basic service description and documentation links.
    """
    return {
        "message": f"Welcome to {APP_NAME} API",
        "status": "online",
        "version": "0.5.0",
        "environment": ENVIRONMENT,
        "docs": "/docs"
    }


@app.get("/health", summary="Consolidated Health Check Endpoint", tags=["System"])
async def health_check():
    """
    Consolidated health check endpoint verifying backend operational readiness.
    Provides non-destructive status for: vector_store, embedding_service,
    safety_engine, database, and LLM configuration.
    """
    vector_status = "ready"
    embedding_status = "ready"
    safety_status = "ready"
    db_status = "ready"
    llm_status = "ready"

    # Non-destructive Vector Store Check
    try:
        from backend.services.vector_store_service import get_vector_store_service
        vs = get_vector_store_service()
        if vs.count() == 0:
            vector_status = "uninitialized"
    except Exception:
        vector_status = "degraded"

    # Non-destructive Embedding Service Check
    try:
        from backend.services.embedding_service import EmbeddingService
        dim = EmbeddingService.get_embedding_dimension()
        if dim <= 0:
            embedding_status = "degraded"
    except Exception:
        embedding_status = "degraded"

    # Non-destructive Safety Engine Check
    try:
        from backend.safety.medical_safety_guard import MedicalSafetyGuard
        _ = MedicalSafetyGuard
    except Exception:
        safety_status = "degraded"

    # Non-destructive Database Connectivity Check
    try:
        db_health = check_db_connection()
        db_status = db_health.get("status", "ready")
    except Exception:
        db_status = "degraded"

    # Non-destructive LLM Configuration Check
    try:
        from backend.config import settings
        masked_key = settings.get_masked_api_key()
        if masked_key == "NOT_CONFIGURED":
            llm_status = "unconfigured"
    except Exception:
        llm_status = "degraded"

    overall_status = "healthy"
    if any(s == "degraded" for s in [vector_status, embedding_status, safety_status]):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "service": "AI-Healthcare-Agent",
        "environment": ENVIRONMENT,
        "vector_store": vector_status,
        "embedding_service": embedding_status,
        "safety_engine": safety_status,
        "database": db_status,
        "llm_config": llm_status,
    }


@app.get("/health/liveness", summary="Kubernetes / Docker Liveness Probe", tags=["System"])
async def liveness_probe():
    """
    Lightweight, fast liveness probe for container orchestrators.
    Confirms the HTTP server process is running and responding to requests.
    """
    return {
        "status": "alive",
        "service": "AI-Healthcare-Agent"
    }


@app.get("/health/readiness", summary="Kubernetes / Docker Readiness Probe", tags=["System"])
async def readiness_probe():
    """
    Comprehensive non-destructive readiness probe for container orchestrators.
    Confirms all core subsystems (vector store 744 vectors, embeddings, safety)
    are initialized before routing live patient/clinical traffic.
    """
    from fastapi.responses import JSONResponse
    from backend.services.vector_store_service import get_vector_store_service
    from backend.services.embedding_service import EmbeddingService
    from backend.safety.medical_safety_guard import MedicalSafetyGuard

    checks = {}
    is_ready = True

    try:
        vs = get_vector_store_service()
        count = vs.count()
        checks["vector_store"] = {
            "status": "ready" if count == 744 else ("degraded" if count == 0 else "mismatched"),
            "vector_count": count,
            "expected_count": 744
        }
        if count != 744:
            is_ready = False
    except Exception as exc:
        checks["vector_store"] = {"status": "degraded", "error": str(exc)}
        is_ready = False

    try:
        dim = EmbeddingService.get_embedding_dimension()
        checks["embedding_service"] = {
            "status": "ready" if dim == 384 else "degraded",
            "dimension": dim
        }
        if dim != 384:
            is_ready = False
    except Exception as exc:
        checks["embedding_service"] = {"status": "degraded", "error": str(exc)}
        is_ready = False

    try:
        _ = MedicalSafetyGuard
        checks["safety_engine"] = {"status": "ready"}
    except Exception as exc:
        checks["safety_engine"] = {"status": "degraded", "error": str(exc)}
        is_ready = False

    status_code = 200 if is_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if is_ready else "not_ready",
            "service": "AI-Healthcare-Agent",
            "environment": ENVIRONMENT,
            "checks": checks
        }
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
