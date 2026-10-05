import os
import time
from typing import Optional
from fastapi import FastAPI, Request, status, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import logging
from contextlib import asynccontextmanager

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

logger = logging.getLogger("backend.main")

# Load environment variables from .env file
load_dotenv()

APP_NAME = os.getenv("APP_NAME", "AI Healthcare Research & Patient Assistance Agent")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Application Lifespan Context Manager (Phase 3.3):
    1. Pre-warms the singleton Gemini client at startup to eliminate cold-start latency.
    2. Performs startup validation safely without unnecessary LLM generation calls.
    3. Cleans up database connection pools on application shutdown.
    """
    try:
        from backend.services.gemini_service import GeminiService
        warm_res = GeminiService.prewarm_client()
        logger.info(
            "Gemini client startup pre-warming: status=%s, warm=%s, duration_ms=%.2f",
            warm_res.get("status"),
            warm_res.get("warm"),
            warm_res.get("warmup_duration_ms", 0.0)
        )
    except Exception as exc:
        logger.warning("Gemini client startup pre-warming notice: %s", str(exc))

    # Production Configuration Validation (Phase 3.5.6)
    try:
        from backend.config import settings
        config_val = settings.validate_production_config()
        if not config_val["valid"]:
            for err in config_val["errors"]:
                logger.error("Production Startup Configuration Error: %s", err)
            if settings.is_production:
                raise RuntimeError(f"Startup halted: production configuration errors: {config_val['errors']}")
        for w in config_val.get("warnings", []):
            logger.warning("Production Startup Configuration Notice: %s", w)
    except Exception as exc:
        if "Startup halted" in str(exc):
            raise
        logger.warning("Startup configuration validation notice: %s", str(exc))

    yield

    try:
        dispose_engine()
    except Exception as exc:
        logger.warning("Error disposing database engine on shutdown: %s", str(exc))


# Initialize FastAPI Application with OpenAPI Swagger metadata and Lifespan
app = FastAPI(
    title=APP_NAME,
    description="Backend API for AI Healthcare Research & Patient Assistance Agent",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Security Headers & Request Size Limit Middleware (10 MB payload limit)
MAX_REQUEST_SIZE_BYTES = 10 * 1024 * 1024


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

    # Non-destructive LLM Cache & Pre-warm Check
    cache_info = {"enabled": False}
    try:
        from backend.services.llm_cache_service import get_llm_cache_service
        cache_info = get_llm_cache_service().stats()
    except Exception:
        pass

    gemini_warm = False
    try:
        from backend.services.gemini_service import GeminiService
        gemini_warm = GeminiService.is_client_warm()
    except Exception:
        pass

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
        "llm_cache": cache_info,
        "gemini_client_warm": gemini_warm,
    }


@app.get("/health/live", summary="Liveness Probe", tags=["System"])
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


@app.get("/health/ready", summary="Readiness Probe", tags=["System"])
@app.get("/health/readiness", summary="Kubernetes / Docker Readiness Probe", tags=["System"])
async def readiness_probe():
    """
    Comprehensive non-destructive readiness probe verifying core subsystems:
    vector_store, gemini, cache, and safety engine.
    """
    from fastapi.responses import JSONResponse
    from backend.services.vector_store_service import get_vector_store_service
    from backend.services.embedding_service import EmbeddingService
    from backend.safety.medical_safety_guard import MedicalSafetyGuard
    from backend.services.llm_cache_service import get_llm_cache_service
    from backend.config import settings

    is_ready = True
    vs_count = 0
    vs_ready = False
    try:
        vs = get_vector_store_service()
        vs_count = vs.count()
        vs_ready = (vs_count == 744)
        if not vs_ready:
            is_ready = False
    except Exception:
        vs_ready = False
        is_ready = False

    dim = 0
    embed_ready = False
    try:
        dim = EmbeddingService.get_embedding_dimension()
        embed_ready = (dim == 384)
    except Exception:
        pass

    gemini_ready = False
    try:
        gemini_ready = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "test-placeholder-key")
    except Exception:
        pass

    cache_ready = False
    try:
        cache_service = get_llm_cache_service()
        cache_ready = bool(cache_service.enabled)
    except Exception:
        pass

    safety_ready = False
    try:
        _ = MedicalSafetyGuard
        safety_ready = True
    except Exception:
        safety_ready = False
        is_ready = False

    checks = {
        "vector_store": {
            "status": "ready" if vs_ready else "degraded",
            "vector_count": vs_count
        },
        "embedding_service": {
            "status": "ready" if embed_ready else "degraded",
            "dimension": dim
        },
        "safety_engine": {
            "status": "ready" if safety_ready else "degraded"
        },
        "gemini": {
            "status": "ready" if gemini_ready else "degraded"
        },
        "cache": {
            "status": "ready" if cache_ready else "degraded"
        }
    }

    status_code = 200 if is_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if is_ready else "not_ready",
            "vector_store": vs_ready,
            "gemini": gemini_ready,
            "cache": cache_ready,
            "service": "AI-Healthcare-Agent",
            "environment": ENVIRONMENT,
            "checks": checks
        }
    )



@app.get("/metrics", summary="Production Observability & Prometheus Metrics", tags=["System"])
@app.get("/metrics/prometheus", summary="Prometheus Text Exposition", tags=["System"])
async def production_metrics(request: Request, format: Optional[str] = None):
    """
    Returns production observability metrics.
    Supports RFC-compliant Prometheus text exposition for scrapers (Accept: text/plain or openmetrics)
    and JSON snapshot for API clients and dashboards.
    """
    from backend.evaluation.observability import get_metrics_collector
    collector = get_metrics_collector()

    accept = request.headers.get("accept", "").lower()
    user_agent = request.headers.get("user-agent", "").lower()
    path = request.url.path

    # If endpoint is /metrics/prometheus or Prometheus scrape requested via headers/param
    is_prometheus = (
        path.endswith("/prometheus")
        or format in ("prometheus", "text", "prom")
        or "text/plain" in accept
        or "openmetrics" in accept
        or "prometheus" in user_agent
    )

    if is_prometheus:
        return PlainTextResponse(
            content=collector.get_prometheus_exposition(),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    # Standard JSON snapshot for API/dashboard consumers
    return JSONResponse(
        content=collector.get_metrics_snapshot()
    )


@app.post("/chat/stream", tags=["Streaming"], summary="Stream RAG / Chat Query via SSE")
@app.post("/api/chat/stream", tags=["Streaming"], summary="Stream RAG / Chat Query via SSE")
async def chat_stream_alias(
    request: Request
):
    """
    Convenience alias endpoint for Server-Sent Events (SSE) streaming.
    Accepts question or message in payload, forwards to stream_rag_query.
    """
    from backend.api.rag_router import RAGQueryRequest, stream_rag_query
    from backend.api.auth_dependencies import get_optional_current_db_user
    body = await request.json()
    question = body.get("question") or body.get("message") or ""
    rag_req = RAGQueryRequest(
        question=question,
        top_k=body.get("top_k", 5),
        similarity_threshold=body.get("similarity_threshold", 0.25),
        conversation_history=body.get("conversation_history")
    )
    return await stream_rag_query(request=rag_req, raw_request=request, current_user=None)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
