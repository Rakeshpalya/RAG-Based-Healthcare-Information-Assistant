# Phase 6: Production Containerization & Deployment Plan

**Project**: `AI-Healthcare-Agent`  
**Milestone**: Phase 6 — Production Containerization, Docker Security, Health Probes, Startup Validation & CI/CD  
**Document Status**: **ACTIVE IMPLEMENTATION BLUEPRINT**  
**Target Invariant**: Strict Read-Only Vector Store (744 FAISS Vectors == 744 Metadata Records)  
**Quality Target**: 100% Pass Rate on 501 Pytest Suite & 5-Track Evaluation Runner  

---

## 1. System Architecture Overview

The `AI-Healthcare-Agent` is an enterprise clinical decision support and patient assistance backend built with FastAPI, FAISS, SentenceTransformers, and Google Gemini / OpenAI LLMs.

```
                           +----------------------------------------+
                           |           Client / Frontend            |
                           |   (Streamlit, Web App, API Clients)    |
                           +-------------------+--------------------+
                                               |
                                        HTTP / JSON
                                               v
+-----------------------------------------------------------------------------------+
|                         FastAPI Application Container                             |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Routers: /health, /documents, /rag, /agent, /auth, /api/db, /api/query      |  |
|  +-----------------------------------------------------------------------------+  |
|                                          |                                        |
|         +--------------------------------+-------------------------------+        |
|         |                                |                               |        |
|         v                                v                               v        |
|  +---------------+             +-------------------+           +---------------+  |
|  | Medical Safety|             |    Query Router   |           |  Database     |  |
|  | Guard & Agent |             |   (RAG vs Agno)   |           |  (PostgreSQL/ |  |
|  | (18 categories|             +---------+---------+           |   Supabase /  |  |
|  | zero FN risk) |                       |                     |   SQLite fallback)
|  +---------------+                       v                     +---------------+  |
|                                +-------------------+                              |
|                                |    RAG Service    |                              |
|                                +---+-----------+---+                              |
|                                    |           |                                  |
|                  +-----------------+           +-----------------+                |
|                  v                                               v                |
|  +-------------------------------+             +-------------------------------+  |
|  |   EmbeddingService            |             |   FAISS Vector Store          |  |
|  |   (MiniLM-L6-v2, 384-dim)     |             |   (IndexFlatIP, 744 vectors,  |  |
|  |   Baked cache in image        |             |   744 metadata records, :ro)  |  |
|  +-------------------------------+             +-------------------------------+  |
|                  |                                               |                |
|                  +-----------------------+-----------------------+                |
|                                          v                                        |
|                        +-----------------------------------+                      |
|                        |   Grounded Generation & Validation|                      |
|                        |   - Prompt Builder                |                      |
|                        |   - GeminiService (Google GenAI)  |                      |
|                        |   - CitationValidator ([Source N])|                      |
|                        |   - HallucinationGuard (Audit)    |                      |
|                        |   - Structured Observability      |                      |
|                        +-----------------------------------+                      |
+-----------------------------------------------------------------------------------+
```

---

## 2. Dependencies and Runtime Specifications

### 2.1 Python Environment
- **Runtime Version**: Python 3.11-slim or 3.12-slim (Debian Bookworm base image).
- **Architecture**: `linux/amd64` (and compatible with Apple Silicon `linux/arm64` via multi-arch build).

### 2.2 Core Package Stack (`requirements.txt`)
- **Web & Server**: `fastapi>=0.110.0`, `uvicorn>=0.28.0`, `python-dotenv>=1.0.1`, `python-multipart>=0.0.9`.
- **Vector Search & ML**: `faiss-cpu>=1.8.0`, `sentence-transformers>=2.2.0`, `torch` (CPU wheel).
- **LLM Integrations**: `google-genai>=1.0.0`, `openai>=1.0.0` (optional chatbot).
- **Document Processing**: `pypdf>=4.0.0`.
- **Database & Migration**: `sqlalchemy>=2.0.0`, `alembic>=1.13.0`, `psycopg2-binary>=2.9.9`, `supabase>=2.0.0`.
- **Testing & Quality Assurance**: `pytest>=7.4.0`, `pytest-asyncio>=0.21.0`.

### 2.3 System-Level Runtime Dependencies (Debian Slim)
- `libgomp1`: Required by FAISS and PyTorch OpenMP for parallelized vector similarity search.
- `curl`: Required for Docker native container health checks.
- `ca-certificates`: Required for secure outbound HTTPS communication with Google GenAI and Supabase endpoints.

---

## 3. Configuration & Environment Variables

All settings are managed centrally via `backend/config.py` using `Settings` with strict secret masking and environment loading.

| Environment Variable | Required | Default in Docker | Purpose |
| :--- | :---: | :---: | :--- |
| `HOST` | No | `0.0.0.0` | Container bind address |
| `PORT` | No | `8000` | HTTP service port |
| `ENVIRONMENT` | No | `production` | Deployment mode flag (`production`, `development`) |
| `APP_NAME` | No | `AI Healthcare Research & Patient Assistance Agent` | Application display title |
| `GEMINI_API_KEY` | **Yes (for LLM)** | *None* | Google GenAI API key for medical answer generation |
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Primary LLM model identifier |
| `GEMINI_TEMPERATURE` | No | `0.2` | Sampling temperature for factual consistency |
| `DATABASE_URL` | No | `postgresql://postgres:postgres@postgres:5432/ai_healthcare` | PostgreSQL connection URL |
| `SUPABASE_URL` | No | *None* | Optional Supabase project URL |
| `SUPABASE_PUBLISHABLE_KEY` | No | *None* | Optional Supabase anonymous/service key |
| `OPENAI_API_KEY` | No | *None* | Optional OpenAI key for secondary conversational agents |
| `RAG_TOP_K` | No | `5` | Maximum retrieved context chunks per inquiry |
| `RAG_SIMILARITY_THRESHOLD` | No | `0.25` | Minimum cosine similarity score for relevance gate |
| `LOG_FORMAT` | No | `json` | Structured logging format (`json` or `text`) |
| `RAG_LATENCY_P95_BUDGET_MS` | No | `3500.0` | Latency SLA budget threshold |
| `HF_HOME` | No | `/app/cache/huggingface` | Hugging Face offline model cache path |

---

## 4. Docker Strategy & Container Architecture

### 4.1 Multi-Stage Build Architecture
1. **Stage 1 (`builder`)**:
   - Base: `python:3.11-slim-bookworm`
   - Installs build utilities: `gcc`, `g++`, `build-essential`, `libgomp1`.
   - Creates a virtual environment in `/opt/venv`.
   - Installs all dependencies from `requirements.txt` with wheel caching.
   - Pre-downloads and bakes the `sentence-transformers/all-MiniLM-L6-v2` model weights into `/opt/huggingface_cache` to ensure completely offline, instant runtime startup.
2. **Stage 2 (`runtime`)**:
   - Base: `python:3.11-slim-bookworm`
   - Installs minimal runtime dependencies only: `libgomp1`, `curl`, `ca-certificates`.
   - Copies `/opt/venv` and `/opt/huggingface_cache` from `builder`.
   - Creates a dedicated non-root user `appuser` (UID 10001, GID 10001).
   - Sets environment variables:
     - `PYTHONUNBUFFERED=1`
     - `PYTHONDONTWRITEBYTECODE=1`
     - `PATH="/opt/venv/bin:$PATH"`
     - `HF_HOME="/app/cache/huggingface"`
     - `PORT=8000`
     - `HOST=0.0.0.0`
   - Copies application source code: `backend/`, `data/vector_store/`, `alembic/`, `alembic.ini`.
   - Configures non-root ownership (`chown -R appuser:appuser /app`).
   - Exposes port `8000`.
   - Healthcheck: `curl -f http://localhost:8000/health || exit 1`.
   - Entrypoint: executes production startup validation, then launches `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2`.

---

## 5. Startup Validation Mechanism (`backend/startup_validation.py`)

To prevent corrupted or incomplete containers from entering service discovery or load-balancer pools, a dedicated startup validation module will execute before accepting HTTP traffic:

```
Container Startup -> backend/startup_validation.py
  |
  +--> 1. Validate environment configuration
  +--> 2. Validate FAISS index exists on disk
  +--> 3. Validate metadata.json exists on disk
  +--> 4. Validate exact 744 FAISS vector invariant
  +--> 5. Validate exact 744 metadata record invariant
  +--> 6. Validate 1:1 vector_id alignment
  +--> 7. Validate embedding service (dim == 384)
  +--> 8. Validate medical safety engine (18 categories)
  |
  +--> If all pass: LOG OK -> Launch Uvicorn
  +--> If any fail: LOG CRITICAL ERROR -> Exit code 1 (Fast Fail)
```

---

## 6. Health & Readiness Strategy

The existing `/health` endpoint will be augmented with dedicated `/health/liveness` and `/health/readiness` sub-routes:
- **`GET /health`**: Consolidated backward-compatible health report (retains existing JSON contract for Phase 5 tests).
- **`GET /health/liveness`**: Ultra-fast, zero-overhead Kubernetes/Docker liveness probe returning `{"status": "alive"}`.
- **`GET /health/readiness`**: Non-destructive operational readiness check confirming vector store (744 count), embedding dimension (384), safety engine, and database connectivity status.

---

## 7. Deployment Risks & Safeguards

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Vector Store Mutation** | Invariant corruption (744 vectors altered) | Docker Compose mounts `data/vector_store` as `:ro` (read-only); startup validation enforces exact 744 count. |
| **Missing Model Weights** | Container stalls or downloads 90MB on every boot | Model weights baked into Docker image during the build stage; `HF_HOME` points to local pre-cached directory. |
| **Missing Libgomp on Debian Slim** | `ImportError: libgomp.so.1` causing fatal container crash | Explicit installation of `libgomp1` in runtime stage. |
| **Credential Exposure in Images** | Leaked API keys in image layers | Zero `.env` files copied into image via `.dockerignore`; non-root user execution; masked logging. |
| **Database Network Delay** | Service startup timeout if remote DB is slow | Non-blocking database connection with pool pre-ping and automated SQLite fallback for local resilience. |

---

## 8. Files to Create and Modify

### Files to Create:
1. `PHASE_6_PRODUCTION_PLAN.md` (this blueprint)
2. `Dockerfile` (multi-stage production container)
3. `.dockerignore` (strict exclusion of secrets, tests, caches; preservation of FAISS data)
4. `docker-compose.yml` (multi-service orchestration: API + PostgreSQL + volumes + health checks)
5. `.env.example` (clean production template with zero secrets)
6. `backend/startup_validation.py` (pre-flight operational invariant validator)
7. `.github/workflows/ci.yml` (automated CI/CD quality gate pipeline)
8. `scripts/validate_production.ps1` (Windows-friendly validation script)
9. `scripts/validate_production.py` (Cross-platform Python validation runner)
10. `PHASE_6_PRODUCTION_DEPLOYMENT_REPORT.md` (final production audit report)

### Files to Modify:
1. `backend/main.py`: Add startup event lifecycle executing startup validation, and expose `/health/liveness` and `/health/readiness`.
2. `README.md`: Add production Docker and Compose deployment guide.
