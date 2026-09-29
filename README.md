# AI-Healthcare-Agent

> **An evidence-grounded, production-hardened clinical AI research and workflow assistance platform.**

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pytest Suite](https://img.shields.io/badge/tests-576%20passed%20(100%25)-brightgreen.svg)](https://docs.pytest.org)
[![Security Hardened](https://img.shields.io/badge/security-41%2F41%20passed-success.svg)](#8-security-features)
[![FAISS Vectors](https://img.shields.io/badge/FAISS%20Index-744%20vectors%20(read--only)-informational.svg)](#11-faiss--vector-store-architecture)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#21-github-project-information)

---

## 1. Project Title
### **AI Healthcare Research & Clinical Workflow Assistance Agent**

---

## 2. Short Project Description
The **AI Healthcare Agent** is a multi-tier clinical AI platform engineered to assist healthcare professionals, clinical researchers, and patient workflow coordinators with evidence-grounded medical document retrieval, query synthesis, and clinical research summarization.

Operating under a strict **Retrieval-Augmented Generation (RAG)** paradigm with zero-tolerance medical safety guardrails, the system strictly grounds every generated response in verified reference clinical guidelines. If relevant documentation is unavailable, the agent halts synthesis rather than generating speculative or ungrounded assertions.

---

## 3. Problem Statement
Modern healthcare professionals and clinical researchers are inundated with clinical literature, evolving treatment protocols, and extensive clinical documentation. General-purpose large language models (LLMs) present substantial risks in clinical settings:
- **Hallucinations & Confabulations**: Generating medically plausible yet incorrect clinical guidance, contraindicated drug combinations, or erroneous pharmacological dosages.
- **Unverified & Phantom Citations**: Citing non-existent journal articles, phantom guideline chapters, or misattributing recommendations.
- **Lack of Life-Safety Boundaries**: Failing to immediately triage acute life-threatening medical emergencies (e.g., myocardial infarction, acute ischemic stroke, anaphylaxis).
- **Privacy & Compliance Exposure**: Leaking sensitive Protected Health Information (PHI) or credentials into centralized logging systems.

**AI-Healthcare-Agent** addresses these challenges through a deterministic, retrieval-first architecture that couples local dense vector embeddings, immutable FAISS indexing, algorithmic citation enforcement, PII/PHI redaction, and strict medical guardrails.

---

## 4. Key Features
- **Deterministic Evidence-Grounded RAG**: Queries local 384-dimensional embeddings across an immutable knowledge base of clinical guidelines.
- **Strict Citation Validation (`CitationValidator`)**: Synthesized answers must explicitly cite retrieved document chunks (`[Source N]`); citations are verified against ground-truth vector metadata.
- **Hallucination Protection (`HallucinationGuard`)**: Post-generation clinical contradiction and entity distortion detection to safeguard against factual drift.
- **Zero-Tolerance Emergency Interception**: Pre-retrieval medical safety engine intercepts acute emergency symptoms with zero false negatives (FN = 0), returning immediate emergency directives.
- **Immutable Read-Only Vector Store**: Production vector store maintains a fixed invariant of **744 FAISS vectors** and **744 metadata records** verified by SHA256 checksums.
- **Multi-Agent Clinical Triage**: Coordinated agent architecture featuring Orchestrator, Research, Document, Explanation, and Safety agents.
- **Enterprise Security Middleware**: 10 MB payload enforcement, RFC-compliant sliding-window rate limiting, and XML prompt injection boundary escaping.
- **PII / PHI & Credential Scrubbing**: Recursive log sanitization masking database credentials, bearer tokens, JWTs, API keys, emails, phone numbers, and SSNs.
- **Resilient Multi-Engine Persistence**: PostgreSQL with SQLAlchemy 2.0 connection pooling (`QueuePool`), automatic offline SQLite failover, and Alembic migrations.
- **Production Observability**: Correlated request tracing (`X-Request-ID`) and machine-readable JSON logging format (`HTTPRequestLogEvent`, `RAGStructuredLogEvent`).
- **Interactive Clinical User Interface**: Streamlit web frontend (`frontend/`) supporting document upload, multi-turn triage dialogues, and grounded source inspection.

---

## 5. Architecture

```
                                  [ Client Layer ]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         Streamlit Web Frontend                    External API Consumers
       (frontend/ : Port 8501)                     (HTTP / REST Clients)
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                      [ Edge & Gateway Security Layer ]
                    ├── Security Headers (CSP, HSTS, X-Frame-Options)
                    ├── 10 MB Gateway Payload Size Enforcement (HTTP 413)
                    ├── Sliding-Window Rate Limiter (Tiered Quotas)
                    └── Request Tracing (X-Request-ID Header Injection)
                                         │
                                         ▼
                      [ Application Routing Layer (FastAPI) ]
                    ├── /rag/query & /rag/retrieve (RAG Core)
                    ├── /health, /health/liveness, /health/readiness (Probes)
                    ├── /documents/upload & /documents/list (Ingestion)
                    ├── /agents/run (Multi-Agent Clinical Triage)
                    └── /db/* (Persistence CRUD Endpoints)
                                         │
                                         ▼
                      [ Clinical Safety & Guardrail Layer ]
                    ├── MedicalSafetyEngine (15–18 Categories)
                    ├── Emergency Interception (Zero False Negatives)
                    └── XML Boundary Tag Escaping (<, >, ", ')
                                         │
                                         ▼
                     [ Retrieval-Augmented Generation (RAG) ]
                    ├── Query Expander (Decomposition)
                    ├── SentenceTransformers (all-MiniLM-L6-v2, 384-dim)
                    ├── FAISS Vector Store (IndexFlatIP, 744 Vectors, Read-Only)
                    ├── Relevance Threshold Filtering (Cosine Similarity >= 0.25)
                    ├── Context Assembly & Multi-Document Deduplication
                    ├── Google Gemini LLM API (gemini-3.5-flash-lite, temp=0.2)
                    ├── CitationValidator (Strict [Source N] Verification)
                    └── HallucinationGuard (Clinical Contradiction Detection)
                                         │
                                         ▼
                       [ Persistence & Storage Layer ]
                    ├── PostgreSQL 16 (Relational Chat, Documents, Users)
                    ├── SQLAlchemy Connection Pooling (QueuePool + Health Cache)
                    ├── Automatic Local SQLite Failover (data/ai_healthcare.db)
                    └── Read-Only Vector Store Assets (index.faiss, metadata.json)
```

---

## 6. RAG Pipeline

The RAG pipeline strictly decouples document retrieval from generative synthesis:

1. **User Query Submission**: Natural language inquiry submitted via `POST /rag/query` or `POST /rag/retrieve`.
2. **Boundary Sanitization**: Input text sanitized; XML boundary tags escaped to prevent prompt injection.
3. **Emergency Symptom Screening**: Evaluated against clinical taxonomy categories (cardiovascular, cerebrovascular, respiratory distress, anaphylaxis). If critical, generation halts and immediate emergency care instructions are returned.
4. **Local Query Embedding**: Vectorized locally using Hugging Face `sentence-transformers/all-MiniLM-L6-v2` into a normalized 384-dimensional dense vector ($L_2$ norm = 1.0).
5. **Exact Semantic Vector Search**: Queried against Meta's `faiss.IndexFlatIP` across 744 normalized clinical guideline vectors.
6. **Relevance Threshold Cutoff**: Chunks scoring below cosine similarity **0.25** are rejected.
7. **Context Construction & Grounding**: Relevant chunks deduplicated and formatted with explicit `[SOURCE N]` boundary headers including document ID, chunk ID, and page references.
8. **Generative LLM Synthesis**: Prompts Google Gemini (`gemini-3.5-flash-lite`, temperature=0.2) with strict instructions to synthesize responses exclusively from the provided reference context.
9. **Citation Verification**: Post-generation regex parser validates that every inline citation matches an actual retrieved context chunk.
10. **Contradiction Screening**: Evaluated by `HallucinationGuard` to detect entity distortion.
11. **Structured Payload Delivery**: Delivers clinical answer, safety disclaimers, performance latency metrics, and independent ground-truth source objects.

---

## 7. Technology Stack

| Category | Component / Technology | Specification & Version |
|---|---|---|
| **Core Framework** | Python | 3.11 / 3.12 / 3.13 |
| **API Server** | FastAPI & Uvicorn | Asynchronous ASGI, OpenAPI 3.1 / Swagger |
| **Embeddings** | Hugging Face SentenceTransformers | `all-MiniLM-L6-v2` (384-dimensional local tensors) |
| **Vector Store** | Meta FAISS | `faiss-cpu` (IndexFlatIP exact cosine similarity) |
| **Generative LLM** | Google GenAI SDK | `gemini-3.5-flash-lite` (temperature=0.2) |
| **Relational Database** | PostgreSQL & SQLite | PostgreSQL 16 / SQLite failover (`ai_healthcare.db`) |
| **ORM & Migrations** | SQLAlchemy 2.0 & Alembic | Declarative mapping, QueuePool, connection pooling |
| **Document Ingestion** | `pypdf` | PDF text extraction and whitespace normalization |
| **Frontend UI** | Streamlit | Multi-page clinical dashboard (`frontend/`) |
| **Testing** | Pytest, TestClient, AnyIO | 576 automated unit, integration, and load tests |
| **Containerization** | Docker & Docker Compose | Multi-stage build, non-root user (`appuser:10001`) |

---

## 8. Security Features
- **Payload Size Enforcement**: Gateway middleware strictly caps incoming request bodies at **10 MB** (`HTTP 413 Content Too Large`).
- **HTTP Security Headers**: Enforces `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, and `Referrer-Policy: strict-origin-when-cross-origin`.
- **Sliding-Window Rate Limiting (`SlidingWindowRateLimiter`)**: Thread-safe in-memory sliding-window limiter with RFC-compliant headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `Retry-After`).
  - Tiered quotas: `auth_login` (10/min), `rag_query` (30/min), `agent_query` (30/min), `doc_upload` (15/min), `rag_retrieve` (60/min), `default` (60/min).
- **Prompt Injection Neutralization**: Neutralizes XML boundary injection attempts by escaping delimiter characters (`<`, `>`, `"`, `'`).
- **PII & PHI Scrubbing (`sanitize_value`)**: Recursive sanitization scrubs:
  - Database URLs with embedded passwords (`postgresql://user:[REDACTED_PASSWORD]@host/db`)
  - Google Gemini API keys (`[REDACTED_API_KEY]`) and OpenAI keys (`[REDACTED_OPENAI_KEY]`)
  - Bearer tokens (`Bearer [REDACTED]`) and JWTs (`[REDACTED_JWT]`)
  - Social Security Numbers (`[REDACTED_SSN]`) and Credit Card numbers (`[REDACTED_CARD]`)
  - Email addresses (`[REDACTED_EMAIL]`) and Phone numbers (`[REDACTED_PHONE]`)
- **Safe Logging Mode (`SAFE_LOG_MODE`)**: Hashes and truncates clinical user queries (`preview... [MASKED_PHI len=N hash=...]`) to prevent PHI retention in cloud logging systems.
- **Container Isolation**: Multi-stage Docker container runs as non-root user `appuser` (UID 10001) with vector assets mounted strictly read-only (`:ro`).

---

## 9. Medical Safety Features
- **Zero Critical False Negatives**: Evaluated against gold-standard clinical emergency benchmarks with 100% sensitivity and zero false negatives.
- **Emergency Interception Taxonomy**: Screens for life-threatening presentations:
  - Acute Coronary Syndrome (crushing chest pain radiating to jaw/left arm, diaphoresis)
  - Cerebrovascular Accidents (FAST criteria: unilateral weakness, facial droop, slurred speech)
  - Severe Respiratory Compromise (stridor, cyanosis, severe acute dyspnea)
  - Systemic Anaphylaxis (acute urticaria, airway constriction, hypotension)
- **Mandatory Medical Disclaimer**: Every generated answer programmatically includes the educational and non-diagnostic disclaimer.
- **Deterministic Fallback**: If retrieved evidence is insufficient or contradictory, synthesis is aborted with a transparent message indicating that reference documents lack adequate coverage.

---

## 10. Database Architecture
The relational persistence layer is managed by **SQLAlchemy 2.0** with **Alembic** schema migrations:
- **Primary Engine**: PostgreSQL with production connection pooling (`QueuePool`).
  - `pool_size=5`, `max_overflow=10`, `pool_recycle=1800s`, `pool_timeout=30s`, `pool_pre_ping=True`.
- **Health Check Caching**: Database health probe incorporates a 30-second TTL cache (`max_age_seconds=30.0`) to protect remote cloud database instances from connection storms during high-frequency health probes.
- **Automatic Local Failover**: Transparently falls back to local SQLite (`data/ai_healthcare.db`) if primary PostgreSQL connection fails during startup.
- **Core Entities (`Base.metadata`)**:
  - `users`: User identity, email, authentication identifiers, and timestamps.
  - `documents`: Document catalog, filenames, file sizes, hashes, and upload metadata.
  - `document_chunks`: Extracted text segments, chunk indices, and foreign keys to parent documents.
  - `conversations`: Clinical chat session threads and user associations.
  - `messages`: Historical dialogue turns, role classifications (`user`/`assistant`), and timestamps.

---

## 11. FAISS / Vector-Store Architecture
- **Index Implementation**: `faiss.IndexFlatIP` (Inner Product).
- **Mathematical Cosine Equivalence**: Dense 384-dimensional embedding vectors are $L_2$-normalized prior to indexing and querying, making the inner product calculation mathematically identical to cosine similarity:
  $$\text{sim}(\mathbf{q}, \mathbf{d}) = \langle \hat{\mathbf{q}}, \hat{\mathbf{d}} \rangle = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$
- **Read-Only Production Invariant**:
  - **FAISS Vectors**: Exactly **744** normalized vectors.
  - **Metadata Records**: Exactly **744** corresponding JSON chunk records.
  - **Dimension**: **384** dimensions.
  - **Cryptographic Hashes**:
    - `index.faiss`: `7ecc494c66fd93971990a6a297fea1052fda49d2ce897ce37543d51317ddce19`
    - `metadata.json`: `67cdf7ad248264fff40a15dab51747bb05b3164dc4417abed0b46087c799065a`

---

## 12. API Overview

| Method | Endpoint | Description | Auth / Access |
|---|---|---|:---:|
| `GET` | `/` | API service metadata, online status, and version | Public |
| `GET` | `/health` | Consolidated subsystem operational readiness probe | Public |
| `GET` | `/health/liveness` | Kubernetes liveness probe (immediate 200 alive state) | Public |
| `GET` | `/health/readiness` | Subsystem readiness probe (verifies 744 vector invariant) | Public |
| `POST` | `/rag/retrieve` | Semantic similarity retrieval returning top-$K$ evidence chunks | Public / Token |
| `POST` | `/rag/query` | End-to-end evidence-grounded RAG query synthesis | Public / Token |
| `POST` | `/documents/upload` | Medical PDF document upload, validation, and text extraction | Authenticated |
| `POST` | `/agents/run` | Multi-agent clinical diagnostic triage pipeline | Authenticated |
| `GET` | `/docs` | Interactive Swagger OpenAPI documentation UI | Public |
| `GET` | `/redoc` | Interactive ReDoc API documentation UI | Public |

---

## 13. Testing and Evaluation

The repository maintains an automated test suite with **100% pass rate** across all functional, security, and performance domains:

```text
============================== test session starts ==============================
collected 576 items

tests/ ................................................................. [100%]

============================== 576 passed in 175.38s ============================
```

### Test Suite Breakdown:
- **Core Functional & Integration Tests**: 501 tests covering chunking, embeddings, vector search, Gemini synthesis, multi-agent orchestration, repository persistence, and Streamlit client communications.
- **Phase 8 Dedicated Security & Load Suite**: 41 tests verifying prompt injection defense, PII scrubbing, RAG poisoning isolation, adversarial clinical inputs, and concurrency load performance (P95 `/health` latency: 66.61 ms).
- **Phase 6 Production Readiness**: 7 tests validating health checks, liveness/readiness probes, and startup invariants.
- **Phase 9 Architecture & Hardening Suite**: 27 tests validating environment configuration, database connection pooling, structured logging, tiered rate limiting, credential auditing, containerization syntax, and Alembic migrations.
- **Production Pre-Flight Harness (`scripts/validate_production.py`)**: All 5 operational quality gates verified.

---

## 14. Project Structure

```
AI-Healthcare-Agent/
├── alembic/                       # Alembic database migration scripts
│   ├── versions/                  # Revision versions (001_initial_schema.py)
│   ├── env.py                     # Migration environment configuration
│   └── script.py.mako             # Revision template
├── alembic.ini                    # Alembic settings configuration
├── backend/                       # Core FastAPI application package
│   ├── agents/                    # Multi-agent clinical triage implementation
│   ├── api/                       # REST API route handlers
│   ├── database/                  # SQLAlchemy models, repositories, and pooling
│   ├── evaluation/                # Observability, citation validation, metrics
│   ├── rag/                       # RAG prompt construction and retrieval logic
│   ├── safety/                    # Medical safety engine and emergency guards
│   ├── security/                  # Rate limiting, header security, PII hasher
│   ├── services/                  # Embeddings, FAISS, Gemini, ingestion services
│   ├── config.py                  # Multi-environment configuration manager
│   ├── main.py                    # Application entry point and middleware
│   └── startup_validation.py      # Non-destructive startup pre-flight validator
├── data/                          # Data persistence directory
│   ├── documents/                 # PDF storage (git-ignored)
│   └── vector_store/              # Read-only FAISS index and metadata
│       ├── index.faiss            # FAISS index (744 vectors, 384 dimensions)
│       └── metadata.json          # Positional chunk metadata (744 records)
├── frontend/                      # Streamlit Clinical Web UI
│   ├── components/                # Reusable UI widgets (chat, sidebar, cards)
│   ├── pages/                     # Dashboard, document upload, history views
│   ├── utils/                     # Formatting and helper utilities
│   ├── api_client.py              # Centralized backend HTTP client
│   └── app.py                     # Streamlit application entry point
├── scripts/                       # Operational validation and load testing tools
│   ├── load_test.py               # Concurrent load benchmark harness
│   ├── validate_production.ps1    # PowerShell production validation runner
│   └── validate_production.py     # Cross-platform production validation harness
├── tests/                         # Comprehensive Pytest test suite (576 tests)
│   ├── test_phase8_security.py    # Dedicated security and vulnerability tests
│   ├── test_phase9_config.py      # Configuration management tests
│   ├── test_phase9_db_pool.py     # Connection pool resilience tests
│   └── ...                        # Additional test modules (74 test files)
├── _backup_react_frontend/        # Archived React 18 / Vite SPA (Protected backup)
├── .dockerignore                  # Docker build exclusions
├── .env.example                   # Environment variable template with placeholders
├── .gitignore                     # Git repository exclusions
├── docker-compose.yml             # Container orchestration (API + PostgreSQL)
├── Dockerfile                     # Multi-stage non-root container definition
├── pytest.ini                     # Pytest runner configuration
├── README.md                      # Developer-focused documentation
└── requirements.txt               # Python package dependencies
```

---

## 15. Installation

### Prerequisites
- **Python**: Version 3.11, 3.12, or 3.13 installed.
- **Git**: Installed for source control.
- **C++ Build Tools** (Windows only): Required for native `faiss-cpu` compilation if wheel is not cached.

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Rakeshpalya/RAG-Based-Healthcare-Information-Assistant.git
   cd AI-Healthcare-Agent
   ```

2. **Create and Activate Virtual Environment**:
   ```powershell
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   ```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 16. Environment Variable Setup

Copy the template file to create your local `.env`:
```powershell
copy .env.example .env     # Windows
cp .env.example .env       # Linux / macOS
```

Configure your local `.env` using **placeholders only**. Never commit live credentials to version control:

```ini
# Application & Environment Settings
APP_NAME="AI Healthcare Research & Patient Assistance Agent"
ENVIRONMENT="development"              # Options: development, test, production
HOST="127.0.0.1"
PORT=8000

# Google Gemini LLM API Configuration
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_MODEL="gemini-3.5-flash-lite"
GEMINI_TEMPERATURE="0.2"

# RAG Retrieval Configuration
RAG_TOP_K="5"
RAG_SIMILARITY_THRESHOLD="0.25"
RAG_LATENCY_P95_BUDGET_MS="3500.0"

# Relational Database Configuration
# Local SQLite fallback: sqlite:///data/ai_healthcare.db
# PostgreSQL / Supabase: postgresql://username:password@hostname:5432/dbname
DATABASE_URL="postgresql://postgres:your_password_here@localhost:5432/ai_healthcare"
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=1800
DB_CONNECT_TIMEOUT=5

# Supabase Auth Configuration (Optional)
SUPABASE_URL="https://your_project_id_here.supabase.co"
SUPABASE_PUBLISHABLE_KEY="your_supabase_publishable_key_here"

# Observability & Safety Settings
LOG_FORMAT="json"                      # Options: json, text
SAFE_LOG_MODE="false"                  # Set to true in production to hash clinical queries
```

---

## 17. How to Run the Backend

### Local Development (Uvicorn)
```powershell
# Windows
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
```bash
# Linux / macOS
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Swagger OpenAPI Documentation: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

### Production Pre-Flight Validation
Execute the standalone operational validation harness before serving traffic:
```bash
python scripts/validate_production.py
```

### Containerized Execution (Docker Compose)
```bash
# Build and launch backend API and PostgreSQL services
docker-compose up -d --build

# Follow API service logs
docker-compose logs -f api

# Verify running container health status
docker-compose ps
```

---

## 18. How to Run the Original Frontend

The **original active frontend** for this project is the multi-page **Streamlit** clinical interface located in `frontend/`.

```powershell
# Windows
.\venv\Scripts\streamlit.exe run frontend/app.py
```
```bash
# Linux / macOS
streamlit run frontend/app.py
```

- Accessible locally at: `http://localhost:8501`
- Features: Clinical search and query interface, interactive source card inspector, PDF document upload, and historical conversation viewing.

*(Note: The previous React 18 / Vite single-page application is preserved in `_backup_react_frontend/` and is not part of the active frontend runtime).*

---

## 19. Limitations / Disclaimer

> [!CAUTION]
> ### **CLINICAL & MEDICAL DISCLAIMER**
> **AI-Healthcare-Agent is an educational, research, and clinical workflow assistance tool.**  
> - It **DOES NOT** provide medical diagnoses, treatment decisions, clinical prognoses, or pharmaceutical prescriptions.
> - It is **NOT** a certified medical device and **MUST NOT** be used as a substitute for the clinical judgment of licensed physicians, nurses, or certified healthcare practitioners.
> - If a patient describes symptoms suggestive of an acute medical emergency, they must immediately contact emergency medical services (e.g., 911 in the US) or present to an emergency department.

### Known Technical Limitations
1. **Remote Cloud Latency**: When connecting to remote cloud-hosted PostgreSQL instances (e.g., Supabase), uncached live database pings are subject to network round-trip time. The 30-second connection health cache prevents probe storms during high-frequency checks.
2. **Generative LLM Availability**: When external network access to the Google Gemini API is unavailable, the RAG engine continues to perform semantic retrieval from local FAISS, but generative answer synthesis falls back to structured evidence excerpts.
3. **English Clinical Vocabulary**: The default embedding model (`all-MiniLM-L6-v2`) is optimized for English-language medical literature and clinical guidelines.

---

## 20. Future Improvements
- **Multi-Modal Diagnostic Support**: Integration of medical imaging analysis (DICOM / radiological scans) alongside clinical narrative text.
- **Distributed Redis Cache**: Shared caching and token-bucket rate limiting across horizontally scaled multi-instance deployments.
- **FHIR / HL7 Interoperability**: Direct integration with Fast Healthcare Interoperability Resources (FHIR) endpoints for seamless Electronic Health Record (EHR) data retrieval.
- **Quantized Local LLM Fallback**: Local on-premise generative inference fallback (e.g., Med-Gemma via ONNX / llama.cpp) for air-gapped clinical deployments.

---

## 21. GitHub Project Information

- **Repository**: `AI-Healthcare-Agent`
- **Maintainer**: Autonomous System Engineer & Clinical AI Pair Programmers
- **License**: [MIT License](LICENSE)
- **Contribution Guidelines**: 
  - Ensure all modifications preserve the **744 vector FAISS invariant**.
  - All submitted pull requests must maintain a **100% pass rate** on the 576-test suite (`pytest -q`).
  - Never stage or commit active `.env` files or credentials.
