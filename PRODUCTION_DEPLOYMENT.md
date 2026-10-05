# Production Deployment & Operational Guide
## AI Healthcare Agent — Phase 3.6 Production Hardening

This comprehensive production deployment guide outlines the operational architecture, deployment topology, secret management, monitoring, high availability, backup & disaster recovery, and security models for the **AI Healthcare Agent** production platform.

---

### 1. Architecture

The AI Healthcare Agent is structured as a resilient, horizontally scalable microservice pipeline:

```
[ Client Request ]
       │
       ▼
[ Nginx / Cloudflare Ingress ] (TLS Termination, IP rate limiting)
       │
       ▼
[ FastAPI Application Pods (Non-root user UID 10001, Gunicorn/Uvicorn) ]
   ├── Authentication & DB User Resolution (Supabase JWT / PostgreSQL)
   ├── Distributed Rate Limiting (Sliding Window via Redis L2, local in-memory fallback)
   ├── Medical Safety Pre-Screen (Immediate emergency / suicide / harm interception)
   ├── Semantic Retrieval (FAISS index + SentenceTransformers all-MiniLM-L6-v2)
   ├── Sufficiency & Relevance Gate (Cosine threshold ≥ 0.25)
   ├── LLM Response Cache (L1 Memory LRU + L2 Redis Distributed Cache)
   ├── LLM Concurrency Controller (Bounded semaphores + queue management)
   ├── Gemini Service (Exponential backoff, fallback models: gemini-3.5-flash-lite, gemini-2.5-flash)
   ├── Citation & Grounding Validator (Source provenance verification)
   ├── Hallucination & Contradiction Guard (Negation, entity, & numerical checks)
   ├── Medical Safety Post-Screen (Clinical boundaries & disclaimer enforcement)
   └── Structured Observability & Audit Logger (PHI masked, Prometheus exporter on :9090)
       │
       ├──► [ Redis Cluster / Replica ] (L2 Cache + Distributed Rate Limiting)
       ├──► [ PostgreSQL 15 / Supabase ] (Patient metadata, document registry, RLS)
       └──► [ Prometheus & Grafana ] (Metrics scraping on /metrics, telemetry dashboard)
```

---

### 2. Environment Variables

All configuration is centralized in `backend/config.py` using `Settings` loaded safely from the environment:

| Variable | Type | Default | Description |
|---|---|---|---|
| `ENVIRONMENT` | string | `production` | Deployment environment (`production`, `staging`, `development`). |
| `DEBUG` | boolean | `false` | Must NEVER be `true` in production mode. |
| `GEMINI_API_KEY` | string | *Secret* | Google GenAI API key for medical answer synthesis. |
| `GEMINI_MODEL` | string | `gemini-3.5-flash-lite` | Primary production LLM model. |
| `GEMINI_TEMPERATURE` | float | `0.0` | Deterministic sampling temperature for clinical precision. |
| `GEMINI_MAX_OUTPUT_TOKENS` | int | `1024` | Token generation ceiling. |
| `GEMINI_TIMEOUT_SECONDS` | float | `30.0` | Max upstream request duration before timeout. |
| `DATABASE_URL` | string | *Secret* | PostgreSQL connection string (`postgresql://user:pass@host:5432/db`). |
| `REDIS_URL` | string | `redis://localhost:6379/0` | Redis connection URL with optional authentication. |
| `REDIS_CACHE_ENABLED` | boolean | `true` | Enables L2 distributed Redis caching. |
| `REDIS_RATE_LIMITER_ENABLED`| boolean | `true` | Enables distributed sliding-window rate limiting. |
| `RATE_LIMIT_RAG_PER_MINUTE` | int | `30` | Maximum RAG queries permitted per minute per user/IP. |
| `LLM_CACHE_ENABLED` | boolean | `true` | Enables L1 memory + L2 Redis caching. |
| `LLM_CACHE_TTL_SECONDS` | int | `3600` | Time-to-live for validated healthcare responses (1 hour). |
| `LLM_CACHE_MAX_ENTRIES` | int | `1000` | In-memory L1 cache capacity. |
| `SUPABASE_URL` | string | *Secret* | Supabase project URL (`https://<project-ref>.supabase.co`). |
| `SUPABASE_PUBLISHABLE_KEY` | string | *Secret* | Public anon key for authentication token verification. |
| `FAISS_STORAGE_DIR` | string | `data/vector_store` | Persistent filesystem path for FAISS index and metadata. |
| `EMBEDDING_DIMENSION` | int | `384` | Vector dimension for `sentence-transformers/all-MiniLM-L6-v2`. |
| `PROMETHEUS_METRICS_ENABLED`| boolean | `true` | Exposes `/metrics` Prometheus scrape endpoint. |

---

### 3. Secret Management

1. **Zero Secret Leakage Principle**:
   - Secrets (`GEMINI_API_KEY`, `DATABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `REDIS_URL`) are sanitized in all log messages, JSON outputs, exceptions, and Prometheus labels.
   - Raw secrets are masked via `Settings.get_sanitized_config_dict()` and `GeminiService.sanitize_secret_static()`.
2. **KMS & Vault Integration**:
   - Inject secrets as container environment variables via AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault.
   - Never commit `.env` or plain-text credentials to git. The `.gitignore` enforces exclusion.
3. **File System Permissions**:
   - Configuration files mounted in containers must have permissions `0400` or `0440`, owned by the unprivileged user (`healthcare` UID `10001`).

---

### 4. Docker Deployment

The application container uses a multi-stage, hardened Dockerfile:

```bash
# 1. Build the production image
docker compose build api

# 2. Inspect the image security attributes
docker run --rm ai-healthcare-agent id
# Output: uid=10001(healthcare) gid=10001(healthcare) groups=10001(healthcare)

# 3. Launch full production stack with API, Redis, Prometheus, and Grafana
docker compose up -d
```

Key Docker hardening features:
- Runs strictly as non-root user `healthcare` (UID `10001`).
- Vector store is mounted `:ro` (read-only) for serving instances to prevent accidental index corruption.
- Health checks execute every 30s with a 5s timeout and 3 retries (`curl -f http://localhost:8000/health/live`).
- Restart policy `unless-stopped` applied to all containers.

---

### 5. Redis Deployment

Redis operates as the shared distributed state layer for L2 response caching and rate limiting:
- **Persistence**: Append-Only File (`appendonly yes`) enabled with persistent named volume `redis_data`.
- **Degradation**: If Redis restarts, experiences network partition, or fails, the application automatically degrades to in-memory L1 LRU caching and in-process rate limiting without downtime.
- **Eviction Policy**: `allkeys-lru` with a bounded memory ceiling (e.g. `maxmemory 512mb`).

---

### 6. Prometheus Deployment

Prometheus continuously scrapes the FastAPI metrics endpoint:
- **Scrape Interval**: 15s (`monitoring/prometheus.yml`).
- **Endpoint**: `http://api:8000/metrics`.
- **Exposed Metrics**:
  - `rag_requests_total`, `rag_requests_successful`, `rag_requests_failed`, `rag_requests_blocked`
  - `rag_cache_hits`, `rag_cache_misses`, `rag_cache_hit_ratio`
  - `rag_llm_calls_total`, `rag_llm_failures_total`, `rag_llm_retries_total`, `rag_llm_latency_seconds`
  - `rag_safety_blocked_total`, `rag_safety_passed_total`
  - `rag_retrieval_latency_seconds`, `rag_total_latency_seconds`
  - `process_resident_memory_bytes`, `process_cpu_seconds_total`

---

### 7. Grafana Setup

Grafana is provisioned automatically with pre-configured dashboards:
- **Port**: `3000` (Default credentials configured via environment).
- **Datasource**: Prometheus (`http://prometheus:9090`).
- **Dashboard**: `monitoring/grafana/dashboards/ai_healthcare_dashboard.json`.
- **Panels**:
  1. Request Throughput & Status Distribution (Stacked 200/429/500).
  2. Cache Hit Rate & Efficiency Ratio.
  3. Latency Percentiles (p50, p95, p99 across retrieval, LLM, and end-to-end).
  4. Medical Safety & Harm Interceptions.
  5. System Resource Utilization (RSS Memory, CPU, and active threads).

---

### 8. Health Checks

- **Liveness Endpoint**: `GET /health/live`
  - Returns HTTP 200 `{"status": "alive"}` if the HTTP server event loop is responsive.
  - Used by Kubernetes / Docker daemon for container restart decisions.

---

### 9. Readiness Checks

- **Readiness Endpoint**: `GET /health/ready`
  - Validates critical dependencies before accepting traffic:
    - Vector Store: Verifies FAISS index is loaded (744 vectors, dimension 384).
    - Embedding Model: Verifies encoder is loaded and functional.
    - Database / Local Store: Verifies connection.
    - Redis (Optional): Reports status (`degraded` if offline, `healthy` if connected).
  - Returns HTTP 200 when ready to serve traffic; HTTP 503 if vector store or embedding model is uninitialized.

---

### 10. Backup Procedure

Vector store snapshots are generated using `VectorStoreSnapshotService`:

```bash
# Execute snapshot from CLI or automated cron
python -c "
from backend.services.snapshot_service import VectorStoreSnapshotService
svc = VectorStoreSnapshotService()
manifest = svc.create_snapshot(reason='Daily automated production backup')
print('Snapshot created:', manifest['snapshot_id'])
"
```

- Verifies SHA-256 checksums of `index.faiss` and `metadata.json`.
- Generates signed `manifest.json` containing vector count, metadata count, dimension, and checksums.
- Snapshots are written to `backups/vector_store/<timestamp>_snapshot/` and can be synced to offsite S3/GCS buckets.

---

### 11. Restore Procedure

To restore from a verified snapshot:

```bash
# Restore snapshot into target directory
python -c "
from backend.services.snapshot_service import VectorStoreSnapshotService
from pathlib import Path
svc = VectorStoreSnapshotService()
manifest = svc.restore_snapshot(
    snapshot_id='<snapshot_id>',
    destination_dir=Path('data/vector_store')
)
print('Restored successfully. Invariant verify: vectors=', manifest['vector_count'])
"
```

- Validates SHA-256 integrity of snapshot files before restoring.
- Refuses restore if manifest checksums or record counts do not match.
- Atomically replaces target vector store files.

---

### 12. Disaster Recovery Procedure

| Scenario | System Impact | Automated Recovery / Operational Action |
|---|---|---|
| **Redis Node Outage** | Cache & rate limiting lose centralized state | Application automatically falls back to in-memory L1 LRU cache and local process sliding-window rate limiting. Zero downtime. Restart Redis: `docker compose restart redis`. |
| **Redis Data Loss** | Cache wiped on reboot | On next queries, cache transparently repopulates from authoritative vector store + Gemini calls. Zero data corruption. |
| **Upstream Gemini Outage** | LLM generation unavailable | Pipeline returns safe fallback response citing lack of available AI generation. Safety and grounding guards prevent fabrication. |
| **Vector Store Corruption** | Similarity retrieval fails | Health check returns 503. Trigger automated restore from latest snapshot via `VectorStoreSnapshotService.restore_snapshot()`. |
| **Application Crash** | Process termination | Docker/K8s restarts pod with `restart: unless-stopped`. Persistent vector store mounts cleanly. |

---

### 13. Scaling Procedure

1. **Horizontal Pod Autoscaling**:
   - Scale API pods based on CPU utilization (> 70%) or p95 latency (> 500ms).
   - In Kubernetes: configure HPA with minimum 3 replicas, maximum 15 replicas.
2. **Read-Only Vector Store Sharing**:
   - In multi-instance deployments, mount the verified FAISS store volume as `ReadOnlyMany` (ROX).
   - All worker pods query the identical immutable 744-vector index.

---

### 14. Rate Limits

- **RAG Query Tier**: 30 requests per minute per authenticated user or client IP.
- **Document Ingestion Tier**: 10 uploads per hour per user.
- **Authentication Tier**: 5 failed login attempts per 15 minutes before temporary IP block.
- **Responses**: Standard HTTP 429 Too Many Requests with `Retry-After: <seconds>` header.

---

### 15. LLM Concurrency

- Managed by `LLMConcurrencyController` using bounded semaphores (default: 8 concurrent upstream calls per pod).
- Protects downstream Gemini quotas and prevents API thread starvation under load spikes.
- Excess requests queue up to bounded buffer capacity, returning clean 503 or 429 when buffer is saturated.

---

### 16. Cache Behavior

- **Two-Tier Architecture**:
  - **L1**: In-memory LRU cache (`OrderedDict`) with bounded capacity (1000 items).
  - **L2**: Distributed Redis cache with TTL (1 hour).
- **Isolation Invariant**: Cache keys are generated with SHA-256 hashing across `(user_scope, normalized_query, document_signature, model, temperature, prompt_version)`.
- **Validation Invariant**: Unsafe, ungrounded, or error responses are NEVER cached (`validated_only=True`).

---

### 17. Security Model

1. **Authentication**: Supabase JWT tokens via `Authorization: Bearer <token>`.
2. **Authorization**: PostgreSQL Row-Level Security (RLS) ensures users can only read/write their own patient documents.
3. **Input Sanitization**:
   - Query text bounded to 2000 characters. Extra JSON fields strictly rejected (`extra="forbid"`).
   - Uploaded filenames sanitized with `sanitize_filename()` (path traversal and null bytes eliminated).
   - Document upload size bounded to 20MB.

---

### 18. Medical Safety Model

Execution Order Invariant:
```
1. Authentication & Rate Limiting
2. MedicalSafetyGuard (Emergency & Harm Pre-Screen)  ◄── MUST EXECUTE BEFORE CACHE & LLM
3. FAISS Semantic Retrieval
4. Sufficiency & Cosine Relevance Gate (≥ 0.25)
5. LLM Response Cache Lookup
6. LLM Concurrency Semaphore
7. Gemini Generation (Deterministic, temp=0.0)
8. Citation Validation (Ground truth metadata check)
9. Hallucination & Contradiction Guard
10. Medical Safety Post-Screen
11. Response Cache Storage (Only if all validations passed)
```

---

### 19. Monitoring

- Prometheus URL: `http://localhost:9090`
- Grafana URL: `http://localhost:3000`
- Alerting Rules:
  - High Error Rate: `rate(rag_requests_failed[5m]) / rate(rag_requests_total[5m]) > 0.05`
  - High Latency: `histogram_quantile(0.95, sum(rate(rag_total_latency_seconds_bucket[5m])) by (le)) > 2.0`
  - Vector Store Invariant Breach: `rag_vector_count != 744`

---

### 20. Troubleshooting

- **Check API logs**: `docker compose logs -f api`
- **Check Redis connectivity**: `docker compose exec redis redis-cli ping`
- **Verify vector store alignment**:
  `python -c "from backend.services.vector_store_service import VectorStoreService; vs = VectorStoreService(); vs.load(); print(vs.count())"`
- **Test health endpoint**: `curl -i http://localhost:8000/health/ready`

---

### 21. Known Limitations

1. **Single-GPU / CPU Embeddings**: SentenceTransformer embeddings run on CPU in default Docker image. High concurrency (> 100 concurrent un-cached queries) will encounter CPU saturation.
2. **Fixed Baseline Index**: The baseline production corpus contains exactly 744 verified clinical chunks. Dynamic ingestion requires explicit snapshotting.
3. **Supabase Dependency**: Cloud Supabase Auth requires outbound HTTPS access to Supabase API.

---

### 22. Rollback Procedure

In the event of an unsuccessful deployment:
1. Revert container image tag to previous stable release: `docker compose down && docker tag ai-healthcare-agent:v3.5 ai-healthcare-agent:latest && docker compose up -d`.
2. If vector index was updated, restore previous snapshot using `VectorStoreSnapshotService.restore_snapshot("<previous_snapshot_id>")`.
3. Flush transient L2 Redis cache keys if prompt template changed: `redis-cli KEYS "rag:cache:*" | xargs redis-cli DEL`.
