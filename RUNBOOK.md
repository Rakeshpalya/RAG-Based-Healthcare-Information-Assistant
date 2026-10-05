# Operational Runbook
## AI Healthcare Agent — Incident Response & Triage

This runbook provides actionable step-by-step procedures for on-call engineers managing the AI Healthcare Agent production environment.

---

### Incident Severity Levels

- **SEV-1 (Critical)**: Complete outage of API, inability to answer medical queries, or vector store corruption.
- **SEV-2 (Major)**: Upstream Gemini degradation, Redis outage with fallback active, elevated p95 latency (> 2s).
- **SEV-3 (Minor)**: Elevated rate-limiting, transient cache misses, non-critical metrics scrape timeout.

---

### 1. Scenario: API Down (HTTP 500 / 502 / Connection Refused)

**Symptoms**:
- `curl http://localhost:8000/health/live` returns connection refused or 502 Bad Gateway.
- Uptime monitoring alerts on endpoint failure.

**Diagnostic Steps**:
```bash
# 1. Check container status
docker compose ps

# 2. Check recent fatal logs
docker compose logs --tail=100 api

# 3. Check if out-of-memory killed (OOMKilled)
docker inspect $(docker compose ps -q api) --format '{{.State.OOMKilled}}'
```

**Remediation Steps**:
1. If container exited due to OOM: Increase container memory limit in `docker-compose.yml` to 2GB or 4GB, then restart:
   ```bash
   docker compose up -d api
   ```
2. If container crashed during startup due to missing environment variable or vector store file:
   - Check `.env` contains all required variables.
   - Verify `data/vector_store/index.faiss` exists and is readable.
3. Once running, verify liveness:
   ```bash
   curl -i http://localhost:8000/health/live
   ```

---

### 2. Scenario: Redis Down

**Symptoms**:
- Prometheus reports `redis_connected == 0`.
- Application logs show: `Redis cache read error (falling back to memory)`.

**Diagnostic Steps**:
```bash
# Test direct Redis ping
docker compose exec redis redis-cli ping
```

**System Behavior**:
- **Automatic Fallback Active**: The application continues to serve traffic without disruption using local in-memory L1 LRU caching and in-process sliding-window rate limiting.

**Remediation Steps**:
1. Restart the Redis container:
   ```bash
   docker compose restart redis
   ```
2. Check Redis logs for persistence or memory errors:
   ```bash
   docker compose logs --tail=50 redis
   ```
3. Once Redis responds with `PONG`, the application will automatically resume L2 distributed caching on subsequent requests.

---

### 3. Scenario: Upstream Gemini Unavailable / 503 / 429

**Symptoms**:
- Application logs report: `The AI generation service is temporarily unavailable due to high demand`.
- Prometheus metric `rag_llm_failures_total` increasing.

**Diagnostic Steps**:
```bash
# Check Google Cloud Status Dashboard for Vertex AI / GenAI outages
# Inspect sanitized API error responses in API logs
docker compose logs api | grep -i "Gemini API"
```

**System Behavior**:
- The service automatically triggers bounded exponential backoff and iterates across fallback models (`gemini-flash-lite-latest`, `gemini-2.5-flash`, `gemini-flash-latest`).
- If all models fail, returns a safe grounded limitation without hallucinating medical facts.

**Remediation Steps**:
1. Verify API key quota in Google Cloud Console.
2. If quota is exhausted, rotate `GEMINI_API_KEY` in secrets manager and restart API:
   ```bash
   docker compose restart api
   ```

---

### 4. Scenario: High Request Latency (p95 > 2000ms)

**Symptoms**:
- Latency alert fires in Grafana (`rag_total_latency_seconds`).

**Diagnostic Steps**:
```bash
# Check latency breakdown in Grafana:
# 1. Retrieval Latency (SentenceTransformers + FAISS)
# 2. Upstream Gemini Latency (API network latency)
```

**Remediation Steps**:
1. **If Retrieval Latency is High**:
   - Check CPU utilization on the API host: `docker stats`.
   - Scale API containers horizontally: `docker compose up -d --scale api=3`.
2. **If Upstream LLM Latency is High**:
   - Upstream Google API network delay. Ensure `GEMINI_TIMEOUT_SECONDS` is set to `30.0` to terminate hung connections.
   - Cache hit rate might be low. Verify Redis is operational.

---

### 5. Scenario: High Process Memory (RSS > 1.5 GB)

**Symptoms**:
- Container memory usage approaches host memory limits.

**Diagnostic Steps**:
```bash
# Run resource profile check
python scripts/profile_phase3_6.py

# Check process memory inside container
docker compose exec api ps aux
```

**Remediation Steps**:
1. Check cache entries: Verify L1 cache is bounded to `LLM_CACHE_MAX_ENTRIES` (default: 1000).
2. Trigger cache purge if stale memory needs immediate reclamation:
   ```bash
   curl -X POST -H "Authorization: Bearer <ADMIN_TOKEN>" http://localhost:8000/cache/clear
   ```
3. Restart container to release memory: `docker compose restart api`.

---

### 6. Scenario: High CPU Utilization (> 90%)

**Symptoms**:
- CPU alerts triggered, request queues backing up.

**Diagnostic Steps**:
```bash
docker stats --no-stream
```

**Remediation Steps**:
1. Heavy embedding computation during concurrent queries without cache hits.
2. Scale API worker processes or increase container CPU allotment.
3. Check for abusive clients or bot traffic in access logs; adjust rate limiter thresholds if needed.

---

### 7. Scenario: Response Cache Failure / Invalidation Loop

**Symptoms**:
- Cache hits remain at 0% despite repeated identical queries.

**Diagnostic Steps**:
```bash
# Check if cache is globally disabled
grep "LLM_CACHE_ENABLED" .env

# Verify Redis key storage
docker compose exec redis redis-cli KEYS "rag:cache:*"
```

**Remediation Steps**:
1. If Redis keys are corrupted: Flush Redis cache:
   ```bash
   docker compose exec redis redis-cli FLUSHDB
   ```
2. Verify that queries pass medical safety pre-screen. (Remember: unvalidated or error responses are never cached by design).

---

### 8. Scenario: Vector Store Corruption or Count Discrepancy

**Symptoms**:
- Health readiness probe returns 503 Service Unavailable: `{"status": "unhealthy", "vector_store": "count mismatch"}`.
- Vector count is not 744.

**Diagnostic Steps**:
```bash
# Check current FAISS vector count
python -c "
from backend.services.vector_store_service import VectorStoreService
vs = VectorStoreService()
vs.load()
print('Vector count:', vs.count(), 'Metadata count:', len(vs.metadata_store))
"
```

**Remediation Steps**:
1. **Restore from Latest Verified Snapshot**:
   ```bash
   python -c "
   from backend.services.snapshot_service import VectorStoreSnapshotService
   svc = VectorStoreSnapshotService()
   snapshots = svc.list_snapshots()
   latest = snapshots[0]['snapshot_id']
   print('Restoring from latest snapshot:', latest)
   svc.restore_snapshot(latest)
   "
   ```
2. Verify restored count equals exactly 744:
   ```bash
   python -c "from backend.services.vector_store_service import VectorStoreService; vs = VectorStoreService(); vs.load(); assert vs.count() == 744"
   ```
3. Restart API container to reload index into memory: `docker compose restart api`.

---

### 9. Scenario: Failed Deployment

**Symptoms**:
- Newly deployed container fails healthcheck on startup.
- Container status shows `unhealthy`.

**Diagnostic Steps**:
```bash
# Inspect container healthcheck failure reason
docker inspect $(docker compose ps -q api) --format '{{json .State.Health}}'
```

**Remediation Steps**:
1. Check if migrations or dependency installations failed.
2. Execute immediate rollback (see Scenario 10).

---

### 10. Scenario: Emergency Rollback Procedure

**Execution Steps**:
```bash
# 1. Stop current containers
docker compose down

# 2. Re-tag previous known-stable Docker image
docker tag ai-healthcare-agent:v3.5 ai-healthcare-agent:latest

# 3. Restore previous vector store snapshot if modified
python -c "
from backend.services.snapshot_service import VectorStoreSnapshotService
svc = VectorStoreSnapshotService()
svc.restore_snapshot('<previous_snapshot_id>')
"

# 4. Launch previous stable release
docker compose up -d

# 5. Verify readiness
curl -f http://localhost:8000/health/ready
```
