"""
Distributed & In-Memory Rate Limiting Engine for AI Healthcare Agent (Phase 3.4 & Phase 3.5).

Provides:
1. Atomic, distributed sliding-window rate limiting backed by Redis sorted sets (ZSET) and Lua scripts.
2. Zero race conditions across multiple FastAPI instances.
3. Automatic TTL expiration to prevent memory growth on idle client keys.
4. Fail-safe transparent fallback to local thread-safe in-memory sliding window if Redis fails or is disabled.
5. Strict protection against unlimited requests: local rate limiting remains strictly enforced during Redis outages.
6. Tiered limits:
   - General RAG: 30 requests/minute/user
   - Streaming SSE: 10 requests/minute/user
   - Auth login: 10 requests/minute
   - Document upload: 15 requests/minute
   - Default: 60 requests/minute
7. RFC-compliant headers: Retry-After, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset.
"""

import time
import uuid
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from fastapi import Request, HTTPException, status

logger = logging.getLogger("security.rate_limiter")

# Atomic sliding-window Lua script for Redis
LUA_SLIDING_WINDOW_RATE_LIMIT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
local clear_before = now - window

-- 1. Evict timestamps outside rolling window
redis.call('ZREMRANGEBYSCORE', key, '-inf', clear_before)

-- 2. Count current elements in window
local current_count = redis.call('ZCARD', key)

if current_count >= limit then
    -- Find oldest entry in current window for Retry-After
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local oldest_ts = tonumber(oldest[2]) or (now - window)
    local retry_after = math.max(1, math.ceil(oldest_ts + window - now))
    local reset_ts = math.ceil(oldest_ts + window)
    return {0, current_count, retry_after, reset_ts}
else
    -- Add this request timestamp
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, math.ceil(window * 2))
    local remaining = math.max(0, limit - current_count - 1)
    local reset_ts = math.ceil(now + window)
    return {1, remaining, 0, reset_ts}
end
"""


class SlidingWindowRateLimiter:
    """
    Hybrid distributed (Redis) and thread-safe local (in-memory) sliding-window
    rate limiter with RFC-compliant headers.
    """

    TIER_LIMITS: Dict[str, int] = {
        "default": 60,
        "auth_login": 10,
        "rag_query": 30,
        "rag_retrieve": 60,
        "rag_stream": 10,
        "agent_query": 30,
        "doc_upload": 15,
    }

    def __init__(
        self,
        requests_per_minute: int = 60,
        window_seconds: int = 60,
        enabled: bool = True,
        redis_service: Optional[Any] = None
    ):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._redis = redis_service
        self._history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _get_redis(self) -> Optional[Any]:
        """Returns the Redis service adapter if configured and available."""
        if self._redis is not None:
            return self._redis
        try:
            from backend.services.redis_service import get_redis_service
            return get_redis_service()
        except Exception:
            return None

    def get_client_key(
        self,
        request: Request,
        endpoint_type: str = "default",
        user_id: Optional[Any] = None
    ) -> str:
        """
        Derives an isolated client identifier combining endpoint tier and client identity
        (explicit user_id, Authorization bearer hash, X-Forwarded-For IP, or client host).
        """
        if user_id is not None:
            return f"{endpoint_type}:user_{user_id}"

        client_id = "unknown_client"

        auth = request.headers.get("Authorization")
        if auth and len(auth) > 15:
            client_id = f"user_{abs(hash(auth)) % 10000000}"
        else:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_id = forwarded.split(",")[0].strip()
            elif request.client and request.client.host:
                client_id = request.client.host

        return f"{endpoint_type}:{client_id}"

    def check_rate_limit(
        self,
        request: Request,
        endpoint_type: str = "default",
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
        user_id: Optional[Any] = None
    ) -> Dict[str, str]:
        """
        Evaluates the rate limit for the incoming request using Redis or in-memory fallback.
        Returns a dict of standard rate limit headers for successful requests.
        Raises HTTPException(429) with Retry-After and X-RateLimit-* headers if quota exceeded.
        """
        if not self.enabled:
            return {}

        from backend.config import settings

        if max_requests is not None:
            effective_limit = max_requests
        elif endpoint_type == "rag_stream":
            effective_limit = min(self.requests_per_minute, getattr(settings, "STREAMING_REQUESTS_PER_MINUTE", 10))
        elif endpoint_type in ("rag_query", "agent_query"):
            effective_limit = min(self.requests_per_minute, getattr(settings, "GENERAL_REQUESTS_PER_MINUTE", 30))
        elif endpoint_type != "default" and endpoint_type in self.TIER_LIMITS:
            effective_limit = min(self.requests_per_minute, self.TIER_LIMITS[endpoint_type])
        else:
            effective_limit = min(self.requests_per_minute, getattr(settings, "RATE_LIMIT_PER_MINUTE", self.requests_per_minute))

        effective_window = window_seconds or self.window_seconds
        client_key = self.get_client_key(request, endpoint_type, user_id=user_id)
        now = time.time()

        # 1. Attempt Redis Distributed Rate Limiting
        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                redis_key = f"{redis_svc.RATE_LIMIT_PREFIX}{client_key}"
                member = f"{now:.6f}:{uuid.uuid4().hex[:6]}"
                res = redis_svc.execute_lua(
                    LUA_SLIDING_WINDOW_RATE_LIMIT,
                    1,
                    redis_key,
                    str(now),
                    str(effective_window),
                    str(effective_limit),
                    member
                )
                if res and isinstance(res, (list, tuple)) and len(res) >= 4:
                    allowed = int(res[0])
                    remaining_or_curr = int(res[1])
                    retry_after = int(res[2])
                    reset_timestamp = int(res[3])

                    if allowed == 1:
                        return {
                            "X-RateLimit-Limit": str(effective_limit),
                            "X-RateLimit-Remaining": str(remaining_or_curr),
                            "X-RateLimit-Reset": str(reset_timestamp),
                        }
                    else:
                        headers = {
                            "Retry-After": str(retry_after),
                            "X-RateLimit-Limit": str(effective_limit),
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(reset_timestamp),
                        }
                        logger.warning(
                            "Distributed rate limit exceeded for client '%s' on tier '%s'. Retry-After: %ds",
                            client_key, endpoint_type, retry_after
                        )
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded. Too many requests. Please wait before retrying.",
                            headers=headers
                        )
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning(
                    "Redis rate limiting failed (falling back safely to local in-memory quota): %s",
                    str(exc)
                )

        # 2. Local Thread-Safe In-Memory Fallback
        cutoff = now - effective_window
        with self._lock:
            timestamps = self._history.get(client_key, [])
            timestamps = [ts for ts in timestamps if ts > cutoff]

            if len(timestamps) >= effective_limit:
                oldest_in_window = timestamps[0]
                retry_after = max(1, int(oldest_in_window + effective_window - now))
                reset_timestamp = int(oldest_in_window + effective_window)

                headers = {
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(effective_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_timestamp),
                }

                logger.warning(
                    "Rate limit exceeded for client '%s' on tier '%s'. %d/%d reqs. Retry-After: %ds",
                    client_key, endpoint_type, len(timestamps), effective_limit, retry_after
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Too many requests. Please wait before retrying.",
                    headers=headers
                )

            timestamps.append(now)
            self._history[client_key] = timestamps
            remaining = max(0, effective_limit - len(timestamps))
            reset_timestamp = int(now + effective_window)

            return {
                "X-RateLimit-Limit": str(effective_limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_timestamp),
            }

    def reset(self) -> None:
        """Clears all tracking history locally and in Redis. Primarily used in unit tests."""
        with self._lock:
            self._history.clear()

        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                redis_svc.delete_pattern(f"{redis_svc.RATE_LIMIT_PREFIX}*")
            except Exception:
                pass


# Default singleton instance: 60 requests per minute
rate_limiter = SlidingWindowRateLimiter(requests_per_minute=60, window_seconds=60, enabled=True)
