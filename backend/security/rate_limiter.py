import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from fastapi import Request, HTTPException, status

logger = logging.getLogger("security.rate_limiter")


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter with RFC-compliant headers.
    
    Tracks timestamps of requests per client key (IP address or authenticated user identifier)
    within rolling time windows. Supports endpoint-specific tiered quotas and emits:
      - Retry-After
      - X-RateLimit-Limit
      - X-RateLimit-Remaining
      - X-RateLimit-Reset
    """

    TIER_LIMITS: Dict[str, int] = {
        "default": 60,
        "auth_login": 10,
        "rag_query": 30,
        "rag_retrieve": 60,
        "agent_query": 30,
        "doc_upload": 15,
    }

    def __init__(
        self,
        requests_per_minute: int = 60,
        window_seconds: int = 60,
        enabled: bool = True
    ):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def get_client_key(self, request: Request, endpoint_type: str = "default") -> str:
        """
        Derives an isolated client identifier combining endpoint tier and client identity
        (Authorization bearer hash, X-Forwarded-For IP, or client host).
        """
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
        window_seconds: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Evaluates the rate limit for the incoming request.
        Returns a dict of standard rate limit headers for successful requests.
        Raises HTTPException(429) with Retry-After and X-RateLimit-* headers if quota exceeded.
        """
        if not self.enabled:
            return {}

        if max_requests is not None:
            effective_limit = max_requests
        elif endpoint_type != "default" and endpoint_type in self.TIER_LIMITS:
            effective_limit = min(self.requests_per_minute, self.TIER_LIMITS[endpoint_type])
        else:
            effective_limit = self.requests_per_minute

        effective_window = window_seconds or self.window_seconds

        client_key = self.get_client_key(request, endpoint_type)
        now = time.time()
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
        """Clears all tracking history. Primarily used in unit tests."""
        with self._lock:
            self._history.clear()


# Default singleton instance: 60 requests per minute
rate_limiter = SlidingWindowRateLimiter(requests_per_minute=60, window_seconds=60, enabled=True)
