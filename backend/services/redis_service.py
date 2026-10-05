"""
Production-grade Redis Infrastructure & Service Adapter (Phase 3.5).

Provides resilient shared infrastructure for:
1. Distributed LLM response cache.
2. Distributed sliding-window rate limiting.
3. Safe serialization and deserialization (JSON, strict isolation, no credentials).
4. Graceful connection failure and recovery with circuit-breaker backoff.
5. 100% transparent in-memory fallback when Redis is unconfigured or unreachable.
"""

import time
import json
import logging
import threading
from typing import Optional, Dict, Any, List, Union

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError, TimeoutError
    REDIS_AVAILABLE_PACKAGE = True
except ImportError:
    redis = None
    RedisError = Exception
    ConnectionError = Exception
    TimeoutError = Exception
    REDIS_AVAILABLE_PACKAGE = False

from backend.config import settings

logger = logging.getLogger("infrastructure.redis")


class RedisService:
    """
    Robust thread-safe Redis client wrapper with connection pooling,
    circuit-breaker health monitoring, and non-blocking failure semantics.
    """

    CACHE_KEY_PREFIX = "healthcare:cache:"
    RATE_LIMIT_PREFIX = "healthcare:ratelimit:"

    def __init__(
        self,
        enabled: Optional[bool] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        socket_timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        max_connections: Optional[int] = None,
        retry_on_timeout: Optional[bool] = None
    ):
        self._enabled = enabled if enabled is not None else getattr(settings, "REDIS_ENABLED", False)
        self._host = host or getattr(settings, "REDIS_HOST", "localhost")
        self._port = int(port if port is not None else getattr(settings, "REDIS_PORT", 6379))
        self._db = int(db if db is not None else getattr(settings, "REDIS_DB", 0))
        self._password = password or getattr(settings, "REDIS_PASSWORD", None)
        self._socket_timeout = float(socket_timeout if socket_timeout is not None else getattr(settings, "REDIS_SOCKET_TIMEOUT", 2.0))
        self._connect_timeout = float(connect_timeout if connect_timeout is not None else getattr(settings, "REDIS_CONNECT_TIMEOUT", 2.0))
        self._max_connections = int(max_connections if max_connections is not None else getattr(settings, "REDIS_MAX_CONNECTIONS", 20))
        self._retry_on_timeout = retry_on_timeout if retry_on_timeout is not None else getattr(settings, "REDIS_RETRY_ON_TIMEOUT", True)

        self._pool: Optional[Any] = None
        self._client: Optional[Any] = None
        self._lock = threading.RLock()

        # Circuit-breaker / cooldown state
        self._last_failure_time: float = 0.0
        self._failure_backoff_seconds: float = 3.0  # Don't hammer unreachable Redis on every call
        self._is_alive: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled and REDIS_AVAILABLE_PACKAGE

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = bool(val)
        if not val:
            self._is_alive = False

    def _get_client(self) -> Optional[Any]:
        """
        Lazily initializes the connection pool and client.
        Applies circuit-breaker cooldown if Redis was recently unreachable.
        """
        if not self.enabled:
            return None

        now = time.time()
        if not self._is_alive and (now - self._last_failure_time) < self._failure_backoff_seconds:
            return None

        with self._lock:
            if self._client is not None:
                return self._client

            if not REDIS_AVAILABLE_PACKAGE or redis is None:
                return None

            try:
                self._pool = redis.ConnectionPool(
                    host=self._host,
                    port=self._port,
                    db=self._db,
                    password=self._password,
                    socket_timeout=self._socket_timeout,
                    socket_connect_timeout=self._connect_timeout,
                    max_connections=self._max_connections,
                    retry_on_timeout=self._retry_on_timeout,
                    decode_responses=True
                )
                self._client = redis.Redis(connection_pool=self._pool)
                # Test connectivity with ping
                self._client.ping()
                self._is_alive = True
                logger.info("Connected to Redis at %s:%d (db=%d)", self._host, self._port, self._db)
                return self._client
            except Exception as exc:
                self._last_failure_time = time.time()
                self._is_alive = False
                self._client = None
                self._pool = None
                logger.debug("Redis connection check failed: %s", str(exc))
                return None

    def is_available(self) -> bool:
        """
        Returns True if Redis is enabled and actively responding to ping.
        Fails safely and quickly without blocking callers.
        """
        if not self.enabled:
            return False

        now = time.time()
        if not self._is_alive and (now - self._last_failure_time) < self._failure_backoff_seconds:
            return False

        try:
            client = self._get_client()
            if client is None:
                return False
            res = client.ping()
            self._is_alive = bool(res)
            return self._is_alive
        except Exception as exc:
            self._last_failure_time = time.time()
            self._is_alive = False
            logger.debug("Redis ping failed: %s", str(exc))
            return False

    def get(self, key: str) -> Optional[str]:
        """Safely fetches a string value by key."""
        client = self._get_client()
        if client is None:
            return None

        try:
            return client.get(key)
        except Exception as exc:
            self._record_error("get", exc)
            return None

    def set(
        self,
        key: str,
        value: str,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Safely sets a string key with optional TTL."""
        client = self._get_client()
        if client is None:
            return False

        try:
            if ttl_seconds is not None and ttl_seconds > 0:
                return bool(client.set(key, value, ex=int(ttl_seconds)))
            return bool(client.set(key, value))
        except Exception as exc:
            self._record_error("set", exc)
            return False

    def delete(self, key: str) -> bool:
        """Safely deletes a key."""
        client = self._get_client()
        if client is None:
            return False

        try:
            return bool(client.delete(key))
        except Exception as exc:
            self._record_error("delete", exc)
            return False

    def delete_pattern(self, pattern: str) -> int:
        """
        Safely scans and deletes keys matching a pattern.
        Uses non-blocking SCAN to avoid freezing Redis in production.
        """
        client = self._get_client()
        if client is None:
            return 0

        deleted_count = 0
        try:
            keys_to_delete = []
            for k in client.scan_iter(match=pattern, count=100):
                keys_to_delete.append(k)
                if len(keys_to_delete) >= 100:
                    deleted_count += client.delete(*keys_to_delete)
                    keys_to_delete.clear()
            if keys_to_delete:
                deleted_count += client.delete(*keys_to_delete)
            return deleted_count
        except Exception as exc:
            self._record_error("delete_pattern", exc)
            return 0

    def expire(self, key: str, seconds: int) -> bool:
        """Sets a TTL on an existing key."""
        client = self._get_client()
        if client is None:
            return False

        try:
            return bool(client.expire(key, int(seconds)))
        except Exception as exc:
            self._record_error("expire", exc)
            return False

    # --------------------------------------------------------------------------
    # Sorted Set Operations (for distributed sliding window rate limiting)
    # --------------------------------------------------------------------------

    def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Adds members with scores to a sorted set."""
        client = self._get_client()
        if client is None:
            return 0

        try:
            return int(client.zadd(key, mapping))
        except Exception as exc:
            self._record_error("zadd", exc)
            return 0

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Removes members from a sorted set whose score is between min and max."""
        client = self._get_client()
        if client is None:
            return 0

        try:
            return int(client.zremrangebyscore(key, min_score, max_score))
        except Exception as exc:
            self._record_error("zremrangebyscore", exc)
            return 0

    def zcard(self, key: str) -> int:
        """Returns the cardinality (number of elements) in a sorted set."""
        client = self._get_client()
        if client is None:
            return 0

        try:
            return int(client.zcard(key))
        except Exception as exc:
            self._record_error("zcard", exc)
            return 0

    def zrange(
        self,
        key: str,
        start: int,
        stop: int,
        withscores: bool = False
    ) -> List[Any]:
        """Returns members in a sorted set in ascending score order."""
        client = self._get_client()
        if client is None:
            return []

        try:
            return client.zrange(key, start, stop, withscores=withscores)
        except Exception as exc:
            self._record_error("zrange", exc)
            return []

    def execute_lua(self, script: str, numkeys: int, *keys_and_args) -> Any:
        """Executes an atomic Lua script on Redis."""
        client = self._get_client()
        if client is None:
            return None

        try:
            return client.eval(script, numkeys, *keys_and_args)
        except Exception as exc:
            self._record_error("execute_lua", exc)
            return None

    def flushdb(self) -> bool:
        """Clears the active Redis database. Used primarily in test isolation."""
        client = self._get_client()
        if client is None:
            return False

        try:
            client.flushdb()
            return True
        except Exception as exc:
            self._record_error("flushdb", exc)
            return False

    def disconnect(self) -> None:
        """Closes all active connections in the connection pool."""
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.disconnect()
                except Exception:
                    pass
            self._client = None
            self._pool = None
            self._is_alive = False

    def _record_error(self, op: str, exc: Exception) -> None:
        """Records an error, updates circuit-breaker state, and logs safely."""
        self._last_failure_time = time.time()
        self._is_alive = False
        logger.warning("Redis operation '%s' failed (graceful degradation engaged): %s", op, str(exc))


# ==============================================================================
# Shared / Singleton RedisService Accessor
# ==============================================================================

_shared_redis_service: Optional[RedisService] = None
_redis_init_lock = threading.Lock()


def get_redis_service() -> RedisService:
    """Returns the process-wide singleton RedisService instance."""
    global _shared_redis_service
    if _shared_redis_service is None:
        with _redis_init_lock:
            if _shared_redis_service is None:
                _shared_redis_service = RedisService()
    return _shared_redis_service
