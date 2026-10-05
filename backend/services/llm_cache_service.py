"""
Production-safe LLM Response Cache Service (Phase 3.3 & Phase 3.5).

Provides layered caching architecture:
1. L1 Fast Thread-Safe In-Memory LRU Cache with bounded memory and TTL.
2. L2 Distributed Redis Cache for multi-instance scaling and shared state.
3. Transparent, zero-downtime graceful fallback to in-memory cache if Redis is down/disabled.
4. Strict per-user and document isolation: user_id and document signatures prevent cross-user leakage.
5. Invariant: Only validated, grounded responses passing medical safety checks are cached.
6. Invariant: Medical safety guard always runs BEFORE cache lookup (pre-screen) and AFTER generation (post-screen).
7. Cache invalidation on document updates, model parameter changes, or prompt version bumps.
"""

import time
import copy
import json
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional, Union, List

from backend.config import settings

logger = logging.getLogger("services.llm_cache")

# Canonical prompt version for cache key generation. Bump on prompt changes.
PROMPT_VERSION = "v3.3.0"


class LLMCacheEntry:
    """Represents a cached validated healthcare response item."""
    __slots__ = (
        "key",
        "value",
        "user_scope",
        "document_signature",
        "created_at",
        "expires_at",
        "access_count",
        "last_accessed",
        "validated"
    )

    def __init__(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: int,
        user_scope: str = "anon",
        document_signature: str = "",
        validated: bool = True
    ):
        now = time.time()
        self.key = key
        self.value = value
        self.user_scope = str(user_scope)
        self.document_signature = str(document_signature)
        self.created_at = now
        self.expires_at = now + float(ttl_seconds)
        self.access_count = 0
        self.last_accessed = now
        self.validated = validated

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def age_ms(self) -> float:
        return max(0.0, (time.time() - self.created_at) * 1000.0)


class LLMCacheService:
    """
    Thread-safe, layered cache for validated clinical RAG responses.
    L1: Local LRU memory cache
    L2: Distributed Redis cache with graceful degradation
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        ttl_seconds: Optional[int] = None,
        max_entries: Optional[int] = None,
        prompt_version: str = PROMPT_VERSION,
        redis_service: Optional[Any] = None
    ):
        self._enabled = enabled if enabled is not None else settings.LLM_CACHE_ENABLED
        self._ttl_seconds = int(ttl_seconds if ttl_seconds is not None else settings.LLM_CACHE_TTL_SECONDS)
        self._max_entries = int(max_entries if max_entries is not None else settings.LLM_CACHE_MAX_ENTRIES)
        self._prompt_version = prompt_version
        self._redis = redis_service

        self._cache: OrderedDict[str, LLMCacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Telemetry metrics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._expirations: int = 0
        self._redis_hits: int = 0
        self._redis_errors: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = bool(val)

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def _get_redis(self) -> Optional[Any]:
        """Returns the Redis service adapter if available, or None."""
        if self._redis is not None:
            return self._redis
        try:
            from backend.services.redis_service import get_redis_service
            return get_redis_service()
        except Exception:
            return None

    @staticmethod
    def generate_cache_key(
        normalized_query: str = "",
        user_scope: Optional[Union[int, str]] = "anon",
        document_signature: str = "",
        model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        prompt_version: str = PROMPT_VERSION,
        query_plan_strategy: Optional[str] = None,
        query: Optional[str] = None,
        synthesis_strategy: Optional[str] = None
    ) -> str:
        """
        Generates a deterministic SHA-256 cache key preventing cross-user or cross-document collisions.

        Key components:
        - user_scope: User ID or 'anon'
        - normalized_query: Cleaned, lower-cased semantic query text
        - document_signature: Hash of relevant document IDs / chunk hashes
        - model: LLM model name
        - temperature: Sampling temperature
        - max_output_tokens: Token ceiling
        - prompt_version: Prompt template identifier
        - query_plan_strategy: Optional retrieval/query-plan strategy (Phase 6.2)
        - synthesis_strategy: Optional clinical answer synthesis strategy (Phase 6.4)
        """
        target_q = query if query is not None else normalized_query
        user_str = f"user:{user_scope}" if user_scope is not None else "user:anon"
        norm_q = str(target_q).strip().lower()
        doc_sig = str(document_signature).strip()
        mod_str = str(model).strip()
        temp_str = f"{float(temperature):.2f}"
        tok_str = str(int(max_output_tokens))
        pv_str = str(prompt_version).strip()

        raw_key = (
            f"scope={user_str}|q={norm_q}|doc={doc_sig}|m={mod_str}|"
            f"t={temp_str}|max={tok_str}|pv={pv_str}"
        )
        if query_plan_strategy:
            raw_key += f"|plan={str(query_plan_strategy).strip()}"
        if synthesis_strategy:
            raw_key += f"|synth={str(synthesis_strategy).strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _set_local(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: int,
        user_scope: Optional[Union[int, str]] = "anon",
        document_signature: str = ""
    ) -> None:
        """Internal helper to insert or update an item in L1 local memory cache."""
        entry = LLMCacheEntry(
            key=key,
            value=copy.deepcopy(value),
            ttl_seconds=ttl_seconds,
            user_scope=str(user_scope) if user_scope is not None else "anon",
            document_signature=document_signature,
            validated=True
        )
        with self._lock:
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
                return

            if len(self._cache) >= self._max_entries:
                oldest_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug("Evicted L1 cache entry: %s", oldest_key)

            self._cache[key] = entry

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a validated cached response if present and unexpired.
        1. Checks L1 in-memory cache.
        2. If miss or expired, checks L2 distributed Redis cache (if active).
        3. Returns a deep copy of the cached payload to prevent caller mutation.
        """
        if not self._enabled:
            with self._lock:
                self._misses += 1
            return None

        # 1. Check L1 in-memory cache
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                if entry.is_expired:
                    del self._cache[key]
                    self._expirations += 1
                else:
                    self._cache.move_to_end(key)
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    self._hits += 1

                    val = copy.deepcopy(entry.value)
                    if isinstance(val, dict):
                        timings = val.setdefault("timings", {})
                        if isinstance(timings, dict):
                            timings["cache_hit"] = True
                            timings["cache_age_ms"] = round(entry.age_ms, 2)
                            timings["cache_key_version"] = self._prompt_version
                            timings["llm_called"] = False
                            timings["gemini_calls_count"] = 0
                            timings["distributed_cache"] = False
                    return val

        # 2. Check L2 distributed Redis cache
        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                redis_key = f"{redis_svc.CACHE_KEY_PREFIX}{key}"
                raw_json = redis_svc.get(redis_key)
                if raw_json:
                    data = json.loads(raw_json)
                    val = data.get("value")
                    if isinstance(val, dict):
                        created_at = float(data.get("created_at", time.time()))
                        age_ms = max(0.0, (time.time() - created_at) * 1000.0)

                        val_copy = copy.deepcopy(val)
                        timings = val_copy.setdefault("timings", {})
                        if isinstance(timings, dict):
                            timings["cache_hit"] = True
                            timings["cache_age_ms"] = round(age_ms, 2)
                            timings["cache_key_version"] = self._prompt_version
                            timings["llm_called"] = False
                            timings["gemini_calls_count"] = 0
                            timings["distributed_cache"] = True

                        with self._lock:
                            self._hits += 1
                            self._redis_hits += 1

                        # Populate local L1 cache with remaining TTL
                        remaining_ttl = max(1, int(float(data.get("expires_at", time.time() + self._ttl_seconds)) - time.time()))
                        self._set_local(
                            key=key,
                            value=val_copy,
                            ttl_seconds=remaining_ttl,
                            user_scope=data.get("user_scope", "anon"),
                            document_signature=data.get("document_signature", "")
                        )
                        return val_copy
            except Exception as exc:
                self._redis_errors += 1
                logger.warning("Redis cache read error (falling back to memory): %s", str(exc))

        with self._lock:
            self._misses += 1
        return None

    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        user_scope: Optional[Union[int, str]] = "anon",
        document_signature: str = "",
        validated_only: bool = True
    ) -> bool:
        """
        Stores a verified safe response in both L1 memory and L2 Redis distributed cache.

        Safety Invariant:
        Only validated, non-error healthcare responses passing medical safety checks are cached.
        Never cache unvalidated, empty, or error responses.
        """
        if not self._enabled:
            return False

        if not key or not value or not isinstance(value, dict):
            return False

        # Invariant: Never cache errors, empty answers, or ungrounded responses
        if validated_only:
            status = value.get("retrieval_status", "")
            if status not in ("success", "grounded_boundary"):
                logger.debug("Refusing to cache response with unvalidated status: %s", status)
                return False

            answer = value.get("answer", "")
            if not answer or len(answer.strip()) < 10:
                return False

        ttl = int(ttl_seconds if ttl_seconds is not None else self._ttl_seconds)

        # 1. Update L1 Local Memory Cache
        self._set_local(
            key=key,
            value=value,
            ttl_seconds=ttl,
            user_scope=user_scope,
            document_signature=document_signature
        )

        # 2. Update L2 Redis Distributed Cache
        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                now = time.time()
                redis_key = f"{redis_svc.CACHE_KEY_PREFIX}{key}"
                payload = {
                    "key": key,
                    "value": copy.deepcopy(value),
                    "user_scope": str(user_scope) if user_scope is not None else "anon",
                    "document_signature": str(document_signature),
                    "created_at": now,
                    "expires_at": now + ttl,
                    "validated": True
                }
                serialized = json.dumps(payload, ensure_ascii=False)
                redis_svc.set(redis_key, serialized, ttl_seconds=ttl)

                # Index key under user set for fast user invalidation
                u_scope = str(user_scope) if user_scope is not None else "anon"
                if u_scope != "anon":
                    user_tag = f"{redis_svc.CACHE_KEY_PREFIX}idx:user:{u_scope}"
                    redis_svc.zadd(user_tag, {redis_key: now + ttl})
                    redis_svc.expire(user_tag, ttl)

                # Index key under document set for fast document invalidation
                if document_signature:
                    doc_tag = f"{redis_svc.CACHE_KEY_PREFIX}idx:doc:{document_signature}"
                    redis_svc.zadd(doc_tag, {redis_key: now + ttl})
                    redis_svc.expire(doc_tag, ttl)
            except Exception as exc:
                self._redis_errors += 1
                logger.warning("Redis cache write error (L1 memory safe): %s", str(exc))

        return True

    def delete(self, key: str) -> bool:
        """Removes a specific entry from both L1 memory and L2 Redis."""
        deleted = False
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                deleted = True

        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                redis_key = f"{redis_svc.CACHE_KEY_PREFIX}{key}"
                if redis_svc.delete(redis_key):
                    deleted = True
            except Exception as exc:
                self._redis_errors += 1
                logger.warning("Redis cache delete error: %s", str(exc))

        return deleted

    def delete_by_user(self, user_scope: Union[int, str]) -> int:
        """
        Invalidates all cached entries for a given user (e.g. after uploading a document).
        Returns number of invalidated entries across L1 and L2.
        """
        target_scope = f"user:{user_scope}" if str(user_scope).isdigit() else str(user_scope)
        with self._lock:
            to_delete = [
                k for k, v in self._cache.items()
                if v.user_scope in (str(user_scope), target_scope, f"user_{user_scope}")
            ]
            for k in to_delete:
                del self._cache[k]
            deleted_count = len(to_delete)

        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                user_tag = f"{redis_svc.CACHE_KEY_PREFIX}idx:user:{user_scope}"
                keys = redis_svc.zrange(user_tag, 0, -1)
                if keys:
                    for k in keys:
                        redis_svc.delete(k)
                    deleted_count = max(deleted_count, len(keys))
                    redis_svc.delete(user_tag)
            except Exception as exc:
                self._redis_errors += 1
                logger.warning("Redis cache delete_by_user error: %s", str(exc))

        return deleted_count

    def delete_by_document(self, document_id: Union[int, str]) -> int:
        """
        Invalidates all cached entries associated with a document_id or signature.
        Returns number of invalidated entries.
        """
        doc_str = str(document_id)
        with self._lock:
            to_delete = [
                k for k, v in self._cache.items()
                if doc_str in v.document_signature or (isinstance(v.value, dict) and doc_str in str(v.value.get("sources", [])))
            ]
            for k in to_delete:
                del self._cache[k]
            deleted_count = len(to_delete)

        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                doc_tag = f"{redis_svc.CACHE_KEY_PREFIX}idx:doc:{document_id}"
                keys = redis_svc.zrange(doc_tag, 0, -1)
                if keys:
                    for k in keys:
                        redis_svc.delete(k)
                    deleted_count = max(deleted_count, len(keys))
                    redis_svc.delete(doc_tag)
            except Exception as exc:
                self._redis_errors += 1
                logger.warning("Redis cache delete_by_document error: %s", str(exc))

        return deleted_count

    def clear(self) -> None:
        """Clears all entries from both L1 memory and L2 Redis."""
        with self._lock:
            self._cache.clear()

        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                redis_svc.delete_pattern(f"{redis_svc.CACHE_KEY_PREFIX}*")
            except Exception as exc:
                self._redis_errors += 1
                logger.warning("Redis cache clear error: %s", str(exc))

    def contains(self, key: str) -> bool:
        """Checks if an unexpired key exists in cache without recording a hit/miss."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and not entry.is_expired:
                return True

        redis_svc = self._get_redis()
        if redis_svc and redis_svc.is_available():
            try:
                redis_key = f"{redis_svc.CACHE_KEY_PREFIX}{key}"
                return redis_svc.get(redis_key) is not None
            except Exception:
                pass

        return False

    def stats(self) -> Dict[str, Any]:
        """Returns structured cache observability metrics."""
        with self._lock:
            now = time.time()
            expired_keys = [k for k, v in self._cache.items() if now >= v.expires_at]
            for k in expired_keys:
                del self._cache[k]
                self._expirations += 1

            total_reqs = self._hits + self._misses
            hit_rate = round(self._hits / total_reqs, 4) if total_reqs > 0 else 0.0

            redis_svc = self._get_redis()
            redis_connected = bool(redis_svc and redis_svc.is_available())

            return {
                "enabled": self._enabled,
                "total_entries": len(self._cache),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "prompt_version": self._prompt_version,
                "redis_enabled": getattr(settings, "REDIS_ENABLED", False),
                "redis_connected": redis_connected,
                "redis_hits": self._redis_hits,
                "redis_errors": self._redis_errors
            }


# ==============================================================================
# Global / Singleton Cache Service Accessor
# ==============================================================================

_shared_llm_cache: Optional[LLMCacheService] = None
_cache_init_lock = threading.Lock()


def get_llm_cache_service() -> LLMCacheService:
    """Returns the process-wide singleton LLMCacheService instance."""
    global _shared_llm_cache
    if _shared_llm_cache is None:
        with _cache_init_lock:
            if _shared_llm_cache is None:
                _shared_llm_cache = LLMCacheService()
    return _shared_llm_cache


# Module-level convenience export
generate_cache_key = LLMCacheService.generate_cache_key
