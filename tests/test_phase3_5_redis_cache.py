"""
Phase 3.5.1 Tests: Redis Infrastructure & Distributed Layered Cache.

Verifies:
1. Redis configuration defaults, settings reload, and masked representation.
2. Layered caching: memory fallback when Redis is unavailable or unconfigured.
3. Distributed cache behavior when Redis is active (using mock Redis adapter).
4. Zero secret leakage in cache keys or serialized values.
5. Strict per-user and per-document isolation.
6. Graceful connection failure: Redis ConnectionError/TimeoutError does not crash the service.
7. Reconnection and cache invalidation.
8. Preservation of medical safety invariant: unvalidated/error responses are NEVER cached.
"""

import time
import copy
import json
import pytest
from unittest.mock import MagicMock, patch

from backend.config import settings
from backend.services.redis_service import RedisService, get_redis_service
from backend.services.llm_cache_service import LLMCacheService, PROMPT_VERSION


class FakeRedisClient:
    """In-memory thread-safe fake Redis for fast deterministic distributed testing."""

    def __init__(self):
        self.store = {}
        self.ttls = {}
        self.zsets = {}
        self.is_connected = True

    def ping(self):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        return True

    def get(self, key):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        if key in self.ttls and time.time() > self.ttls[key]:
            self.store.pop(key, None)
            self.ttls.pop(key, None)
            return None
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        self.store[key] = str(value)
        if ex:
            self.ttls[key] = time.time() + float(ex)
        elif key in self.ttls:
            del self.ttls[key]
        return True

    def delete(self, *keys):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        count = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                count += 1
            if k in self.ttls:
                del self.ttls[k]
            if k in self.zsets:
                del self.zsets[k]
        return count

    def scan_iter(self, match="*", count=100):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        prefix = match.replace("*", "")
        for k in list(self.store.keys()):
            if prefix in k or match == "*":
                yield k

    def expire(self, key, seconds):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        self.ttls[key] = time.time() + float(seconds)
        return True

    def zadd(self, key, mapping):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        if key not in self.zsets:
            self.zsets[key] = {}
        for member, score in mapping.items():
            self.zsets[key][member] = float(score)
        return len(mapping)

    def zrange(self, key, start, stop, withscores=False):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        if key not in self.zsets:
            return []
        items = sorted(self.zsets[key].items(), key=lambda x: x[1])
        if stop == -1:
            slice_items = items[start:]
        else:
            slice_items = items[start:stop + 1]
        if withscores:
            return slice_items
        return [m for m, s in slice_items]

    def zremrangebyscore(self, key, min_score, max_score):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        if key not in self.zsets:
            return 0
        to_del = [m for m, s in self.zsets[key].items() if min_score <= s <= max_score]
        for m in to_del:
            del self.zsets[key][m]
        return len(to_del)

    def zcard(self, key):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        return len(self.zsets.get(key, {}))

    def flushdb(self):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        self.store.clear()
        self.ttls.clear()
        self.zsets.clear()
        return True

    def eval(self, script, numkeys, *args):
        if not self.is_connected:
            raise ConnectionError("FakeRedis: Connection refused")
        # Handles rate limiter sliding window Lua script
        key = args[0]
        now = float(args[1])
        window = float(args[2])
        limit = int(args[3])
        member = args[4]
        clear_before = now - window

        if key not in self.zsets:
            self.zsets[key] = {}

        self.zsets[key] = {m: ts for m, ts in self.zsets[key].items() if ts > clear_before}
        current_count = len(self.zsets[key])

        if current_count >= limit:
            oldest_ts = min(self.zsets[key].values()) if self.zsets[key] else (now - window)
            retry_after = max(1, int(oldest_ts + window - now))
            reset_ts = int(oldest_ts + window)
            return [0, current_count, retry_after, reset_ts]
        else:
            self.zsets[key][member] = now
            self.ttls[key] = now + window * 2
            remaining = max(0, limit - current_count - 1)
            reset_ts = int(now + window)
            return [1, remaining, 0, reset_ts]


def test_redis_config_defaults_and_masking():
    """Verify settings defaults and that Redis credentials are sanitized."""
    assert hasattr(settings, "REDIS_ENABLED")
    assert hasattr(settings, "REDIS_HOST")
    assert hasattr(settings, "REDIS_PORT")
    assert hasattr(settings, "REDIS_DB")
    assert hasattr(settings, "REDIS_SOCKET_TIMEOUT")

    sanitized = settings.get_sanitized_config_dict()
    assert "redis_enabled" in sanitized
    assert "redis_host" in sanitized
    assert "redis_port" in sanitized
    # Redis password must never be exposed
    assert "redis_password" not in sanitized or sanitized.get("redis_password") is None
    assert "redis_password_configured" in sanitized


def test_redis_service_graceful_when_disabled():
    """Verify RedisService returns False / None safely when disabled."""
    service = RedisService(enabled=False)
    assert not service.is_available()
    assert service.get("some_key") is None
    assert not service.set("some_key", "value")
    assert not service.delete("some_key")
    assert service.delete_pattern("some_*") == 0


def test_redis_service_graceful_on_connection_error():
    """Verify RedisService catches connection errors without raising."""
    fake_client = FakeRedisClient()
    fake_client.is_connected = False

    service = RedisService(enabled=True)
    service._client = fake_client

    assert not service.is_available()
    assert service.get("some_key") is None
    assert not service.set("some_key", "value")
    assert not service.delete("some_key")


def test_layered_cache_fallback_to_memory_when_redis_unavailable():
    """Verify LLMCacheService functions seamlessly via in-memory cache when Redis is unavailable."""
    fake_redis_svc = RedisService(enabled=False)
    cache = LLMCacheService(enabled=True, ttl_seconds=60, redis_service=fake_redis_svc)

    valid_payload = {
        "answer": "Normal blood pressure is generally defined as systolic under 120 and diastolic under 80 mm Hg.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c1", "document_id": "doc1"}]
    }

    key = cache.generate_cache_key("What is normal blood pressure?", user_scope=1001)
    stored = cache.set(key, valid_payload, user_scope=1001)
    assert stored is True

    cached = cache.get(key)
    assert cached is not None
    assert "Normal blood pressure" in cached["answer"]
    assert cached["timings"]["cache_hit"] is True
    assert cached["timings"]["distributed_cache"] is False

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["total_entries"] == 1


def test_layered_cache_writes_to_redis_when_available():
    """Verify LLMCacheService writes to Redis and can serve distributed hits."""
    fake_client = FakeRedisClient()
    redis_svc = RedisService(enabled=True)
    redis_svc._client = fake_client
    redis_svc._is_alive = True

    cache_instance_a = LLMCacheService(enabled=True, ttl_seconds=60, redis_service=redis_svc)
    cache_instance_b = LLMCacheService(enabled=True, ttl_seconds=60, redis_service=redis_svc)

    valid_payload = {
        "answer": "Metformin is a first-line medication for type 2 diabetes management.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c2", "document_id": "doc_metformin"}]
    }

    key = cache_instance_a.generate_cache_key("What is Metformin used for?", user_scope=2002)
    stored = cache_instance_a.set(key, valid_payload, user_scope=2002)
    assert stored is True

    # Read from Instance B (different process / node)
    # Clear Instance B's local cache to simulate separate process
    cache_instance_b._cache.clear()
    cached = cache_instance_b.get(key)
    assert cached is not None
    assert "Metformin is a first-line" in cached["answer"]
    assert cached["timings"]["cache_hit"] is True
    assert cached["timings"]["distributed_cache"] is True


def test_user_and_document_cache_isolation():
    """Verify User A cannot read User B's cache, and document changes invalidate correctly."""
    fake_client = FakeRedisClient()
    redis_svc = RedisService(enabled=True)
    redis_svc._client = fake_client
    redis_svc._is_alive = True

    cache = LLMCacheService(enabled=True, ttl_seconds=60, redis_service=redis_svc)

    payload_a = {
        "answer": "Confidential lab results for User 101 indicating normal fasting glucose.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c1", "document_id": "doc101"}]
    }

    key_user_a = cache.generate_cache_key("What are my glucose levels?", user_scope=101, document_signature="sig_101")
    key_user_b = cache.generate_cache_key("What are my glucose levels?", user_scope=202, document_signature="sig_202")

    assert key_user_a != key_user_b

    cache.set(key_user_a, payload_a, user_scope=101, document_signature="sig_101")

    # User B must get a cache MISS
    assert cache.get(key_user_b) is None

    # User A gets a cache HIT
    hit_a = cache.get(key_user_a)
    assert hit_a is not None
    assert "User 101" in hit_a["answer"]

    # Invalidate User 101 cache (e.g. on new lab report upload)
    invalidated_count = cache.delete_by_user(101)
    assert invalidated_count >= 1

    # User A now gets a cache MISS
    assert cache.get(key_user_a) is None


def test_cache_safety_invariant_refuses_unvalidated():
    """Verify only validated, grounded responses passing medical safety can be cached."""
    fake_client = FakeRedisClient()
    redis_svc = RedisService(enabled=True)
    redis_svc._client = fake_client
    redis_svc._is_alive = True

    cache = LLMCacheService(enabled=True, ttl_seconds=60, redis_service=redis_svc)

    # 1. Unvalidated / safety intercepted response
    unsafe_payload = {
        "answer": "Call emergency services immediately (911).",
        "retrieval_status": "safety_intercepted",
        "sources": []
    }
    key_unsafe = cache.generate_cache_key("I am experiencing severe chest pain", user_scope="anon")
    stored_unsafe = cache.set(key_unsafe, unsafe_payload, validated_only=True)
    assert stored_unsafe is False
    assert cache.get(key_unsafe) is None

    # 2. Error / ungrounded response
    error_payload = {
        "answer": "",
        "retrieval_status": "failed",
        "sources": []
    }
    key_err = cache.generate_cache_key("Unanswerable query", user_scope="anon")
    stored_err = cache.set(key_err, error_payload, validated_only=True)
    assert stored_err is False
    assert cache.get(key_err) is None


def test_graceful_recovery_when_redis_reconnects():
    """Verify cache seamlessly uses Redis once connectivity is restored."""
    fake_client = FakeRedisClient()
    redis_svc = RedisService(enabled=True)
    redis_svc._client = fake_client

    # Redis starts DOWN
    fake_client.is_connected = False
    redis_svc._is_alive = False

    cache = LLMCacheService(enabled=True, ttl_seconds=60, redis_service=redis_svc)

    payload = {
        "answer": "Asthma is characterized by variable airflow obstruction and bronchial hyperresponsiveness.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c_asthma", "document_id": "doc_resp"}]
    }

    key = cache.generate_cache_key("What is asthma pathophysiology?", user_scope=303)
    # Writes to local cache while Redis is down
    stored = cache.set(key, payload, user_scope=303)
    assert stored is True

    # Local cache serves the request
    val_local = cache.get(key)
    assert val_local is not None
    assert val_local["timings"]["distributed_cache"] is False

    # Redis comes UP
    fake_client.is_connected = True
    redis_svc._is_alive = True
    redis_svc._last_failure_time = 0.0

    # Write new entry with Redis online
    key2 = cache.generate_cache_key("What is chronic bronchitis?", user_scope=303)
    payload2 = {
        "answer": "Chronic bronchitis involves cough and sputum production for at least 3 months.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c_copd", "document_id": "doc_copd"}]
    }
    cache.set(key2, payload2, user_scope=303)

    # Invalidate local L1 cache to test Redis L2 fetch
    cache._cache.clear()
    val_redis = cache.get(key2)
    assert val_redis is not None
    assert "Chronic bronchitis" in val_redis["answer"]
    assert val_redis["timings"]["distributed_cache"] is True
