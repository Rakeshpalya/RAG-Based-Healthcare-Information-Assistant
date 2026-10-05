"""
Concurrency and Request Control for AI Healthcare Agent (Phase 3.4).

Provides production-grade async and thread-safe concurrency management:
1. Bounded concurrency semaphore for LLM calls (prevents resource exhaustion).
2. Request and LLM timeout controls.
3. Backpressure handling with graceful rejection (HTTP 503 / ConcurrencyLimitExceeded).
4. Cancellation safety: semaphores are ALWAYS released in finally blocks (no deadlocks).
"""

import time
import asyncio
import logging
import threading
from typing import Optional
from contextlib import contextmanager, asynccontextmanager
from fastapi import HTTPException, status

from backend.config import settings

logger = logging.getLogger("security.concurrency")


class ConcurrencyLimitExceeded(HTTPException):
    """Raised when maximum concurrent LLM requests are running and timeout expires."""
    def __init__(self, detail: str = "Server is processing maximum concurrent AI generation requests. Please retry shortly."):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers={"Retry-After": "5"}
        )


class LLMConcurrencyController:
    """
    Manages concurrency limits for external LLM generation calls.
    Supports both async (FastAPI coroutines) and synchronous (background workers/sync RAG) callers.
    """

    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        acquire_timeout: Optional[float] = None
    ):
        self._max_concurrent = max_concurrent or getattr(settings, "MAX_CONCURRENT_LLM_REQUESTS", 10)
        self._acquire_timeout = acquire_timeout or getattr(settings, "REQUEST_TIMEOUT_SECONDS", 60.0)

        # Thread synchronization
        self._thread_semaphore = threading.BoundedSemaphore(self._max_concurrent)
        self._lock = threading.Lock()

        # Metrics
        self._active_count: int = 0
        self._total_acquired: int = 0
        self._total_rejected: int = 0

        # Lazy async semaphore per event loop
        self._async_semaphores = {}

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @max_concurrent.setter
    def max_concurrent(self, val: int) -> None:
        with self._lock:
            self._max_concurrent = max(1, int(val))
            self._thread_semaphore = threading.BoundedSemaphore(self._max_concurrent)
            self._async_semaphores.clear()

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    @property
    def total_acquired(self) -> int:
        with self._lock:
            return self._total_acquired

    @property
    def total_rejected(self) -> int:
        with self._lock:
            return self._total_rejected

    def _get_async_semaphore(self) -> asyncio.Semaphore:
        """Retrieves or creates an asyncio.Semaphore tied to the current event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        loop_id = id(loop) if loop else "no_loop"
        with self._lock:
            if loop_id not in self._async_semaphores:
                self._async_semaphores[loop_id] = asyncio.Semaphore(self._max_concurrent)
            return self._async_semaphores[loop_id]

    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """
        Synchronous context manager to safely acquire an LLM generation slot.
        Releases semaphore reliably on success, exception, or timeout.
        """
        effective_timeout = timeout if timeout is not None else self._acquire_timeout
        acquired = self._thread_semaphore.acquire(timeout=effective_timeout)

        if not acquired:
            with self._lock:
                self._total_rejected += 1
            logger.warning(
                "LLM concurrency limit reached (%d active). Request timed out after %.2fs waiting for slot.",
                self._max_concurrent, effective_timeout
            )
            raise ConcurrencyLimitExceeded()

        with self._lock:
            self._active_count += 1
            self._total_acquired += 1

        try:
            yield
        finally:
            with self._lock:
                self._active_count = max(0, self._active_count - 1)
            self._thread_semaphore.release()

    @asynccontextmanager
    async def acquire_async(self, timeout: Optional[float] = None):
        """
        Asynchronous context manager to safely acquire an LLM generation slot.
        Releases semaphore reliably on completion, cancellation, or exception.
        """
        effective_timeout = timeout if timeout is not None else self._acquire_timeout
        sem = self._get_async_semaphore()

        try:
            await asyncio.wait_for(sem.acquire(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            with self._lock:
                self._total_rejected += 1
            logger.warning(
                "Async LLM concurrency limit reached (%d capacity). Slot acquisition timed out after %.2fs.",
                self._max_concurrent, effective_timeout
            )
            raise ConcurrencyLimitExceeded()

        with self._lock:
            self._active_count += 1
            self._total_acquired += 1

        try:
            yield
        finally:
            with self._lock:
                self._active_count = max(0, self._active_count - 1)
            sem.release()

    def reset(self) -> None:
        """Resets active counters and semaphores. Primarily for test isolation."""
        with self._lock:
            self._active_count = 0
            self._total_acquired = 0
            self._total_rejected = 0
            self._thread_semaphore = threading.BoundedSemaphore(self._max_concurrent)
            self._async_semaphores.clear()


# Global Singleton Concurrency Controller
concurrency_controller = LLMConcurrencyController()
