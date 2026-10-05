"""
Pytest configuration and global fixtures for AI Healthcare Agent test suite.
"""
import pytest
from backend.services.llm_cache_service import get_llm_cache_service


@pytest.fixture(autouse=True)
def reset_llm_cache_between_tests():
    """
    Ensure each test runs with a pristine, isolated LLM response cache.
    Prevents cross-test response leakage or interference with mock assertions.
    """
    cache = get_llm_cache_service()
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def anyio_backend():
    """Default anyio backend to asyncio."""
    return "asyncio"
