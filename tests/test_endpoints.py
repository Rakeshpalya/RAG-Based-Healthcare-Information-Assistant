import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.main import root, health_check


def test_root_endpoint():
    result = asyncio.run(root())
    assert result["status"] == "online"
    assert "message" in result
    assert result["docs"] == "/docs"
    print("[PASS] GET / test passed:", result)


def test_health_endpoint():
    result = asyncio.run(health_check())
    assert result["status"] == "healthy"
    assert result["service"] == "AI-Healthcare-Agent"
    print("[PASS] GET /health test passed:", result)


if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    print("[SUCCESS] All endpoint unit tests passed successfully!")

