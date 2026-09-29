"""
Automated Secret Audit and Environment Hygiene Test Suite for Phase 9.
Verifies that:
1. No real API keys or private credentials exist in source code.
2. No developer-specific absolute paths are hardcoded in backend services.
3. .env.example strictly contains template placeholders.
4. .gitignore prevents sensitive credential and database file leaks.
"""

import os
import re
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_hardcoded_api_keys_in_backend():
    """Scans all backend Python files for exposed live API key formats."""
    # Matches real Google AI keys not containing 'Mock' or 'Test'
    gemini_pattern = re.compile(r"['\"](AIzaSy[a-zA-Z0-9_\-]{33})['\"]")
    openai_pattern = re.compile(r"['\"](sk-[a-zA-Z0-9]{32,})['\"]")
    aws_pattern = re.compile(r"['\"](AKIA[0-9A-Z]{16})['\"]")

    backend_dir = os.path.join(REPO_ROOT, "backend")

    for root, _, files in os.walk(backend_dir):
        for f in files:
            if f.endswith(".py"):
                file_path = os.path.join(root, f)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                    
                    # Gemini key check
                    for m in gemini_pattern.finditer(content):
                        key = m.group(1)
                        if "mock" not in key.lower() and "test" not in key.lower():
                            pytest.fail(f"Potential real Gemini key detected in {f}: {key[:8]}...")

                    # OpenAI key check
                    for m in openai_pattern.finditer(content):
                        key = m.group(1)
                        if "mock" not in key.lower() and "test" not in key.lower():
                            pytest.fail(f"Potential real OpenAI key detected in {f}: {key[:8]}...")

                    # AWS key check
                    for m in aws_pattern.finditer(content):
                        pytest.fail(f"Potential AWS key detected in {f}")


def test_no_developer_absolute_paths_in_backend():
    """Scans backend code for hardcoded Windows/Linux absolute home/user paths."""
    # Pattern detects c:\users\ or /home/username/ hardcoded paths
    abs_path_pattern = re.compile(r"['\"](?:[a-zA-Z]:\\\\Users\\\\[a-zA-Z0-9_]+|/home/[a-zA-Z0-9_]+)['\"]", re.IGNORECASE)

    backend_dir = os.path.join(REPO_ROOT, "backend")

    for root, _, files in os.walk(backend_dir):
        for f in files:
            if f.endswith(".py"):
                file_path = os.path.join(root, f)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                    matches = abs_path_pattern.findall(content)
                    assert len(matches) == 0, f"Hardcoded developer path found in {f}: {matches}"


def test_env_example_has_no_secrets():
    """Verifies that .env.example contains only empty values or documented placeholders."""
    env_example_path = os.path.join(REPO_ROOT, ".env.example")
    assert os.path.exists(env_example_path), ".env.example must exist"

    with open(env_example_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            if "KEY" in key or "SECRET" in key:
                # Key/Secret values in .env.example must be empty or generic placeholders
                assert val == "" or "placeholder" in val.lower() or "your_" in val.lower(), (
                    f"Sensitive key {key} in .env.example has unexpected value: '{val}'"
                )


def test_gitignore_protects_secrets_and_databases():
    """Ensures .gitignore includes critical secret and storage exclusions."""
    gitignore_path = os.path.join(REPO_ROOT, ".gitignore")
    assert os.path.exists(gitignore_path), ".gitignore must exist"

    with open(gitignore_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    assert ".env" in content
    assert "*.pem" in content
    assert "*.key" in content
    assert "*.db" in content
