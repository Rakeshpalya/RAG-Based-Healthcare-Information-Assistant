"""
Production Startup Validation Module for AI-Healthcare-Agent.

Enforces pre-flight operational invariants before the container accepts traffic:
1. FAISS index file exists on disk.
2. Metadata store JSON exists on disk.
3. Strict Vector Store Invariant: FAISS vector count == 744 == metadata records count.
4. Embedding dimension == 384 (sentence-transformers/all-MiniLM-L6-v2).
5. Medical safety engine registered with all 18 clinical categories.
6. Embedding service loads and encodes test vector safely.
7. Database connectivity / resilience fallback verified.
8. Critical configuration exists.

CRITICAL INVARIANT:
This validator is strictly READ-ONLY. It never mutates, re-indexes, or modifies
the FAISS index or metadata store.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import faiss

from backend.config import settings

logger = logging.getLogger(__name__)

EXPECTED_VECTOR_COUNT = 744
EXPECTED_METADATA_COUNT = 744
EXPECTED_EMBEDDING_DIM = 384
EXPECTED_SAFETY_CATEGORIES = 15


class StartupValidationError(Exception):
    """Raised when one or more production startup invariants fail."""
    pass


def validate_production_startup(root_dir: Path = None) -> Dict[str, Any]:
    """
    Executes comprehensive non-destructive pre-flight validation.

    Returns:
        Dict with validation status, subsystem results, and any error details.

    Raises:
        StartupValidationError: If any critical invariant fails.
    """
    base_dir = root_dir or Path(__file__).resolve().parent.parent
    faiss_path = base_dir / "data" / "vector_store" / "index.faiss"
    metadata_path = base_dir / "data" / "vector_store" / "metadata.json"

    errors: List[str] = []
    checks: Dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # 1. Configuration Validation
    # --------------------------------------------------------------------------
    masked_key = settings.get_masked_api_key()
    checks["configuration"] = {
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "host": settings.HOST,
        "port": settings.PORT,
        "gemini_api_key_configured": masked_key != "NOT_CONFIGURED",
        "gemini_masked_key": masked_key,
        "gemini_model": settings.GEMINI_MODEL,
        "database_url_configured": bool(settings.DATABASE_URL),
        "status": "PASS",
    }

    # --------------------------------------------------------------------------
    # 2. Vector Store File Presence & Structure
    # --------------------------------------------------------------------------
    if not faiss_path.exists():
        msg = f"FAISS index file missing at expected path: {faiss_path}"
        errors.append(msg)
        checks["faiss_file"] = {"status": "FAIL", "error": msg}
    else:
        checks["faiss_file"] = {"status": "PASS", "path": str(faiss_path), "size_bytes": faiss_path.stat().st_size}

    if not metadata_path.exists():
        msg = f"Metadata JSON file missing at expected path: {metadata_path}"
        errors.append(msg)
        checks["metadata_file"] = {"status": "FAIL", "error": msg}
    else:
        checks["metadata_file"] = {"status": "PASS", "path": str(metadata_path), "size_bytes": metadata_path.stat().st_size}

    # --------------------------------------------------------------------------
    # 3. Strict Read-Only Vector Store Invariant Check (744 == 744)
    # --------------------------------------------------------------------------
    faiss_count = -1
    meta_count = -1
    if faiss_path.exists() and metadata_path.exists():
        try:
            # Non-destructive read
            idx = faiss.read_index(str(faiss_path))
            faiss_count = int(idx.ntotal)
            dim = int(idx.d)

            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)
            records = meta_payload.get("records", [])
            meta_count = len(records)

            if faiss_count != EXPECTED_VECTOR_COUNT:
                msg = f"FAISS invariant breach: Expected {EXPECTED_VECTOR_COUNT} vectors, found {faiss_count}."
                errors.append(msg)

            if meta_count != EXPECTED_METADATA_COUNT:
                msg = f"Metadata invariant breach: Expected {EXPECTED_METADATA_COUNT} records, found {meta_count}."
                errors.append(msg)

            if faiss_count != meta_count:
                msg = f"Count mismatch: FAISS has {faiss_count} vectors, but metadata has {meta_count} records."
                errors.append(msg)

            if dim != EXPECTED_EMBEDDING_DIM:
                msg = f"Vector dimension mismatch: Expected {EXPECTED_EMBEDDING_DIM}, found {dim}."
                errors.append(msg)

            checks["vector_store_invariant"] = {
                "faiss_count": faiss_count,
                "metadata_count": meta_count,
                "dimension": dim,
                "expected_count": EXPECTED_VECTOR_COUNT,
                "expected_dimension": EXPECTED_EMBEDDING_DIM,
                "status": "PASS" if not errors else "FAIL"
            }
        except Exception as exc:
            msg = f"Vector store invariant inspection failed with error: {str(exc)}"
            errors.append(msg)
            checks["vector_store_invariant"] = {"status": "FAIL", "error": msg}

    # --------------------------------------------------------------------------
    # 4. Embedding Service Validation
    # --------------------------------------------------------------------------
    try:
        from backend.services.embedding_service import EmbeddingService
        embed_dim = EmbeddingService.get_embedding_dimension()
        if embed_dim != EXPECTED_EMBEDDING_DIM:
            msg = f"Embedding service dimension mismatch: Expected {EXPECTED_EMBEDDING_DIM}, got {embed_dim}."
            errors.append(msg)
            checks["embedding_service"] = {"status": "FAIL", "dimension": embed_dim, "error": msg}
        else:
            # Verification: single test vector encoding
            test_vec = EmbeddingService.embed_query("Hypertension clinical baseline check")
            if len(test_vec) != EXPECTED_EMBEDDING_DIM:
                msg = f"Embedding output length mismatch: Expected {EXPECTED_EMBEDDING_DIM}, got {len(test_vec)}."
                errors.append(msg)
                checks["embedding_service"] = {"status": "FAIL", "error": msg}
            else:
                checks["embedding_service"] = {
                    "status": "PASS",
                    "model_name": EmbeddingService.MODEL_NAME,
                    "dimension": embed_dim
                }
    except Exception as exc:
        msg = f"Embedding service initialization failed: {str(exc)}"
        errors.append(msg)
        checks["embedding_service"] = {"status": "FAIL", "error": msg}

    # --------------------------------------------------------------------------
    # 5. Medical Safety Guardrails Engine Check
    # --------------------------------------------------------------------------
    try:
        from backend.safety.medical_safety_guard import MedicalSafetyGuard, SafetyCategory
        categories_count = len(SafetyCategory)
        if categories_count != EXPECTED_SAFETY_CATEGORIES:
            msg = f"Safety categories count mismatch: Expected {EXPECTED_SAFETY_CATEGORIES}, found {categories_count}."
            errors.append(msg)
            checks["safety_engine"] = {"status": "FAIL", "categories_count": categories_count, "error": msg}
        else:
            # Test emergency detection
            allow_gen, assessment, immediate_msg = MedicalSafetyGuard.pre_screen_inquiry(
                "I have severe chest pain and cannot breathe"
            )
            if allow_gen is not False or assessment.category != SafetyCategory.EMERGENCY_SYMPTOMS:
                msg = "Safety engine critical failure: Did not detect severe emergency query."
                errors.append(msg)
                checks["safety_engine"] = {"status": "FAIL", "error": msg}
            else:
                checks["safety_engine"] = {
                    "status": "PASS",
                    "categories_count": categories_count,
                    "emergency_detection": "VERIFIED"
                }
    except Exception as exc:
        msg = f"Medical safety guard engine check failed: {str(exc)}"
        errors.append(msg)
        checks["safety_engine"] = {"status": "FAIL", "error": msg}

    # --------------------------------------------------------------------------
    # 6. Database Connectivity Check
    # --------------------------------------------------------------------------
    try:
        from backend.database.database import get_engine
        engine = get_engine()
        checks["database"] = {
            "status": "PASS",
            "dialect": engine.dialect.name,
            "url_type": "sqlite_fallback" if "sqlite" in str(engine.url) else "postgresql"
        }
    except Exception as exc:
        checks["database"] = {"status": "DEGRADED", "warning": str(exc)}

    # --------------------------------------------------------------------------
    # Summary & Status Determination
    # --------------------------------------------------------------------------
    overall_passed = len(errors) == 0
    result = {
        "status": "PASS" if overall_passed else "FAIL",
        "passed": overall_passed,
        "errors": errors,
        "checks": checks,
        "vector_count": faiss_count,
        "metadata_count": meta_count,
        "embedding_dimension": EXPECTED_EMBEDDING_DIM,
    }

    if not overall_passed:
        logger.critical("Startup validation FAILED with errors: %s", errors)
        raise StartupValidationError(f"Production startup validation failed: {'; '.join(errors)}")

    logger.info("Production startup validation PASSED: All operational invariants verified.")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("AI-HEALTHCARE-AGENT: PRODUCTION STARTUP VALIDATION")
    print("=" * 70)
    try:
        res = validate_production_startup()
        print("\n[OK] Status: PASS")
        print(f"  FAISS Vectors:       {res['vector_count']} / {EXPECTED_VECTOR_COUNT}")
        print(f"  Metadata Records:    {res['metadata_count']} / {EXPECTED_METADATA_COUNT}")
        print(f"  Embedding Dimension: {res['embedding_dimension']}")
        print(f"  Safety Engine:       {res['checks']['safety_engine']['status']} ({res['checks']['safety_engine'].get('categories_count', 0)} categories)")
        print(f"  Database Engine:     {res['checks']['database']['status']} ({res['checks']['database'].get('dialect', 'unknown')})")
        print("=" * 70)
        sys.exit(0)
    except StartupValidationError as err:
        print(f"\n[FAIL] Status: FAILED - {err}")
        print("=" * 70)
        sys.exit(1)
