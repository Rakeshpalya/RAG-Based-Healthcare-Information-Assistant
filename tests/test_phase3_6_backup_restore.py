"""
Phase 3.6.5 Tests: Production Vector Store Backup & Restore Drill.

Verifies:
1. Snapshot creation from production vector store.
2. Manifest completeness and metadata.
3. Individual and compound SHA-256 checksum validation.
4. FAISS vector count verification (744).
5. Metadata record count verification (744).
6. Snapshot restoration into an isolated temporary directory.
7. Restored vector count verification (744).
8. Restored metadata count verification (744).
9. Restored vector dimension verification (384).
10. Retrieval consistency between the original production index and the restored index.

CRITICAL INVARIANT:
The production data/vector_store is NEVER mutated or overwritten.
All restore targets and snapshot staging occur in tmp_path.
"""

import hashlib
import json
import pytest
from pathlib import Path

from backend.services.snapshot_service import VectorStoreSnapshotService
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from backend.services.embedding_service import EmbeddingService


@pytest.fixture
def prod_store_dir():
    p = Path("data/vector_store")
    assert p.exists() and (p / "index.faiss").exists() and (p / "metadata.json").exists()
    return p


@pytest.fixture
def backup_drill_env(tmp_path, prod_store_dir):
    snapshots_dir = tmp_path / "snapshots"
    restore_target_dir = tmp_path / "restored_store"

    service = VectorStoreSnapshotService(
        vector_store_dir=prod_store_dir,
        snapshots_dir=snapshots_dir
    )
    return {
        "service": service,
        "snapshots_dir": snapshots_dir,
        "restore_dir": restore_target_dir,
        "prod_dir": prod_store_dir
    }


def test_01_through_05_snapshot_creation_and_integrity(backup_drill_env):
    """Steps 1-5: Create snapshot, verify manifest, checksums, vector count (744), metadata (744)."""
    service = backup_drill_env["service"]
    snaps_dir = backup_drill_env["snapshots_dir"]

    # 1. Create snapshot
    manifest = service.create_snapshot(
        snapshot_id="drill_snap_001",
        validate_production_invariants=True
    )

    # 2. Verify manifest structure
    assert manifest["snapshot_id"] == "drill_snap_001"
    assert "created_at" in manifest
    assert "faiss_checksum" in manifest
    assert "metadata_checksum" in manifest
    assert "checksum" in manifest

    # 3. Verify SHA-256 checksums
    snap_path = snaps_dir / "drill_snap_001"
    assert snap_path.exists()
    assert (snap_path / "index.faiss").exists()
    assert (snap_path / "metadata.json").exists()
    assert (snap_path / "manifest.json").exists()

    faiss_data = (snap_path / "index.faiss").read_bytes()
    meta_data = (snap_path / "metadata.json").read_bytes()

    assert hashlib.sha256(faiss_data).hexdigest() == manifest["faiss_checksum"]
    assert hashlib.sha256(meta_data).hexdigest() == manifest["metadata_checksum"]

    # 4. Verify FAISS vector count invariant
    assert manifest["vector_count"] == 744, f"Expected 744 vectors, found {manifest['vector_count']}"

    # 5. Verify Metadata count invariant
    assert manifest["metadata_count"] == 744, f"Expected 744 records, found {manifest['metadata_count']}"
    assert manifest["dimension"] == 384


def test_06_through_09_restore_into_isolated_target(backup_drill_env):
    """Steps 6-9: Restore into isolated target, verify vector count (744), metadata (744), dim (384)."""
    service = backup_drill_env["service"]
    snaps_dir = backup_drill_env["snapshots_dir"]
    restore_dir = backup_drill_env["restore_dir"]

    # First take snapshot
    service.create_snapshot("drill_snap_002", validate_production_invariants=True)
    snap_path = snaps_dir / "drill_snap_002"

    # 6. Restore snapshot into an isolated temporary directory
    res = service.restore_snapshot(
        snapshot_dir=snap_path,
        target_dir=restore_dir,
        activate_in_memory=False
    )
    assert res["restored"] is True

    # 7. Verify restored vector count
    assert res["vector_count"] == 744

    # 8. Verify restored metadata count
    assert res["metadata_count"] == 744

    # 9. Verify vector dimension
    assert res["dimension"] == 384


def test_10_retrieval_consistency_between_original_and_restored(backup_drill_env):
    """Step 10: Verify retrieval results on restored index match the production index exactly."""
    service = backup_drill_env["service"]
    snaps_dir = backup_drill_env["snapshots_dir"]
    restore_dir = backup_drill_env["restore_dir"]

    service.create_snapshot("drill_snap_003", validate_production_invariants=True)
    service.restore_snapshot(
        snapshot_dir=snaps_dir / "drill_snap_003",
        target_dir=restore_dir,
        activate_in_memory=False
    )

    # Instantiate services pointing to prod vs restored
    prod_vs = get_vector_store_service()
    restored_vs = VectorStoreService(storage_dir=restore_dir, dimension=384)
    loaded = restored_vs.load()
    assert loaded is True
    assert restored_vs.count() == 744

    test_queries = [
        "What is the recommended management for stage 1 hypertension?",
        "What are the diagnostic fasting glucose criteria for type 2 diabetes?",
        "What triggers bronchial wheezing and airway constriction in asthma?"
    ]

    for q in test_queries:
        q_vec = EmbeddingService.embed_query(q)
        assert q_vec is not None

        prod_results = prod_vs.search(q_vec, top_k=3)
        rest_results = restored_vs.search(q_vec, top_k=3)

        assert len(prod_results) == len(rest_results)
        for p_res, r_res in zip(prod_results, rest_results):
            assert p_res["chunk_id"] == r_res["chunk_id"]
            assert pytest.approx(p_res["similarity_score"], abs=1e-4) == r_res["similarity_score"]
            assert p_res["text"] == r_res["text"]
