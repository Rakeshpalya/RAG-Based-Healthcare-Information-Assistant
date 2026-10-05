"""
Phase 3.5.4 Tests: Vector Store Snapshot & Restore Suite.

Verifies:
1. Atomic snapshot creation without modifying or locking active production index.
2. Verified SHA-256 checksum calculation and manifest structure.
3. Metadata and FAISS vector count alignment.
4. Two-phase restoration (temporary staging validation before activation).
5. Error handling on corrupted snapshot (checksum tamper).
6. Error handling on incomplete snapshot (missing files).
7. Error handling on wrong embedding dimension.
8. Error handling on vector/metadata count mismatch.
9. Verification of production invariant: exactly 744 FAISS vectors and 744 metadata records.
"""

import json
import shutil
import pytest
from pathlib import Path
import numpy as np
import faiss

from backend.services.snapshot_service import (
    VectorStoreSnapshotService,
    compute_file_sha256
)
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service


@pytest.fixture
def temp_store_and_snapshots(tmp_path):
    """Fixture providing isolated temporary directories and a dummy vector store."""
    store_dir = tmp_path / "active_store"
    snaps_dir = tmp_path / "snapshots"
    restore_dir = tmp_path / "restore_target"

    store_dir.mkdir(parents=True)
    snaps_dir.mkdir(parents=True)
    restore_dir.mkdir(parents=True)

    # Create dummy vector store with 10 vectors of dimension 384
    vs = VectorStoreService(dimension=384, storage_dir=store_dir)
    embeddings = np.random.randn(10, 384).astype(np.float32)
    # L2 normalize
    faiss.normalize_L2(embeddings)
    metadatas = [
        {"chunk_id": f"chunk_{i}", "text": f"Medical evidence text {i}", "document_id": f"doc_{i % 2}"}
        for i in range(10)
    ]
    vs.add_embeddings(embeddings, metadatas)
    vs.save(store_dir)

    return {
        "store_dir": store_dir,
        "snaps_dir": snaps_dir,
        "restore_dir": restore_dir,
        "service": VectorStoreSnapshotService(vector_store_dir=store_dir, snapshots_dir=snaps_dir)
    }


def test_snapshot_creation_and_manifest(temp_store_and_snapshots):
    """Verify snapshot is created atomically with valid manifest."""
    env = temp_store_and_snapshots
    service = env["service"]

    manifest = service.create_snapshot(snapshot_id="test_snap_001")

    assert manifest["snapshot_id"] == "test_snap_001"
    assert manifest["vector_count"] == 10
    assert manifest["metadata_count"] == 10
    assert manifest["dimension"] == 384
    assert "checksum" in manifest
    assert "faiss_checksum" in manifest
    assert "metadata_checksum" in manifest

    snap_dir = env["snaps_dir"] / "test_snap_001"
    assert (snap_dir / "index.faiss").exists()
    assert (snap_dir / "metadata.json").exists()
    assert (snap_dir / "manifest.json").exists()

    # Active index remains untouched and intact
    active_vs = VectorStoreService(dimension=384, storage_dir=env["store_dir"])
    active_vs.load(env["store_dir"])
    assert active_vs.count() == 10


def test_checksum_validation(temp_store_and_snapshots):
    """Verify SHA-256 checksum matching and tampering detection."""
    env = temp_store_and_snapshots
    service = env["service"]

    service.create_snapshot(snapshot_id="snap_checksum_test")
    snap_dir = env["snaps_dir"] / "snap_checksum_test"

    val_res = service.validate_snapshot(snap_dir)
    assert val_res["valid"] is True
    assert len(val_res["errors"]) == 0

    # Tamper with metadata.json to simulate corruption
    meta_path = snap_dir / "metadata.json"
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(" ")

    tampered_res = service.validate_snapshot(snap_dir)
    assert tampered_res["valid"] is False
    assert any("Checksum mismatch" in err for err in tampered_res["errors"])


def test_restore_two_phase_activation(temp_store_and_snapshots):
    """Verify restore validates in staging before activating to target directory."""
    env = temp_store_and_snapshots
    service = env["service"]

    service.create_snapshot(snapshot_id="snap_restore_test")
    snap_dir = env["snaps_dir"] / "snap_restore_test"

    restore_target = env["restore_dir"]
    restore_res = service.restore_snapshot(
        snapshot_dir=snap_dir,
        target_dir=restore_target,
        activate_in_memory=False
    )

    assert restore_res["restored"] is True
    assert restore_res["vector_count"] == 10
    assert restore_res["metadata_count"] == 10

    # Verify restored vector store can be loaded and searched
    restored_vs = VectorStoreService(dimension=384, storage_dir=restore_target)
    assert restored_vs.load(restore_target) is True
    assert restored_vs.count() == 10

    query = np.random.randn(384).astype(np.float32)
    faiss.normalize_L2(query.reshape(1, -1))
    hits = restored_vs.search(query, top_k=3)
    assert len(hits) == 3


def test_incomplete_snapshot_rejected(temp_store_and_snapshots):
    """Verify restore and validation fail safely on incomplete snapshot."""
    env = temp_store_and_snapshots
    service = env["service"]

    incomplete_dir = env["snaps_dir"] / "incomplete_snap"
    incomplete_dir.mkdir(parents=True)
    # Only write manifest.json, missing index.faiss and metadata.json
    with open(incomplete_dir / "manifest.json", "w") as f:
        json.dump({"schema_version": "v3.5.0"}, f)

    val = service.validate_snapshot(incomplete_dir)
    assert val["valid"] is False
    assert any("Incomplete snapshot" in err for err in val["errors"])

    with pytest.raises(ValueError) as exc_info:
        service.restore_snapshot(incomplete_dir, target_dir=env["restore_dir"])
    assert "incomplete" in str(exc_info.value).lower()


def test_wrong_dimension_rejected(temp_store_and_snapshots, tmp_path):
    """Verify validation rejects index with wrong dimension (e.g. 512 instead of 384)."""
    env = temp_store_and_snapshots
    wrong_dim_dir = tmp_path / "wrong_dim_store"
    wrong_dim_dir.mkdir(parents=True)

    vs = VectorStoreService(dimension=512, storage_dir=wrong_dim_dir)
    embeddings = np.random.randn(5, 512).astype(np.float32)
    faiss.normalize_L2(embeddings)
    metadatas = [{"chunk_id": f"c_{i}", "text": "text"} for i in range(5)]
    vs.add_embeddings(embeddings, metadatas)
    vs.save(wrong_dim_dir)

    service = VectorStoreSnapshotService(vector_store_dir=wrong_dim_dir, snapshots_dir=env["snaps_dir"])
    service.create_snapshot(snapshot_id="wrong_dim_snap")

    snap_dir = env["snaps_dir"] / "wrong_dim_snap"
    val = service.validate_snapshot(snap_dir, expected_dimension=384)
    assert val["valid"] is False
    assert any("Wrong dimension" in err for err in val["errors"])


def test_vector_metadata_count_mismatch_rejected(temp_store_and_snapshots, tmp_path):
    """Verify validation and restore reject corrupted snapshot where vectors != metadata records."""
    env = temp_store_and_snapshots
    mismatch_dir = tmp_path / "mismatch_store"
    mismatch_dir.mkdir(parents=True)

    vs = VectorStoreService(dimension=384, storage_dir=mismatch_dir)
    embeddings = np.random.randn(8, 384).astype(np.float32)
    faiss.normalize_L2(embeddings)
    metadatas = [{"chunk_id": f"c_{i}", "text": "text"} for i in range(8)]
    vs.add_embeddings(embeddings, metadatas)
    vs.save(mismatch_dir)

    # Intentionally corrupt metadata file by dropping records
    meta_file = mismatch_dir / "metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    meta["records"] = meta["records"][:4]  # 4 records vs 8 vectors!
    with open(meta_file, "w") as f:
        json.dump(meta, f)

    service = VectorStoreSnapshotService(vector_store_dir=mismatch_dir, snapshots_dir=env["snaps_dir"])

    # Creation should fail immediately due to mismatch
    with pytest.raises(ValueError) as exc_info:
        service.create_snapshot(snapshot_id="mismatch_snap")
    assert "mismatch" in str(exc_info.value).lower()


def test_production_vector_store_invariants():
    """
    CRITICAL INVARIANT VERIFICATION:
    Verifies that the production vector store in data/vector_store
    contains EXACTLY 744 vectors and 744 metadata records, with dimension 384,
    and creates a production-grade verified snapshot.
    """
    prod_store_dir = Path("data/vector_store")
    assert prod_store_dir.exists(), "Production vector store data/vector_store must exist"

    service = VectorStoreSnapshotService(vector_store_dir=prod_store_dir)
    manifest = service.create_snapshot(
        snapshot_id="prod_phase3_5_test_snapshot",
        validate_production_invariants=True
    )

    # Check 744/744 production invariant
    assert manifest["vector_count"] == 744
    assert manifest["metadata_count"] == 744
    assert manifest["dimension"] == 384
    assert manifest["schema_version"] == "v3.5.0"

    # Validate the created production snapshot
    snap_path = service.snapshots_dir / "prod_phase3_5_test_snapshot"
    val = service.validate_snapshot(snap_path, expected_dimension=384)
    assert val["valid"] is True
    assert val["actual_vectors"] == 744
    assert val["actual_metadata"] == 744
    assert len(val["errors"]) == 0

    # Clean up test snapshot
    shutil.rmtree(snap_path, ignore_errors=True)
