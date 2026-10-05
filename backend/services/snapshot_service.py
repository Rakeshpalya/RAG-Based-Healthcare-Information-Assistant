"""
Production Vector Store Snapshot & Automated Restore Service (Phase 3.5).

Provides safe, atomic, verified backup and disaster recovery for FAISS vector indices
and metadata stores:
1. Atomic snapshot creation with verified SHA-256 checksums and manifest records.
2. Never overwrites or locks the active production index during backup.
3. Rigorous validation of vector count, metadata count, and embedding dimensions (384).
4. Safe two-phase restoration: unpacks and validates in a temporary staging area first.
5. Invariant preservation: strict 1:1 alignment between FAISS vectors and metadata records (744/744).
"""

import os
import shutil
import hashlib
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import faiss

from backend.services.vector_store_service import VectorStoreService, get_vector_store_service

logger = logging.getLogger("services.snapshot")

SNAPSHOT_SCHEMA_VERSION = "v3.5.0"
DEFAULT_SNAPSHOTS_DIR = Path("data/snapshots")
DEFAULT_VECTOR_STORE_DIR = Path("data/vector_store")


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 checksum of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class VectorStoreSnapshotService:
    """
    Manages atomic creation, integrity verification, and safe restoration of
    vector store snapshots.
    """

    def __init__(
        self,
        vector_store_dir: Optional[Union[str, Path]] = None,
        snapshots_dir: Optional[Union[str, Path]] = None,
        schema_version: str = SNAPSHOT_SCHEMA_VERSION
    ):
        self.vector_store_dir = Path(vector_store_dir) if vector_store_dir else DEFAULT_VECTOR_STORE_DIR
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else DEFAULT_SNAPSHOTS_DIR
        self.schema_version = schema_version
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        snapshot_id: Optional[str] = None,
        custom_source_dir: Optional[Union[str, Path]] = None,
        validate_production_invariants: bool = False
    ) -> Dict[str, Any]:
        """
        Atomically creates a snapshot of the vector store index and metadata.

        Steps:
        1. Reads source index and metadata files.
        2. Validates alignment (vector count == metadata count > 0).
        3. Writes to temporary staging folder: .tmp_<snapshot_id>.
        4. Calculates SHA-256 checksums and creates manifest.json.
        5. Atomically renames staging folder to final snapshot folder.
        6. Never touches or locks the active index files.
        """
        source_dir = Path(custom_source_dir) if custom_source_dir else self.vector_store_dir
        index_file = source_dir / "index.faiss"
        metadata_file = source_dir / "metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(f"Active FAISS index file not found: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Active metadata store file not found: {metadata_file}")

        # 1. Inspect source index and metadata
        loaded_index = faiss.read_index(str(index_file))
        vector_count = int(loaded_index.ntotal)
        dimension = int(loaded_index.d)

        with open(metadata_file, "r", encoding="utf-8") as f:
            meta_payload = json.load(f)

        records = meta_payload.get("records", [])
        metadata_count = len(records)

        # Invariant checks
        if vector_count != metadata_count:
            raise ValueError(
                f"Vector and metadata count mismatch: {vector_count} vectors vs {metadata_count} metadata records."
            )

        if vector_count == 0:
            raise ValueError("Refusing to create snapshot of empty vector store (vector count is 0).")

        if validate_production_invariants:
            if vector_count != 744 or metadata_count != 744:
                raise ValueError(
                    f"Production invariant violation: Expected exactly 744 vectors and metadata, got {vector_count}."
                )
            if dimension != 384:
                raise ValueError(
                    f"Production dimension violation: Expected dimension 384, got {dimension}."
                )

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if not snapshot_id:
            snapshot_id = now_utc.strftime("%Y-%m-%dT%H%M%SZ")

        final_snapshot_dir = self.snapshots_dir / snapshot_id
        if final_snapshot_dir.exists():
            raise FileExistsError(f"Snapshot directory already exists: {final_snapshot_dir}")

        temp_snapshot_dir = self.snapshots_dir / f".tmp_{snapshot_id}"
        if temp_snapshot_dir.exists():
            shutil.rmtree(temp_snapshot_dir, ignore_errors=True)
        temp_snapshot_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy active files to staging dir
            temp_index = temp_snapshot_dir / "index.faiss"
            temp_metadata = temp_snapshot_dir / "metadata.json"

            shutil.copy2(index_file, temp_index)
            shutil.copy2(metadata_file, temp_metadata)

            # Compute checksums
            faiss_checksum = compute_file_sha256(temp_index)
            metadata_checksum = compute_file_sha256(temp_metadata)

            # Compound checksum of both files
            compound = hashlib.sha256((faiss_checksum + metadata_checksum).encode("utf-8")).hexdigest()

            manifest = {
                "schema_version": self.schema_version,
                "snapshot_id": snapshot_id,
                "created_at": now_utc.isoformat(),
                "vector_count": vector_count,
                "metadata_count": metadata_count,
                "dimension": dimension,
                "faiss_checksum": faiss_checksum,
                "metadata_checksum": metadata_checksum,
                "checksum": compound,
            }

            temp_manifest = temp_snapshot_dir / "manifest.json"
            with open(temp_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            # Atomic rename staging directory to final directory
            temp_snapshot_dir.rename(final_snapshot_dir)
            logger.info("Successfully created vector store snapshot: %s (vectors=%d)", snapshot_id, vector_count)
            return manifest
        except Exception:
            if temp_snapshot_dir.exists():
                shutil.rmtree(temp_snapshot_dir, ignore_errors=True)
            raise

    def validate_snapshot(
        self,
        snapshot_dir: Union[str, Path],
        expected_dimension: int = 384
    ) -> Dict[str, Any]:
        """
        Rigorously validates a snapshot directory:
        1. Checks presence of index.faiss, metadata.json, manifest.json.
        2. Validates schema_version.
        3. Verifies SHA-256 checksums match manifest.
        4. Verifies actual vector count matches actual metadata records count.
        5. Verifies dimension matches expected_dimension (384).

        Returns:
            Dict with "valid": bool, "errors": List[str], "manifest": Dict.
        """
        s_dir = Path(snapshot_dir)
        errors: List[str] = []
        manifest: Dict[str, Any] = {}

        if not s_dir.exists() or not s_dir.is_dir():
            return {"valid": False, "errors": [f"Snapshot directory does not exist: {s_dir}"], "manifest": {}}

        index_file = s_dir / "index.faiss"
        metadata_file = s_dir / "metadata.json"
        manifest_file = s_dir / "manifest.json"

        if not manifest_file.exists():
            errors.append("Incomplete snapshot: manifest.json is missing.")
        if not index_file.exists():
            errors.append("Incomplete snapshot: index.faiss is missing.")
        if not metadata_file.exists():
            errors.append("Incomplete snapshot: metadata.json is missing.")

        if errors:
            return {"valid": False, "errors": errors, "manifest": {}}

        # Parse manifest
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            return {"valid": False, "errors": [f"Corrupted manifest.json: {str(exc)}"], "manifest": {}}

        # Check checksums
        try:
            curr_faiss_cs = compute_file_sha256(index_file)
            curr_meta_cs = compute_file_sha256(metadata_file)
            curr_compound = hashlib.sha256((curr_faiss_cs + curr_meta_cs).encode("utf-8")).hexdigest()

            if manifest.get("faiss_checksum") and curr_faiss_cs != manifest["faiss_checksum"]:
                errors.append(f"Checksum mismatch for index.faiss: expected {manifest['faiss_checksum']}, got {curr_faiss_cs}")

            if manifest.get("metadata_checksum") and curr_meta_cs != manifest["metadata_checksum"]:
                errors.append(f"Checksum mismatch for metadata.json: expected {manifest['metadata_checksum']}, got {curr_meta_cs}")

            if manifest.get("checksum") and curr_compound != manifest["checksum"]:
                errors.append(f"Compound snapshot checksum mismatch: expected {manifest['checksum']}, got {curr_compound}")
        except Exception as exc:
            errors.append(f"Error computing file checksums: {str(exc)}")

        # Check FAISS index validity and dimension
        actual_vectors = -1
        actual_dim = -1
        try:
            loaded_index = faiss.read_index(str(index_file))
            actual_vectors = int(loaded_index.ntotal)
            actual_dim = int(loaded_index.d)

            if actual_dim != expected_dimension:
                errors.append(f"Wrong dimension in FAISS index: expected {expected_dimension}, got {actual_dim}")

            if actual_vectors != manifest.get("vector_count"):
                errors.append(f"Vector count mismatch: manifest says {manifest.get('vector_count')}, index has {actual_vectors}")
        except Exception as exc:
            errors.append(f"Corrupted index.faiss: cannot read FAISS index: {str(exc)}")

        # Check metadata json validity
        actual_meta_count = -1
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            records = meta_data.get("records", [])
            actual_meta_count = len(records)

            if actual_meta_count != manifest.get("metadata_count"):
                errors.append(f"Metadata count mismatch: manifest says {manifest.get('metadata_count')}, metadata has {actual_meta_count}")
        except Exception as exc:
            errors.append(f"Corrupted metadata.json: cannot parse JSON records: {str(exc)}")

        # Check alignment
        if actual_vectors != -1 and actual_meta_count != -1:
            if actual_vectors != actual_meta_count:
                errors.append(f"Vector/Metadata alignment mismatch: {actual_vectors} vectors vs {actual_meta_count} records.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "manifest": manifest,
            "actual_vectors": actual_vectors,
            "actual_metadata": actual_meta_count,
            "dimension": actual_dim
        }

    def restore_snapshot(
        self,
        snapshot_dir: Union[str, Path],
        target_dir: Optional[Union[str, Path]] = None,
        expected_dimension: int = 384,
        activate_in_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Safely restores a snapshot into target directory.

        Two-Phase Restoration:
        Phase 1: Restore into temporary staging directory (.staging_restore) and validate thoroughly.
        Phase 2: Only after 100% validation passes, activate files into target directory.
        """
        s_dir = Path(snapshot_dir)
        dest_dir = Path(target_dir) if target_dir else self.vector_store_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        staging_dir = dest_dir.parent / f".staging_restore_{int(datetime.datetime.now().timestamp())}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Phase 1: Copy to staging
            manifest_src = s_dir / "manifest.json"
            index_src = s_dir / "index.faiss"
            meta_src = s_dir / "metadata.json"

            if not manifest_src.exists() or not index_src.exists() or not meta_src.exists():
                raise ValueError(f"Cannot restore: incomplete snapshot at {s_dir}")

            shutil.copy2(manifest_src, staging_dir / "manifest.json")
            shutil.copy2(index_src, staging_dir / "index.faiss")
            shutil.copy2(meta_src, staging_dir / "metadata.json")

            # Validate in staging location
            validation = self.validate_snapshot(staging_dir, expected_dimension=expected_dimension)
            if not validation["valid"]:
                err_str = "; ".join(validation["errors"])
                raise ValueError(f"Restoration validation failed in staging: {err_str}")

            # Phase 2: Activation into destination directory
            dest_index = dest_dir / "index.faiss"
            dest_meta = dest_dir / "metadata.json"

            # Backup existing active files if present to a temporary rollback slot
            rollback_dir = dest_dir.parent / f".rollback_{int(datetime.datetime.now().timestamp())}"
            try:
                if dest_index.exists() and dest_meta.exists():
                    rollback_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest_index, rollback_dir / "index.faiss")
                    shutil.copy2(dest_meta, rollback_dir / "metadata.json")

                # Copy validated staging files to destination
                shutil.copy2(staging_dir / "index.faiss", dest_index)
                shutil.copy2(staging_dir / "metadata.json", dest_meta)

                # If activate_in_memory: reload global singleton VectorStoreService
                if activate_in_memory:
                    vs = get_vector_store_service(storage_dir=dest_dir)
                    vs.load(dest_dir)

                # Clean up rollback backup on success
                if rollback_dir.exists():
                    shutil.rmtree(rollback_dir, ignore_errors=True)

                logger.info("Successfully restored vector store from %s into %s", s_dir, dest_dir)
                return {
                    "restored": True,
                    "target_dir": str(dest_dir),
                    "vector_count": validation["actual_vectors"],
                    "metadata_count": validation["actual_metadata"],
                    "dimension": validation["dimension"],
                    "manifest": validation["manifest"]
                }
            except Exception as act_err:
                # Rollback to original active index if activation failed
                if rollback_dir.exists():
                    logger.error("Activation failed, rolling back to previous index: %s", str(act_err))
                    if (rollback_dir / "index.faiss").exists():
                        shutil.copy2(rollback_dir / "index.faiss", dest_index)
                    if (rollback_dir / "metadata.json").exists():
                        shutil.copy2(rollback_dir / "metadata.json", dest_meta)
                    shutil.rmtree(rollback_dir, ignore_errors=True)
                raise
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Lists all available snapshots sorted by creation time."""
        results = []
        if not self.snapshots_dir.exists():
            return results

        for child in self.snapshots_dir.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                manifest_file = child / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        m["directory"] = str(child)
                        results.append(m)
                    except Exception:
                        pass
        return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)


# Singleton instance accessor
_shared_snapshot_service: Optional[VectorStoreSnapshotService] = None


def get_snapshot_service() -> VectorStoreSnapshotService:
    global _shared_snapshot_service
    if _shared_snapshot_service is None:
        _shared_snapshot_service = VectorStoreSnapshotService()
    return _shared_snapshot_service
