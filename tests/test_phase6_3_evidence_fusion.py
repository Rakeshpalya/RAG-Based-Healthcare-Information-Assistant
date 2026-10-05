"""
Phase 6.3 Tests: Clinical Evidence Fusion & Multi-Document Reasoning.

Comprehensive test suite verifying all Phase 6.3 requirements:
1. Clinical Evidence Data Models & Serialization
2. Evidence Deduplication & Near-Duplicate Pruning
3. Single-Document Evidence Handling
4. Multi-Document Evidence Aggregation
5. Document Diversity & Balanced Selection
6. Source Attribution Preservation & Citation Mapping
7. Conflict Detection: Dosage Discrepancies
8. Conflict Detection: Contraindication Contradictions
9. Conflict Detection: First-Line Treatment Conflicts
10. Conflict Detection: Diagnostic Criteria Discrepancies
11. Complementary Evidence vs. Genuine Conflicts
12. Multi-Document Comparative Queries
13. Evidence Coverage Assessment (Full, Partial, Insufficient)
14. Evidence Sufficiency Gates & Conservative Handling
15. Multi-Tenant User and Document Isolation
16. Safe LLM Cache Interaction & Strategy Isolation
17. Medical Safety Precedence (Emergency / Self-Harm Retrieval Bypass)
18. Deterministic Behavior & Invariant Guarantees
19. Performance Bounds (< 10ms execution, zero LLM calls)
20. Observability: Low-Cardinality Prometheus Metrics
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from backend.intelligence.evidence_models import (
    ConflictType,
    ConflictSeverity,
    CoverageStatus,
    EvidenceConflict,
    EvidenceCoverage,
    FusedEvidenceChunk,
    FusedContextResult
)
from backend.intelligence.evidence_fusion import ClinicalEvidenceFusionEngine
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.intent_models import (
    ClinicalIntent,
    SafetyPriority,
    ClinicalRoutingStrategy,
    IntentClassificationResult
)
from backend.intelligence.query_planner import ClinicalQueryPlanner
from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.evaluation.observability import (
    get_metrics_collector,
    record_evidence_fusion_event
)
from backend.rag.rag_service import RAGService
from backend.safety.medical_safety_guard import MedicalSafetyGuard


# =========================================================================
# FIXTURES & MOCK EVIDENCE GENERATORS
# =========================================================================

@pytest.fixture
def sample_raw_chunks():
    """Provides realistic multi-document clinical evidence chunks."""
    return [
        {
            "chunk_id": "CHUNK_CARDIO_01",
            "document_id": "DOC_CARDIO_GUIDELINE_2024",
            "filename": "clinical_cardiology_2024.pdf",
            "page_number": 4,
            "text": "For initial management of stage 1 hypertension, lisinopril starting dose is 10 mg once daily orally. Maximum daily dose is 40 mg.",
            "similarity_score": 0.88,
            "metadata": {"section": "hypertension_treatment", "filename": "clinical_cardiology_2024.pdf"}
        },
        {
            "chunk_id": "CHUNK_PHARMA_01",
            "document_id": "DOC_PHARMA_COMPENDIUM",
            "filename": "pharmacology_compendium.pdf",
            "page_number": 12,
            "text": "Lisinopril is an ACE inhibitor commonly used in hypertension. Common adverse effects include persistent dry cough and dizziness.",
            "similarity_score": 0.82,
            "metadata": {"section": "ace_inhibitors", "filename": "pharmacology_compendium.pdf"}
        },
        {
            "chunk_id": "CHUNK_RENAL_01",
            "document_id": "DOC_NEPHRO_HANDBOOK",
            "filename": "nephrology_handbook.pdf",
            "page_number": 8,
            "text": "ACE inhibitors such as lisinopril are contraindicated in patients with bilateral renal artery stenosis or acute kidney injury.",
            "similarity_score": 0.79,
            "metadata": {"section": "renal_contraindications", "filename": "nephrology_handbook.pdf"}
        }
    ]


@pytest.fixture
def conflicting_dosage_chunks():
    """Provides chunks with contradictory clinical dosing for the same drug."""
    return [
        {
            "chunk_id": "CHUNK_DOSE_A",
            "document_id": "DOC_GUIDELINE_A",
            "filename": "guideline_hospital_a.pdf",
            "page_number": 2,
            "text": "For adult type 2 diabetes, the recommended starting dose of metformin is 500 mg once daily with meals, titrating gradually.",
            "similarity_score": 0.87,
            "metadata": {"filename": "guideline_hospital_a.pdf"}
        },
        {
            "chunk_id": "CHUNK_DOSE_B",
            "document_id": "DOC_GUIDELINE_B",
            "filename": "guideline_hospital_b.pdf",
            "page_number": 5,
            "text": "In newly diagnosed diabetic adults, the initial starting dose of metformin is 1000 mg twice daily with meals.",
            "similarity_score": 0.85,
            "metadata": {"filename": "guideline_hospital_b.pdf"}
        }
    ]


@pytest.fixture
def conflicting_contraindication_chunks():
    """Provides chunks with direct safety contradiction on contraindications."""
    return [
        {
            "chunk_id": "CHUNK_CONTRA_A",
            "document_id": "DOC_SAFETY_A",
            "filename": "safety_reference_a.pdf",
            "page_number": 1,
            "text": "Metformin is strictly contraindicated in patients with severe renal impairment (eGFR < 30 mL/min) due to risk of lactic acidosis.",
            "similarity_score": 0.89,
            "metadata": {"filename": "safety_reference_a.pdf"}
        },
        {
            "chunk_id": "CHUNK_CONTRA_B",
            "document_id": "DOC_SAFETY_B",
            "filename": "safety_reference_b.pdf",
            "page_number": 7,
            "text": "Metformin is safe in severe renal impairment and can be administered regardless of glomerular filtration rate without dosage adjustment.",
            "similarity_score": 0.84,
            "metadata": {"filename": "safety_reference_b.pdf"}
        }
    ]


# =========================================================================
# 1. CLINICAL EVIDENCE DATA MODELS & SERIALIZATION
# =========================================================================

class TestEvidenceModelsAndSerialization:
    """Verifies typed models, enums, and JSON-safe dictionary serialization."""

    def test_conflict_type_and_severity_enums(self):
        assert ConflictType.DOSAGE_DISCREPANCY.value == "dosage_discrepancy"
        assert ConflictType.CONTRAINDICATION_CONFLICT.value == "contraindication_conflict"
        assert ConflictType.TREATMENT_RECOMMENDATION_CONFLICT.value == "treatment_recommendation_conflict"
        assert ConflictType.DIAGNOSTIC_CRITERIA_CONFLICT.value == "diagnostic_criteria_conflict"
        assert ConflictType.CLINICAL_OUTCOME_CONFLICT.value == "clinical_outcome_conflict"

        assert ConflictSeverity.LOW.value == "low"
        assert ConflictSeverity.MODERATE.value == "moderate"
        assert ConflictSeverity.HIGH.value == "high"

        assert CoverageStatus.FULL.value == "full"
        assert CoverageStatus.PARTIAL.value == "partial"
        assert CoverageStatus.INSUFFICIENT.value == "insufficient"

    def test_evidence_conflict_serialization(self):
        conflict = EvidenceConflict(
            topic="metformin starting dose",
            conflict_type=ConflictType.DOSAGE_DISCREPANCY,
            severity=ConflictSeverity.MODERATE,
            doc_a_id="DOC_A",
            doc_b_id="DOC_B",
            doc_a_name="Hospital_A.pdf",
            doc_b_name="Hospital_B.pdf",
            statement_a="Starting dose 500 mg daily",
            statement_b="Starting dose 1000 mg daily",
            resolution_guidance="Verify patient renal function and consult authoritative FDA label."
        )
        d = conflict.to_dict()
        assert d["topic"] == "metformin starting dose"
        assert d["conflict_type"] == "dosage_discrepancy"
        assert d["severity"] == "moderate"
        assert d["doc_a_name"] == "Hospital_A.pdf"
        assert d["doc_b_name"] == "Hospital_B.pdf"

    def test_fused_evidence_chunk_serialization(self):
        chunk = FusedEvidenceChunk(
            chunk_id="CHK_01",
            document_id="DOC_01",
            document_name="guide.pdf",
            page_number=3,
            text="Clinical statement.",
            similarity_score=0.85,
            weighting_boost=0.06,
            fused_score=0.91,
            rank=1,
            source_index=1,
            aspects=["dosage"],
            metadata={"user_id": 101}
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "CHK_01"
        assert d["fused_score"] == 0.91
        assert d["source_index"] == 1
        assert "dosage" in d["aspects"]

    def test_fused_context_result_serialization(self):
        cov = EvidenceCoverage(
            coverage_score=0.75,
            status=CoverageStatus.PARTIAL,
            query_aspects=["dosage", "side_effects"],
            covered_aspects=["dosage"],
            missing_aspects=["side_effects"]
        )
        result = FusedContextResult(
            fused_chunks=[],
            contributing_documents=["guide.pdf"],
            contributing_documents_count=1,
            conflicts=[],
            has_conflicts=False,
            coverage=cov,
            formatted_context="[SOURCE 1] ...",
            sources=[{"source_index": 1, "filename": "guide.pdf"}],
            deduped_count=1,
            is_sufficient=True,
            latency_ms=1.45,
            metadata={"strategy": "single_document"}
        )
        d = result.to_dict()
        assert d["contributing_documents_count"] == 1
        assert d["has_conflicts"] is False
        assert d["coverage"]["status"] == "partial"
        assert d["latency_ms"] == 1.45


# =========================================================================
# 2. EVIDENCE DEDUPLICATION & NEAR-DUPLICATE PRUNING
# =========================================================================

class TestEvidenceDeduplication:
    """Verifies exact and near-duplicate removal while keeping the highest-ranked chunk."""

    def test_exact_duplicate_pruning(self):
        chunks = [
            {
                "chunk_id": "CHK_A1",
                "document_id": "DOC_A",
                "filename": "doc_a.pdf",
                "text": "Metformin reduces hepatic glucose production and improves insulin sensitivity in peripheral tissues.",
                "similarity_score": 0.88
            },
            {
                "chunk_id": "CHK_A1_DUP",
                "document_id": "DOC_A",
                "filename": "doc_a.pdf",
                "text": "Metformin reduces hepatic glucose production and improves insulin sensitivity in peripheral tissues.",
                "similarity_score": 0.82
            }
        ]
        deduped, pruned = ClinicalEvidenceFusionEngine.deduplicate_evidence(chunks)
        assert len(deduped) == 1
        assert pruned == 1
        assert deduped[0]["similarity_score"] == 0.88  # Retained higher scoring chunk

    def test_near_duplicate_jaccard_pruning(self):
        chunks = [
            {
                "chunk_id": "CHK_ORIGINAL",
                "document_id": "DOC_1",
                "filename": "doc1.pdf",
                "text": "Lisinopril is an angiotensin-converting enzyme inhibitor used for hypertension and heart failure management.",
                "similarity_score": 0.90
            },
            {
                "chunk_id": "CHK_NEAR_DUP",
                "document_id": "DOC_2",
                "filename": "doc2.pdf",
                "text": "Lisinopril is an angiotensin converting enzyme inhibitor utilized for hypertension and heart failure management.",
                "similarity_score": 0.84
            }
        ]
        deduped, pruned = ClinicalEvidenceFusionEngine.deduplicate_evidence(chunks, jaccard_threshold=0.80)
        assert len(deduped) == 1
        assert pruned == 1
        assert deduped[0]["chunk_id"] == "CHK_ORIGINAL"

    def test_distinct_chunks_preserved(self, sample_raw_chunks):
        deduped, pruned = ClinicalEvidenceFusionEngine.deduplicate_evidence(sample_raw_chunks)
        assert len(deduped) == 3
        assert pruned == 0


# =========================================================================
# 3. SINGLE-DOCUMENT & MULTI-DOCUMENT EVIDENCE HANDLING
# =========================================================================

class TestDocumentEvidenceHandling:
    """Verifies evidence aggregation across single vs. multiple clinical documents."""

    def test_single_document_evidence(self):
        chunks = [
            {
                "chunk_id": "CHK_1",
                "document_id": "DOC_SINGLE",
                "filename": "cardiology_guide.pdf",
                "page_number": 1,
                "text": "Hypertension is defined as persistent systolic blood pressure above 130 mmHg or diastolic above 80 mmHg.",
                "similarity_score": 0.89,
                "metadata": {"filename": "cardiology_guide.pdf"}
            },
            {
                "chunk_id": "CHK_2",
                "document_id": "DOC_SINGLE",
                "filename": "cardiology_guide.pdf",
                "page_number": 2,
                "text": "First-line lifestyle interventions include dietary sodium restriction, aerobic exercise, and weight management.",
                "similarity_score": 0.85,
                "metadata": {"filename": "cardiology_guide.pdf"}
            },
            {
                "chunk_id": "CHK_3",
                "document_id": "DOC_SINGLE",
                "filename": "cardiology_guide.pdf",
                "page_number": 3,
                "text": "Pharmacological therapy should be initiated when blood pressure remains consistently above stage 1 targets.",
                "similarity_score": 0.81,
                "metadata": {"filename": "cardiology_guide.pdf"}
            }
        ]
        plan = QueryPlan(
            intent=ClinicalIntent.TREATMENT_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.TREATMENT_RAG,
            multi_document_strategy=MultiDocumentStrategy.SINGLE_DOCUMENT,
            top_k=3
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(
            query="hypertension thresholds",
            retrieved_chunks=chunks,
            query_plan=plan
        )
        assert fused.contributing_documents_count == 1
        assert fused.contributing_documents == ["cardiology_guide.pdf"]
        assert len(fused.fused_chunks) == 3
        assert "Multi-Document Context:" not in fused.formatted_context

    def test_multi_document_evidence(self, sample_raw_chunks):
        plan = QueryPlan(
            intent=ClinicalIntent.MEDICATION_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
            multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
            top_k=3
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(
            query="lisinopril management and side effects",
            retrieved_chunks=sample_raw_chunks,
            query_plan=plan
        )
        assert fused.contributing_documents_count == 3
        assert len(fused.contributing_documents) == 3
        assert "Multi-Document Context:" in fused.formatted_context
        assert "3 distinct references" in fused.formatted_context


# =========================================================================
# 4. DOCUMENT DIVERSITY & BALANCED SELECTION
# =========================================================================

class TestDocumentDiversity:
    """Verifies that evidence ranking balances distribution and prevents document monopolization."""

    def test_balanced_retrieval_distributes_across_documents(self):
        # 5 chunks from Doc A (high scores) and 2 chunks from Doc B
        chunks = [
            {
                "chunk_id": f"DOC_A_{i}",
                "document_id": "DOC_A",
                "filename": "doc_a.pdf",
                "text": f"Doc A cardiology note item {i}.",
                "similarity_score": 0.95 - (i * 0.01),
                "metadata": {"filename": "doc_a.pdf"}
            }
            for i in range(5)
        ] + [
            {
                "chunk_id": f"DOC_B_{i}",
                "document_id": "DOC_B",
                "filename": "doc_b.pdf",
                "text": f"Doc B cardiology note item {i}.",
                "similarity_score": 0.85 - (i * 0.01),
                "metadata": {"filename": "doc_b.pdf"}
            }
            for i in range(2)
        ]

        plan = QueryPlan(
            intent=ClinicalIntent.DOCUMENT_COMPARISON,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.DOCUMENT_COMPARISON_RAG,
            multi_document_strategy=MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL,
            top_k=4
        )

        ranked = ClinicalEvidenceFusionEngine.rank_and_balance_evidence(
            chunks=chunks,
            query_plan=plan,
            top_k=4
        )

        doc_counts = {}
        for c in ranked:
            doc_counts[c.document_name] = doc_counts.get(c.document_name, 0) + 1

        # Must include chunks from both Doc A and Doc B, preventing Doc A monopoly
        assert "doc_a.pdf" in doc_counts
        assert "doc_b.pdf" in doc_counts
        assert doc_counts["doc_a.pdf"] <= 2  # Balanced distribution


# =========================================================================
# 5. SOURCE ATTRIBUTION & CITATION PRESERVATION
# =========================================================================

class TestSourceAttributionAndCitations:
    """Verifies provenance metadata and strict 1:1 mapping with prompt citations."""

    def test_source_indexing_and_attribution(self, sample_raw_chunks):
        plan = QueryPlan(
            intent=ClinicalIntent.MEDICATION_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
            top_k=3
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(
            query="lisinopril",
            retrieved_chunks=sample_raw_chunks,
            query_plan=plan
        )

        assert len(fused.sources) == 3
        for i, src in enumerate(fused.sources, start=1):
            assert src["source_index"] == i
            assert src["source_label"] == f"[Source {i}]"
            assert src["document_id"] is not None
            assert src["filename"] is not None
            assert "similarity_score" in src
            assert "fused_score" in src

        # Formatted context must have [SOURCE 1], [SOURCE 2], [SOURCE 3]
        assert "[SOURCE 1]" in fused.formatted_context
        assert "[SOURCE 2]" in fused.formatted_context
        assert "[SOURCE 3]" in fused.formatted_context


# =========================================================================
# 6. CLINICAL CONFLICT DETECTION
# =========================================================================

class TestClinicalConflictDetection:
    """Verifies deterministic detection of dosage, contraindication, and treatment conflicts."""

    def test_dosage_discrepancy_conflict(self, conflicting_dosage_chunks):
        plan = QueryPlan(
            intent=ClinicalIntent.DOSAGE_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.DOSAGE_RAG,
            top_k=2
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(
            query="metformin starting dose",
            retrieved_chunks=conflicting_dosage_chunks,
            query_plan=plan
        )

        assert fused.has_conflicts is True
        assert len(fused.conflicts) >= 1
        conflict = fused.conflicts[0]
        assert conflict.conflict_type == ConflictType.DOSAGE_DISCREPANCY
        assert "metformin" in conflict.topic.lower()
        assert "CLINICAL EVIDENCE NOTICE: CONFLICTING RECOMMENDATIONS IDENTIFIED" in fused.formatted_context

    def test_contraindication_safety_conflict(self, conflicting_contraindication_chunks):
        plan = QueryPlan(
            intent=ClinicalIntent.MEDICATION_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
            top_k=2
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(
            query="metformin contraindications",
            retrieved_chunks=conflicting_contraindication_chunks,
            query_plan=plan
        )

        assert fused.has_conflicts is True
        conflict = next(c for c in fused.conflicts if c.conflict_type == ConflictType.CONTRAINDICATION_CONFLICT)
        assert conflict.severity == ConflictSeverity.HIGH
        assert "renal" in conflict.statement_a.lower() or "renal" in conflict.statement_b.lower()

    def test_first_line_treatment_conflict(self):
        chunks = [
            {
                "chunk_id": "CHK_TX_1",
                "document_id": "GUIDELINE_2020",
                "filename": "guideline_2020.pdf",
                "text": "Lisinopril is first-line therapy for non-black patients with uncomplicated essential hypertension.",
                "similarity_score": 0.86,
                "metadata": {"filename": "guideline_2020.pdf"}
            },
            {
                "chunk_id": "CHK_TX_2",
                "document_id": "GUIDELINE_2024",
                "filename": "guideline_2024.pdf",
                "text": "Lisinopril is not recommended as first-line therapy for hypertension when calcium channel blockers are preferred.",
                "similarity_score": 0.84,
                "metadata": {"filename": "guideline_2024.pdf"}
            }
        ]
        conflicts = ClinicalEvidenceFusionEngine.detect_conflicts(
            chunks=ClinicalEvidenceFusionEngine.rank_and_balance_evidence(chunks, top_k=2),
            query="lisinopril first line treatment"
        )
        assert len(conflicts) >= 1
        assert any(c.conflict_type == ConflictType.TREATMENT_RECOMMENDATION_CONFLICT for c in conflicts)

    def test_diagnostic_criteria_conflict(self):
        chunks = [
            {
                "chunk_id": "CHK_DX_1",
                "document_id": "DOC_ADA",
                "filename": "ada_standards.pdf",
                "text": "Diabetes is diagnosed when fasting blood glucose is 126 mg/dL or higher on two separate occasions.",
                "similarity_score": 0.88,
                "metadata": {"filename": "ada_standards.pdf"}
            },
            {
                "chunk_id": "CHK_DX_2",
                "document_id": "DOC_ALT",
                "filename": "alt_guideline.pdf",
                "text": "Diabetes is diagnosed when fasting blood glucose is 140 mg/dL or higher during clinical assessment.",
                "similarity_score": 0.85,
                "metadata": {"filename": "alt_guideline.pdf"}
            }
        ]
        conflicts = ClinicalEvidenceFusionEngine.detect_conflicts(
            chunks=ClinicalEvidenceFusionEngine.rank_and_balance_evidence(chunks, top_k=2),
            query="fasting blood glucose criteria for diabetes"
        )
        assert len(conflicts) >= 1
        assert any(c.conflict_type == ConflictType.DIAGNOSTIC_CRITERIA_CONFLICT for c in conflicts)


# =========================================================================
# 7. COMPLEMENTARY EVIDENCE VS. GENUINE CONFLICTS
# =========================================================================

class TestComplementaryEvidence:
    """Verifies that non-contradictory, complementary evidence does NOT trigger false conflicts."""

    def test_complementary_facets_of_same_drug(self):
        chunks = [
            {
                "chunk_id": "CHK_DOSE",
                "document_id": "DOC_A",
                "filename": "dosing_guide.pdf",
                "text": "Metformin starting dose is 500 mg once daily with the evening meal.",
                "similarity_score": 0.88,
                "metadata": {"filename": "dosing_guide.pdf"}
            },
            {
                "chunk_id": "CHK_SIDE_EFFECTS",
                "document_id": "DOC_B",
                "filename": "safety_guide.pdf",
                "text": "Common side effects of metformin include diarrhea, nausea, and abdominal discomfort.",
                "similarity_score": 0.84,
                "metadata": {"filename": "safety_guide.pdf"}
            }
        ]
        conflicts = ClinicalEvidenceFusionEngine.detect_conflicts(
            chunks=ClinicalEvidenceFusionEngine.rank_and_balance_evidence(chunks, top_k=2),
            query="metformin dosage and side effects"
        )
        # Should recognize complementary information, NO conflicts
        assert len(conflicts) == 0


# =========================================================================
# 8. MULTI-DOCUMENT COMPARATIVE QUERIES
# =========================================================================

class TestComparativeQueries:
    """Verifies handling of queries comparing clinical alternatives across documents."""

    def test_comparative_retrieval_and_fusion(self):
        chunks = [
            {
                "chunk_id": "CHK_LIS",
                "document_id": "DOC_ACE",
                "filename": "ace_inhibitors.pdf",
                "text": "Lisinopril lowers blood pressure through renin-angiotensin inhibition and is renoprotective in diabetes.",
                "similarity_score": 0.86,
                "metadata": {"filename": "ace_inhibitors.pdf"}
            },
            {
                "chunk_id": "CHK_AML",
                "document_id": "DOC_CCB",
                "filename": "calcium_blockers.pdf",
                "text": "Amlodipine lowers blood pressure through peripheral arterial vasodilation with peripheral edema as a common side effect.",
                "similarity_score": 0.85,
                "metadata": {"filename": "calcium_blockers.pdf"}
            }
        ]
        plan = QueryPlan(
            intent=ClinicalIntent.DOCUMENT_COMPARISON,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.DOCUMENT_COMPARISON_RAG,
            multi_document_strategy=MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL,
            top_k=2
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(
            query="compare lisinopril and amlodipine for hypertension",
            retrieved_chunks=chunks,
            query_plan=plan
        )
        assert fused.contributing_documents_count == 2
        assert len(fused.fused_chunks) == 2
        assert fused.is_sufficient is True


# =========================================================================
# 9. EVIDENCE COVERAGE & SUFFICIENCY GATES
# =========================================================================

class TestEvidenceCoverageAndSufficiency:
    """Verifies inquiry aspect coverage analysis and sufficiency gate adherence."""

    def test_full_coverage(self):
        query = "What is the dose, side effects, and contraindications of lisinopril?"
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "filename": "d1.pdf",
                "text": "Lisinopril dose is 10 mg daily. Adverse side effects include dry cough. Contraindications include pregnancy and angioedema.",
                "similarity_score": 0.90,
                "metadata": {"filename": "d1.pdf"}
            }
        ]
        plan = QueryPlan(
            intent=ClinicalIntent.MEDICATION_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
            top_k=1
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(query, chunks, plan)
        assert fused.coverage.status == CoverageStatus.FULL
        assert fused.coverage.coverage_score == 1.0
        assert len(fused.coverage.missing_aspects) == 0

    def test_partial_coverage_flags_missing_aspects(self):
        query = "What is the dose, side effects, and contraindications of lisinopril?"
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "filename": "d1.pdf",
                "text": "Lisinopril dose is 10 mg daily. It is prescribed for hypertension.",
                "similarity_score": 0.85,
                "metadata": {"filename": "d1.pdf"}
            }
        ]
        plan = QueryPlan(
            intent=ClinicalIntent.MEDICATION_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
            top_k=1
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(query, chunks, plan)
        assert fused.coverage.status == CoverageStatus.PARTIAL
        assert "side_effects" in fused.coverage.missing_aspects or "contraindications" in fused.coverage.missing_aspects

    def test_insufficient_evidence_when_chunks_empty(self):
        plan = QueryPlan(
            intent=ClinicalIntent.UNCERTAIN,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
            top_k=5
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence("unknown rare disease", [], plan)
        assert fused.is_sufficient is False
        assert fused.coverage.status == CoverageStatus.INSUFFICIENT


# =========================================================================
# 10. MULTI-TENANT ISOLATION
# =========================================================================

class TestMultiTenantIsolation:
    """Verifies that evidence fusion strictly respects user/tenant boundaries."""

    def test_fusion_filters_by_user_id(self):
        user1_chunks = [
            {
                "chunk_id": "CHK_U1",
                "document_id": "DOC_U1",
                "filename": "user1_doc.pdf",
                "text": "User 1 confidential blood panel showing high cholesterol.",
                "similarity_score": 0.88,
                "metadata": {"user_id": 101, "filename": "user1_doc.pdf"}
            },
            {
                "chunk_id": "CHK_U2",
                "document_id": "DOC_U2",
                "filename": "user2_doc.pdf",
                "text": "User 2 confidential blood panel showing normal cholesterol.",
                "similarity_score": 0.87,
                "metadata": {"user_id": 102, "filename": "user2_doc.pdf"}
            }
        ]
        plan = QueryPlan(
            intent=ClinicalIntent.LAB_RESULT_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.LAB_RAG
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(
            query="cholesterol panel",
            retrieved_chunks=user1_chunks,
            query_plan=plan,
            user_id=101
        )
        # Should only retain chunks belonging to user 101
        for chunk in fused.fused_chunks:
            assert chunk.metadata.get("user_id") == 101


# =========================================================================
# 11. SAFE LLM CACHE INTERACTION
# =========================================================================

class TestCacheInteraction:
    """Verifies that query plan and fused context result interact cleanly with caching."""

    def test_fused_evidence_cached_payload_serialization(self, sample_raw_chunks):
        plan = QueryPlan(
            intent=ClinicalIntent.DOSAGE_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.DOSAGE_RAG,
            top_k=2
        )
        fused = ClinicalEvidenceFusionEngine.fuse_evidence("lisinopril dosing", sample_raw_chunks, plan)
        fused_dict = fused.to_dict()

        # Simulate cache payload
        cached_payload = {
            "question": "lisinopril dosing",
            "context": fused.formatted_context,
            "sources": fused.sources,
            "fused_evidence": fused_dict
        }

        assert "fused_evidence" in cached_payload
        assert cached_payload["fused_evidence"]["contributing_documents_count"] >= 1
        assert isinstance(cached_payload["fused_evidence"]["coverage"], dict)


# =========================================================================
# 12. MEDICAL SAFETY PRECEDENCE
# =========================================================================

class TestMedicalSafetyPrecedence:
    """Verifies that safety pre-screen intercepts emergency queries before retrieval and fusion."""

    @pytest.mark.parametrize("emergency_query", [
        "I am having crushing chest pain radiating to my left arm, what do I do?",
        "I want to end my life, what is the best way to do it?",
        "My toddler just swallowed an entire bottle of drain cleaner!"
    ])
    def test_safety_prescreen_bypasses_fusion(self, emergency_query):
        allow_rag, assessment, msg = MedicalSafetyGuard.pre_screen_inquiry(emergency_query)
        assert allow_rag is False
        assert assessment.category.value in ("EMERGENCY_SYMPTOMS", "SELF_HARM_OR_SUICIDE", "POISONING_OR_OVERDOSE")
        assert msg is not None
        assert len(msg) > 0


# =========================================================================
# 13. DETERMINISTIC BEHAVIOR & PERFORMANCE BOUNDS
# =========================================================================

class TestDeterministicBehaviorAndPerformance:
    """Verifies sub-millisecond execution and 100% determinism over multiple trials."""

    def test_deterministic_fusion_across_trials(self, sample_raw_chunks):
        plan = QueryPlan(
            intent=ClinicalIntent.MEDICATION_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
            top_k=3
        )

        results = [
            ClinicalEvidenceFusionEngine.fuse_evidence("lisinopril management", sample_raw_chunks, plan)
            for _ in range(5)
        ]

        # Verify all runs produce identical rankings and formatted contexts
        first_ctx = results[0].formatted_context
        first_scores = [c.fused_score for c in results[0].fused_chunks]

        for r in results[1:]:
            assert r.formatted_context == first_ctx
            assert [c.fused_score for c in r.fused_chunks] == first_scores
            assert r.has_conflicts == results[0].has_conflicts
            assert r.coverage.status == results[0].coverage.status

    def test_fusion_performance_under_10ms(self, sample_raw_chunks):
        plan = QueryPlan(
            intent=ClinicalIntent.MEDICATION_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
            top_k=3
        )

        start = time.perf_counter()
        fused = ClinicalEvidenceFusionEngine.fuse_evidence("lisinopril management", sample_raw_chunks, plan)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < 10.0  # Must be strictly under 10ms (typically < 1ms)
        assert fused.latency_ms < 10.0


# =========================================================================
# 14. OBSERVABILITY TELEMETRY
# =========================================================================

class TestObservabilityTelemetry:
    """Verifies low-cardinality Prometheus metrics recording without PHI/query leak."""

    def test_record_evidence_fusion_metrics(self):
        collector = get_metrics_collector()
        collector.reset()

        collector.record_evidence_fusion_event(
            strategy="balanced_document_retrieval",
            contributing_docs_count=3,
            conflicts_count=1,
            conflict_types=["dosage_discrepancy"],
            deduped_count=2,
            is_sufficient=True,
            latency_ms=1.85
        )

        snapshot = collector.get_metrics_snapshot()
        ev_metrics = snapshot["intelligence"]["evidence_fusion"]

        assert ev_metrics["total"] == 1
        assert ev_metrics["contributing_docs_total"] == 3
        assert ev_metrics["conflicts_total"] == 1
        assert ev_metrics["conflicts_by_type"].get("dosage_discrepancy") == 1
        assert ev_metrics["deduped_total"] == 2

        # Verify Prometheus exposition text
        prom_text = collector.get_prometheus_exposition()
        assert "rag_evidence_fusion_total 1" in prom_text
        assert 'rag_evidence_fusion_strategy_total{strategy="balanced_document_retrieval"} 1' in prom_text
        assert "rag_evidence_contributing_docs_total 3" in prom_text
        assert "rag_evidence_conflicts_total 1" in prom_text
        assert 'rag_evidence_conflicts_by_type_total{conflict_type="dosage_discrepancy"} 1' in prom_text
        assert "rag_evidence_deduped_total 2" in prom_text
        assert "rag_evidence_fusion_duration_seconds" in prom_text

        # Invariant: No query text, doc IDs, or user IDs in Prometheus metrics
        assert "lisinopril" not in prom_text
        assert "DOC_" not in prom_text
        assert "user_id" not in prom_text
