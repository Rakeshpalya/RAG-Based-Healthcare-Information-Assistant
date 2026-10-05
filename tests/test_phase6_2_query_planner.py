"""
Phase 6.2 Tests: Clinical Query Planning & Adaptive Retrieval Strategies.

Verifies all 20 required specifications:
1. Valid plan generated for every intent type
2. Emergency, self-harm, and poisoning queries result in retrieval being completely disabled
3. Dosage and diagnosis queries require high-confidence evidence
4. Medication queries use MEDICATION_PRIORITY weighting
5. Lab queries use LAB_PRIORITY weighting
6. Document summary queries use MAP_REDUCE multi-document strategy
7. Document comparison queries use BALANCED_DOCUMENT_RETRIEVAL strategy
8. Out-of-scope queries bypass retrieval
9. Low-confidence/uncertain intents produce conservative plans
10. Query expansion produces bounded, deterministic output
11. Query expansion handles edge cases (empty strings, special chars, numbers)
12. Multi-tenant isolation: different users/tenants receive independent, uncorrupted plans
13. Cache isolation: plans with different strategies don't collide in cache
14. End-to-end integration: query planning works with RAGService
15. Retrieval parameters in plan are respected by RAGService
16. Relevance verification uses plan-specified thresholds and high-confidence gating
17. Chunk weighting actually affects chunk ordering
18. Multi-document evidence selection respects document limits and distribution
19. Performance: query planning completes in < 5ms (and expansion in < 2ms)
20. Observability: query planning records low-cardinality Prometheus metrics without error
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from backend.intelligence.intent_models import (
    ClinicalIntent,
    SafetyPriority,
    ClinicalRoutingStrategy,
    IntentClassificationResult
)
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.query_planner import ClinicalQueryPlanner
from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.evaluation.observability import get_metrics_collector
from backend.services.llm_cache_service import generate_cache_key
from backend.rag.rag_service import (
    RAGService,
    apply_chunk_weighting,
    select_multi_document_evidence
)


# =========================================================================
# 1. PLAN DATA MODEL & SERIALIZATION
# =========================================================================

class TestQueryPlanModelAndSerialization:
    """Verifies QueryPlan data model integrity, defaults, and dictionary serialization."""

    def test_plan_instantiation_and_to_dict(self):
        plan = QueryPlan(
            intent=ClinicalIntent.MEDICATION_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
            top_k=5,
            similarity_threshold=0.25,
            max_chunks=7,
            chunk_weighting_strategy=ChunkWeightingStrategy.MEDICATION_PRIORITY,
            document_filter_strategy=DocumentFilterStrategy.ALL_AVAILABLE,
            query_expansion_enabled=True,
            query_expansions=["metformin side effects", "metformin adverse effects"],
            multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
            context_budget=3500,
            requires_high_confidence_evidence=False,
            safety_priority=SafetyPriority.NORMAL,
            generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
            scoped_document_name=None,
            latency_ms=1.234
        )

        d = plan.to_dict()
        assert d["intent"] == "MEDICATION_QUERY"
        assert d["retrieval_required"] is True
        assert d["retrieval_strategy"] == "medication_rag"
        assert d["top_k"] == 5
        assert d["similarity_threshold"] == 0.25
        assert d["max_chunks"] == 7
        assert d["chunk_weighting_strategy"] == "medication_priority"
        assert d["document_filter_strategy"] == "all_available"
        assert d["query_expansion_enabled"] is True
        assert len(d["query_expansions"]) == 2
        assert d["multi_document_strategy"] == "multi_document"
        assert d["context_budget"] == 3500
        assert d["requires_high_confidence_evidence"] is False
        assert d["safety_priority"] == "normal"
        assert d["generation_strategy"] == "standard_grounded"
        assert d["scoped_document_name"] is None
        assert d["latency_ms"] == 1.234


# =========================================================================
# 2. VALID PLAN GENERATED FOR EVERY INTENT TYPE (REQ 1, 4, 5, 6, 7)
# =========================================================================

class TestIntentToPlanMapping:
    """Verifies that every intent in the taxonomy produces a valid, specialized plan."""

    @pytest.mark.parametrize("intent,expected_strat,expected_weight,expected_multi", [
        (ClinicalIntent.MEDICATION_QUERY, ClinicalRoutingStrategy.MEDICATION_RAG, ChunkWeightingStrategy.MEDICATION_PRIORITY, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.DOSAGE_QUERY, ClinicalRoutingStrategy.DOSAGE_RAG, ChunkWeightingStrategy.DOSAGE_PRIORITY, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.LAB_RESULT_QUERY, ClinicalRoutingStrategy.LAB_RAG, ChunkWeightingStrategy.LAB_PRIORITY, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.DIAGNOSIS_QUERY, ClinicalRoutingStrategy.DIAGNOSIS_RAG, ChunkWeightingStrategy.DIAGNOSTIC_PRIORITY, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.TREATMENT_QUERY, ClinicalRoutingStrategy.TREATMENT_RAG, ChunkWeightingStrategy.TREATMENT_PRIORITY, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.PREVENTION_QUERY, ClinicalRoutingStrategy.PREVENTION_RAG, ChunkWeightingStrategy.PREVENTION_PRIORITY, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.SYMPTOM_QUERY, ClinicalRoutingStrategy.SYMPTOM_RAG, ChunkWeightingStrategy.SYMPTOM_PRIORITY, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.DOCUMENT_SUMMARY, ClinicalRoutingStrategy.DOCUMENT_SUMMARY_RAG, ChunkWeightingStrategy.BROAD_COVERAGE, MultiDocumentStrategy.MAP_REDUCE),
        (ClinicalIntent.DOCUMENT_COMPARISON, ClinicalRoutingStrategy.DOCUMENT_COMPARISON_RAG, ChunkWeightingStrategy.BALANCED_MULTI_DOCUMENT, MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL),
        (ClinicalIntent.GENERAL_HEALTH, ClinicalRoutingStrategy.GENERAL_HEALTH_RAG, ChunkWeightingStrategy.STANDARD, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.UNCERTAIN, ClinicalRoutingStrategy.STANDARD_RAG, ChunkWeightingStrategy.STANDARD, MultiDocumentStrategy.MULTI_DOCUMENT),
        (ClinicalIntent.OUT_OF_SCOPE, ClinicalRoutingStrategy.OUT_OF_SCOPE_RESPONSE, ChunkWeightingStrategy.STANDARD, MultiDocumentStrategy.SINGLE_DOCUMENT),
        (ClinicalIntent.EMERGENCY, ClinicalRoutingStrategy.EMERGENCY_SAFETY, ChunkWeightingStrategy.STANDARD, MultiDocumentStrategy.SINGLE_DOCUMENT),
        (ClinicalIntent.SELF_HARM, ClinicalRoutingStrategy.SELF_HARM_SAFETY, ChunkWeightingStrategy.STANDARD, MultiDocumentStrategy.SINGLE_DOCUMENT),
        (ClinicalIntent.POISONING, ClinicalRoutingStrategy.POISONING_SAFETY, ChunkWeightingStrategy.STANDARD, MultiDocumentStrategy.SINGLE_DOCUMENT),
    ])
    def test_plan_generated_for_each_intent(self, intent, expected_strat, expected_weight, expected_multi):
        intent_res = IntentClassificationResult(
            intent=intent,
            confidence=0.85,
            safety_priority=SafetyPriority.NORMAL,
            requires_retrieval=intent not in (ClinicalIntent.EMERGENCY, ClinicalIntent.SELF_HARM, ClinicalIntent.POISONING, ClinicalIntent.OUT_OF_SCOPE),
            routing_strategy=expected_strat
        )
        plan = ClinicalQueryPlanner.plan("Sample clinical question about health", intent_res)

        assert isinstance(plan, QueryPlan)
        assert plan.intent == intent
        assert plan.retrieval_strategy == expected_strat
        assert plan.chunk_weighting_strategy == expected_weight
        assert plan.multi_document_strategy == expected_multi

    def test_medication_query_specifics(self):
        """Req 4: Medication queries use MEDICATION_PRIORITY weighting."""
        intent_res = ClinicalIntentClassifier.classify("What are the side effects and contraindications of metformin?")
        plan = ClinicalQueryPlanner.plan("What are the side effects and contraindications of metformin?", intent_res)
        assert plan.chunk_weighting_strategy == ChunkWeightingStrategy.MEDICATION_PRIORITY
        assert plan.retrieval_required is True
        assert plan.top_k == 5
        assert plan.max_chunks == 7

    def test_lab_query_specifics(self):
        """Req 5: Lab queries use LAB_PRIORITY weighting."""
        intent_res = ClinicalIntentClassifier.classify("What does an elevated HbA1c test result of 8.5 mean?")
        plan = ClinicalQueryPlanner.plan("What does an elevated HbA1c test result of 8.5 mean?", intent_res)
        assert plan.chunk_weighting_strategy == ChunkWeightingStrategy.LAB_PRIORITY
        assert plan.retrieval_required is True
        assert plan.top_k == 6

    def test_document_summary_specifics(self):
        """Req 6: Document summary queries use MAP_REDUCE multi-document strategy."""
        intent_res = ClinicalIntentClassifier.classify("Can you summarize the attached clinical discharge summary document?")
        plan = ClinicalQueryPlanner.plan("Can you summarize the attached clinical discharge summary document?", intent_res)
        assert plan.multi_document_strategy == MultiDocumentStrategy.MAP_REDUCE
        assert plan.chunk_weighting_strategy == ChunkWeightingStrategy.BROAD_COVERAGE
        assert plan.generation_strategy == GenerationStrategy.DOCUMENT_SUMMARY_SYNTHESIS
        assert plan.top_k == 8
        assert plan.context_budget >= 4000

    def test_document_comparison_specifics(self):
        """Req 7: Document comparison queries use BALANCED_DOCUMENT_RETRIEVAL strategy."""
        query = "Compare these two medical reports and highlight the difference between both studies."
        intent_res = ClinicalIntentClassifier.classify(query)
        plan = ClinicalQueryPlanner.plan(query, intent_res)
        assert plan.multi_document_strategy == MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL
        assert plan.chunk_weighting_strategy == ChunkWeightingStrategy.BALANCED_MULTI_DOCUMENT
        assert plan.generation_strategy == GenerationStrategy.COMPARATIVE_SYNTHESIS
        assert plan.top_k == 8


# =========================================================================
# 3. SAFETY AND SCOPE OVERRIDES (REQ 2, 8)
# =========================================================================

class TestSafetyAndScopeOverrides:
    """Verifies absolute safety precedence and out-of-scope retrieval bypass."""

    def test_emergency_query_disables_retrieval(self):
        """Req 2: Emergency queries result in retrieval being completely disabled."""
        intent_res = ClinicalIntentClassifier.classify("Severe crushing chest pain and difficulty breathing right now")
        plan = ClinicalQueryPlanner.plan("Severe crushing chest pain and difficulty breathing right now", intent_res)

        assert plan.retrieval_required is False
        assert plan.top_k == 0
        assert plan.max_chunks == 0
        assert plan.query_expansion_enabled is False
        assert plan.query_expansions == []
        assert plan.context_budget == 0
        assert plan.safety_priority == SafetyPriority.CRITICAL
        assert plan.generation_strategy == GenerationStrategy.SAFETY_REFUSAL
        assert plan.retrieval_strategy == ClinicalRoutingStrategy.EMERGENCY_SAFETY

    def test_self_harm_query_disables_retrieval(self):
        """Req 2: Self-harm queries result in retrieval being completely disabled."""
        intent_res = ClinicalIntentClassifier.classify("I want to end my life and overdose on pills")
        plan = ClinicalQueryPlanner.plan("I want to end my life and overdose on pills", intent_res)

        assert plan.retrieval_required is False
        assert plan.top_k == 0
        assert plan.max_chunks == 0
        assert plan.safety_priority == SafetyPriority.CRITICAL
        assert plan.generation_strategy == GenerationStrategy.SAFETY_REFUSAL
        assert plan.retrieval_strategy == ClinicalRoutingStrategy.SELF_HARM_SAFETY

    def test_poisoning_query_disables_retrieval(self):
        """Req 2: Poisoning queries result in retrieval being completely disabled."""
        intent_res = ClinicalIntentClassifier.classify("My child swallowed bleach and toxic drain cleaner")
        plan = ClinicalQueryPlanner.plan("My child swallowed bleach and toxic drain cleaner", intent_res)

        assert plan.retrieval_required is False
        assert plan.top_k == 0
        assert plan.max_chunks == 0
        assert plan.safety_priority == SafetyPriority.CRITICAL
        assert plan.generation_strategy == GenerationStrategy.SAFETY_REFUSAL
        assert plan.retrieval_strategy == ClinicalRoutingStrategy.POISONING_SAFETY

    def test_out_of_scope_bypasses_retrieval(self):
        """Req 8: Out-of-scope queries bypass retrieval."""
        intent_res = ClinicalIntentClassifier.classify("Who won the soccer world cup final in 2022?")
        plan = ClinicalQueryPlanner.plan("Who won the soccer world cup final in 2022?", intent_res)

        assert plan.retrieval_required is False
        assert plan.top_k == 0
        assert plan.max_chunks == 0
        assert plan.query_expansion_enabled is False
        assert plan.query_expansions == []
        assert plan.generation_strategy == GenerationStrategy.OUT_OF_SCOPE_REFUSAL
        assert plan.safety_priority == SafetyPriority.NONE


# =========================================================================
# 4. HIGH-CONFIDENCE EVIDENCE GATING & UNCERTAINTY (REQ 3, 9)
# =========================================================================

class TestHighConfidenceEvidenceAndUncertainty:
    """Verifies high-confidence requirements for critical clinical intents and conservative fallback."""

    def test_dosage_query_requires_high_confidence(self):
        """Req 3: Dosage queries require high-confidence evidence."""
        intent_res = ClinicalIntentClassifier.classify("What is the starting dose of lisinopril for hypertension?")
        plan = ClinicalQueryPlanner.plan("What is the starting dose of lisinopril for hypertension?", intent_res)

        assert plan.requires_high_confidence_evidence is True
        assert plan.safety_priority == SafetyPriority.HIGH
        assert plan.similarity_threshold == 0.30
        assert plan.generation_strategy == GenerationStrategy.HIGH_CONFIDENCE_GROUNDED

    def test_diagnosis_query_requires_high_confidence(self):
        """Req 3: Diagnosis queries require high-confidence evidence."""
        query = "Do I have Type 2 diabetes based on frequent urination, fatigue, and high blood sugar?"
        intent_res = ClinicalIntentClassifier.classify(query)
        plan = ClinicalQueryPlanner.plan(query, intent_res)

        assert plan.requires_high_confidence_evidence is True
        assert plan.safety_priority == SafetyPriority.HIGH
        assert plan.similarity_threshold == 0.28
        assert plan.generation_strategy == GenerationStrategy.HIGH_CONFIDENCE_GROUNDED

    def test_uncertain_intent_produces_conservative_plan(self):
        """Req 9: Low-confidence/uncertain intents produce conservative plans."""
        intent_res = IntentClassificationResult(
            intent=ClinicalIntent.UNCERTAIN,
            confidence=0.30,
            safety_priority=SafetyPriority.NONE,
            requires_retrieval=True,
            routing_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
            metadata={"is_uncertain": True}
        )
        plan = ClinicalQueryPlanner.plan("some vague ambiguous health query", intent_res)

        assert plan.intent == ClinicalIntent.UNCERTAIN
        assert plan.retrieval_required is True
        assert plan.retrieval_strategy == ClinicalRoutingStrategy.STANDARD_RAG
        assert plan.top_k == 5
        assert plan.max_chunks == 5
        assert plan.requires_high_confidence_evidence is True
        assert plan.query_expansion_enabled is False
        assert plan.metadata.get("conservative_mode") is True


# =========================================================================
# 5. QUERY EXPANSION BOUNDS & EDGE CASES (REQ 10, 11)
# =========================================================================

class TestQueryExpansion:
    """Verifies that query expansion is bounded, deterministic, and handles edge cases safely."""

    def test_bounded_and_deterministic_expansion(self):
        """Req 10: Query expansion produces bounded (<= 4), deterministic output."""
        query = "What are the common side effects of metformin?"
        expansions1 = ClinicalQueryPlanner.generate_expansions(query, ClinicalIntent.MEDICATION_QUERY)
        expansions2 = ClinicalQueryPlanner.generate_expansions(query, ClinicalIntent.MEDICATION_QUERY)

        assert isinstance(expansions1, list)
        assert 0 < len(expansions1) <= 4
        assert expansions1 == expansions2, "Expansions must be deterministic"
        assert any("adverse effects" in e or "side effects" in e for e in expansions1)

    def test_zero_expansion_for_safety_and_out_of_scope(self):
        """Query expansion is strictly disabled for emergencies and out-of-scope."""
        for intent in (ClinicalIntent.EMERGENCY, ClinicalIntent.SELF_HARM, ClinicalIntent.POISONING, ClinicalIntent.OUT_OF_SCOPE):
            expansions = ClinicalQueryPlanner.generate_expansions("Chest pain or suicide", intent)
            assert expansions == []

    @pytest.mark.parametrize("edge_query", [
        "",
        "   ",
        "???!!!...",
        "12345 67890",
        "a",
        "@#$%^&*()_+",
        "a" * 500,
        "null\x00byte\r\n\t",
    ])
    def test_query_expansion_edge_cases(self, edge_query):
        """Req 11: Handles empty strings, special chars, numbers, and gibberish without crashing."""
        expansions = ClinicalQueryPlanner.generate_expansions(edge_query, ClinicalIntent.MEDICATION_QUERY)
        assert isinstance(expansions, list)
        assert len(expansions) <= 4


# =========================================================================
# 6. MULTI-TENANT ISOLATION & DOCUMENT SCOPING (REQ 12)
# =========================================================================

class TestMultiTenancyAndDocumentScoping:
    """Verifies document filtering and multi-tenant isolation in query plans."""

    def test_tenant_isolation_filter(self):
        """Req 12: Plan with user_id enforces USER_DOCUMENTS_ONLY; without user_id uses ALL_AVAILABLE."""
        intent_res = ClinicalIntentClassifier.classify("What is the dose for metformin?")
        plan_user1 = ClinicalQueryPlanner.plan("What is the dose for metformin?", intent_res, user_id=101)
        plan_user2 = ClinicalQueryPlanner.plan("What is the dose for metformin?", intent_res, user_id=202)
        plan_anon = ClinicalQueryPlanner.plan("What is the dose for metformin?", intent_res, user_id=None)

        assert plan_user1.document_filter_strategy == DocumentFilterStrategy.USER_DOCUMENTS_ONLY
        assert plan_user2.document_filter_strategy == DocumentFilterStrategy.USER_DOCUMENTS_ONLY
        assert plan_anon.document_filter_strategy == DocumentFilterStrategy.ALL_AVAILABLE

    def test_explicit_document_scoping(self):
        """Explicit document tag in query scopes the retrieval to that document."""
        scoped_query = "What does [Doc: lab_results_2024.pdf] say about my cholesterol?"
        intent_res = ClinicalIntentClassifier.classify(scoped_query)
        plan = ClinicalQueryPlanner.plan(scoped_query, intent_res, user_id=101)

        assert plan.document_filter_strategy == DocumentFilterStrategy.SCOPED_DOCUMENT
        assert plan.scoped_document_name == "lab_results_2024.pdf"


# =========================================================================
# 7. CACHE ISOLATION (REQ 13)
# =========================================================================

class TestCacheIsolation:
    """Verifies that query plans with differing retrieval strategies do not collide in cache."""

    def test_cache_key_differentiation_by_strategy(self):
        """Req 13: Different query_plan_strategy values produce distinct hash keys."""
        query = "What is the clinical management for hypertension?"
        key_medication = generate_cache_key(
            normalized_query=query,
            query_plan_strategy=ClinicalRoutingStrategy.MEDICATION_RAG.value
        )
        key_dosage = generate_cache_key(
            normalized_query=query,
            query_plan_strategy=ClinicalRoutingStrategy.DOSAGE_RAG.value
        )
        key_none = generate_cache_key(
            normalized_query=query,
            query_plan_strategy=None
        )

        assert key_medication != key_dosage
        assert key_medication != key_none
        assert key_dosage != key_none

    def test_cache_key_deterministic_for_same_strategy(self):
        query = "What is the clinical management for hypertension?"
        k1 = generate_cache_key(normalized_query=query, query_plan_strategy="dosage_rag")
        k2 = generate_cache_key(normalized_query=query, query_plan_strategy="dosage_rag")
        assert k1 == k2


# =========================================================================
# 8. ADAPTIVE RETRIEVAL MECHANISMS (REQ 15, 16, 17, 18)
# =========================================================================

class TestAdaptiveRetrievalMechanisms:
    """Verifies chunk weighting, multi-doc evidence selection, and high-confidence gating."""

    def test_chunk_weighting_medication_priority(self):
        """Req 17: Chunk weighting prioritizes medication chunks when MEDICATION_PRIORITY is active."""
        chunks = [
            {"chunk_id": "c1", "text": "General exercise and diet tips for health.", "similarity_score": 0.85, "metadata": {}},
            {"chunk_id": "c2", "text": "Metformin side effects include nausea and contraindications in renal impairment.", "similarity_score": 0.84, "metadata": {"category": "medication"}},
            {"chunk_id": "c3", "text": "Hospital billing policies.", "similarity_score": 0.80, "metadata": {}},
        ]
        weighted = apply_chunk_weighting(chunks, ChunkWeightingStrategy.MEDICATION_PRIORITY, "metformin side effects")

        # c2 should receive a boost and move to top
        assert weighted[0]["chunk_id"] == "c2"
        assert weighted[0]["similarity_score"] > 0.84

    def test_chunk_weighting_dosage_priority(self):
        """Req 17: Chunk weighting prioritizes dosage and numerical chunks when DOSAGE_PRIORITY is active."""
        chunks = [
            {"chunk_id": "c1", "text": "Lisinopril is an ACE inhibitor for cardiovascular health.", "similarity_score": 0.82, "metadata": {}},
            {"chunk_id": "c2", "text": "The recommended starting dose of lisinopril is 10 mg orally once daily.", "similarity_score": 0.80, "metadata": {}},
        ]
        weighted = apply_chunk_weighting(chunks, ChunkWeightingStrategy.DOSAGE_PRIORITY, "lisinopril dose")

        assert weighted[0]["chunk_id"] == "c2"
        assert weighted[0]["similarity_score"] > 0.80

    def test_balanced_multi_document_selection(self):
        """Req 18: BALANCED_DOCUMENT_RETRIEVAL distributes chunks across distinct documents round-robin."""
        chunks = [
            {"chunk_id": "d1_c1", "document_id": "doc1", "text": "Doc 1 Chunk 1", "similarity_score": 0.95},
            {"chunk_id": "d1_c2", "document_id": "doc1", "text": "Doc 1 Chunk 2", "similarity_score": 0.94},
            {"chunk_id": "d1_c3", "document_id": "doc1", "text": "Doc 1 Chunk 3", "similarity_score": 0.93},
            {"chunk_id": "d2_c1", "document_id": "doc2", "text": "Doc 2 Chunk 1", "similarity_score": 0.90},
            {"chunk_id": "d2_c2", "document_id": "doc2", "text": "Doc 2 Chunk 2", "similarity_score": 0.89},
            {"chunk_id": "d3_c1", "document_id": "doc3", "text": "Doc 3 Chunk 1", "similarity_score": 0.85},
        ]
        selected = select_multi_document_evidence(chunks, MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL, top_k=4)

        # Should pick from doc1, doc2, doc3, then doc1
        doc_ids = [c["document_id"] for c in selected]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids
        assert len(selected) == 4

    def test_dosage_high_confidence_evidence_gating(self):
        """Req 16: Dosage query fails sufficiency if retrieved chunks contain zero numerical dosages or units."""
        rag = RAGService()
        plan = QueryPlan(
            intent=ClinicalIntent.DOSAGE_QUERY,
            retrieval_required=True,
            retrieval_strategy=ClinicalRoutingStrategy.DOSAGE_RAG,
            requires_high_confidence_evidence=True
        )

        # Chunks mentioning drug but NO dosage numbers or units
        insufficient_chunks = [
            {"chunk_id": "c1", "text": "Lisinopril is an effective medication for lowering blood pressure.", "similarity_score": 0.75}
        ]
        is_sufficient, reason = rag.verify_relevance_and_sufficiency(
            "What is the starting dose of lisinopril?",
            insufficient_chunks,
            query_plan=plan
        )
        assert is_sufficient is False, "Dosage queries must fail sufficiency without explicit dosage numbers/units"

        # Chunks WITH dosage numbers and units
        sufficient_chunks = [
            {"chunk_id": "c1", "text": "The starting dose of lisinopril is 10 mg once daily.", "similarity_score": 0.75}
        ]
        is_sufficient, reason = rag.verify_relevance_and_sufficiency(
            "What is the starting dose of lisinopril?",
            sufficient_chunks,
            query_plan=plan
        )
        assert is_sufficient is True


# =========================================================================
# 9. END-TO-END RAGSERVICE INTEGRATION (REQ 14, 15)
# =========================================================================

class TestRAGServiceIntegration:
    """Verifies end-to-end integration of ClinicalQueryPlanner within RAGService."""

    def test_generate_rag_answer_includes_query_plan(self):
        """Req 14: RAGService.generate_rag_answer attaches query_plan dictionary to response."""
        rag = RAGService()
        mock_gemini = MagicMock()
        mock_gemini.generate_answer.return_value = {
            "answer": "Metformin reduces hepatic glucose production [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "status": "success",
            "disclaimer": "MEDICAL DISCLAIMER: Consult a physician.",
            "input_tokens": 100,
            "output_tokens": 20,
            "api_request_time_ms": 10.0,
            "gemini_calls_count": 1
        }

        res = rag.generate_rag_answer(
            question="What is the pharmacology and side effects of metformin?",
            gemini_service=mock_gemini,
            use_cache=False
        )

        assert "query_plan" in res
        plan_dict = res["query_plan"]
        assert plan_dict["intent"] == "MEDICATION_QUERY"
        assert plan_dict["retrieval_strategy"] == "medication_rag"
        assert plan_dict["chunk_weighting_strategy"] == "medication_priority"
        assert plan_dict["retrieval_required"] is True

    def test_emergency_interception_includes_query_plan(self):
        """Req 14: Safety intercepted response attaches query_plan with retrieval_required=False."""
        rag = RAGService()
        emergency_q = "I have severe crushing chest pain radiating to my left arm right now"
        res = rag.generate_rag_answer(question=emergency_q)

        assert res["retrieval_status"] == "safety_intercepted"
        assert "query_plan" in res
        assert res["query_plan"]["intent"] == "EMERGENCY"
        assert res["query_plan"]["retrieval_required"] is False
        assert res["query_plan"]["generation_strategy"] == "safety_refusal"

    def test_generate_rag_stream_yields_query_plan(self):
        """Req 14: RAGService.generate_rag_stream yields query_plan event."""
        rag = RAGService()
        mock_gemini = MagicMock()
        mock_gemini.generate_stream.return_value = iter(["Metformin works by ", "inhibiting gluconeogenesis [Source 1]."])

        events = list(rag.generate_rag_stream(
            question="What is the mechanism of action of metformin?",
            gemini_service=mock_gemini,
            use_cache=False
        ))

        event_types = [event_type for event_type, _ in events]
        assert "query_plan" in event_types

        plan_payload = next(payload for event_type, payload in events if event_type == "query_plan")
        assert "query_plan" in plan_payload
        assert plan_payload["query_plan"]["intent"] == "MEDICATION_QUERY"


# =========================================================================
# 10. PERFORMANCE & OBSERVABILITY (REQ 19, 20)
# =========================================================================

class TestPerformanceAndObservability:
    """Verifies low-latency execution (< 5ms) and Prometheus observability metrics."""

    def test_query_planning_latency_under_5ms(self):
        """Req 19: ClinicalQueryPlanner.plan execution latency is < 5ms (p95)."""
        intent_res = ClinicalIntentClassifier.classify("What is the typical dose of amlodipine?")

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            plan = ClinicalQueryPlanner.plan("What is the typical dose of amlodipine?", intent_res)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]

        assert p50 < 2.0, f"p50 latency {p50:.3f}ms exceeded 2ms target"
        assert p95 < 5.0, f"p95 latency {p95:.3f}ms exceeded 5ms limit"

    def test_query_expansion_latency_under_2ms(self):
        """Req 19: Query expansion executes in < 2ms."""
        query = "What are the common side effects of lisinopril?"
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            expansions = ClinicalQueryPlanner.generate_expansions(query, ClinicalIntent.MEDICATION_QUERY)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 2.0, f"Query expansion p95 {p95:.3f}ms exceeded 2ms limit"

    def test_observability_metrics_recorded(self):
        """Req 20: Observability metrics are recorded in Prometheus registry without errors."""
        collector = get_metrics_collector()
        collector.reset()

        collector.record_query_plan(
            intent="MEDICATION_QUERY",
            strategy="medication_rag",
            expansions_count=3,
            requires_high_evidence=False,
            latency_ms=0.8
        )
        collector.record_query_plan(
            intent="DOSAGE_QUERY",
            strategy="dosage_rag",
            expansions_count=2,
            requires_high_evidence=True,
            latency_ms=1.1
        )

        snap = collector.get_metrics_snapshot()
        assert "intelligence" in snap
        assert snap["intelligence"]["query_plans"]["intents"]["MEDICATION_QUERY"] == 1
        assert snap["intelligence"]["query_plans"]["intents"]["DOSAGE_QUERY"] == 1
        assert snap["intelligence"]["query_plans"]["high_evidence_total"] == 1

        expo = collector.get_prometheus_exposition()
        assert 'rag_query_plan_intent_total{intent="MEDICATION_QUERY"} 1' in expo
        assert 'rag_query_plan_intent_total{intent="DOSAGE_QUERY"} 1' in expo
        assert 'rag_query_plan_total{strategy="medication_rag"} 1' in expo
        assert "rag_query_plan_high_evidence_total 1" in expo
        assert "rag_query_plan_expansion_total 2" in expo

    def test_metrics_contain_zero_phi_or_user_ids(self):
        """Zero PHI, user queries, user_id, or patient data in metric labels."""
        collector = get_metrics_collector()
        expo = collector.get_prometheus_exposition()

        forbidden_tokens = ["patient", "John", "Doe", "user_id", "password", "secret", "prescription", "ssn"]
        for token in forbidden_tokens:
            assert token not in expo.lower(), f"Forbidden token {token} found in Prometheus metrics!"
