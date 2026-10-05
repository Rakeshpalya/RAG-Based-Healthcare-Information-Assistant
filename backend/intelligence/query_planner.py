"""
Clinical Query Planner for AI-Healthcare-Agent (Phase 6.2).

Consumes the IntentClassificationResult from Phase 6.1 and deterministically produces
a comprehensive QueryPlan orchestrating retrieval strategy, Top-K, thresholds,
evidence requirements, chunk weighting, and query expansion.
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional

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
from backend.rag.query_expander import MedicalQueryExpander

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# DETERMINISTIC CLINICAL EXPANSION PATTERNS (BOUNDED & PURE PYTHON)
# -------------------------------------------------------------------------

CLINICAL_EXPANSION_TEMPLATES: Dict[ClinicalIntent, List[str]] = {
    ClinicalIntent.MEDICATION_QUERY: [
        "{entity} adverse effects",
        "{entity} side effects",
        "{entity} contraindications",
        "{entity} pharmacology mechanism"
    ],
    ClinicalIntent.DOSAGE_QUERY: [
        "{entity} dosage recommendations",
        "{entity} daily dose",
        "{entity} titration schedule",
        "{entity} administration guidelines"
    ],
    ClinicalIntent.LAB_RESULT_QUERY: [
        "{entity} reference range",
        "{entity} clinical interpretation",
        "{entity} diagnostic significance",
        "{entity} abnormal levels"
    ],
    ClinicalIntent.DIAGNOSIS_QUERY: [
        "{entity} diagnostic criteria",
        "{entity} clinical evaluation",
        "{entity} differential diagnosis",
        "{entity} diagnostic guidelines"
    ],
    ClinicalIntent.TREATMENT_QUERY: [
        "{entity} first-line therapy",
        "{entity} clinical management",
        "{entity} therapeutic options",
        "{entity} treatment protocol"
    ],
    ClinicalIntent.PREVENTION_QUERY: [
        "{entity} lifestyle modifications",
        "{entity} prevention guidelines",
        "{entity} risk reduction",
        "{entity} prophylaxis"
    ],
    ClinicalIntent.SYMPTOM_QUERY: [
        "{entity} clinical manifestations",
        "{entity} warning signs",
        "{entity} early symptoms",
        "{entity} presentation"
    ],
}


class ClinicalQueryPlanner:
    """
    Deterministic clinical query planning engine for AI-Healthcare-Agent.
    """

    @classmethod
    def extract_primary_clinical_entity(cls, query: str) -> Optional[str]:
        """
        Extracts the central clinical subject or drug name from the query.
        """
        if not query:
            return None
        q = query.lower()

        # Known common drugs
        drug_matches = re.findall(
            r'\b(metformin|lisinopril|atorvastatin|amlodipine|losartan|albuterol|levothyroxine|omeprazole|amoxicillin|hydrochlorothiazide|ibuprofen|acetaminophen|aspirin|warfarin|apixaban|clopidogrel|gabapentin|sertraline|metoprolol|furosemide)\b',
            q
        )
        if drug_matches:
            return drug_matches[0].strip()

        # Known common conditions / biomarkers
        condition_matches = re.findall(
            r'\b(hypertension|high blood pressure|type 2 diabetes|type 1 diabetes|diabetes|asthma|heart failure|cardiovascular disease|stroke|cholesterol|hba1c|creatinine|troponin|pneumonia|kidney disease|cancer|arrhythmia|headache|migraine)\b',
            q
        )
        if condition_matches:
            return condition_matches[0].strip()

        # Fallback: extract key noun phrase after question word
        clean = re.sub(r'^(what is|what are|what does|how to|how is|tell me about|can you explain|summarize|compare)\s+', '', q)
        clean = re.sub(r'[?!.,]', '', clean).strip()
        tokens = [w for w in clean.split() if len(w) > 3 and w not in ("dose", "dosage", "report", "document", "test", "result", "recommended", "usually", "prescribed")]
        if tokens:
            return " ".join(tokens[:2])
        return None

    @classmethod
    def generate_expansions(cls, query: str, intent: ClinicalIntent) -> List[str]:
        """
        Generates deterministic, bounded query expansions (max 3-5).
        Never expands safety violations or out-of-scope inquiries.
        """
        if intent in (ClinicalIntent.EMERGENCY, ClinicalIntent.SELF_HARM, ClinicalIntent.POISONING, ClinicalIntent.OUT_OF_SCOPE):
            return []

        entity = cls.extract_primary_clinical_entity(query)
        expansions: List[str] = []

        if entity and intent in CLINICAL_EXPANSION_TEMPLATES:
            for tmpl in CLINICAL_EXPANSION_TEMPLATES[intent]:
                expansions.append(tmpl.format(entity=entity))

        # Check existing synonym expander for additional high-signal clinical synonyms
        synonyms = MedicalQueryExpander.get_expanded_terms(query)
        for syn in synonyms:
            if syn.lower() not in [e.lower() for e in expansions]:
                expansions.append(syn)

        # Strictly bounded to 4 expansions
        return expansions[:4]

    @classmethod
    def plan(
        cls,
        query: str,
        intent_result: IntentClassificationResult,
        user_id: Optional[int] = None
    ) -> QueryPlan:
        """
        Generates a strongly typed execution plan based on query and intent classification.

        Args:
            query: User's raw question string.
            intent_result: Result of ClinicalIntentClassifier.classify.
            user_id: Optional user identifier for tenancy.

        Returns:
            QueryPlan controlling downstream retrieval, thresholds, and evidence gating.
        """
        t0 = time.perf_counter()
        intent = intent_result.intent
        safety_prio = intent_result.safety_priority

        # -------------------------------------------------------------
        # STEP 1: SAFETY ABSOLUTE PRECEDENCE (EMERGENCY / HARM / TOXIC)
        # -------------------------------------------------------------
        if intent == ClinicalIntent.EMERGENCY:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return QueryPlan(
                intent=intent,
                retrieval_required=False,
                retrieval_strategy=ClinicalRoutingStrategy.EMERGENCY_SAFETY,
                top_k=0,
                similarity_threshold=1.0,
                max_chunks=0,
                chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
                document_filter_strategy=DocumentFilterStrategy.ALL_AVAILABLE,
                query_expansion_enabled=False,
                query_expansions=[],
                multi_document_strategy=MultiDocumentStrategy.SINGLE_DOCUMENT,
                context_budget=0,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.CRITICAL,
                generation_strategy=GenerationStrategy.SAFETY_REFUSAL,
                latency_ms=elapsed_ms,
                metadata={"plan_reason": "Emergency symptoms intercepted. Zero retrieval."}
            )

        if intent == ClinicalIntent.SELF_HARM:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return QueryPlan(
                intent=intent,
                retrieval_required=False,
                retrieval_strategy=ClinicalRoutingStrategy.SELF_HARM_SAFETY,
                top_k=0,
                similarity_threshold=1.0,
                max_chunks=0,
                chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
                document_filter_strategy=DocumentFilterStrategy.ALL_AVAILABLE,
                query_expansion_enabled=False,
                query_expansions=[],
                multi_document_strategy=MultiDocumentStrategy.SINGLE_DOCUMENT,
                context_budget=0,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.CRITICAL,
                generation_strategy=GenerationStrategy.SAFETY_REFUSAL,
                latency_ms=elapsed_ms,
                metadata={"plan_reason": "Self-harm ideation intercepted. Zero retrieval."}
            )

        if intent == ClinicalIntent.POISONING:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return QueryPlan(
                intent=intent,
                retrieval_required=False,
                retrieval_strategy=ClinicalRoutingStrategy.POISONING_SAFETY,
                top_k=0,
                similarity_threshold=1.0,
                max_chunks=0,
                chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
                document_filter_strategy=DocumentFilterStrategy.ALL_AVAILABLE,
                query_expansion_enabled=False,
                query_expansions=[],
                multi_document_strategy=MultiDocumentStrategy.SINGLE_DOCUMENT,
                context_budget=0,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.CRITICAL,
                generation_strategy=GenerationStrategy.SAFETY_REFUSAL,
                latency_ms=elapsed_ms,
                metadata={"plan_reason": "Toxic ingestion intercepted. Zero retrieval."}
            )

        # -------------------------------------------------------------
        # STEP 2: OUT-OF-SCOPE REFUSAL
        # -------------------------------------------------------------
        if intent == ClinicalIntent.OUT_OF_SCOPE:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return QueryPlan(
                intent=intent,
                retrieval_required=False,
                retrieval_strategy=ClinicalRoutingStrategy.OUT_OF_SCOPE_RESPONSE,
                top_k=0,
                similarity_threshold=1.0,
                max_chunks=0,
                chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
                document_filter_strategy=DocumentFilterStrategy.ALL_AVAILABLE,
                query_expansion_enabled=False,
                query_expansions=[],
                multi_document_strategy=MultiDocumentStrategy.SINGLE_DOCUMENT,
                context_budget=0,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NONE,
                generation_strategy=GenerationStrategy.OUT_OF_SCOPE_REFUSAL,
                latency_ms=elapsed_ms,
                metadata={"plan_reason": "Non-medical query. Retrieval bypassed."}
            )

        # -------------------------------------------------------------
        # STEP 3: DOCUMENT SCOPING & TENANT FILTERING
        # -------------------------------------------------------------
        scoped_doc: Optional[str] = None
        # Check if user query explicitly scopes a specific document
        from backend.rag.rag_service import RAGService
        scoped_doc = RAGService.extract_scoped_document_name(query)

        if scoped_doc:
            doc_filter = DocumentFilterStrategy.SCOPED_DOCUMENT
        elif user_id is not None:
            doc_filter = DocumentFilterStrategy.USER_DOCUMENTS_ONLY
        else:
            doc_filter = DocumentFilterStrategy.ALL_AVAILABLE

        # -------------------------------------------------------------
        # STEP 4: INTENT-SPECIFIC RETRIEVAL & WEIGHTING POLICIES
        # -------------------------------------------------------------
        expansions = cls.generate_expansions(query, intent)
        has_expansions = len(expansions) > 0

        if intent == ClinicalIntent.DOSAGE_QUERY:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.DOSAGE_RAG,
                top_k=6,
                similarity_threshold=0.30,
                max_chunks=8,
                chunk_weighting_strategy=ChunkWeightingStrategy.DOSAGE_PRIORITY,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=has_expansions,
                query_expansions=expansions,
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3500,
                requires_high_confidence_evidence=True,
                safety_priority=SafetyPriority.HIGH,
                generation_strategy=GenerationStrategy.HIGH_CONFIDENCE_GROUNDED,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.DIAGNOSIS_QUERY:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.DIAGNOSIS_RAG,
                top_k=7,
                similarity_threshold=0.28,
                max_chunks=8,
                chunk_weighting_strategy=ChunkWeightingStrategy.DIAGNOSTIC_PRIORITY,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=has_expansions,
                query_expansions=expansions,
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3800,
                requires_high_confidence_evidence=True,
                safety_priority=SafetyPriority.HIGH,
                generation_strategy=GenerationStrategy.HIGH_CONFIDENCE_GROUNDED,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.MEDICATION_QUERY:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.MEDICATION_RAG,
                top_k=5,
                similarity_threshold=0.25,
                max_chunks=7,
                chunk_weighting_strategy=ChunkWeightingStrategy.MEDICATION_PRIORITY,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=has_expansions,
                query_expansions=expansions,
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3500,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NORMAL,
                generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.LAB_RESULT_QUERY:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.LAB_RAG,
                top_k=6,
                similarity_threshold=0.25,
                max_chunks=8,
                chunk_weighting_strategy=ChunkWeightingStrategy.LAB_PRIORITY,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=has_expansions,
                query_expansions=expansions,
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3500,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NORMAL,
                generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.DOCUMENT_SUMMARY:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.DOCUMENT_SUMMARY_RAG,
                top_k=8,
                similarity_threshold=0.20,
                max_chunks=10,
                chunk_weighting_strategy=ChunkWeightingStrategy.BROAD_COVERAGE,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=False,
                query_expansions=[],
                multi_document_strategy=MultiDocumentStrategy.MAP_REDUCE,
                context_budget=4200,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NORMAL,
                generation_strategy=GenerationStrategy.DOCUMENT_SUMMARY_SYNTHESIS,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.DOCUMENT_COMPARISON:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.DOCUMENT_COMPARISON_RAG,
                top_k=8,
                similarity_threshold=0.20,
                max_chunks=10,
                chunk_weighting_strategy=ChunkWeightingStrategy.BALANCED_MULTI_DOCUMENT,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=False,
                query_expansions=[],
                multi_document_strategy=MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL,
                context_budget=4500,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NORMAL,
                generation_strategy=GenerationStrategy.COMPARATIVE_SYNTHESIS,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.TREATMENT_QUERY:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.TREATMENT_RAG,
                top_k=6,
                similarity_threshold=0.25,
                max_chunks=7,
                chunk_weighting_strategy=ChunkWeightingStrategy.TREATMENT_PRIORITY,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=has_expansions,
                query_expansions=expansions,
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3500,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NORMAL,
                generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.PREVENTION_QUERY:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.PREVENTION_RAG,
                top_k=5,
                similarity_threshold=0.25,
                max_chunks=6,
                chunk_weighting_strategy=ChunkWeightingStrategy.PREVENTION_PRIORITY,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=has_expansions,
                query_expansions=expansions,
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3200,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NORMAL,
                generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.SYMPTOM_QUERY:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.SYMPTOM_RAG,
                top_k=5,
                similarity_threshold=0.25,
                max_chunks=6,
                chunk_weighting_strategy=ChunkWeightingStrategy.SYMPTOM_PRIORITY,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=has_expansions,
                query_expansions=expansions,
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3200,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NORMAL,
                generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
                scoped_document_name=scoped_doc
            )
        elif intent == ClinicalIntent.UNCERTAIN:
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
                top_k=5,
                similarity_threshold=0.25,
                max_chunks=5,
                chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=False,
                query_expansions=[],
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3000,
                requires_high_confidence_evidence=True,
                safety_priority=SafetyPriority.NONE,
                generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
                scoped_document_name=scoped_doc,
                metadata={"conservative_mode": True}
            )
        else:  # GENERAL_HEALTH and fallbacks
            plan = QueryPlan(
                intent=intent,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.GENERAL_HEALTH_RAG,
                top_k=5,
                similarity_threshold=0.25,
                max_chunks=6,
                chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
                document_filter_strategy=doc_filter,
                query_expansion_enabled=False,
                query_expansions=[],
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                context_budget=3200,
                requires_high_confidence_evidence=False,
                safety_priority=SafetyPriority.NORMAL,
                generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
                scoped_document_name=scoped_doc
            )

        plan.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return plan
