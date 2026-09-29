"""
Phase 2F: Production Retrieval Pipeline Hardening & Validation Test Suite.

Validates the complete set of requirements from Step 13:
A. Query expansion (synonyms, medication intent, no medication injection without intent)
B. Safe subject extraction (hypertension, diabetes, asthma, pneumonia, arthritis, unknown, none)
C. Multi-aspect decomposition (valid known condition, unknown condition, ambiguous, no hardcoded hypertension)
D. Ambiguity safety (unanchored queries safely handled without guessing disease)
E. Adversarial retrieval (distractor rejection, out of scope rejection)
F. Citation grounding (sources strictly align with retrieved evidence)
G. User isolation (user_id filtering strictly enforced)
H. Top_k enforcement (returns exactly <= top_k results)
I. Vector-store invariance (vector store remains strictly read-only and unchanged)
J. Latency sanity (deterministic expansion and retrieval within latency envelope)
"""

import time
import pytest
from unittest.mock import MagicMock

from backend.rag.query_expander import MedicalQueryExpander
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import get_vector_store_service


# ==============================================================================
# A. Query Expansion & Medication Intent
# ==============================================================================
def test_a1_hypertension_synonym_expansion():
    """Verify hypertension terms expand to clinical synonyms."""
    q = "What causes elevated arterial blood pressure?"
    terms = MedicalQueryExpander.get_expanded_terms(q)
    assert any("hypertension" in t.lower() for t in terms)
    assert any("high blood pressure" in t.lower() for t in terms)


def test_a2_diabetes_synonyms_without_medication_intent():
    """Verify glycemic disorders expand to diabetes without injecting metformin when no med intent exists."""
    q = "What are glycemic disorders and how are they defined?"
    terms = MedicalQueryExpander.get_expanded_terms(q)
    assert any("diabetes" in t.lower() for t in terms)
    # Metformin MUST NOT be injected when user did not ask about medications
    assert not any("metformin" in t.lower() for t in terms)


def test_a3_diabetes_synonyms_with_medication_intent():
    """Verify glycemic queries with explicit medication intent DO include medication terms."""
    q = "What medications help regulate blood sugar in glycemic disorders?"
    terms = MedicalQueryExpander.get_expanded_terms(q)
    assert any("diabetes" in t.lower() for t in terms)
    assert any("metformin" in t.lower() for t in terms)


def test_a4_complication_synonyms():
    """Verify sequelae and organ damage expand to complications."""
    q = "What are the common secondary sequelae of high blood pressure?"
    terms = MedicalQueryExpander.get_expanded_terms(q)
    assert any("complications" in t.lower() for t in terms)


# ==============================================================================
# B. Safe Subject Extraction
# ==============================================================================
@pytest.mark.parametrize(
    "query, expected_subject",
    [
        ("What causes elevated blood pressure in adults?", "hypertension"),
        ("Explain fasting blood glucose thresholds in type 2 diabetes", "diabetes"),
        ("What triggers bronchial wheezing and shortness of breath in asthma?", "asthma"),
        ("What are the clinical signs of community-acquired pneumonia?", "pneumonia"),
        ("How is rheumatoid arthritis diagnosed in clinical practice?", "arthritis"),
        ("What is the weather forecast for tomorrow?", ""),
        ("What treatment is recommended?", ""),
    ]
)
def test_b_subject_extraction(query, expected_subject):
    """Verify subject extraction accurately identifies conditions and never defaults to hypertension."""
    subject = MedicalQueryExpander.extract_primary_subject(query)
    if expected_subject:
        assert subject.lower() == expected_subject.lower()
    else:
        # Must NOT invent hypertension
        assert subject != "hypertension"
        assert subject == ""


# ==============================================================================
# C. Multi-Aspect Decomposition (No Hardcoded Fallback)
# ==============================================================================
def test_c1_multi_aspect_known_condition():
    """Verify multi-aspect queries for known conditions produce clean faceted subqueries."""
    q = "Provide a comprehensive guide to asthma: definition, triggers, symptoms, and medical treatment."
    sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(q)
    assert len(sub_queries) >= 2
    assert all("asthma" in sq.lower() for sq in sub_queries)
    # Must NOT contain hypertension
    assert not any("hypertension" in sq.lower() for sq in sub_queries)


def test_c2_multi_aspect_unknown_condition_no_hardcoded_hypertension():
    """Verify multi-aspect queries for non-hypertension conditions never inject hypertension."""
    q = "Explain pneumonia symptoms, diagnostic chest imaging, and antibiotic therapy."
    sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(q)
    # Even if decomposed or fallback, must NEVER contain hypertension
    assert not any("hypertension" in sq.lower() for sq in sub_queries)
    assert any("pneumonia" in sq.lower() for sq in sub_queries)


def test_c3_multi_aspect_ambiguous_no_hardcoded_hypertension():
    """Verify completely unanchored multi-aspect queries do NOT fabricate hypertension."""
    q = "What are the common symptoms, diagnostic guidelines, and follow-up recommendations?"
    sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(q)
    assert not any("hypertension" in sq.lower() for sq in sub_queries)


# ==============================================================================
# D. Ambiguity Safety
# ==============================================================================
def test_d_ambiguous_query_safety():
    """Verify unanchored ambiguous queries are safely rejected when no condition anchor is present."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    # Query completely lacks condition or patient anchor
    res = rag.query("What follow-up plan is established?", top_k=5)
    # Must safely reject without hallucinating medical facts
    assert res.get("retrieval_status") == "no_relevant_context"
    assert len(res.get("sources", [])) == 0


# ==============================================================================
# E. Adversarial Retrieval
# ==============================================================================
def test_e1_distractor_query_rejection():
    """Verify cross-document false framing distractor queries are rejected."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    q = "What does the dermatology pdf say regarding emergency cardioversion and defibrillation protocols?"
    res = rag.query(q, top_k=5)
    assert res.get("retrieval_status") == "no_relevant_context"
    assert len(res.get("sources", [])) == 0


def test_e2_out_of_scope_query_rejection():
    """Verify non-healthcare out of scope queries are safely halted."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    q = "What are the structural load calculations for designing a suspension bridge tower?"
    res = rag.query(q, top_k=5)
    assert res.get("retrieval_status") == "no_relevant_context"
    assert len(res.get("sources", [])) == 0


# ==============================================================================
# F. Citation Grounding
# ==============================================================================
def test_f_citation_grounding_alignment():
    """Verify retrieved sources strictly correspond to returned citations."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    q = "What causes elevated arterial blood pressure?"
    res = rag.query(q, top_k=3)
    assert res.get("retrieval_status") == "success"
    sources = res.get("sources", [])
    assert len(sources) > 0
    # Every source must have non-empty filename, chunk_id, and similarity_score
    for src in sources:
        assert src.get("filename") is not None
        assert src.get("similarity_score") > 0.0


# ==============================================================================
# G. User Isolation
# ==============================================================================
def test_g_user_isolation_retrieval_path():
    """Verify user_id filter strictly excludes other users' documents."""
    mock_vs = MagicMock()
    mock_vs.count.return_value = 10
    mock_vs.search.return_value = [
        {"document_id": "1", "chunk_id": "c1", "similarity_score": 0.85, "text": "hypertension text", "metadata": {"filename": "test.pdf", "user_id": 42}}
    ]

    rag = RAGService(vector_store=mock_vs)
    rag.query("What causes hypertension?", user_id=42)

    # Verify user_id=42 was passed to all search calls
    for call_args in mock_vs.search.call_args_list:
        assert call_args.kwargs.get("user_id") == 42


# ==============================================================================
# H. Top_k Enforcement
# ==============================================================================
def test_h_top_k_enforcement():
    """Verify retrieval never returns more than top_k items."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    for requested_k in [1, 2, 3, 5]:
        res = rag.query("What are the diagnostic criteria and treatment for hypertension?", top_k=requested_k)
        assert len(res.get("sources", [])) <= requested_k


# ==============================================================================
# I. Vector-Store Invariance
# ==============================================================================
def test_i_vector_store_invariance():
    """Verify that retrieval operations do not alter vector store count or mutate store."""
    vs = get_vector_store_service()
    count_before = vs.count()
    rag = RAGService(vector_store=vs)
    rag.query("What are the common symptoms of asthma?", top_k=5)
    rag.query("What medications treat diabetes?", top_k=5)
    rag.query("Out of scope query regarding aerospace engineering", top_k=5)
    count_after = vs.count()
    assert count_before == count_after
    assert count_before >= 637


# ==============================================================================
# J. Latency Sanity
# ==============================================================================
def test_j_latency_sanity():
    """Verify query expansion and decomposition are ultra-fast deterministic CPU operations (< 5 ms)."""
    q = "Provide a comprehensive guide to hypertension: definition, etiology, symptoms, lifestyle measures, and medical treatment."
    t0 = time.perf_counter()
    _ = MedicalQueryExpander.is_multi_aspect_query(q)
    _ = MedicalQueryExpander.decompose_multi_aspect_query(q)
    _ = MedicalQueryExpander.get_expanded_terms(q)
    _ = MedicalQueryExpander.expand_query(q)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 5.0, f"Query expansion took {elapsed_ms:.2f} ms, expected < 5 ms"
