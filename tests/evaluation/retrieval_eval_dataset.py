"""
Retrieval Evaluation Dataset for Phase 2D Benchmark.

Contains benchmark queries mapped to expected document IDs, relevant documents,
irrelevant documents, query type, and difficulty.

Categories covered:
A. Single-document factual queries
B. Multi-document queries
C. Synonym queries
D. Multiple-chunk queries
E. Out-of-scope queries
F. Ambiguous medical queries
G. Irrelevant medical-document queries
H. Exact terminology queries
I. Natural-language patient questions

All queries are grounded in medical topics from actual project documents:
- HealthAI_RAG_Test_Document.pdf
- synthetic_hypertension_test.pdf
- hypertension_summary.pdf
- cardio.pdf
- trial_report.pdf (Metformin & Type 2 Diabetes)
- asthma.pdf
- onco_pathology.pdf
- pheo_case.pdf
- ct_scan.pdf
- derma.pdf
"""

from typing import List, Dict, Any

EVALUATION_DATASET: List[Dict[str, Any]] = [
    # =========================================================================
    # Category A: Single-Document Factual Queries
    # =========================================================================
    {
        "id": "A1",
        "query": "What are the common risk factors for hypertension according to the synthetic hypertension document?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["derma.pdf", "ct_scan.pdf", "DSA 1.pdf"],
        "query_type": "single_document",
        "difficulty": "easy"
    },
    {
        "id": "A2",
        "query": "What diagnosis and vitals are recorded for patient John Doe in the hypertension summary?",
        "expected_document_ids": ["hypertension_summary.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["hypertension_summary.pdf"],
        "irrelevant_documents": ["asthma.pdf", "onco_pathology.pdf"],
        "query_type": "single_document",
        "difficulty": "easy"
    },
    {
        "id": "A3",
        "query": "What is pheochromocytoma and what does it secrete?",
        "expected_document_ids": ["pheo_case.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["pheo_case.pdf"],
        "irrelevant_documents": ["derma.pdf", "asthma.pdf"],
        "query_type": "single_document",
        "difficulty": "easy"
    },
    {
        "id": "A4",
        "query": "What was the efficacy of Metformin evaluated for in the clinical trial report?",
        "expected_document_ids": ["trial_report.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["trial_report.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "cardio.pdf"],
        "query_type": "single_document",
        "difficulty": "easy"
    },

    # =========================================================================
    # Category B: Multi-Document Queries
    # =========================================================================
    {
        "id": "B1",
        "query": "What clinical guidelines and medication management are documented for hypertension across cardiology and patient records?",
        "expected_document_ids": ["hypertension_summary.pdf", "synthetic_hypertension_test.pdf", "cardio.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["hypertension_summary.pdf", "synthetic_hypertension_test.pdf", "cardio.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["derma.pdf", "onco_pathology.pdf"],
        "query_type": "multi_document",
        "difficulty": "medium"
    },
    {
        "id": "B2",
        "query": "How is Type 2 Diabetes managed and what treatments are studied in the clinical trials and general guidelines?",
        "expected_document_ids": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "derma.pdf"],
        "query_type": "multi_document",
        "difficulty": "medium"
    },
    {
        "id": "B3",
        "query": "What lifestyle and clinical interventions reduce cardiovascular and blood pressure risks?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "cardio.pdf", "hypertension_summary.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "cardio.pdf", "hypertension_summary.pdf"],
        "irrelevant_documents": ["derma.pdf", "triage.pdf"],
        "query_type": "multi_document",
        "difficulty": "hard"
    },
    {
        "id": "B4",
        "query": "What diagnostic evaluations are recorded for oncology and pathology findings?",
        "expected_document_ids": ["onco_pathology.pdf", "alice_chart.pdf", "pheo_case.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["onco_pathology.pdf", "alice_chart.pdf", "pheo_case.pdf"],
        "irrelevant_documents": ["asthma.pdf", "triage.pdf"],
        "query_type": "multi_document",
        "difficulty": "medium"
    },

    # =========================================================================
    # Category C: Synonym Queries
    # =========================================================================
    {
        "id": "C1",
        "query": "What are the common causes and triggers of high blood pressure?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf", "hypertension_summary.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf", "hypertension_summary.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "derma.pdf"],
        "query_type": "synonym",
        "difficulty": "medium"
    },
    {
        "id": "C2",
        "query": "What are the long-term sequelae and organ damage from elevated arterial blood pressure?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["asthma.pdf", "derma.pdf"],
        "query_type": "synonym",
        "difficulty": "hard"
    },
    {
        "id": "C3",
        "query": "What medications help regulate blood sugar in glycemic disorders?",
        "expected_document_ids": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "triage.pdf"],
        "query_type": "synonym",
        "difficulty": "medium"
    },
    {
        "id": "C4",
        "query": "What adrenal gland neoplasm causes excessive catecholamine release?",
        "expected_document_ids": ["pheo_case.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["pheo_case.pdf"],
        "irrelevant_documents": ["asthma.pdf", "cardio.pdf"],
        "query_type": "synonym",
        "difficulty": "medium"
    },

    # =========================================================================
    # Category D: Multiple-Chunk Queries
    # =========================================================================
    {
        "id": "D1",
        "query": "Provide an overview of hypertension definition, risk factors, lifestyle measures, and complications.",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["derma.pdf", "ct_scan.pdf"],
        "query_type": "multi_chunk",
        "difficulty": "medium"
    },
    {
        "id": "D2",
        "query": "What are all the sections covered in the test health research document regarding chronic diseases?",
        "expected_document_ids": ["HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["alice_chart.pdf", "derma.pdf"],
        "query_type": "multi_chunk",
        "difficulty": "medium"
    },
    {
        "id": "D3",
        "query": "What are the lifestyle recommendations and diagnostic criteria described for high blood pressure?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "irrelevant_documents": ["onco_pathology.pdf", "triage.pdf"],
        "query_type": "multi_chunk",
        "difficulty": "medium"
    },

    # =========================================================================
    # Category E: Out-of-Scope Queries (Explicit Negative Boundary)
    # =========================================================================
    {
        "id": "E1",
        "query": "What are the symptoms and transmission vector of malaria according to the uploaded PDF?",
        "expected_document_ids": ["HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["HealthAI_RAG_Test_Document.pdf"],  # Explicit boundary chunk exists in this document
        "irrelevant_documents": ["cardio.pdf", "hypertension_summary.pdf"],
        "query_type": "out_of_scope",
        "difficulty": "hard"
    },
    {
        "id": "E2",
        "query": "What does the uploaded test document state about tuberculosis treatments?",
        "expected_document_ids": ["HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["HealthAI_RAG_Test_Document.pdf"],  # Boundary statement covers tuberculosis
        "irrelevant_documents": ["asthma.pdf", "derma.pdf"],
        "query_type": "out_of_scope",
        "difficulty": "hard"
    },
    {
        "id": "E3",
        "query": "What are the chemotherapy guidelines for pancreatic carcinoma in the general hypertension guideline?",
        "expected_document_ids": [],
        "expected_chunk_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "query_type": "out_of_scope",
        "difficulty": "hard"
    },

    # =========================================================================
    # Category F: Ambiguous Medical Queries
    # =========================================================================
    {
        "id": "F1",
        "query": "What are the common secondary complications mentioned in the report?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["triage.pdf", "derma.pdf"],
        "query_type": "ambiguous",
        "difficulty": "hard"
    },
    {
        "id": "F2",
        "query": "What treatment was prescribed for the patient?",
        "expected_document_ids": ["hypertension_summary.pdf", "asthma.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["hypertension_summary.pdf", "asthma.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "DSA 1.pdf"],
        "query_type": "ambiguous",
        "difficulty": "hard"
    },
    {
        "id": "F3",
        "query": "What does section 2 discuss regarding health management?",
        "expected_document_ids": ["triage.pdf", "HealthAI_RAG_Test_Document.pdf", "synthetic_hypertension_test.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["triage.pdf", "HealthAI_RAG_Test_Document.pdf", "synthetic_hypertension_test.pdf"],
        "irrelevant_documents": ["pheo_case.pdf"],
        "query_type": "ambiguous",
        "difficulty": "hard"
    },

    # =========================================================================
    # Category G: Irrelevant Medical-Document Queries (Zero True Positive Support)
    # =========================================================================
    {
        "id": "G1",
        "query": "What are the surgical reduction techniques for comminuted femoral fractures?",
        "expected_document_ids": [],
        "expected_chunk_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "cardio.pdf", "asthma.pdf"],
        "query_type": "irrelevant",
        "difficulty": "easy"
    },
    {
        "id": "G2",
        "query": "What is the pediatric vaccination schedule for measles and mumps?",
        "expected_document_ids": [],
        "expected_chunk_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["hypertension_summary.pdf", "pheo_case.pdf"],
        "query_type": "irrelevant",
        "difficulty": "easy"
    },
    {
        "id": "G3",
        "query": "What are the dental extraction protocols for third molar impaction?",
        "expected_document_ids": [],
        "expected_chunk_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["HealthAI_RAG_Test_Document.pdf", "trial_report.pdf"],
        "query_type": "irrelevant",
        "difficulty": "easy"
    },
    {
        "id": "G4",
        "query": "What ophthalmic surgical procedures are indicated for open-angle glaucoma?",
        "expected_document_ids": [],
        "expected_chunk_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "onco_pathology.pdf"],
        "query_type": "irrelevant",
        "difficulty": "easy"
    },

    # =========================================================================
    # Category H: Exact Terminology Queries
    # =========================================================================
    {
        "id": "H1",
        "query": "Amlodipine 5mg daily prescription details",
        "expected_document_ids": ["hypertension_summary.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["hypertension_summary.pdf"],
        "irrelevant_documents": ["asthma.pdf", "ct_scan.pdf"],
        "query_type": "exact_terminology",
        "difficulty": "easy"
    },
    {
        "id": "H2",
        "query": "Metformin efficacy in Type 2 Diabetes clinical trial",
        "expected_document_ids": ["trial_report.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["trial_report.pdf"],
        "irrelevant_documents": ["derma.pdf", "cardio.pdf"],
        "query_type": "exact_terminology",
        "difficulty": "easy"
    },
    {
        "id": "H3",
        "query": "catecholamines secreted by adrenal medulla neuroendocrine tumor",
        "expected_document_ids": ["pheo_case.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["pheo_case.pdf"],
        "irrelevant_documents": ["triage.pdf", "asthma.pdf"],
        "query_type": "exact_terminology",
        "difficulty": "easy"
    },
    {
        "id": "H4",
        "query": "Stage 2 Essential Hypertension 158/96 mmHg",
        "expected_document_ids": ["hypertension_summary.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["hypertension_summary.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "onco_pathology.pdf"],
        "query_type": "exact_terminology",
        "difficulty": "easy"
    },

    # =========================================================================
    # Category I: Natural-Language Patient Questions
    # =========================================================================
    {
        "id": "I1",
        "query": "Why is my blood pressure so high and what everyday lifestyle habits can help lower it?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["derma.pdf", "onco_pathology.pdf"],
        "query_type": "natural_language",
        "difficulty": "medium"
    },
    {
        "id": "I2",
        "query": "I was diagnosed with asthma, what medicine or inhaler was recommended for me?",
        "expected_document_ids": ["asthma.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["asthma.pdf"],
        "irrelevant_documents": ["hypertension_summary.pdf", "ct_scan.pdf"],
        "query_type": "natural_language",
        "difficulty": "medium"
    },
    {
        "id": "I3",
        "query": "Can high blood pressure cause heart problems or strokes if I don't treat it?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["derma.pdf", "triage.pdf"],
        "query_type": "natural_language",
        "difficulty": "medium"
    },
    {
        "id": "I4",
        "query": "What did the doctor find on my chest CT scan and are my lungs clear?",
        "expected_document_ids": ["ct_scan.pdf"],
        "expected_chunk_ids": [],
        "relevant_documents": ["ct_scan.pdf"],
        "irrelevant_documents": ["hypertension_summary.pdf", "pheo_case.pdf"],
        "query_type": "natural_language",
        "difficulty": "easy"
    }
]
