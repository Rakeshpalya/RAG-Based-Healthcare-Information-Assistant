"""
Phase 2F: Adversarial Retrieval Benchmark Dataset.

Contains at least 40 rigorous adversarial queries designed to stress-test:
- Medical synonym generalization
- Ambiguous clinical intent handling
- Out-of-scope safety rejection
- Multi-aspect topic coverage
- Multi-document evidence synthesis
- Distractor resistance (lexical bait matching document titles)
- Very short medical keyword handling
- Long, noisy natural-language patient inquiries

Ground truth documents indexed in the project vector store:
- HealthAI_RAG_Test_Document.pdf
- synthetic_hypertension_test.pdf
- hypertension_summary.pdf
- trial_report.pdf (Metformin & Type 2 Diabetes)
- cardio.pdf
- asthma.pdf
- pheo_case.pdf
- onco_pathology.pdf
- ct_scan.pdf
- derma.pdf
- triage.pdf
"""

from typing import List, Dict, Any

PHASE2F_ADVERSARIAL_DATASET: List[Dict[str, Any]] = [
    # =========================================================================
    # Category A: Synonyms & Lexical Variants (5 queries)
    # =========================================================================
    {
        "id": "ADV_A1",
        "query": "What triggers sudden elevations in systemic arterial tension?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["derma.pdf", "ct_scan.pdf"],
        "query_type": "synonym",
        "category": "A_synonym",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_A2",
        "query": "What interventions regulate glycemic control in insulin resistance syndromes?",
        "expected_document_ids": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["asthma.pdf", "triage.pdf"],
        "query_type": "synonym",
        "category": "A_synonym",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_A3",
        "query": "What acute management is indicated for acute coronary occlusion and heart attack?",
        "expected_document_ids": ["cardio.pdf"],
        "relevant_documents": ["cardio.pdf"],
        "irrelevant_documents": ["derma.pdf", "trial_report.pdf"],
        "query_type": "synonym",
        "category": "A_synonym",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_A4",
        "query": "What are the clinical markers indicating progressive renal failure and loss of kidney function?",
        "expected_document_ids": ["HealthAI_RAG_Test_Document.pdf", "synthetic_hypertension_test.pdf"],
        "relevant_documents": ["HealthAI_RAG_Test_Document.pdf", "synthetic_hypertension_test.pdf"],
        "irrelevant_documents": ["asthma.pdf", "derma.pdf"],
        "query_type": "synonym",
        "category": "A_synonym",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_A5",
        "query": "What are the common secondary sequelae and target organ damages of uncontrolled vascular pressure?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "triage.pdf"],
        "query_type": "synonym",
        "category": "A_synonym",
        "difficulty": "hard",
        "should_reject": False
    },

    # =========================================================================
    # Category B: Ambiguous Clinical Queries (5 queries)
    # =========================================================================
    {
        "id": "ADV_B1",
        "query": "What complications are mentioned in the patient charts?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["triage.pdf", "derma.pdf"],
        "query_type": "ambiguous",
        "category": "B_ambiguous",
        "difficulty": "hard",
        "should_reject": False
    },
    {
        "id": "ADV_B2",
        "query": "What medications are used for treating the patient?",
        "expected_document_ids": ["hypertension_summary.pdf", "asthma.pdf", "trial_report.pdf"],
        "relevant_documents": ["hypertension_summary.pdf", "asthma.pdf", "trial_report.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "DSA 1.pdf"],
        "query_type": "ambiguous",
        "category": "B_ambiguous",
        "difficulty": "hard",
        "should_reject": False
    },
    {
        "id": "ADV_B3",
        "query": "What are the main risk factors described?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "derma.pdf"],
        "query_type": "ambiguous",
        "category": "B_ambiguous",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_B4",
        "query": "What does the clinical report recommend regarding lifestyle?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["onco_pathology.pdf", "ct_scan.pdf"],
        "query_type": "ambiguous",
        "category": "B_ambiguous",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_B5",
        "query": "What follow-up plan is established for the patient?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["derma.pdf", "ct_scan.pdf", "security_test.pdf"],
        "query_type": "ambiguous",
        "category": "B_ambiguous",
        "difficulty": "hard",
        "should_reject": True  # Unanchored ambiguous query with no follow-up in indexed summaries
    },

    # =========================================================================
    # Category C: Out-of-Scope Queries (Must be rejected) (8 queries)
    # =========================================================================
    {
        "id": "ADV_C1",
        "query": "What is the recommended treatment and mosquito eradication protocol for malaria?",
        "expected_document_ids": ["HealthAI_RAG_Test_Document.pdf"],  # Contains explicit negative boundary chunk
        "relevant_documents": ["HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["cardio.pdf", "hypertension_summary.pdf"],
        "query_type": "out_of_scope",
        "category": "C_out_of_scope",
        "difficulty": "hard",
        "should_reject": False  # Handled by grounded boundary
    },
    {
        "id": "ADV_C2",
        "query": "What are the surgical steps for total knee replacement arthroplasty in orthopedic surgery?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf", "cardio.pdf"],
        "query_type": "out_of_scope",
        "category": "C_out_of_scope",
        "difficulty": "hard",
        "should_reject": True
    },
    {
        "id": "ADV_C3",
        "query": "What are the trimester-specific ultrasound guidelines during normal pregnancy and prenatal care?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "asthma.pdf", "trial_report.pdf"],
        "query_type": "out_of_scope",
        "category": "C_out_of_scope",
        "difficulty": "hard",
        "should_reject": True
    },
    {
        "id": "ADV_C4",
        "query": "How do you calculate thermal resistance and finite element stress in structural mechanical engineering?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "cardio.pdf"],
        "query_type": "irrelevant",
        "category": "C_out_of_scope",
        "difficulty": "easy",
        "should_reject": True
    },
    {
        "id": "ADV_C5",
        "query": "What will tomorrow's weather forecast be in Seattle Washington?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "derma.pdf"],
        "query_type": "irrelevant",
        "category": "C_out_of_scope",
        "difficulty": "easy",
        "should_reject": True
    },
    {
        "id": "ADV_C6",
        "query": "How do you implement an asynchronous B-tree index in Rust or C++?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "cardio.pdf"],
        "query_type": "irrelevant",
        "category": "C_out_of_scope",
        "difficulty": "easy",
        "should_reject": True
    },
    {
        "id": "ADV_C7",
        "query": "What antibiotic dosage is indicated for acute bacterial meningitis in infants under 6 months?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "query_type": "out_of_scope",
        "category": "C_out_of_scope",
        "difficulty": "hard",
        "should_reject": True
    },
    {
        "id": "ADV_C8",
        "query": "What is the recommended chemotherapy regimen for metastatic glioblastoma multiforme?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf", "cardio.pdf"],
        "query_type": "out_of_scope",
        "category": "C_out_of_scope",
        "difficulty": "hard",
        "should_reject": True
    },

    # =========================================================================
    # Category D: Multi-Aspect Clinical Queries (5 queries)
    # =========================================================================
    {
        "id": "ADV_D1",
        "query": "Provide a comprehensive guide to hypertension: definition, etiology, symptoms, lifestyle measures, and medical treatment.",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["derma.pdf", "ct_scan.pdf"],
        "query_type": "multi_aspect",
        "category": "D_multi_aspect",
        "difficulty": "hard",
        "should_reject": False
    },
    {
        "id": "ADV_D2",
        "query": "Detail the risk factors, dietary lifestyle modifications, and vascular complications for elevated arterial pressure.",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["onco_pathology.pdf", "triage.pdf"],
        "query_type": "multi_aspect",
        "category": "D_multi_aspect",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_D3",
        "query": "What are the diagnostic blood pressure criteria, medication choices, and long-term monitoring recommendations for hypertension?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "derma.pdf"],
        "query_type": "multi_aspect",
        "category": "D_multi_aspect",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_D4",
        "query": "Explain type 2 diabetes pathophysiology, glycemic diagnostic thresholds, metformin efficacy, and secondary organ damage.",
        "expected_document_ids": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["cardio.pdf", "derma.pdf"],
        "query_type": "multi_aspect",
        "category": "D_multi_aspect",
        "difficulty": "hard",
        "should_reject": False
    },
    {
        "id": "ADV_D5",
        "query": "What are asthma triggers, clinical manifestations, inhaler pharmacology, and exacerbation risks?",
        "expected_document_ids": ["asthma.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["asthma.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["hypertension_summary.pdf", "triage.pdf"],
        "query_type": "multi_aspect",
        "category": "D_multi_aspect",
        "difficulty": "hard",
        "should_reject": False
    },

    # =========================================================================
    # Category E: Multi-Document Queries (5 queries)
    # =========================================================================
    {
        "id": "ADV_E1",
        "query": "Compare the general lifestyle guidance in the synthetic test document with patient John Doe's actual clinical vitals.",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "irrelevant_documents": ["derma.pdf", "ct_scan.pdf"],
        "query_type": "multi_document",
        "category": "E_multi_document",
        "difficulty": "hard",
        "should_reject": False
    },
    {
        "id": "ADV_E2",
        "query": "What overlap exists between hypertension management and cardiovascular disease prevention guidelines?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "cardio.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "cardio.pdf"],
        "irrelevant_documents": ["derma.pdf", "asthma.pdf"],
        "query_type": "multi_document",
        "category": "E_multi_document",
        "difficulty": "hard",
        "should_reject": False
    },
    {
        "id": "ADV_E3",
        "query": "How do chronic respiratory diseases like asthma compare to metabolic conditions like diabetes in the health research document?",
        "expected_document_ids": ["HealthAI_RAG_Test_Document.pdf", "asthma.pdf", "trial_report.pdf"],
        "relevant_documents": ["HealthAI_RAG_Test_Document.pdf", "asthma.pdf", "trial_report.pdf"],
        "irrelevant_documents": ["ct_scan.pdf", "triage.pdf"],
        "query_type": "multi_document",
        "category": "E_multi_document",
        "difficulty": "hard",
        "should_reject": False
    },
    {
        "id": "ADV_E4",
        "query": "Synthesize blood pressure diagnostic criteria from the synthetic hypertension test with the cardiology guidelines.",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "cardio.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "cardio.pdf"],
        "irrelevant_documents": ["onco_pathology.pdf", "derma.pdf"],
        "query_type": "multi_document",
        "category": "E_multi_document",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_E5",
        "query": "What clinical findings are documented across the oncology pathology report and abdominal CT scan imaging?",
        "expected_document_ids": ["onco_pathology.pdf", "ct_scan.pdf"],
        "relevant_documents": ["onco_pathology.pdf", "ct_scan.pdf"],
        "irrelevant_documents": ["asthma.pdf", "synthetic_hypertension_test.pdf"],
        "query_type": "multi_document",
        "category": "E_multi_document",
        "difficulty": "hard",
        "should_reject": False
    },

    # =========================================================================
    # Category F: Distractor Queries (Lexical Bait & False Framing) (5 queries)
    # =========================================================================
    {
        "id": "ADV_F1",
        "query": "What are the chemotherapy protocols for pancreatic adenocarcinoma in the general hypertension guideline?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "query_type": "distractor",
        "category": "F_distractor",
        "difficulty": "hard",
        "should_reject": True
    },
    {
        "id": "ADV_F2",
        "query": "According to the asthma management document, what is the surgical resection margin for lung cancer?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["asthma.pdf", "onco_pathology.pdf"],
        "query_type": "distractor",
        "category": "F_distractor",
        "difficulty": "hard",
        "should_reject": True
    },
    {
        "id": "ADV_F3",
        "query": "In the synthetic hypertension test document, what does the report state about metformin dosages for diabetes?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["synthetic_hypertension_test.pdf", "trial_report.pdf"],
        "query_type": "distractor",
        "category": "F_distractor",
        "difficulty": "hard",
        "should_reject": True
    },
    {
        "id": "ADV_F4",
        "query": "What does the dermatology pdf say regarding emergency cardioversion and defibrillation protocols?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["derma.pdf", "cardio.pdf"],
        "query_type": "distractor",
        "category": "F_distractor",
        "difficulty": "hard",
        "should_reject": True
    },
    {
        "id": "ADV_F5",
        "query": "Based on John Doe's hypertension clinical summary, what are the insulin sliding scale units prescribed?",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["hypertension_summary.pdf", "trial_report.pdf"],
        "query_type": "distractor",
        "category": "F_distractor",
        "difficulty": "hard",
        "should_reject": True
    },

    # =========================================================================
    # Category G: Very Short Keyword Queries (5 queries)
    # =========================================================================
    {
        "id": "ADV_G1",
        "query": "hypertension",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["derma.pdf", "ct_scan.pdf"],
        "query_type": "short_keyword",
        "category": "G_short_keyword",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_G2",
        "query": "diabetes",
        "expected_document_ids": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["asthma.pdf", "derma.pdf"],
        "query_type": "short_keyword",
        "category": "G_short_keyword",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_G3",
        "query": "complications",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["ct_scan.pdf", "triage.pdf"],
        "query_type": "short_keyword",
        "category": "G_short_keyword",
        "difficulty": "hard",
        "should_reject": True  # Single unanchored keyword cannot uniquely establish condition; safely halts
    },
    {
        "id": "ADV_G4",
        "query": "medications",
        "expected_document_ids": [],
        "relevant_documents": [],
        "irrelevant_documents": ["ct_scan.pdf", "DSA 1.pdf"],
        "query_type": "short_keyword",
        "category": "G_short_keyword",
        "difficulty": "hard",
        "should_reject": True  # Single unanchored keyword cannot uniquely establish condition; safely halts
    },
    {
        "id": "ADV_G5",
        "query": "asthma",
        "expected_document_ids": ["asthma.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["asthma.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["cardio.pdf", "hypertension_summary.pdf"],
        "query_type": "short_keyword",
        "category": "G_short_keyword",
        "difficulty": "easy",
        "should_reject": False
    },

    # =========================================================================
    # Category H: Long Natural-Language Patient Queries (5 queries)
    # =========================================================================
    {
        "id": "ADV_H1",
        "query": "Hello doctor, my father is 58 years old and his blood pressure reading came back as 145 over 95 mmHg during a routine clinic visit. What do these numbers mean according to the hypertension document and should we be worried?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "hypertension_summary.pdf"],
        "irrelevant_documents": ["derma.pdf", "ct_scan.pdf"],
        "query_type": "natural_language",
        "category": "H_natural_language",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_H2",
        "query": "I have been recently diagnosed with high blood pressure and I really want to avoid heavy pills if possible. Does your reference document describe specific lifestyle habits, exercises, or dietary choices like reducing sodium that could help me manage it?",
        "expected_document_ids": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["synthetic_hypertension_test.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["onco_pathology.pdf", "triage.pdf"],
        "query_type": "natural_language",
        "category": "H_natural_language",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_H3",
        "query": "My mother has diabetes and her physician mentioned metformin in a clinical study report. Can you explain what dosage was tested, how effectively it lowers hemoglobin A1c levels, and what adverse reactions were reported?",
        "expected_document_ids": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["trial_report.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["cardio.pdf", "derma.pdf"],
        "query_type": "natural_language",
        "category": "H_natural_language",
        "difficulty": "hard",
        "should_reject": False
    },
    {
        "id": "ADV_H4",
        "query": "Whenever I exercise outdoors in cold weather I experience severe wheezing, chest tightness, and shortness of breath. Based on the pulmonology asthma guidelines what type of rescue inhaler or controller medication is typically recommended?",
        "expected_document_ids": ["asthma.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "relevant_documents": ["asthma.pdf", "HealthAI_RAG_Test_Document.pdf"],
        "irrelevant_documents": ["hypertension_summary.pdf", "ct_scan.pdf"],
        "query_type": "natural_language",
        "category": "H_natural_language",
        "difficulty": "medium",
        "should_reject": False
    },
    {
        "id": "ADV_H5",
        "query": "I am reviewing laboratory results for an adrenal nodule that secretes high levels of epinephrine and norepinephrine causing severe episodic headache, palpitations, and diaphoresis. What rare condition does this case report describe?",
        "expected_document_ids": ["pheo_case.pdf"],
        "relevant_documents": ["pheo_case.pdf"],
        "irrelevant_documents": ["asthma.pdf", "derma.pdf"],
        "query_type": "natural_language",
        "category": "H_natural_language",
        "difficulty": "medium",
        "should_reject": False
    },
]
