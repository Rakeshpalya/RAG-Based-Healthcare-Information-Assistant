# RAG Retrieval Error Analysis & Sensitivity Report

**Phase**: Phase 8 — Retrieval Accuracy & Sensitivity  
**Date**: September 2026  
**Corpus**: Synthetic de-identified medical knowledge base (16 chunks, 8 domains)  
**Dataset**: 24 semantic evaluation queries (`data/evaluation/rag_evaluation.json`)  
**Index**: FAISS `IndexFlatIP` (Cosine Similarity, L2-normalized 384-dimensional embeddings)  
**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`  

---

## 1. Executive Summary

This report provides a granular examination of the vector retrieval component of the AI Healthcare Agent. Using an evaluation dataset of 24 multi-domain clinical queries, we assessed retrieval precision, recall, ranking quality (MRR), and similarity score distributions across varying sensitivity cutoffs.

> [!CAUTION]
> **Scientific & Clinical Honesty Notice**:
> Metric performance on a synthetic de-identified evaluation benchmark demonstrates mathematical retrieval relevance within a controlled vector space. It does **not** prove clinical infallibility, medical efficacy, or readiness for autonomous diagnostic decision-making.

---

## 2. Quantitative Metric Definitions

1. **Precision@K**:
   $$\text{Precision@}K = \frac{|\text{Retrieved}_K \cap \text{Expected}|}{K}$$
   *Interpretation*: Proportion of retrieved top-$K$ items that are relevant. When a query targets a single specific chunk ($|\text{Expected}|=1$), Precision@K mathematically tops out at $1/K$ (e.g., max Precision@3 = $0.333$, max Precision@5 = $0.200$). This is expected behavior and not an algorithmic failure.

2. **Recall@K**:
   $$\text{Recall@}K = \frac{|\text{Retrieved}_K \cap \text{Expected}|}{|\text{Expected}|}$$
   *Interpretation*: Fraction of ground-truth relevant chunks successfully brought into the top-$K$ context window.

3. **Mean Reciprocal Rank (MRR@K)**:
   $$\text{MRR@}K = \frac{1}{\text{rank of first relevant chunk}}$$
   *Interpretation*: Assesses whether the most relevant evidence is ranked in the top position (Rank 1 = 1.0, Rank 2 = 0.5, Rank 3 = 0.333, unretrieved = 0.0).

---

## 3. Benchmark Retrieval Results ($K \in \{1, 3, 5\}$, Threshold $= 0.25$)

| Metric | Cutoff $K=1$ | Cutoff $K=3$ | Cutoff $K=5$ | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Precision@K** | **1.0000** | 0.3472 | 0.2083 | Exactly reflects single-target query design ($1/3 \approx 0.33$, $1/5 = 0.20$) |
| **Mean Recall@K** | **0.9792** | **1.0000** | **1.0000** | $100\%$ of all required chunks retrieved within top 3 |
| **Mean MRR@K** | **1.0000** | **1.0000** | **1.0000** | Every query had an expected relevant chunk at **Rank 1** |

---

## 4. Query-by-Query Analysis & Multi-Chunk Edge Cases

### Case Study: Multi-Document Query Q019
- **Query**: *"How does high arterial blood pressure accelerate chronic kidney disease progression?"*
- **Expected Chunks**:
  1. `MED_CHUNK_4` (Nephrology: CKD progression accelerated by arterial hypertension)
  2. `MED_CHUNK_0` (Cardiology: Definition of hypertension and endothelial remodeling)
- **Observed Ranks**:
  - Rank 1: `MED_CHUNK_4` (Score: $0.6284$) — Nephrology direct hit
  - Rank 2: `MED_CHUNK_0` (Score: $0.5140$) — Cardiology secondary hit
  - Rank 3: `MED_CHUNK_10` (Score: $0.3412$) — Microalbuminuria in diabetic nephropathy
- **Metric Outcome**:
  - Precision@1: $1.0$ (1 / 1)
  - Recall@1: $0.5$ (1 / 2 expected)
  - Recall@3: $1.0$ (2 / 2 expected)
  - MRR: $1.0$
- **Analysis**: The dense retriever effectively ranked the nephrology chunk highest due to the specific query focus on kidney disease, while appropriately pulling the broader hypertension pathophysiology chunk into Rank 2.

---

## 5. Similarity Threshold Sensitivity Analysis

We performed sensitivity sweeps across five cosine similarity thresholds ($[0.20, 0.25, 0.30, 0.35, 0.40]$):

| Threshold | Precision@1 | Precision@3 | Recall@5 | MRR@5 | Excluded/Failed Queries | Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.20** | 1.0000 | 0.3472 | 1.0000 | 1.0000 | 0 | Permissive; accepts lower-similarity context chunks into prompt. |
| **0.25** | **1.0000** | **0.3472** | **1.0000** | **1.0000** | **0** | **Recommended baseline**; cleanly filters noise while retaining relevant clinical context. |
| **0.30** | 1.0000 | 0.3472 | 1.0000 | 1.0000 | 0 | Highly selective; robust on this curated corpus. |
| **0.35** | 1.0000 | 0.3472 | 1.0000 | 1.0000 | 0 | Strict; top-1 chunks remain well above $0.45$. |
| **0.40** | 1.0000 | 0.3472 | 1.0000 | 1.0000 | 0 | Very aggressive; risk of false negatives on paraphrased or patient-layman queries. |

### Trade-Off Assessment:
- **Low Threshold ($< 0.20$)**: Maximizes recall but risks injecting irrelevant text into the LLM prompt, increasing prompt token costs and distraction.
- **High Threshold ($> 0.35$)**: Minimizes false positive context but risks false negative drops (triggering the `no_relevant_context` safety stop on valid user queries phrased with non-technical vocabulary).
- **Design Decision**: Retain **0.25** as the default threshold in `backend/config.py`.

---

## 6. Failure Modes & Root Cause Analysis

While the synthetic benchmark achieved $1.0$ MRR, production medical corpora introduce common failure modes that must be proactively engineered for:

1. **Vocabulary Mismatch / Layman Phrasing**:
   - *Symptom*: Patients query *"sugar in blood"* or *"chest tightness"* while documents discuss *"hyperglycemia"* or *"angina pectoris"*.
   - *Remedy*: Pre-retrieval query rewriting / clinical term expansion in downstream agentic phases.
2. **Chunk Boundary Splitting**:
   - *Symptom*: A symptom and its contraindicating medication are split across chunk borders.
   - *Remedy*: Semantic chunking and chunk overlap ($50-100$ words).
3. **Cross-Domain Semantic Confusion**:
   - *Symptom*: Overlapping terminology (e.g., *"ACE inhibitor cough"* being retrieved under Pulmonology rather than Cardiology pharmacology).
   - *Remedy*: Hybrid search (dense vector embeddings + BM25 sparse keyword search) and cross-encoder re-ranking.
4. **Out-of-Distribution / Unsupported Queries**:
   - *Symptom*: Questions about unsupported conditions (e.g. oncology, rare genetic mutations).
   - *Remedy*: The Phase 7 safety stop successfully identifies maximum similarity $< 0.25$ and immediately halts before calling Gemini.

---

## 7. Latency Profile

- **Embedding Generation**: $\sim 20-38\ \text{ms}$ (CPU-bound `all-MiniLM-L6-v2`)
- **FAISS Search**: $\sim 0.15-0.25\ \text{ms}$ (SIMD inner-product vector search)
- **Total Retrieval Overhead**: $\mathbf{\sim 20-38\ \text{ms}}$
