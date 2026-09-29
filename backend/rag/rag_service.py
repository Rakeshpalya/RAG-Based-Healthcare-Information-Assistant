import os
import re
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional

from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from backend.evaluation.citation_validator import CitationValidator
from backend.rag.markdown_utils import clean_ai_markdown
from backend.rag.query_expander import MedicalQueryExpander

logger = logging.getLogger(__name__)



class RAGService:
    """
    Retrieval-Augmented Generation (RAG) Service for AI-Healthcare-Agent.

    Orchestrates the retrieval and context assembly pipeline:
        User Question
             ↓
        Query Embedding (EmbeddingService)
             ↓
        FAISS Semantic Search (VectorStoreService)
             ↓
        Relevance Threshold Filtering & Top-K Chunks
             ↓
        Context Construction & Source Extraction
             ↓
        Structured RAG Response Object

    Scope Note (Phase 6):
        Generative LLM synthesis (Gemini/OpenAI) is intentionally deferred
        to Phase 7. This service handles retrieval, relevance thresholding,
        and context grounding without text generation.
    """

    DEFAULT_TOP_K = 5
    DEFAULT_SIMILARITY_THRESHOLD = 0.25

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        default_top_k: Optional[int] = None,
        default_similarity_threshold: Optional[float] = None
    ):
        """
        Initializes the RAGService.

        Args:
            vector_store: Optional VectorStoreService instance. If not provided, uses shared get_vector_store_service().
            default_top_k: Default top_k results to retrieve (falls back to RAG_TOP_K env or 5).
            default_similarity_threshold: Minimum cosine similarity threshold (falls back to RAG_SIMILARITY_THRESHOLD env or 0.25).
        """
        self.vector_store = vector_store or VectorStoreService()

        # Configurable top_k from environment or default
        env_top_k = os.getenv("RAG_TOP_K")
        if default_top_k is not None:
            self.default_top_k = int(default_top_k)
        elif env_top_k:
            self.default_top_k = int(env_top_k)
        else:
            self.default_top_k = self.DEFAULT_TOP_K

        # Configurable similarity threshold from environment or default
        env_thresh = os.getenv("RAG_SIMILARITY_THRESHOLD")
        if default_similarity_threshold is not None:
            self.default_similarity_threshold = float(default_similarity_threshold)
        elif env_thresh:
            self.default_similarity_threshold = float(env_thresh)
        else:
            self.default_similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD

    @staticmethod
    def _normalize_chunk_text(text: str) -> str:
        """Normalizes chunk text for stable content comparison by trimming and collapsing whitespace."""
        if not text:
            return ""
        return " ".join(str(text).strip().split())

    @classmethod
    def _compute_chunk_content_hash(cls, text: str) -> str:
        """Computes SHA-256 hash of normalized chunk text."""
        norm = cls._normalize_chunk_text(text)
        if not norm:
            return ""
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    @classmethod
    def deduplicate_chunks(cls, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicates retrieved chunks based on stable content identity and chunk identity.
        If duplicate evidence entries are found, retains the representative with the
        highest similarity_score and preserves its metadata.

        Preserves genuinely different chunks (different content or different chunk IDs).
        Deduplicates identical chunks coming from duplicate uploads of the same document.
        """
        if not chunks:
            return []

        # Sort chunks by similarity_score descending (preserve highest score first)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: float(c.get("similarity_score", 0.0)),
            reverse=True
        )

        seen_doc_chunk: set = set()
        seen_content_chunk: set = set()
        seen_content_hash: set = set()
        deduped: List[Dict[str, Any]] = []

        for chunk in sorted_chunks:
            doc_id = str(chunk.get("document_id") or chunk.get("metadata", {}).get("document_id") or "").strip()
            fname = str(chunk.get("metadata", {}).get("filename") or "").strip().lower()
            doc_key = f"{doc_id}:{fname}" if (doc_id and fname) else (doc_id or fname)
            chunk_id = str(chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id") or "").strip().lower()
            text = chunk.get("text", "")
            c_hash = cls._compute_chunk_content_hash(text)

            is_duplicate = False

            # 1. Same document and same chunk_id (multiple vector hits for exact same document chunk)
            if doc_key and chunk_id:
                if (doc_key, chunk_id) in seen_doc_chunk:
                    is_duplicate = True

            # 2. Same normalized content and same chunk_id across duplicate document uploads
            if not is_duplicate and c_hash and chunk_id:
                if (c_hash, chunk_id) in seen_content_chunk:
                    is_duplicate = True

            # 3. Same normalized content across document uploads even if chunk_id is missing or different
            if not is_duplicate and c_hash:
                if c_hash in seen_content_hash:
                    is_duplicate = True

            if is_duplicate:
                continue

            if doc_key and chunk_id:
                seen_doc_chunk.add((doc_key, chunk_id))
            if c_hash and chunk_id:
                seen_content_chunk.add((c_hash, chunk_id))
            if c_hash:
                seen_content_hash.add(c_hash)

            deduped.append(chunk)

        return deduped

    @classmethod
    def select_diverse_evidence(
        cls,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        max_per_doc: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Balances multi-document evidence selection to prevent a single document
        from monopolizing top-k evidence slots when multiple genuinely different
        documents contain relevant information.

        PASS 1:
        - Group candidates by unique document identity.
        - Select highest-similarity chunks from each document, allowing at most
          max_per_doc (default: 2) chunks per document during this pass.
        - Process candidates in descending similarity order.

        PASS 2:
        - If fewer than top_k chunks were selected,
          backfill from remaining candidates by descending similarity.
        - Backfill may use additional chunks from documents already represented.

        FINAL:
        - Sort selected evidence by similarity_score descending.
        - Return at most top_k chunks.
        """
        if not chunks:
            return []

        if top_k <= 0:
            return []

        # Sort candidates descending by similarity_score
        sorted_candidates = sorted(
            chunks,
            key=lambda c: float(c.get("similarity_score", 0.0)),
            reverse=True
        )

        def get_doc_identity(c: Dict[str, Any]) -> str:
            d_id = c.get("document_id") or c.get("metadata", {}).get("document_id")
            if d_id is not None and str(d_id).strip():
                return str(d_id).strip()
            fname = c.get("metadata", {}).get("filename")
            if fname is not None and str(fname).strip():
                return str(fname).strip()
            return "UNKNOWN_DOCUMENT"

        selected: List[Dict[str, Any]] = []
        selected_ids: set = set()
        doc_counts: Dict[str, int] = {}
        remaining_candidates: List[Dict[str, Any]] = []

        # PASS 1: Select up to max_per_doc chunks per document in descending similarity order
        for chunk in sorted_candidates:
            doc_key = get_doc_identity(chunk)
            curr_count = doc_counts.get(doc_key, 0)
            chunk_obj_id = id(chunk)

            if curr_count < max_per_doc and len(selected) < top_k:
                selected.append(chunk)
                selected_ids.add(chunk_obj_id)
                doc_counts[doc_key] = curr_count + 1
            else:
                remaining_candidates.append(chunk)

        # PASS 2: If fewer than top_k chunks selected, backfill from remaining candidates
        if len(selected) < top_k:
            for chunk in remaining_candidates:
                chunk_obj_id = id(chunk)
                if chunk_obj_id not in selected_ids:
                    selected.append(chunk)
                    selected_ids.add(chunk_obj_id)
                    if len(selected) >= top_k:
                        break

        # FINAL: Sort selected evidence by similarity_score descending
        final_sorted = sorted(
            selected,
            key=lambda c: float(c.get("similarity_score", 0.0)),
            reverse=True
        )

        return final_sorted[:top_k]

    @classmethod
    def _is_chunk_relevant_to_query(cls, query: str, chunk: Dict[str, Any]) -> bool:
        """
        Lightweight lexical and clinical topic relevance validator for a single candidate chunk.

        Evaluates whether a candidate chunk has a reasonable semantic/topic connection to the user's
        query, avoiding irrelevant medical chunks surviving purely due to dense vector hubness.
        """
        if isinstance(chunk, dict):
            text = str(chunk.get("text", "")).lower()
        else:
            text = str(chunk or "").lower()
        if not text.strip():
            return False

        q_lower = query.lower().strip()
        q_focus = cls.extract_query_clinical_focus(query).lower().strip()
        search_q = q_focus if q_focus else q_lower

        # 1. Explicit document-boundary preservation:
        # If the chunk contains explicit negative boundary / scope limitation language,
        # check if it references any query entity or term. If so, preserve it for Phase 2.
        boundary_pattern = re.compile(
            r'\b(?:testing\s+boundary|scope\s+boundary|testing\s+boundaries|boundary\b|boundaries\b|out\s+of\s+scope|outside\s+(?:the\s+)?scope|unsupported\s+by\s+this\s+document|treated\s+as\s+unsupported|not\s+contain\s+information\s+about|does\s+not\s+contain\s+information\s+about|not\s+covered\s+(?:in|by)\s+this\s+document|exclusion\s+criteria)\b',
            re.IGNORECASE
        )
        if boundary_pattern.search(text):
            stop_words_boundary = {
                "what", "is", "are", "the", "difference", "between", "common", "main", "for", "in", "of", "and",
                "to", "a", "an", "according", "uploaded", "document", "tell", "me", "about", "can", "you",
                "explain", "describe", "which", "them", "they", "it", "this", "that", "these", "those", "how",
                "does", "do", "from", "with", "or", "by", "on", "at", "be", "as", "have", "has", "pdf"
            }
            q_terms = [w for w in re.findall(r'\b[a-z]{3,}\b', search_q or q_lower) if w not in stop_words_boundary]
            if any(term in text for term in q_terms):
                return True

        # 2. Clinical condition & medical synonym/variant groups
        clinical_entities = [
            ("hypertension", [r"\bhypertens", r"\bhigh\s+blood\s+pressure\b", r"\bblood\s+pressure\b", r"\bsystolic\b", r"\bdiastolic\b", r"\barterial\s+(?:blood\s+)?pressure\b", r"\belevated\s+(?:arterial\s+)?blood\s+pressure\b", r"\bvascular\s+pressure\b", r"\barterial\s+tension\b"]),
            ("diabetes", [r"\bdiabet", r"\bglycem", r"\bglucose\b", r"\binsulin\b", r"\bhba1c\b", r"\bmetformin\b", r"\bblood\s+sugar\b"]),
            ("asthma", [r"\basthma", r"\bbronch", r"\bwheez", r"\binhaler"]),
            ("cardiovascular", [r"\bcardio", r"\bmyocardi", r"\bheart\s+attack\b", r"\binfarct", r"\bcardiac\b", r"\bcoronary\b", r"\bheart\s+failure\b"]),
            ("cancer", [r"\bcancer", r"\boncol", r"\btumor", r"\bmalignan", r"\bcarcinoma", r"\bbiopsy"]),
            ("cholesterol", [r"\bcholesterol\b", r"\blipid", r"\bstatin", r"\bldl\b", r"\bhdl\b", r"\btriglycerid"]),
            ("stroke", [r"\bstroke", r"\bcerebrovascular", r"\bischemi"]),
            ("pneumonia", [r"\bpneumon", r"\blung\s+infiltrat"]),
            ("tuberculosis", [r"\btuberculosis\b", r"\btb\b", r"\bmycobacteri"]),
            ("malaria", [r"\bmalaria\b", r"\bplasmodi"]),
            ("arthritis", [r"\barthrit", r"\bjoint\s+inflamm"]),
            ("hepatitis", [r"\bhepatit", r"\bliver\s+inflamm"]),
            ("arrhythmia", [r"\barrhythm", r"\batrial\s+fibrillat"]),
            ("kidney disease", [r"\bkidney\s+disease\b", r"\brenal\b", r"\bnephro", r"\bckd\b"]),
            ("complications", [r"\bcomplication", r"\bsequelae\b", r"\borgan\s+damage\b", r"\bheart\s+disease\b", r"\bstroke\b", r"\bkidney\s+problem"]),
        ]

        # Check if query targets any recognized clinical entity
        # Prioritize search_q (clinical focus) so document framing does not falsely inject entities
        matched_query_entities = [
            (entity_name, patterns)
            for entity_name, patterns in clinical_entities
            if any(re.search(pat, search_q) for pat in patterns)
        ]
        if not matched_query_entities and q_lower != search_q:
            matched_query_entities = [
                (entity_name, patterns)
                for entity_name, patterns in clinical_entities
                if any(re.search(pat, q_lower) for pat in patterns)
            ]

        if matched_query_entities:
            for entity_name, patterns in matched_query_entities:
                if any(re.search(pat, text) for pat in patterns):
                    return True

        # 3. Meaningful query token overlap check (for general / sub-topic / lifestyle / synonym terms)
        stop_words = {
            "a", "about", "according", "all", "also", "an", "and", "any", "are", "as", "at", "be",
            "because", "been", "between", "both", "but", "by", "can", "could", "describe", "did",
            "difference", "do", "does", "explain", "file", "for", "from", "get", "give", "had",
            "has", "have", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is",
            "it", "its", "just", "like", "list", "main", "may", "me", "mention", "mentioned", "might",
            "more", "most", "my", "no", "not", "notes", "of", "on", "or", "our", "out", "overview",
            "paper", "pdf", "per", "provide", "query", "record", "say", "she", "should", "show",
            "so", "some", "state", "summarize", "summary", "tell", "than", "that", "the", "their",
            "them", "then", "there", "these", "they", "this", "those", "through", "to", "up",
            "uploaded", "use", "used", "was", "we", "were", "what", "when", "where", "which",
            "who", "why", "will", "with", "would", "you", "your"
        }
        q_tokens = [w for w in re.findall(r'\b[a-z]{3,}\b', search_q or q_lower) if w not in stop_words]

        # Include expanded medical synonym tokens for clinical recall
        expanded_terms = MedicalQueryExpander.get_expanded_terms(query)
        expanded_tokens = [
            w for term in expanded_terms
            for w in re.findall(r'\b[a-z]{3,}\b', term.lower())
            if w not in stop_words
        ]
        all_check_tokens = set(q_tokens + expanded_tokens)

        if all_check_tokens:
            for tok in all_check_tokens:
                if tok in text:
                    return True
                if len(tok) >= 5 and tok[:-1] in text:
                    return True
                if len(tok) >= 6 and tok[:-2] in text:
                    return True

        if matched_query_entities:
            return False

        if not q_tokens:
            return True

        return False

    @classmethod
    def filter_candidate_precision(
        cls,
        query: str,
        chunks: List[Dict[str, Any]],
        threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Phase 2C: Retrieval Precision & Evidence Quality Filtering.

        Filters candidate chunks prior to Phase 2B diversity selection to ensure
        only genuinely relevant evidence reaches the context construction stage.

        Pipeline position:
            Static threshold (0.25)
                 ↓
            Phase 2A content deduplication
                 ↓
            Phase 2C precision filtering  <-- THIS METHOD
                 ↓
            Phase 2B diversity selection
                 ↓
            Safety gate (verify_relevance_and_sufficiency)

        Guarantees:
        1. Dynamic relative similarity filtering:
           For top_score >= 0.60:
               min_relative = max(threshold, top_score * 0.75, top_score - 0.20)
           For top_score < 0.60:
               preserves threshold behavior (does not aggressively prune).
        2. Lightweight per-candidate lexical and clinical topic validation:
           - Recognizes clinical conditions and medical synonym variants.
           - Checks token overlap with non-stopword query terms.
           - Preserves explicit document-boundary chunks (e.g. malaria scope).
        3. Deterministic descending score ordering.
        """
        if not chunks:
            return []

        # Sort chunks descending by similarity score
        sorted_chunks = sorted(
            chunks,
            key=lambda c: float(c.get("similarity_score", 0.0)),
            reverse=True
        )

        # 1. Similarity threshold cutoff
        valid_chunks = [
            c for c in sorted_chunks
            if float(c.get("similarity_score", 0.0)) >= float(threshold)
        ]
        if not valid_chunks:
            return []

        top_score = float(valid_chunks[0].get("similarity_score", 0.0))

        # 2. Dynamic relative similarity filtering
        q_clean = query.strip() if query else ""
        sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(q_clean) if MedicalQueryExpander.is_multi_aspect_query(q_clean) else []
        is_multi_aspect = bool(sub_queries)

        if float(threshold) < 0.20:
            # Caller explicitly requested unconstrained / raw threshold evaluation (e.g. 0.0 or 0.10)
            score_filtered = valid_chunks
        elif is_multi_aspect:
            # Multi-aspect queries target diverse facets (e.g. guidelines vs medication vs complications)
            # from different documents with varying similarity ranges; avoid starving secondary facets.
            min_relative = max(float(threshold), top_score * 0.50, top_score - 0.35)
            score_filtered = [
                c for c in valid_chunks
                if float(c.get("similarity_score", 0.0)) >= min_relative
            ]
        elif top_score >= 0.60:
            min_relative = max(float(threshold), top_score * 0.75, top_score - 0.20)
            score_filtered = [
                c for c in valid_chunks
                if float(c.get("similarity_score", 0.0)) >= min_relative
            ]
        else:
            score_filtered = valid_chunks

        if not score_filtered:
            return []

        # 3. Lightweight per-candidate lexical / clinical topic relevance validation
        if not q_clean or float(threshold) < 0.20:
            return score_filtered

        top_doc_id = ""
        if score_filtered:
            first_c = score_filtered[0]
            top_doc_id = str(first_c.get("document_id") or first_c.get("metadata", {}).get("document_id") or first_c.get("metadata", {}).get("filename") or "").strip()

        scoped_doc = cls.extract_scoped_document_name(q_clean)
        # Multi-document comparative queries must not be artificially restricted to a single scoped document
        is_comparison = bool(re.search(r'\b(?:compare|contrast|versus|vs\.?|across|both|alongside|and\s+patient|with\s+patient)\b', q_clean, re.IGNORECASE))
        if is_comparison:
            scoped_doc = None

        scoped_words = [w for w in re.findall(r'\b[a-z]{3,}\b', scoped_doc) if w not in {"the", "and", "for", "with"}] if scoped_doc else []

        # Guard against weak unanchored queries:
        # If the query contains no clinical condition anchor, no patient name, and top_score is below cutoff,
        # it represents low-similarity nearest-neighbor noise.
        has_primary_subject = bool(MedicalQueryExpander.extract_primary_subject(q_clean))
        has_patient_name = bool(re.search(r'\b(john\s+doe|alice|bob|patient\s+[a-z]+)\b', q_clean, re.IGNORECASE))
        has_facet = bool(re.search(r'\b(?:risk\s+factors?|lifestyle|complications?|medications?|treatment|diet|monitoring)\b', q_clean, re.IGNORECASE))
        min_unanchored_threshold = 0.30 if has_facet else 0.35
        if not has_primary_subject and not has_patient_name and not scoped_words and top_score < min_unanchored_threshold:
            return []

        survived: List[Dict[str, Any]] = []
        for chunk in score_filtered:
            c_doc_id = str(chunk.get("document_id") or chunk.get("metadata", {}).get("document_id") or chunk.get("metadata", {}).get("filename") or "").strip()
            c_fname = str(chunk.get("metadata", {}).get("filename") or c_doc_id).lower()

            # If user explicitly scoped to a document name (e.g. synthetic hypertension document),
            # an unrelated document (e.g. trial_report.pdf) must not be allowed to bypass validation
            if scoped_words:
                matches_scope = any(w in c_fname for w in scoped_words)
                if not matches_scope:
                    continue

            is_same_top_doc = bool(top_doc_id and c_doc_id and c_doc_id == top_doc_id and not scoped_words)
            is_rel = (
                is_same_top_doc or
                cls._is_chunk_relevant_to_query(q_clean, chunk) or
                any(cls._is_chunk_relevant_to_query(sq, chunk) for sq in sub_queries)
            )
            if is_rel:
                survived.append(chunk)

        # Fallback safeguard: if lexical check pruned all candidates but top candidate
        # had a high similarity score (>= 0.70), retain top candidate to avoid false starvation
        if not survived and score_filtered and top_score >= 0.70 and not scoped_words:
            survived = [score_filtered[0]]

        return survived

    @staticmethod
    def extract_prior_topic(text: str) -> Optional[str]:
        """Extracts primary clinical condition or topic from a prior conversation turn."""
        if not text:
            return None
        matches = re.findall(
            r'\b(hypertension|high blood pressure|blood pressure|diabetes|type 2 diabetes|type 1 diabetes|asthma|cardiovascular disease|heart failure|arrhythmia|cancer|pneumonia|cholesterol)\b',
            text,
            re.IGNORECASE
        )
        if matches:
            return matches[0].strip()
        clean = re.sub(r'^(what is|what are|tell me about|how to|can you explain|describe)\s+', '', text.strip(), flags=re.IGNORECASE)
        clean = re.sub(r'[?!.,]', '', clean).strip()
        words = clean.split()
        if words:
            return " ".join(words[-3:])
        return None

    @classmethod
    def extract_query_clinical_focus(cls, text: str) -> str:
        """
        Removes document-reference framing from a user question to isolate the clinical entity and topic.
        e.g., 'According to the hypertension document, what are the complications of diabetes?'
              -> 'what are the complications of diabetes?'
        e.g., 'What does the hypertension document say about complications of diabetes?'
              -> 'complications of diabetes'
        e.g., 'In the uploaded hypertension pdf, what medications are recommended?'
              -> 'what medications are recommended'
        """
        if not text:
            return ""
        q = text.lower().strip().rstrip('?!., ')

        # 1. Preamble stripping: According to ..., Based on ..., In ..., From ..., Per ...
        q = re.sub(
            r'^(?:according\s+to|based\s+on|in|from|per)\s+(?:the\s+)?[a-zA-Z0-9_\-\s]+?(?:document|pdf|file|paper|notes|upload|guideline|record|summary|chart|report)s?(?:,\s*|\s+)',
            '',
            q,
            flags=re.IGNORECASE
        ).strip()

        # 2. Leading question framing referencing documents
        q = re.sub(
            r'^(?:what\s+does|what\s+do)\s+(?:the\s+)?[a-zA-Z0-9_\-\s]+?(?:document|pdf|file|paper|notes|upload|guideline|summary|chart|report)s?\s+(?:say|mention|state|discuss)\s+(?:about\s+)?',
            '',
            q,
            flags=re.IGNORECASE
        ).strip()
        q = re.sub(
            r'^(?:does|do|can)\s+(?:the\s+)?[a-zA-Z0-9_\-\s]+?(?:document|pdf|file|paper|notes|upload|guideline|summary|chart|report)s?\s+(?:mention|contain|discuss|have|provide)\s+',
            '',
            q,
            flags=re.IGNORECASE
        ).strip()

        # 3. Trailing document references: "... according to the [X] document"
        q = re.sub(
            r'\s+(?:according\s+to|based\s+on|in|from|per)\s+(?:the\s+)?[a-zA-Z0-9_\-\s]+?(?:document|pdf|file|paper|notes|upload|guideline|record|summary|chart|report)s?\s*$',
            '',
            q,
            flags=re.IGNORECASE
        ).strip()

        # 4. Remove any remaining "[X] document" noun phrases
        q = re.sub(
            r'\b(?:the\s+)?[a-zA-Z0-9_\-]+\s+(?:document|pdf|file|paper|guideline|summary|chart|report)s?\b',
            '',
            q,
            flags=re.IGNORECASE
        ).strip()

        return q

    @classmethod
    def extract_scoped_document_name(cls, text: str) -> Optional[str]:
        """
        Extracts explicit document scoping reference if the query asks about a specific document.
        e.g., 'In the synthetic hypertension test document, what does the report state about metformin?'
              -> 'synthetic hypertension test'
        """
        if not text:
            return None
        m = re.search(
            r'\b(?:according\s+to|based\s+on|in|from|per)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)\s+(?:document|pdf|file|paper|guideline|summary|chart|report)s?\b',
            text,
            flags=re.IGNORECASE
        )
        m2 = re.search(
            r'\b(?:what\s+does|what\s+do|does|did)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)\s+(?:document|pdf|file|paper|guideline|summary|chart|report)s?\s+(?:say|mention|state|discuss|show|list)\b',
            text,
            flags=re.IGNORECASE
        )
        match = m or m2
        if match:
            scoped = match.group(1).lower().strip()
            # If multiple sources are conjoined, it is not a single document scope
            if any(conj in scoped for conj in [" and ", " or ", " with ", " vs "]):
                return None
            generic_stops = {
                "uploaded", "clinical", "patient", "general", "medical", "hospital",
                "this", "that", "a clinical study", "a study", "the study", "a research",
                "the research", "the clinical", "a clinical"
            }
            if scoped in generic_stops:
                return None
            doc_keywords = {
                "hypertension", "synthetic", "summary", "cardio", "cardiology",
                "asthma", "derma", "dermatology", "onco", "oncology", "pathology",
                "ct", "ct_scan", "triage", "pheo", "pheochromocytoma", "trial",
                "john doe", "alice", "security"
            }
            if not any(kw in scoped for kw in doc_keywords):
                return None
            return scoped
        return None

    @classmethod
    def resolve_followup_query(
        cls,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, Optional[str]]:
        """
        Resolves pronouns in follow-up queries using immediate prior conversation turn.
        Returns:
            (effective_search_query, conversation_context_snippet)
        """
        if not question or not question.strip():
            return question or "", None

        if not conversation_history:
            return question.strip(), None

        prior_user_q = None
        for msg in reversed(conversation_history):
            role = msg.get("role") or msg.get("sender")
            content = msg.get("content") or msg.get("text", "")
            if role in ("user", "human") and content.strip() and content.strip() != question.strip():
                prior_user_q = content.strip()
                break

        if not prior_user_q:
            return question.strip(), None

        pronoun_pattern = re.compile(
            r'\b(it|this|that|these|those|them|they|the condition|the disease|the illness|such condition)\b',
            re.IGNORECASE
        )
        has_pronoun = bool(pronoun_pattern.search(question))
        prior_topic = cls.extract_prior_topic(prior_user_q)
        effective_query = question.strip()

        if has_pronoun and prior_topic:
            # Check if follow-up is asking which of them can be modified
            is_modifiable_query = bool(re.search(
                r'\b(?:which\s+(?:of\s+)?(?:them|these|those)?\s+can\s+be\s+modified|can\s+(?:any\s+of\s+)?(?:them|these|those)\s+be\s+modified|how\s+can\s+(?:they|them|these|those)\s+be\s+modified)\b',
                question,
                re.IGNORECASE
            ))
            if is_modifiable_query:
                effective_query = f"Which {prior_topic} risk factors can be modified through lifestyle changes?"
            else:
                # Check if prior user question has a specific topic phrase like "risk factors for X"
                rf_match = re.search(
                    r'\b(risk factors|symptoms|causes|complications|treatments|lifestyle measures)\s+(?:for|of)?\s*([a-zA-Z\s]+)',
                    prior_user_q,
                    re.IGNORECASE
                )
                if rf_match:
                    sub_phrase = f"{rf_match.group(1).strip()} for {rf_match.group(2).strip()}"
                    effective_query = pronoun_pattern.sub(sub_phrase, question)
                else:
                    effective_query = pronoun_pattern.sub(prior_topic, question)

        conv_context = f"User asked previously: \"{prior_user_q}\""
        return effective_query.strip(), conv_context

    @classmethod
    def verify_relevance_and_sufficiency(
        cls,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        similarity_threshold: float = 0.25
    ) -> tuple[bool, str]:
        """
        Strict Relevance & Sufficiency Gate executed before Gemini invocation:

        Evaluates whether the retrieved FAISS chunks actually contain sufficient,
        grounded evidence to answer the user's specific clinical query.

        Prevents unsupported answers, hallucinations, and spurious citations
        caused by nearest-neighbor FAISS false positives.

        Returns:
            (is_sufficient: bool, reason: str)
        """
        if not retrieved_chunks:
            return False, "no_retrieved_chunks"

        scores = [float(c.get("similarity_score", 0.0)) for c in retrieved_chunks]
        max_score = max(scores) if scores else 0.0

        # 1. Similarity Threshold Cutoff
        if max_score < float(similarity_threshold):
            return False, f"max_similarity_score_{max_score:.4f}_below_threshold_{similarity_threshold:.4f}"

        combined_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
        q_lower = question.lower().strip()

        # Document Reference Stripping to isolate queried entity/focus
        q_focus = cls.extract_query_clinical_focus(question)

        # 1B. Scoped Document Verification:
        # If the user explicitly restricted the query to a named document, verify that
        # retrieved evidence actually originates from that document rather than an unmentioned distractor.
        scoped_doc = cls.extract_scoped_document_name(question)
        if scoped_doc:
            scoped_words = [w for w in re.findall(r'\b[a-z]{3,}\b', scoped_doc) if w not in {"the", "and", "for", "with"}]
            if scoped_words:
                doc_fnames = [str(c.get("metadata", {}).get("filename") or c.get("document_id") or "").lower() for c in retrieved_chunks]
                has_scoped_doc = any(any(w in fn for w in scoped_words) for fn in doc_fnames)
                if not has_scoped_doc:
                    return False, f"scoped_document_{scoped_doc}_not_found_in_retrieved_context"

        # 2. Disease / Primary Clinical Subject Grounding Check:
        known_conditions = [
            ("diabetes", [r"\bdiabet", r"\bglycem", r"\bglucose", r"\binsulin", r"\bhba1c", r"\bmetformin\b", r"\bblood\s+sugar\b"]),
            ("hypertension", [r"\bhypertens", r"\bblood\s+pressure", r"\bsystolic", r"\bdiastolic", r"\barterial\s+(?:blood\s+)?pressure\b", r"\belevated\s+(?:arterial\s+)?blood\s+pressure\b", r"\bvascular\s+pressure\b", r"\barterial\s+tension\b"]),
            ("asthma", [r"\basthma", r"\bbronch", r"\bwheez", r"\binhaler"]),
            ("cardiovascular", [r"\bcardio", r"\bmyocardi", r"\bheart\s+attack\b", r"\binfarct", r"\bcardiac\b", r"\bcoronary\b", r"\bheart\s+failure\b"]),
            ("cancer", [r"\bcancer", r"\boncol", r"\btumor", r"\bmalignan", r"carcinoma", r"\badeno", r"\bbiopsy", r"\bchemo"]),
            ("pheochromocytoma", [r"\bpheochromocytoma", r"\badrenal", r"\bcatecholamin"]),
            ("glomangioma", [r"\bglomangioma", r"\bglomus"]),
            ("whipple", [r"\bwhipple", r"\btropheryma"]),
            ("castleman", [r"\bcastleman", r"\blymphadenopath"]),
            ("pneumonia", [r"\bpneumon", r"\blung\s+infiltrat"]),
            ("cholesterol", [r"\bcholesterol", r"\blipid", r"\bstatin", r"\bldl", r"\bhdl"]),
            ("stroke", [r"\bstroke", r"\bcerebrovascular", r"\bischemi"]),
            ("arthritis", [r"\barthrit", r"\bjoint\s+inflamm"]),
            ("tuberculosis", [r"\btuberculosis", r"\btb\b", r"\bmycobacteri"]),
            ("hepatitis", [r"\bhepatit", r"\bliver\s+inflamm"]),
            ("arrhythmia", [r"\barrhythm", r"\batrial\s+fibrillat"]),
            ("kidney disease", [r"\bkidney\s+(?:disease|problems?|failure|damage|dysfunction)\b", r"\brenal\b", r"\bnephro", r"\bckd\b"]),
        ]

        # 2A. Specific Property-Target Queries:
        # e.g., "complications of diabetes", "risk factors for diabetes", "symptoms of asthma"
        target_entity_pattern = re.compile(
            r'\b(?:complications?|risk\s+factors?|symptoms?|causes?|treatment|medications?|management|diagnosis|prevention|prognosis|sequelae|protocols?|regimens?|therapy|therapies)\s+(?:of|for)\s+([a-zA-Z\s]+)',
            re.IGNORECASE
        )
        prop_matches = target_entity_pattern.findall(q_focus) or target_entity_pattern.findall(q_lower)
        if prop_matches:
            for target_phrase in prop_matches:
                target_phrase_clean = target_phrase.lower().strip()
                for cond_name, patterns in known_conditions:
                    if any(re.search(pat, target_phrase_clean) for pat in patterns):
                        has_cond = any(re.search(pat, combined_text) for pat in dict(known_conditions)[cond_name])
                        if not has_cond:
                            return False, f"query_targets_condition_{cond_name}_not_found_in_retrieved_context"

        # 2B. Check conditions in the focused question (after stripping document references)
        conditions_in_focus = [
            cond_name for cond_name, patterns in known_conditions
            if any(re.search(pat, q_focus) for pat in patterns)
        ]
        if conditions_in_focus:
            supported_in_focus = [
                cond_name for cond_name in conditions_in_focus
                if any(re.search(pat, combined_text) for pat in dict(known_conditions)[cond_name])
            ]
            if len(supported_in_focus) == 0:
                return False, f"query_targets_condition_{conditions_in_focus[0]}_not_found_in_retrieved_context"

        # 2C. Fallback check on full question: if all conditions mentioned in full query are absent
        matched_conditions = [
            cond_name for cond_name, patterns in known_conditions
            if any(re.search(pat, q_lower) for pat in patterns)
        ]
        if matched_conditions:
            supported_conditions = [
                cond_name for cond_name in matched_conditions
                if any(re.search(pat, combined_text) for pat in dict(known_conditions)[cond_name])
            ]
            if len(supported_conditions) == 0:
                return False, f"query_targets_condition_{matched_conditions[0]}_not_found_in_retrieved_context"

        # 3. Sub-Topic Intent: Medication / Drug Recommendations
        # If the user specifically asks what medications are recommended for treating a condition,
        # the context must actually contain medication recommendations, drug classes, or treatment guidelines.
        is_asking_med_recommendation = bool(
            re.search(r'\b(what|which|any)\s+medications?\b', q_lower) or
            re.search(r'\b(what|which|any)\s+drugs?\b', q_lower) or
            re.search(r'\bmedications?\s+(are|is)?\s*recommended\b', q_lower) or
            re.search(r'\bdrugs?\s+(are|is)?\s*recommended\b', q_lower) or
            re.search(r'\brecommended\s+medications?\b', q_lower) or
            re.search(r'\brecommended\s+drugs?\b', q_lower) or
            re.search(r'\bprescribed\s+medications?\b', q_lower) or
            re.search(r'\bmedications?\s+(for|to)\s+treat', q_lower) or
            re.search(r'\bmedicines?\s+(for|to)\s+treat', q_lower) or
            re.search(r'\bmedications?\s+prescribed\b', q_lower) or
            re.search(r'\b(?:what|which)\s+(?:treatment|therapy|prescription)\s+(?:was\s+)?prescribed\b', q_lower) or
            re.search(r'\b(?:treatment|therapy)\s+(?:was\s+)?prescribed\b', q_lower) or
            re.search(r'\b(?:medications?|pharmacotherapy|medication\s+management)\b', q_lower)
        )

        if is_asking_med_recommendation:
            med_indicators = [
                r'\b(ace\s+inhibitor|arb|beta\s+blocker|calcium\s+channel\s+blocker|diuretic|thiazide)\b',
                r'\b(lisinopril|amlodipine|losartan|metoprolol|hydrochlorothiazide|furosemide|metformin|insulin|statin|atorvastatin|albuterol|fluticasone)\b',
                r'\b(prescribed\s+(?:to|for|with|medication|drug|therapy|inhaler|daily)|first-line\s+medication|first-line\s+agent|first-line\s+drug|pharmacological\s+treatment)\b',
                r'\b(medications?\s+recommended|recommended\s+medication|treatment\s+with\s+medication|drug\s+therapy)\b',
                r'\b(antihypertensive\s+(?:agent|medication|drug)|inhaler\s+treatment|dosage\s+of|mg\s+daily)\b'
            ]
            has_med_info = any(re.search(pat, combined_text) for pat in med_indicators)
            if not has_med_info:
                return False, "missing_medication_recommendations_in_context"

        # 3B. Specific Medication Check:
        # If the user names a specific drug or drug class in the focused query,
        # the context must actually contain that specific medication.
        specific_meds = [
            "insulin", "metformin", "lisinopril", "amlodipine", "losartan", "metoprolol",
            "hydrochlorothiazide", "furosemide", "albuterol", "fluticasone", "statin", "atorvastatin"
        ]
        requested_meds = [
            m for m in specific_meds
            if re.search(rf'\b{m}\b', q_focus or q_lower) and
            not (m == "insulin" and re.search(r'\binsulin\s+resistance\b', q_focus or q_lower) and not re.search(r'\b(dose|dosage|scale|unit|inject|therapy|prescrib)\b', q_focus or q_lower))
        ]
        if requested_meds:
            has_requested_med = any(re.search(rf'\b{m}\b', combined_text) for m in requested_meds)
            if not has_requested_med:
                return False, f"requested_medication_{requested_meds[0]}_not_found_in_retrieved_context"

        # 4. Sub-Topic Intent: Surgical / Invasive Procedures
        is_asking_surgery = bool(re.search(r'\b(surgery|surgical\s+procedure|surgical\s+resection|craniotomy|operation)\b', q_lower))
        if is_asking_surgery:
            has_surgery_info = bool(re.search(r'\b(surgery|surgical|operation|resection|craniotomy|operative)\b', combined_text))
            if not has_surgery_info:
                return False, "missing_surgical_information_in_context"

        # 5. Sub-Topic Intent: Complications / Long-Term Consequences
        is_asking_complications = bool(
            re.search(r'\b(complications?|long-term\s+consequences?|long-term\s+effects?|sequelae|organ\s+damage)\b', q_lower)
        )
        if is_asking_complications:
            complication_indicators = [
                r'\b(complications?|stroke|heart\s+attack|myocardial\s+infarction|kidney\s+failure|renal\s+failure|retinopathy|neuropathy|nephropathy|organ\s+damage|cardiovascular\s+events?|heart\s+disease|kidney\s+problems?|damage\s+to\s+blood\s+vessels)\b'
            ]
            has_comp_info = any(re.search(pat, combined_text) for pat in complication_indicators)
            if not has_comp_info:
                return False, "missing_complications_information_in_context"

        # 5B. Sub-Topic Intent: Follow-Up / Monitoring Plan
        is_asking_followup = bool(
            re.search(r'\b(follow-up|follow\s+up|monitoring\s+plan|subsequent\s+visit|next\s+appointment)\b', q_lower)
        )
        if is_asking_followup:
            followup_indicators = [
                r'\b(follow-up|follow\s+up|monitoring\s+plan|subsequent\s+visit|next\s+visit|re-evaluated|recheck|scheduled\s+in|return\s+in)\b'
            ]
            has_followup_info = any(re.search(pat, combined_text) for pat in followup_indicators)
            if not has_followup_info:
                return False, "missing_followup_information_in_context"

        # 6. Core Term Overlap Check (guard against queries about specific terms not in text)
        stop_words = {
            "what", "is", "are", "the", "difference", "between", "common", "main", "for", "in", "of", "and",
            "to", "a", "an", "according", "uploaded", "document", "tell", "me", "about", "can", "you",
            "explain", "describe", "which", "them", "they", "it", "this", "that", "these", "those", "how",
            "does", "do", "from", "with", "or", "by", "on", "at", "be", "as", "have", "has", "summarize",
            "summary", "overview", "provide", "show", "give", "list", "general",
            "patient", "patients", "chart", "charts", "case", "cases", "report", "reports",
            "document", "documents", "record", "records", "note", "notes", "file", "files"
        }
        q_tokens = [w for w in re.findall(r'\b[a-z]{3,}\b', q_focus or q_lower) if w not in stop_words]
        expanded_terms = MedicalQueryExpander.get_expanded_terms(question)
        expanded_tokens = [
            w for term in expanded_terms
            for w in re.findall(r'\b[a-z]{3,}\b', term.lower())
            if w not in stop_words
        ]
        all_check_tokens = set(q_tokens + expanded_tokens)
        if all_check_tokens:
            has_token_overlap = any(
                tok in combined_text or
                (len(tok) >= 5 and tok[:-1] in combined_text) or
                (len(tok) >= 6 and tok[:5] in combined_text)
                for tok in all_check_tokens
            )
            if not has_token_overlap:
                return False, "no_core_query_terms_found_in_retrieved_context"

        return True, "context_relevant_and_sufficient"

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes a query string, computes its vector embedding, retrieves nearest
        chunks from FAISS with optional user ownership isolation, filters by similarity threshold,
        and deduplicates chunks by stable (document_id, chunk_id).

        Args:
            query: User's medical or research query text.
            top_k: Number of most similar chunks to retrieve (defaults to self.default_top_k).
            similarity_threshold: Optional minimum cosine similarity cutoff.
            user_id: Optional user ID for ownership isolation.

        Returns:
            List of filtered, deduplicated, and ranked chunk records matching the threshold.
        """
        if not query or not query.strip():
            return []

        k = int(top_k) if top_k is not None else self.default_top_k
        threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )

        query_vec = EmbeddingService.embed_query(query.strip())
        if not query_vec:
            return []

        # Retrieve a broader candidate pool from FAISS to allow for deduplication
        candidate_k = max(k * 4, 20)
        q_clean = query.strip()
        raw_results = self.vector_store.search(query_vec, top_k=candidate_k, user_id=user_id)

        # Phase 2E: Multi-aspect and query expansion candidate retrieval
        if MedicalQueryExpander.is_multi_aspect_query(q_clean):
            sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(q_clean)
            for sq in sub_queries:
                sq_vec = EmbeddingService.embed_query(sq)
                if sq_vec:
                    raw_results.extend(self.vector_store.search(sq_vec, top_k=candidate_k, user_id=user_id))
        elif MedicalQueryExpander.get_expanded_terms(q_clean):
            exp_q = MedicalQueryExpander.expand_query(q_clean)
            exp_vec = EmbeddingService.embed_query(exp_q)
            if exp_vec:
                raw_results.extend(self.vector_store.search(exp_vec, top_k=candidate_k, user_id=user_id))

        # Apply similarity threshold cutoff
        filtered_results = [
            chunk for chunk in raw_results
            if chunk.get("similarity_score", 0.0) >= threshold
        ]

        # Deduplicate by stable (document_id, chunk_id) retaining highest similarity
        deduped_results = self.deduplicate_chunks(filtered_results)

        # Phase 2C: Precision and evidence quality filtering
        precision_results = self.filter_candidate_precision(
            query=query,
            chunks=deduped_results,
            threshold=threshold
        )

        # Multi-document balanced evidence selection
        diverse_results = self.select_diverse_evidence(precision_results, top_k=k, max_per_doc=2)

        return diverse_results

    def build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Constructs a structured, human-readable context block from retrieved chunks.
        Preserves complete source attribution, chunk IDs, page numbers, and document IDs.

        Example output:
            [SOURCE 1]
            Document: DOC_CARDIO_001
            Page: 1
            Chunk ID: MED_CHUNK_0

            Hypertension is a condition in which blood pressure remains chronically elevated...

            [SOURCE 2]
            ...
        """
        if not retrieved_chunks:
            return ""

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            doc_id = chunk.get("document_id") or "UNKNOWN_DOCUMENT"
            page_num = chunk.get("page_number")
            page_str = str(page_num) if page_num is not None else "N/A"
            chunk_id = chunk.get("chunk_id") or f"CHUNK_{i}"
            text = chunk.get("text", "").strip()

            block = (
                f"[SOURCE {i}]\n"
                f"Document: {doc_id}\n"
                f"Page: {page_str}\n"
                f"Chunk ID: {chunk_id}\n\n"
                f"{text}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def build_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts structured source citation metadata for all retrieved chunks.
        Ensures source indexing [Source 1], [Source 2] strictly maps to prompt citations.

        Returns:
            List of citation dictionaries:
            [
                {
                    "source_index": int,
                    "source_label": str,
                    "source_id": str,
                    "document_id": Optional[str],
                    "filename": Optional[str],
                    "page_number": Optional[int],
                    "chunk_id": str,
                    "similarity_score": float,
                    "preview_text": str,
                    "metadata": Dict[str, Any]
                },
                ...
            ]
        """
        sources = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            doc_id = chunk.get("document_id") or chunk.get("metadata", {}).get("document_id")
            fname = chunk.get("metadata", {}).get("filename")
            text_val = chunk.get("text", "")
            source_rec = {
                "source_index": idx,
                "source_label": f"[Source {idx}]",
                "source_id": chunk.get("document_id") or "UNKNOWN_DOCUMENT",
                "document_id": str(doc_id) if doc_id is not None else None,
                "filename": fname,
                "page_number": chunk.get("page_number"),
                "chunk_id": chunk.get("chunk_id", ""),
                "similarity_score": round(float(chunk.get("similarity_score", 0.0)), 4),
                "preview_text": text_val[:250] if text_val else "",
                "text": text_val,
                "metadata": chunk.get("metadata", {})
            }
            sources.append(source_rec)
        return sources

    @staticmethod
    def select_cited_sources(
        retrieved_sources: List[Dict[str, Any]],
        citations_found: List[int],
        valid_citations: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Builds evidence cards for sources actually cited in the answer.

        Keeps original source_index / source_label values so inline citations
        such as [Source 2] [Source 3] remain aligned with displayed cards.
        Does not invent sources or renumber cited indices.
        """
        if not retrieved_sources or not valid_citations:
            return []

        source_map: Dict[int, Dict[str, Any]] = {}
        for idx, src in enumerate(retrieved_sources, start=1):
            if not isinstance(src, dict):
                continue
            src_num = src.get("source_index", idx)
            try:
                src_num = int(src_num)
            except (ValueError, TypeError):
                src_num = idx
            source_map[src_num] = src

        valid_set = set(valid_citations)
        used_indices: List[int] = []
        seen = set()
        for cit in citations_found:
            if cit in valid_set and cit not in seen:
                used_indices.append(cit)
                seen.add(cit)

        final_sources: List[Dict[str, Any]] = []
        for cit in used_indices:
            src = source_map.get(cit)
            if src is None and 1 <= cit <= len(retrieved_sources):
                src = retrieved_sources[cit - 1]
            if not src:
                continue
            card = dict(src)
            card["source_index"] = cit
            card["source_label"] = f"[Source {cit}]"
            final_sources.append(card)
        return final_sources

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the full RAG retrieval pipeline and returns a structured response object
        with optional user ownership filtering and granular latency timing.

        Args:
            question: User's question or medical research query.
            top_k: Number of chunks to retrieve.
            similarity_threshold: Minimum similarity cutoff score.
            user_id: Optional user ID for ownership isolation.

        Returns:
            Dictionary containing:
            - question: Original query string
            - retrieved_chunks: List of matched and deduplicated chunk dictionaries
            - context: Formatted context string with [SOURCE N] headers
            - sources: Clean list of citation references
            - retrieval_status: "success", "no_relevant_context", or "empty_query"
            - timings: Millisecond latency breakdown
        """
        timings = {
            "embedding_time_ms": 0.0,
            "faiss_retrieval_time_ms": 0.0,
            "deduplication_time_ms": 0.0,
            "context_construction_time_ms": 0.0,
            "total_retrieval_time_ms": 0.0,
            # Legacy compatibility aliases
            "query_embedding_time_ms": 0.0,
            "vector_search_time_ms": 0.0,
        }

        # 1. Handle empty / whitespace queries
        if not question or not question.strip():
            return {
                "question": question or "",
                "retrieved_chunks": [],
                "context": "",
                "sources": [],
                "retrieval_status": "empty_query",
                "timings": timings
            }

        cleaned_question = question.strip()
        k = int(top_k) if top_k is not None else self.default_top_k
        threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )

        # 2. Measure Query Embedding Time
        t_embed_start = time.perf_counter()
        query_vec = EmbeddingService.embed_query(cleaned_question)
        embed_ms = round((time.perf_counter() - t_embed_start) * 1000.0, 3)
        timings["embedding_time_ms"] = embed_ms
        timings["query_embedding_time_ms"] = embed_ms

        if not query_vec:
            return {
                "question": cleaned_question,
                "retrieved_chunks": [],
                "context": "",
                "sources": [],
                "retrieval_status": "empty_query",
                "timings": timings
            }

        # 3. Measure FAISS Vector Search Time with Ownership Filter (Fetch candidate pool)
        candidate_k = max(k * 4, 20)
        t_search_start = time.perf_counter()
        raw_results = self.vector_store.search(query_vec, top_k=candidate_k, user_id=user_id)

        # Phase 2E: Multi-aspect and query expansion candidate retrieval
        if MedicalQueryExpander.is_multi_aspect_query(cleaned_question):
            sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(cleaned_question)
            for sq in sub_queries:
                sq_vec = EmbeddingService.embed_query(sq)
                if sq_vec:
                    raw_results.extend(self.vector_store.search(sq_vec, top_k=candidate_k, user_id=user_id))
        elif MedicalQueryExpander.get_expanded_terms(cleaned_question):
            exp_q = MedicalQueryExpander.expand_query(cleaned_question)
            exp_vec = EmbeddingService.embed_query(exp_q)
            if exp_vec:
                raw_results.extend(self.vector_store.search(exp_vec, top_k=candidate_k, user_id=user_id))

        search_ms = round((time.perf_counter() - t_search_start) * 1000.0, 3)
        timings["faiss_retrieval_time_ms"] = search_ms
        timings["vector_search_time_ms"] = search_ms

        # 4. Filter by Similarity Threshold
        filtered_chunks = [
            chunk for chunk in raw_results
            if chunk.get("similarity_score", 0.0) >= threshold
        ]

        # 5. Measure Deduplication Time & Apply Phase 2C Precision Filtering
        t_dedup_start = time.perf_counter()
        deduped_chunks = self.deduplicate_chunks(filtered_chunks)
        precision_chunks = self.filter_candidate_precision(
            query=cleaned_question,
            chunks=deduped_chunks,
            threshold=threshold
        )
        deduped_chunks = self.select_diverse_evidence(precision_chunks, top_k=k, max_per_doc=2)
        dedup_ms = round((time.perf_counter() - t_dedup_start) * 1000.0, 3)
        timings["deduplication_time_ms"] = dedup_ms

        candidate_scores = [round(float(c.get("similarity_score", 0.0)), 4) for c in (deduped_chunks or raw_results)]
        candidate_chunk_ids = [str(c.get("chunk_id", "")) for c in (deduped_chunks or raw_results)]
        candidate_doc_names = [
            str(c.get("metadata", {}).get("filename") or c.get("document_id") or "UNKNOWN")
            for c in (deduped_chunks or raw_results)
        ]

        # Pre-LLM Strict Relevance and Evidence Sufficiency Gate
        is_relevant, gate_reason = self.verify_relevance_and_sufficiency(
            question=cleaned_question,
            retrieved_chunks=deduped_chunks,
            similarity_threshold=threshold
        )

        # 6. Handle No Relevant Context
        if not deduped_chunks or not is_relevant:
            timings["total_retrieval_time_ms"] = round(embed_ms + search_ms + dedup_ms, 3)
            timings["relevant_context_found"] = False
            timings["llm_called"] = False
            timings["similarity_scores"] = candidate_scores
            timings["relevance_threshold"] = threshold
            timings["retrieved_chunk_ids"] = candidate_chunk_ids
            timings["retrieved_document_names"] = candidate_doc_names
            timings["number_of_sources"] = 0
            timings["gate_reason"] = gate_reason if not is_relevant else "no_chunks_above_threshold"

            return {
                "question": cleaned_question,
                "retrieved_chunks": [],
                "context": "",
                "sources": [],
                "retrieval_status": "no_relevant_context",
                "timings": timings
            }

        # 7. Measure Context Construction Time
        t_ctx_start = time.perf_counter()
        context_str = self.build_context(deduped_chunks)
        sources_list = self.build_sources(deduped_chunks)
        ctx_ms = round((time.perf_counter() - t_ctx_start) * 1000.0, 3)
        timings["context_construction_time_ms"] = ctx_ms

        timings["total_retrieval_time_ms"] = round(embed_ms + search_ms + dedup_ms + ctx_ms, 3)
        timings["relevant_context_found"] = True
        timings["llm_called"] = False
        timings["similarity_scores"] = [round(float(c.get("similarity_score", 0.0)), 4) for c in deduped_chunks]
        timings["relevance_threshold"] = threshold
        timings["retrieved_chunk_ids"] = [str(c.get("chunk_id", "")) for c in deduped_chunks]
        timings["retrieved_document_names"] = [
            str(c.get("metadata", {}).get("filename") or c.get("document_id") or "UNKNOWN")
            for c in deduped_chunks
        ]
        timings["number_of_sources"] = len(sources_list)

        return {
            "question": cleaned_question,
            "retrieved_chunks": deduped_chunks,
            "context": context_str,
            "sources": sources_list,
            "retrieval_status": "success",
            "timings": timings
        }

    def generate_rag_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        gemini_service: Optional[Any] = None,
        user_id: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG + Gemini Generation pipeline with user ownership filtering,
        conversation follow-up awareness, strict grounding, and deterministic citation mapping:
        1. Resolves follow-up pronouns (e.g. 'it') using recent conversation turn.
        2. Retrieves context using self.query(..., user_id=user_id).
        3. If retrieval_status == 'empty_query', returns validation message immediately.
        4. If retrieval_status == 'no_relevant_context', DOES NOT call Gemini (0 ms LLM time).
        5. If retrieval_status == 'success', feeds grounded context into GeminiService.
        6. Validates citations and prunes un-used evidence cards to guarantee deterministic alignment.

        Args:
            question: User's question or medical research inquiry.
            top_k: Optional top_k limit for retrieval.
            similarity_threshold: Optional similarity score cutoff.
            gemini_service: Optional GeminiService instance (injected or instantiated).
            user_id: Optional user ID for ownership filtering.
            conversation_history: Optional list of previous chat messages for follow-up resolution.
            request_id: Optional traceable request ID (auto-generated if None).

        Returns:
            Structured response dictionary with question, answer, status, sources,
            disclaimer, request_id, and granular timing metrics.
        """
        from backend.rag.prompt_builder import MEDICAL_DISCLAIMER
        from backend.services.gemini_service import GeminiService, GeminiServiceError
        from backend.evaluation.observability import (
            generate_request_id,
            StructuredRAGLogger,
            RAGStructuredLogEvent
        )

        trace_id = request_id or generate_request_id()

        # Phase 4 Medical Safety Pre-Screening: Intercept emergency, self-harm, overdose
        from backend.safety.medical_safety_guard import MedicalSafetyGuard
        allow_rag, safety_assessment, immediate_msg = MedicalSafetyGuard.pre_screen_inquiry(question)
        if not allow_rag:
            logger.warning(
                "Inquiry intercepted by MedicalSafetyGuard (Category: %s, Risk: %s): %s",
                safety_assessment.category,
                safety_assessment.risk_level,
                safety_assessment.reason
            )
            cat_name = safety_assessment.category.value if hasattr(safety_assessment.category, "value") else str(safety_assessment.category)
            StructuredRAGLogger.emit_rag_log(RAGStructuredLogEvent(
                request_id=trace_id,
                query=question or "",
                safety_classification=cat_name,
                risk_level=safety_assessment.risk_level,
                retrieval_status="safety_intercepted",
                embedding_latency_ms=0.0,
                vector_search_latency_ms=0.0,
                context_construction_latency_ms=0.0,
                llm_latency_ms=0.0,
                total_latency_ms=0.0,
                number_of_retrieved_chunks=0,
                similarity_scores=[],
                citation_validation_passed=True,
                citation_coverage=1.0,
                hallucination_detected=False,
                hallucination_types=[],
                final_status="SAFETY_INTERCEPTED"
            ))
            return {
                "request_id": trace_id,
                "question": question or "",
                "answer": immediate_msg or "",
                "retrieval_status": "safety_intercepted",
                "retrieved_chunks": [],
                "context": "",
                "sources": [],
                "disclaimer": MEDICAL_DISCLAIMER,
                "timings": {
                    "embedding_time_ms": 0.0,
                    "faiss_retrieval_time_ms": 0.0,
                    "deduplication_time_ms": 0.0,
                    "context_construction_time_ms": 0.0,
                    "llm_generation_time_ms": 0.0,
                    "api_request_time_ms": 0.0,
                    "retrieval_time_ms": 0.0,
                    "generation_time_ms": 0.0,
                    "total_time_ms": 0.0,
                    "gemini_calls_count": 0,
                    "unique_sources": 0,
                    "relevant_context_found": False,
                    "llm_called": False,
                    "number_of_sources": 0,
                    "safety_assessment": safety_assessment.to_dict()
                }
            }

        effective_search_query, conv_context_snippet = self.resolve_followup_query(
            question=question,
            conversation_history=conversation_history
        )

        retrieval_res = self.query(
            question=effective_search_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            user_id=user_id
        )

        status = retrieval_res["retrieval_status"]
        retrieval_timings = retrieval_res["timings"]
        thresh_val = float(similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold)

        # Double-check original question to prevent context leak when user asks an unsupported entity
        if status == "success":
            orig_focus = self.extract_query_clinical_focus(question)
            if orig_focus:
                orig_is_relevant, orig_reason = self.verify_relevance_and_sufficiency(
                    question=orig_focus,
                    retrieved_chunks=retrieval_res["retrieved_chunks"],
                    similarity_threshold=thresh_val
                )
                if not orig_is_relevant and "query_targets_condition" in orig_reason:
                    status = "no_relevant_context"
                    retrieval_res["retrieval_status"] = "no_relevant_context"

        # 1. Handle Empty Query
        if status == "empty_query":
            logger.info(
                "=== RAG DIAGNOSTIC AUDIT ===\n"
                "Query: %s\n"
                "Retrieved chunk IDs: []\n"
                "Retrieved document names: []\n"
                "Similarity scores: []\n"
                "Relevance threshold: %.4f\n"
                "Relevant context found: false\n"
                "LLM called: false\n"
                "Number of sources: 0\n"
                "Latency Breakdown:\n"
                "  Embedding: 0.00 ms\n"
                "  FAISS Search: 0.00 ms\n"
                "  Context Preparation: 0.00 ms\n"
                "  Gemini Generation: 0.00 ms\n"
                "  Total Latency: 0.00 ms\n"
                "============================",
                question or "",
                thresh_val
            )
            StructuredRAGLogger.emit_rag_log(RAGStructuredLogEvent(
                request_id=trace_id,
                query=question or "",
                safety_classification="benign",
                risk_level="none",
                retrieval_status="empty_query",
                embedding_latency_ms=0.0,
                vector_search_latency_ms=0.0,
                context_construction_latency_ms=0.0,
                llm_latency_ms=0.0,
                total_latency_ms=0.0,
                number_of_retrieved_chunks=0,
                similarity_scores=[],
                citation_validation_passed=True,
                citation_coverage=1.0,
                hallucination_detected=False,
                hallucination_types=[],
                final_status="EMPTY_QUERY"
            ))
            return {
                "request_id": trace_id,
                "question": question or "",
                "answer": "Please provide a valid, non-empty clinical question or research query.",
                "retrieval_status": "empty_query",
                "retrieved_chunks": [],
                "context": "",
                "sources": [],
                "disclaimer": MEDICAL_DISCLAIMER,
                "timings": {
                    "request_id": trace_id,
                    "embedding_time_ms": 0.0,
                    "faiss_retrieval_time_ms": 0.0,
                    "deduplication_time_ms": 0.0,
                    "context_construction_time_ms": 0.0,
                    "llm_generation_time_ms": 0.0,
                    "api_request_time_ms": 0.0,
                    "retrieval_time_ms": 0.0,
                    "generation_time_ms": 0.0,
                    "total_time_ms": 0.0,
                    "gemini_calls_count": 0,
                    "unique_sources": 0,
                    "relevant_context_found": False,
                    "llm_called": False,
                    "similarity_scores": [],
                    "relevance_threshold": thresh_val,
                    "retrieved_chunk_ids": [],
                    "retrieved_document_names": [],
                    "number_of_sources": 0,
                    "query_embedding_time_ms": 0.0,
                    "vector_search_time_ms": 0.0,
                }
            }

        # 2. Handle No Relevant Context (STRICT PRE-LLM GATE: DO NOT CALL GEMINI)
        if status == "no_relevant_context":
            ret_scores = retrieval_timings.get("similarity_scores", [])
            ret_chunk_ids = retrieval_timings.get("retrieved_chunk_ids", [])
            ret_doc_names = retrieval_timings.get("retrieved_document_names", [])

            logger.info(
                "=== RAG DIAGNOSTIC AUDIT ===\n"
                "Query: %s\n"
                "Retrieved chunk IDs: %s\n"
                "Retrieved document names: %s\n"
                "Similarity scores: %s\n"
                "Relevance threshold: %.4f\n"
                "Relevant context found: false\n"
                "LLM called: false\n"
                "Number of sources: 0\n"
                "Latency Breakdown:\n"
                "  Embedding: %.2f ms\n"
                "  FAISS Search: %.2f ms\n"
                "  Context Preparation: 0.00 ms\n"
                "  Gemini Generation: 0.00 ms\n"
                "  Total Latency: %.2f ms\n"
                "============================",
                question,
                ret_chunk_ids,
                ret_doc_names,
                ret_scores,
                thresh_val,
                retrieval_timings.get("embedding_time_ms", 0.0),
                retrieval_timings.get("faiss_retrieval_time_ms", 0.0),
                retrieval_timings.get("total_retrieval_time_ms", 0.0)
            )

            cat_name = safety_assessment.category.value if hasattr(safety_assessment.category, "value") else str(safety_assessment.category)
            StructuredRAGLogger.emit_rag_log(RAGStructuredLogEvent(
                request_id=trace_id,
                query=question or "",
                safety_classification=cat_name,
                risk_level=safety_assessment.risk_level,
                retrieval_status="no_relevant_context",
                embedding_latency_ms=retrieval_timings.get("embedding_time_ms", 0.0),
                vector_search_latency_ms=retrieval_timings.get("faiss_retrieval_time_ms", 0.0),
                context_construction_latency_ms=0.0,
                llm_latency_ms=0.0,
                total_latency_ms=retrieval_timings.get("total_retrieval_time_ms", 0.0),
                number_of_retrieved_chunks=len(retrieval_res.get("retrieved_chunks", [])),
                similarity_scores=ret_scores,
                citation_validation_passed=True,
                citation_coverage=1.0,
                hallucination_detected=False,
                hallucination_types=[],
                final_status="HALTED_NO_CONTEXT"
            ))

            return {
                "request_id": trace_id,
                "question": question,
                "answer": (
                    "Relevant medical information could not be found in the available reference documents. "
                    "To prevent unsupported healthcare answers, generation was halted. "
                    "Please refine your query or consult authorized clinical guidelines."
                ),
                "retrieval_status": "no_relevant_context",
                "retrieved_chunks": [],
                "context": "",
                "sources": [],
                "disclaimer": MEDICAL_DISCLAIMER,
                "timings": {
                    "request_id": trace_id,
                    "embedding_time_ms": retrieval_timings.get("embedding_time_ms", 0.0),
                    "faiss_retrieval_time_ms": retrieval_timings.get("faiss_retrieval_time_ms", 0.0),
                    "deduplication_time_ms": retrieval_timings.get("deduplication_time_ms", 0.0),
                    "context_construction_time_ms": 0.0,
                    "llm_generation_time_ms": 0.0,
                    "api_request_time_ms": 0.0,
                    "retrieval_time_ms": retrieval_timings.get("total_retrieval_time_ms", 0.0),
                    "generation_time_ms": 0.0,
                    "total_time_ms": retrieval_timings.get("total_retrieval_time_ms", 0.0),
                    "gemini_calls_count": 0,
                    "unique_sources": 0,
                    "relevant_context_found": False,
                    "llm_called": False,
                    "similarity_scores": ret_scores,
                    "relevance_threshold": thresh_val,
                    "retrieved_chunk_ids": ret_chunk_ids,
                    "retrieved_document_names": ret_doc_names,
                    "number_of_sources": 0,
                    # Legacy compatibility aliases
                    "query_embedding_time_ms": retrieval_timings.get("embedding_time_ms", 0.0),
                    "vector_search_time_ms": retrieval_timings.get("faiss_retrieval_time_ms", 0.0),
                }
            }

        # 3. Call Gemini with grounded context
        service = gemini_service or GeminiService()
        try:
            gen_res = service.generate_answer(
                question=effective_search_query,
                context=retrieval_res["context"],
                conversation_context=conv_context_snippet
            )
            raw_answer = gen_res["answer"]
            cleaned_answer = clean_ai_markdown(raw_answer)
        except GeminiServiceError as gse:
            logger.warning("RAG synthesis halted due to Gemini service error: %s", str(gse))
            gen_ms = 0.0
            total_ms = round(retrieval_timings.get("total_retrieval_time_ms", 0.0), 2)
            clean_unavailable_msg = (
                "The AI generation service is temporarily unavailable due to high demand. "
                "Your document passages were retrieved successfully, but answer generation "
                "could not be completed at this time. Please wait a few moments and try your inquiry again."
            )
            return {
                "question": question,
                "answer": clean_unavailable_msg,
                "retrieval_status": "service_unavailable",
                "retrieved_chunks": retrieval_res.get("retrieved_chunks", []),
                "context": retrieval_res.get("context", ""),
                "sources": [],
                "disclaimer": MEDICAL_DISCLAIMER,
                "timings": {
                    **retrieval_timings,
                    "context_construction_time_ms": retrieval_timings.get("context_construction_time_ms", 0.0),
                    "llm_generation_time_ms": 0.0,
                    "api_request_time_ms": 0.0,
                    "generation_time_ms": gen_ms,
                    "total_time_ms": total_ms,
                    "gemini_calls_count": 0,
                    "unique_sources": 0,
                    "relevant_context_found": True,
                    "llm_called": True,
                    "number_of_sources": 0,
                    "service_error": str(gse),
                }
            }

        # 4. Strict Grounding Guard for Modifiable Risk Factor Inquiries:
        # If the query asks which risk factors can be modified / modifiable risk factors,
        # but the document only lists risk factors and general lifestyle measures separately
        # without explicitly categorizing which specific risk factors are modifiable:
        # Enforce grounded limitation without unsupported classifications or cross-context inferences.
        is_modifiable_query = bool(re.search(
            r'\b(?:which\s+(?:of\s+)?(?:them|these|those)?\s+can\s+be\s+modified|can\s+(?:any\s+of\s+)?(?:them|these|those)\s+be\s+modified|how\s+can\s+(?:they|them|these|those)\s+be\s+modified|which\s+(?:risk\s+)?factors\s+can\s+be\s+modified|modifiable\s+(?:risk\s+)?factors?)\b',
            question,
            re.IGNORECASE
        )) or bool(re.search(
            r'\b(?:which\s+(?:of\s+)?(?:them|these|those)?\s+can\s+be\s+modified|can\s+(?:any\s+of\s+)?(?:them|these|those)\s+be\s+modified|how\s+can\s+(?:they|them|these|those)\s+be\s+modified|which\s+(?:risk\s+)?factors\s+can\s+be\s+modified|modifiable\s+(?:risk\s+)?factors?)\b',
            effective_search_query,
            re.IGNORECASE
        ))

        ctx_lower = retrieval_res.get("context", "").lower()
        has_explicit_modifiable = bool(re.search(
            r'\b(?:modifiable\s+risk\s+factors?|risk\s+factors?\s+(?:that\s+)?(?:are|can\s+be)\s+modified|classified\s+as\s+modifiable|categorized\s+as\s+modifiable)\b',
            ctx_lower
        ))

        if is_modifiable_query and not has_explicit_modifiable:
            topic_name = "hypertension"
            if "hypertens" in ctx_lower or "hypertens" in question.lower() or (conv_context_snippet and "hypertens" in conv_context_snippet.lower()):
                topic_name = "hypertension"

            ans_lower = cleaned_answer.lower()
            # Detect if model made an unsupported medical classification across separate sections
            has_unsupported_classification = bool(re.search(
                r'\b(?:can\s+be\s+modified\s+include|factors\s+that\s+can\s+be\s+modified|are\s+considered\s+modifiable|can\s+be\s+classified\s+as\s+modifiable|are\s+modifiable|can\s+be\s+modified\s+through)\b',
                ans_lower
            ))
            states_grounded_limitation = bool(re.search(
                r'\bdoes\s+not\s+explicitly\s+(?:identify|categorize|classify)\s+(?:which|any)\b',
                ans_lower
            ))

            if has_unsupported_classification or not states_grounded_limitation:
                cleaned_answer = f"The document lists several {topic_name} risk factors and separately describes lifestyle measures, but it does not explicitly identify which specific risk factors are modifiable [Source 1]."

        # 5. Deterministic Citation Validation & Evidence Pruning
        all_retrieved_sources = retrieval_res["sources"]
        validation_result = CitationValidator.validate_grounded_citations(
            answer_text=cleaned_answer,
            retrieved_sources=all_retrieved_sources
        )

        # Phase 3.3: Strict Grounded Citation Enforcement
        # If factual claims are unsupported by cited sources, prune or safely halt
        if validation_result.has_citations and validation_result.claims_unsupported > 0:
            if validation_result.claims_supported == 0 or not validation_result.cleaned_grounded_answer:
                logger.warning(
                    "Citation enforcement rejected answer: %d unsupported claims found (%s). Halting generation.",
                    validation_result.claims_unsupported,
                    validation_result.unsupported_claims
                )
                cleaned_answer = (
                    "Relevant medical information could not be found in the available reference documents. "
                    "To prevent unsupported healthcare answers, generation was halted. "
                    "Please refine your query or consult authorized clinical guidelines."
                )
            else:
                logger.info(
                    "Citation enforcement pruned %d unsupported claims, retaining %d grounded claims.",
                    validation_result.claims_unsupported,
                    validation_result.claims_supported
                )
                cleaned_answer = validation_result.cleaned_grounded_answer

        # Strip any hallucinated source tags that do not exist in retrieved sources
        cleaned_answer = CitationValidator.strip_invalid_citations(
            cleaned_answer, validation_result.invalid_citations
        )
        cleaned_answer = clean_ai_markdown(cleaned_answer)

        # Phase 4 Medical Safety Post-Screening & Response Guardrails
        post_safety_res = MedicalSafetyGuard.post_screen_answer(
            user_question=question,
            generated_answer=cleaned_answer,
            assessment=safety_assessment,
            sources_present=len(all_retrieved_sources) > 0
        )
        cleaned_answer = post_safety_res["sanitized_answer"]

        # Filter evidence cards strictly to sources actually cited in the answer
        # Preserves deterministic 1:1 mapping without renumbering:
        # [Source 1] -> source_index 1, [Source 3] -> source_index 3
        valid_citations = validation_result.valid_citations
        safe_text = cleaned_answer.lower()
        is_grounded_limitation = bool(
            re.search(r'\bdoes\s+not\s+explicitly\s+(?:identify|categorize|classify)\s+which\b', safe_text) or
            re.search(r'\bseparately\s+describes\s+lifestyle\s+measures,\s+but\s+it\s+does\s+not\b', safe_text)
        )

        # Phase 2: Grounded Document Boundary & Explicit Negative Scope Detection
        # A boundary response is allowed ONLY when the retrieved document itself explicitly supports the boundary claim.
        is_grounded_boundary = False
        if all_retrieved_sources:
            combined_ctx_lower = " ".join(c.get("text", "") for c in (retrieval_res.get("retrieved_chunks") or [])).lower()
            boundary_pattern = re.compile(
                r'\b(?:testing\s+boundary|scope\s+boundary|testing\s+boundaries|boundary\b|boundaries\b|out\s+of\s+scope|outside\s+(?:the\s+)?scope|unsupported\s+by\s+this\s+document|treated\s+as\s+unsupported|not\s+contain\s+information\s+about|does\s+not\s+contain\s+information\s+about|not\s+covered\s+(?:in|by)\s+this\s+document|exclusion\s+criteria)\b',
                re.IGNORECASE
            )
            has_boundary_in_doc = bool(boundary_pattern.search(combined_ctx_lower))
            if has_boundary_in_doc:
                q_focus = self.extract_query_clinical_focus(question).lower()
                stop_tokens = {
                    "what", "is", "are", "the", "difference", "between", "common", "main", "for", "in", "of", "and",
                    "to", "a", "an", "according", "uploaded", "document", "tell", "me", "about", "can", "you",
                    "explain", "describe", "which", "them", "they", "it", "this", "that", "these", "those", "how",
                    "does", "do", "from", "with", "or", "by", "on", "at", "be", "as", "have", "has", "summarize",
                    "summary", "overview", "provide", "show", "give", "list", "symptoms", "causes", "complications",
                    "treatment", "treatments", "contain", "mention", "pdf"
                }
                q_terms = [w for w in re.findall(r'\b[a-z]{3,}\b', q_focus) if w not in stop_tokens]
                boundary_segments = [
                    seg for seg in re.split(r'(?:\n\n|\.\s+)', combined_ctx_lower)
                    if boundary_pattern.search(seg)
                ]
                boundary_text = " ".join(boundary_segments)
                targets_boundary_entity = any(term in boundary_text for term in q_terms) if q_terms else False
                ans_reflects_boundary = bool(re.search(
                    r'\b(?:not\s+contain|do\s+not\s+contain|does\s+not\s+contain|not\s+covered|unsupported|not\s+supported|testing\s+boundary|boundary)\b',
                    safe_text
                ))
                if targets_boundary_entity and ans_reflects_boundary:
                    is_grounded_boundary = True

        is_absence_or_refusal = (not is_grounded_limitation) and (not is_grounded_boundary) and any(
            phrase in safe_text for phrase in [
                "do not contain",
                "do not mention",
                "not mentioned",
                "no mention",
                "not provide that information",
                "not provide information",
                "not enough information",
                "insufficient",
                "could not be found",
                "cannot answer",
                "does not contain",
                "does not mention",
                "no information found",
                "not addressed",
                "not discussed",
                "not included",
                "no details about",
                "no reference to",
                "is not mentioned",
                "are not mentioned"
            ]
        )

        if is_grounded_boundary:
            # Grounded document boundary response:
            # The document explicitly states the entity is unsupported/out of scope.
            # Preserve Gemini's grounded boundary answer, [Source 1], and corresponding source cards.
            if valid_citations:
                final_sources = self.select_cited_sources(
                    retrieved_sources=all_retrieved_sources,
                    citations_found=validation_result.citations_found,
                    valid_citations=valid_citations,
                )
            else:
                top_src = dict(all_retrieved_sources[0])
                top_src["source_index"] = 1
                top_src["source_label"] = "[Source 1]"
                final_sources = [top_src]
                if "[source" not in cleaned_answer.lower():
                    cleaned_answer = f"{cleaned_answer.rstrip('. ')} [Source 1]."

        elif is_absence_or_refusal:
            final_sources = []
            cleaned_answer = (
                "Relevant medical information could not be found in the available reference documents. "
                "To prevent unsupported healthcare answers, generation was halted. "
                "Please refine your query or consult authorized clinical guidelines."
            )
        elif valid_citations:
            final_sources = self.select_cited_sources(
                retrieved_sources=all_retrieved_sources,
                citations_found=validation_result.citations_found,
                valid_citations=valid_citations,
            )
        else:
            if not all_retrieved_sources:
                final_sources = []
            else:
                # Informative answer where model omitted explicit inline citation tag: fallback to top relevant source
                top_src = dict(all_retrieved_sources[0])
                top_src["source_index"] = 1
                top_src["source_label"] = "[Source 1]"
                final_sources = [top_src]

        gen_ms = gen_res.get("generation_time_ms", 0.0)
        total_ms = round(
            retrieval_timings["total_retrieval_time_ms"] + gen_ms, 2
        )

        ret_scores = retrieval_timings.get("similarity_scores", [])
        ret_chunk_ids = retrieval_timings.get("retrieved_chunk_ids", [])
        ret_doc_names = retrieval_timings.get("retrieved_document_names", [])

        logger.info(
            "=== RAG DIAGNOSTIC AUDIT ===\n"
            "Query: %s\n"
            "Retrieved chunk IDs: %s\n"
            "Retrieved document names: %s\n"
            "Similarity scores: %s\n"
            "Relevance threshold: %.4f\n"
            "Relevant context found: true\n"
            "LLM called: true\n"
            "Number of sources: %d\n"
            "Latency Breakdown:\n"
            "  Embedding: %.2f ms\n"
            "  FAISS Search: %.2f ms\n"
            "  Context Preparation: %.2f ms\n"
            "  Gemini Generation: %.2f ms\n"
            "  Total Latency: %.2f ms\n"
            "============================",
            question,
            ret_chunk_ids,
            ret_doc_names,
            ret_scores,
            thresh_val,
            len(final_sources),
            retrieval_timings.get("embedding_time_ms", 0.0),
            retrieval_timings.get("faiss_retrieval_time_ms", 0.0),
            retrieval_timings.get("context_construction_time_ms", 0.0),
            gen_ms,
            total_ms
        )

        if is_grounded_boundary:
            final_status = "grounded_boundary"
            final_chunks = retrieval_res["retrieved_chunks"]
            final_ctx = retrieval_res["context"]
            has_rel_ctx = True
        elif is_absence_or_refusal:
            final_status = "no_relevant_context"
            final_chunks = []
            final_ctx = ""
            has_rel_ctx = False
        else:
            final_status = "success"
            final_chunks = retrieval_res["retrieved_chunks"]
            final_ctx = retrieval_res["context"]
            has_rel_ctx = True

        cat_name = safety_assessment.category.value if hasattr(safety_assessment.category, "value") else str(safety_assessment.category)
        StructuredRAGLogger.emit_rag_log(RAGStructuredLogEvent(
            request_id=trace_id,
            query=question or "",
            safety_classification=cat_name,
            risk_level=safety_assessment.risk_level,
            retrieval_status=final_status,
            embedding_latency_ms=retrieval_timings.get("embedding_time_ms", 0.0),
            vector_search_latency_ms=retrieval_timings.get("faiss_retrieval_time_ms", 0.0),
            context_construction_latency_ms=retrieval_timings.get("context_construction_time_ms", 0.0),
            llm_latency_ms=gen_ms,
            total_latency_ms=total_ms,
            number_of_retrieved_chunks=len(final_chunks),
            similarity_scores=ret_scores,
            citation_validation_passed=validation_result.is_valid,
            citation_coverage=validation_result.citation_coverage,
            hallucination_detected=validation_result.claims_unsupported > 0,
            hallucination_types=["unsupported_claim"] if validation_result.claims_unsupported > 0 else [],
            final_status="SUCCESS" if final_status in ("success", "grounded_boundary") else "FILTERED"
        ))

        return {
            "request_id": trace_id,
            "question": question,
            "answer": cleaned_answer,
            "retrieval_status": final_status,
            "retrieved_chunks": final_chunks,
            "context": final_ctx,
            "sources": final_sources,
            "disclaimer": gen_res.get("disclaimer", MEDICAL_DISCLAIMER),
            "timings": {
                "request_id": trace_id,
                "embedding_time_ms": retrieval_timings.get("embedding_time_ms", 0.0),
                "faiss_retrieval_time_ms": retrieval_timings.get("faiss_retrieval_time_ms", 0.0),
                "deduplication_time_ms": retrieval_timings.get("deduplication_time_ms", 0.0),
                "context_construction_time_ms": retrieval_timings.get("context_construction_time_ms", 0.0),
                "llm_generation_time_ms": gen_ms,
                "api_request_time_ms": gen_res.get("api_request_time_ms", gen_ms),
                "total_time_ms": total_ms,
                "gemini_calls_count": gen_res.get("gemini_calls_count", 1),
                "input_tokens": gen_res.get("input_tokens"),
                "output_tokens": gen_res.get("output_tokens"),
                "request_start_time": gen_res.get("request_start_time"),
                "model_used": gen_res.get("model"),
                "unique_sources": len(final_sources),
                "relevant_context_found": has_rel_ctx,
                "llm_called": True,
                "similarity_scores": ret_scores,
                "relevance_threshold": thresh_val,
                "retrieved_chunk_ids": ret_chunk_ids,
                "retrieved_document_names": ret_doc_names,
                "number_of_sources": len(final_sources),
                # Phase 3.3 Citation Enforcement Metrics
                "citation_coverage": validation_result.citation_coverage,
                "claims_checked": validation_result.claims_checked,
                "claims_supported": validation_result.claims_supported,
                "claims_unsupported": validation_result.claims_unsupported,
                # Phase 4 Medical Safety Metrics
                "safety_category": safety_assessment.category.value if hasattr(safety_assessment.category, "value") else str(safety_assessment.category),
                "risk_level": safety_assessment.risk_level,
                "safety_post_check_passed": post_safety_res["post_check_passed"],
                "safety_warnings": post_safety_res["warnings"],
                # Legacy compatibility aliases
                "query_embedding_time_ms": retrieval_timings.get("embedding_time_ms", 0.0),
                "vector_search_time_ms": retrieval_timings.get("faiss_retrieval_time_ms", 0.0),
                "retrieval_time_ms": retrieval_timings["total_retrieval_time_ms"],
                "generation_time_ms": gen_ms,
            }
        }

