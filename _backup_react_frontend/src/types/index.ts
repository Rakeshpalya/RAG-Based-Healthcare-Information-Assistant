export interface SourceCitation {
  chunk_id: string;
  document_name?: string;
  document_id?: string;
  page_number?: number;
  similarity_score?: number;
  text: string;
}

export interface SafetyAssessment {
  category: string;
  risk_level: string;
  requires_escalation: boolean;
  allow_normal_rag: boolean;
}

export interface RAGQueryTimings {
  total_time_ms?: number;
  embedding_time_ms?: number;
  search_time_ms?: number;
  generation_time_ms?: number;
  llm_called?: boolean;
  safety_category?: string;
  risk_level?: string;
  safety_assessment?: SafetyAssessment;
}

export interface RAGQueryResponse {
  question: string;
  answer: string;
  retrieval_status: 'success' | 'no_relevant_context' | 'safety_intercepted' | 'service_unavailable' | string;
  sources: SourceCitation[];
  context?: string | null;
  disclaimer: string;
  timings?: RAGQueryTimings;
  request_id?: string;
}

export interface RAGQueryRequest {
  question: string;
  top_k?: number;
  similarity_threshold?: number;
  conversation_history?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

export interface SubsystemCheck {
  status: 'ready' | 'degraded' | 'mismatched' | 'uninitialized';
  vector_count?: number;
  expected_count?: number;
  dimension?: number;
  error?: string;
}

export interface ReadinessResponse {
  status: 'ready' | 'not_ready' | string;
  service: string;
  environment: string;
  checks: {
    vector_store?: SubsystemCheck;
    embedding_service?: SubsystemCheck;
    safety_engine?: SubsystemCheck;
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: SourceCitation[];
  retrieval_status?: string;
  safety_assessment?: SafetyAssessment;
  request_id?: string;
  isError?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}
