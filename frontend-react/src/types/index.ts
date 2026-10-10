/**
 * Strongly typed domain models for AI Healthcare Assistant Frontend (Phase 7).
 * Strictly mirrors backend schemas from FastAPI routers, Phase 6.8 & Phase 6.9 pipelines.
 */

// ==============================================================================
// Authentication & User Types
// ==============================================================================

export interface User {
  id: string | number;
  email: string;
  role?: string;
  full_name?: string | null;
  created_at?: string | null;
  user_metadata?: Record<string, any>;
}

export interface AuthSession {
  access_token: string;
  token_type: string;
  expires_in?: number;
  expires_at?: number;
}

export interface AuthResponse {
  message: string;
  user: User;
  session?: AuthSession;
  status?: string;
}

// ==============================================================================
// Document Management Types
// ==============================================================================

export interface DocumentRecord {
  id: number;
  filename: string;
  file_path?: string;
  file_size_bytes?: number | null;
  num_pages?: number | null;
  num_chunks: number;
  status: string; // 'processed', 'uploaded', 'indexed', 'failed'
  user_id?: number | null;
  created_at: string;
}

export interface DocumentUploadResponse {
  document_id?: number | null;
  filename: string;
  status: string;
  pages?: number;
  number_of_pages?: number;
  num_chunks?: number;
  extracted_text_length?: number;
  user_id?: number | null;
}

// ==============================================================================
// Conversation & Dialogue Types
// ==============================================================================

export interface Conversation {
  id: number;
  user_id?: number | null;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  id: number;
  conversation_id: number;
  sender: 'user' | 'assistant' | 'agent';
  text: string;
  agent_type?: string | null;
  citations?: any;
  created_at: string;
}

// ==============================================================================
// RAG & Clinical Intelligence Types
// ==============================================================================

export interface SourceCitation {
  chunk_id: string;
  document_name?: string;
  document_id?: string | number;
  page_number?: number;
  similarity_score?: number;
  text: string;
}

export interface SafetyAssessment {
  category: string;
  risk_level: string;
  requires_escalation?: boolean;
  allow_normal_rag?: boolean;
}

export interface DialogueContextPayload {
  turn_count?: number;
  prior_turns_used?: number;
  is_follow_up?: boolean;
  effective_query?: string;
  contraindications_detected?: number;
  contraindications?: Array<{
    contraindication_id?: string;
    severity?: string;
    reason?: string;
    patient_condition?: string;
    conflicting_entity?: string;
  }>;
  profile_hash?: string;
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

export interface RAGQueryResponse {
  question: string;
  answer: string;
  retrieval_status: 'success' | 'no_relevant_context' | 'safety_intercepted' | 'service_unavailable' | string;
  sources: SourceCitation[];
  context?: string | null;
  disclaimer: string;
  timings?: {
    total_time_ms?: number;
    embedding_time_ms?: number;
    search_time_ms?: number;
    generation_time_ms?: number;
    llm_called?: boolean;
    safety_assessment?: SafetyAssessment;
  };
  request_id?: string;
  intent?: Record<string, any>;
  dialogue_context?: DialogueContextPayload;
}

// ==============================================================================
// System Health Types
// ==============================================================================

export type HealthConnectivityState = 'CONNECTED' | 'DEGRADED' | 'OFFLINE' | 'CHECKING';

export interface BackendHealthResponse {
  status: 'healthy' | 'degraded' | string;
  service: string;
  environment: string;
  vector_store?: string;
  embedding_service?: string;
  safety_engine?: string;
  database?: string;
  llm_config?: string;
}

// ==============================================================================
// API Error Type
// ==============================================================================

export interface ApiError {
  status: number;
  message: string;
  detail?: string;
}
