import { apiClient } from './client';
import { RAGQueryRequest, RAGQueryResponse, ReadinessResponse } from '../types';

/**
 * Executes a context-grounded medical RAG query against FastAPI backend.
 * POST /rag/query
 */
export async function queryRAG(request: RAGQueryRequest): Promise<RAGQueryResponse> {
  if (!request.question || request.question.trim().length < 3) {
    throw new Error('Please enter a clinical question with at least 3 characters.');
  }

  const payload: RAGQueryRequest = {
    question: request.question.trim(),
    top_k: request.top_k ?? 5,
    similarity_threshold: request.similarity_threshold ?? 0.25,
    conversation_history: request.conversation_history,
  };

  const response = await apiClient.post<RAGQueryResponse>('/rag/query', payload);
  return response.data;
}

/**
 * Checks backend subsystem readiness.
 * GET /health/readiness
 */
export async function checkReadiness(): Promise<ReadinessResponse> {
  const response = await apiClient.get<ReadinessResponse>('/health/readiness');
  return response.data;
}

/**
 * Checks basic service liveness.
 * GET /health/liveness
 */
export async function checkLiveness(): Promise<{ status: string; service: string }> {
  const response = await apiClient.get<{ status: string; service: string }>('/health/liveness');
  return response.data;
}
