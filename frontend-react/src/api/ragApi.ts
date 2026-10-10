import { apiClient } from './client';
import {
  RAGQueryRequest,
  RAGQueryResponse,
  BackendHealthResponse,
} from '../types';
import { streamRAGQuery, SSECallbacks } from '../utils/sseParser';

export const ragApi = {
  /**
   * Executes synchronous RAG query against POST /rag/query.
   * Direct integration with Phase 6.1-6.9 clinical intelligence pipeline.
   */
  queryRAG: async (request: RAGQueryRequest): Promise<RAGQueryResponse> => {
    const response = await apiClient.post<RAGQueryResponse>('/rag/query', request);
    return response.data;
  },

  /**
   * Executes streaming RAG query against POST /rag/stream.
   * If streaming is unsupported or network fails, can fallback to queryRAG.
   */
  streamRAG: async (
    request: RAGQueryRequest,
    callbacks: SSECallbacks,
    signal?: AbortSignal
  ): Promise<void> => {
    return streamRAGQuery(request, callbacks, signal);
  },

  /**
   * Queries consolidated system health status.
   */
  checkHealth: async (): Promise<BackendHealthResponse> => {
    const response = await apiClient.get<BackendHealthResponse>('/health');
    return response.data;
  },

  /**
   * Non-destructive probe for Kubernetes/Docker readiness verification.
   */
  checkReadiness: async (): Promise<any> => {
    const response = await apiClient.get('/health/readiness');
    return response.data;
  },
};
