import { describe, it, expect, vi, beforeEach } from 'vitest';
import { queryRAG, checkReadiness, checkLiveness } from '../api/ragApi';
import { apiClient } from '../api/client';

describe('RAG API Client (Phase 7)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('rejects clinical queries with fewer than 3 characters before network dispatch', async () => {
    await expect(queryRAG({ question: 'no' })).rejects.toThrow(
      'Please enter a clinical question with at least 3 characters.'
    );
  });

  it('successfully dispatches POST /rag/query with default parameters', async () => {
    const mockData = {
      question: 'What is hypertension?',
      answer: 'Hypertension is persistent blood pressure elevation.',
      retrieval_status: 'success',
      sources: [],
      disclaimer: 'Educational only',
      request_id: 'req_123',
    };

    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({ data: mockData });

    const res = await queryRAG({ question: 'What is hypertension?' });
    expect(res).toEqual(mockData);
    expect(apiClient.post).toHaveBeenCalledWith('/rag/query', {
      question: 'What is hypertension?',
      top_k: 5,
      similarity_threshold: 0.25,
      conversation_history: undefined,
    });
  });

  it('successfully checks readiness via GET /health/readiness', async () => {
    const mockReadiness = {
      status: 'ready',
      service: 'AI-Healthcare-Agent',
      environment: 'production',
      checks: {},
    };

    vi.spyOn(apiClient, 'get').mockResolvedValueOnce({ data: mockReadiness });

    const res = await checkReadiness();
    expect(res.status).toBe('ready');
    expect(apiClient.get).toHaveBeenCalledWith('/health/readiness');
  });

  it('successfully checks liveness via GET /health/liveness', async () => {
    const mockLiveness = {
      status: 'alive',
      service: 'AI-Healthcare-Agent',
    };

    vi.spyOn(apiClient, 'get').mockResolvedValueOnce({ data: mockLiveness });

    const res = await checkLiveness();
    expect(res.status).toBe('alive');
    expect(apiClient.get).toHaveBeenCalledWith('/health/liveness');
  });
});
