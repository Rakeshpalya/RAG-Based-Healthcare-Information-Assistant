import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from '../App';
import * as ragApi from '../api/ragApi';
import { RAGQueryResponse, ReadinessResponse } from '../types';

vi.mock('../api/ragApi');

describe('AI Healthcare Assistant Frontend (Phase 7)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();

    // Default mock for readiness check
    const mockReadiness: ReadinessResponse = {
      status: 'ready',
      service: 'AI-Healthcare-Agent',
      environment: 'production',
      checks: {
        vector_store: { status: 'ready', vector_count: 744, expected_count: 744 },
        embedding_service: { status: 'ready', dimension: 384 },
        safety_engine: { status: 'ready' },
      },
    };
    vi.mocked(ragApi.checkReadiness).mockResolvedValue(mockReadiness);
  });

  // 1. Chat Rendering & Initial Screen
  it('renders initial application layout with title, empty state, and disclaimer', async () => {
    render(<App />);

    expect(screen.getByText('AI Healthcare Assistant')).toBeInTheDocument();
    expect(screen.getByText(/How can I assist your medical research today\?/i)).toBeInTheDocument();
    expect(screen.getByText(/Educational information only/i)).toBeInTheDocument();
  });

  // 10. Readiness Status Display
  it('renders system ready status indicator upon successful readiness probe', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('System Ready')).toBeInTheDocument();
    });
  });

  // 11. Empty / Short Input Validation
  it('disables send button when input is empty or shorter than 3 characters', async () => {
    render(<App />);

    const sendBtn = screen.getByTitle('Send query (Enter)');
    expect(sendBtn).toBeDisabled();

    const textarea = screen.getByPlaceholderText(/Ask a medical or clinical research question/i);
    fireEvent.change(textarea, { target: { value: 'hi' } });
    expect(sendBtn).toBeDisabled();

    fireEvent.change(textarea, { target: { value: 'What is hypertension?' } });
    expect(sendBtn).not.toBeDisabled();
  });

  // 2, 3, 4, 5, 9. Sending Question, Loading State, RAG Response, Citations, Request ID
  it('sends question, shows loading state, and renders grounded answer with citations and request ID', async () => {
    const mockRAGResponse: RAGQueryResponse = {
      question: 'What is hypertension?',
      answer: 'Hypertension is defined as persistent blood pressure elevation [Source 1].',
      retrieval_status: 'success',
      sources: [
        {
          chunk_id: 'CHUNK_001',
          document_name: 'Hypertension_Guideline.pdf',
          similarity_score: 0.88,
          page_number: 1,
          text: 'Hypertension is defined as persistent systolic BP >= 140 mmHg.',
        },
      ],
      disclaimer: 'Educational information only.',
      request_id: 'req_test_abc123',
    };

    vi.mocked(ragApi.queryRAG).mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(mockRAGResponse), 50))
    );

    render(<App />);

    const textarea = screen.getByPlaceholderText(/Ask a medical or clinical research question/i);
    const sendBtn = screen.getByTitle('Send query (Enter)');

    fireEvent.change(textarea, { target: { value: 'What is hypertension?' } });
    fireEvent.click(sendBtn);

    // Verify User question rendered immediately (in chat stream and sidebar title)
    expect(screen.getAllByText('What is hypertension?').length).toBeGreaterThanOrEqual(1);

    // Verify loading indicator is present
    expect(screen.getByText(/Searching vector database & synthesizing grounded evidence/i)).toBeInTheDocument();

    // Await response
    await waitFor(() => {
      expect(screen.getByText(/Hypertension is defined as persistent blood pressure elevation/i)).toBeInTheDocument();
    });

    // 5. Verify Citation Rendering
    expect(screen.getByText('Source 1')).toBeInTheDocument();
    expect(screen.getByText('Hypertension_Guideline.pdf')).toBeInTheDocument();
    expect(screen.getByText(/88% relevance/i)).toBeInTheDocument();

    // 9. Verify Request ID display
    expect(screen.getByText(/ID: req_test_abc123/i)).toBeInTheDocument();
  });

  // 6, 7. Medical Safety & Acute Emergency Interception
  it('renders prominent emergency advisory banner when backend intercepts acute emergency', async () => {
    const mockEmergencyResponse: RAGQueryResponse = {
      question: 'I have severe chest pain and left arm numbness',
      answer: 'EMERGENCY ADVISORY: Inquiry contains acute life-threatening emergency medical symptoms. Please call 911 or visit the nearest emergency room immediately.',
      retrieval_status: 'safety_intercepted',
      sources: [],
      disclaimer: 'Educational information only.',
      timings: {
        safety_assessment: {
          category: 'EMERGENCY_SYMPTOMS',
          risk_level: 'CRITICAL',
          requires_escalation: true,
          allow_normal_rag: false,
        },
        llm_called: false,
      },
      request_id: 'req_emergency_999',
    };

    vi.mocked(ragApi.queryRAG).mockResolvedValue(mockEmergencyResponse);

    render(<App />);

    const textarea = screen.getByPlaceholderText(/Ask a medical or clinical research question/i);
    fireEvent.change(textarea, { target: { value: 'I have severe chest pain and left arm numbness' } });
    fireEvent.click(screen.getByTitle('Send query (Enter)'));

    await waitFor(() => {
      expect(screen.getByText(/IMPORTANT — SEEK IMMEDIATE EMERGENCY CARE/i)).toBeInTheDocument();
      expect(screen.getByText(/Call 911 \(or local emergency\) Immediately/i)).toBeInTheDocument();
    });
  });

  // 8. Backend Failure & Error Handling
  it('displays friendly UI error and offers retry when backend request fails', async () => {
    vi.mocked(ragApi.queryRAG).mockRejectedValue(
      new Error('Unable to connect to the healthcare assistant. Please verify your connection or backend status.')
    );

    render(<App />);

    const textarea = screen.getByPlaceholderText(/Ask a medical or clinical research question/i);
    fireEvent.change(textarea, { target: { value: 'What causes hypertension?' } });
    fireEvent.click(screen.getByTitle('Send query (Enter)'));

    await waitFor(() => {
      expect(screen.getByText(/Unable to connect to the healthcare assistant/i)).toBeInTheDocument();
      expect(screen.getByText('Retry this query')).toBeInTheDocument();
    });
  });

  // 12. Duplicate Request Prevention
  it('prevents accidental duplicate requests while a query is currently processing', async () => {
    vi.mocked(ragApi.queryRAG).mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve({
        question: 'What is blood pressure?',
        answer: 'Blood pressure is the lateral force on arterial walls.',
        retrieval_status: 'success',
        sources: [],
        disclaimer: '',
      }), 100))
    );

    render(<App />);

    const textarea = screen.getByPlaceholderText(/Ask a medical or clinical research question/i);
    const sendBtn = screen.getByTitle('Send query (Enter)');

    fireEvent.change(textarea, { target: { value: 'What is blood pressure?' } });
    fireEvent.click(sendBtn);

    // Try clicking immediately again
    fireEvent.click(sendBtn);
    fireEvent.click(sendBtn);

    expect(ragApi.queryRAG).toHaveBeenCalledTimes(1);
  });
});
