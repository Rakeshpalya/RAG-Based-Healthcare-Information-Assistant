import { describe, it, expect, vi, afterEach } from 'vitest';
import { streamRAGQuery } from '../utils/sseParser';

describe('Server-Sent Events (SSE) Streaming Parser', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('correctly parses tokens and completes stream', async () => {
    const ssePayload = [
      'event: start\ndata: {"trace_id": "tr_123", "question": "test?"}\n\n',
      'event: status\ndata: {"step": "retrieval", "message": "Searching FAISS"}\n\n',
      'event: token\ndata: {"token": "Hypertension "}\n\n',
      'event: token\ndata: {"token": "is a condition."}\n\n',
      'event: complete\ndata: {"question": "test?", "answer": "Hypertension is a condition.", "sources": [], "retrieval_status": "success", "disclaimer": "Medical note"}\n\n',
    ].join('');

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(ssePayload));
        controller.close();
      },
    });

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      body: stream,
    });

    const tokens: string[] = [];
    const statuses: string[] = [];
    let completedAnswer = '';

    await streamRAGQuery(
      { question: 'test?' },
      {
        onToken: (t) => tokens.push(t),
        onStatus: (s) => statuses.push(s.message || ''),
        onComplete: (res) => {
          completedAnswer = res.answer;
        },
      }
    );

    expect(tokens).toEqual(['Hypertension ', 'is a condition.']);
    expect(statuses).toContain('Searching FAISS');
    expect(completedAnswer).toBe('Hypertension is a condition.');
  });

  it('triggers onError callback on server error event', async () => {
    const ssePayload = 'event: error\ndata: {"error": "LLM rate limit encountered"}\n\n';

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(ssePayload));
        controller.close();
      },
    });

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      body: stream,
    });

    let receivedError = '';
    await streamRAGQuery(
      { question: 'test' },
      {
        onError: (err) => {
          receivedError = err.message;
        },
      }
    );

    expect(receivedError).toBe('LLM rate limit encountered');
  });

  it('respects AbortSignal when user cancels generation', async () => {
    const abortController = new AbortController();
    abortController.abort();

    global.fetch = vi.fn().mockRejectedValue(new DOMException('Aborted', 'AbortError'));

    const onError = vi.fn();
    await streamRAGQuery({ question: 'test' }, { onError }, abortController.signal);

    expect(onError).not.toHaveBeenCalled();
  });
});
