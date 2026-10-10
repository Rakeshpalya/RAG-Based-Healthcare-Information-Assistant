import { RAGQueryRequest, RAGQueryResponse } from '../types';

export interface SSECallbacks {
  onStart?: (data: { trace_id?: string; question?: string; timestamp?: string }) => void;
  onStatus?: (data: { step?: string; message?: string }) => void;
  onToken?: (token: string) => void;
  onComplete?: (response: RAGQueryResponse) => void;
  onError?: (error: Error) => void;
}

/**
 * Executes a streaming RAG query against POST /rag/stream using fetch ReadableStream.
 * Handles SSE event parsing, partial chunk buffering, token emission, and safe completion.
 */
export async function streamRAGQuery(
  request: RAGQueryRequest,
  callbacks: SSECallbacks,
  signal?: AbortSignal
): Promise<void> {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '';
  const url = `${baseUrl}/rag/stream`;

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  const token = localStorage.getItem('auth_token');
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok) {
      let errorDetail = `Streaming request failed with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorDetail = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
        }
      } catch {
        // use default error detail
      }
      throw new Error(errorDetail);
    }

    if (!response.body) {
      throw new Error('Streaming response body is undefined.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let currentEvent = 'message';
    let currentData = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete trailing line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          // Empty line indicates event boundary in SSE specification
          if (currentData) {
            try {
              const parsed = JSON.parse(currentData);
              if (currentEvent === 'start' && callbacks.onStart) {
                callbacks.onStart(parsed);
              } else if (currentEvent === 'status' && callbacks.onStatus) {
                callbacks.onStatus(parsed);
              } else if (currentEvent === 'token') {
                const tokenStr = parsed.token !== undefined ? parsed.token : (parsed.text || '');
                if (callbacks.onToken && tokenStr) {
                  callbacks.onToken(tokenStr);
                }
              } else if (currentEvent === 'complete' && callbacks.onComplete) {
                callbacks.onComplete(parsed as RAGQueryResponse);
              } else if (currentEvent === 'error') {
                const errMsg = parsed.error || 'Server streaming error';
                if (callbacks.onError) {
                  callbacks.onError(new Error(errMsg));
                }
              }
            } catch (jsonErr) {
              // Non-JSON or plaintext data
              if (currentEvent === 'token' && callbacks.onToken) {
                callbacks.onToken(currentData);
              }
            }
          }
          currentEvent = 'message';
          currentData = '';
          continue;
        }

        if (trimmed.startsWith('event:')) {
          currentEvent = trimmed.slice(6).trim();
        } else if (trimmed.startsWith('data:')) {
          const dataContent = trimmed.slice(5).trim();
          currentData = currentData ? currentData + '\n' + dataContent : dataContent;
        }
      }
    }

    // Process any remaining buffered data
    if (currentData && callbacks.onComplete) {
      try {
        const parsed = JSON.parse(currentData);
        if (currentEvent === 'complete') {
          callbacks.onComplete(parsed as RAGQueryResponse);
        }
      } catch {
        // ignore trailing parse error
      }
    }
  } catch (err: any) {
    if (signal?.aborted) {
      // Stream deliberately cancelled by user
      return;
    }
    if (callbacks.onError) {
      callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    } else {
      throw err;
    }
  }
}
