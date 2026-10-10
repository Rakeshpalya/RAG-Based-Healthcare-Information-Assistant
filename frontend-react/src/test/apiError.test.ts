import { describe, it, expect } from 'vitest';
import { normalizeApiError } from '../api/client';
import { AxiosError } from 'axios';

describe('Centralized API Error Normalization', () => {
  it('handles 401 Unauthorized session expiration cleanly', () => {
    const mockAxiosError = {
      isAxiosError: true,
      response: {
        status: 401,
        data: { detail: 'Could not validate credentials' },
      },
    } as unknown as AxiosError;

    const result = normalizeApiError(mockAxiosError);
    expect(result.status).toBe(401);
    expect(result.message).toContain('credentials');
  });

  it('handles 403 Forbidden access denied', () => {
    const mockAxiosError = {
      isAxiosError: true,
      response: {
        status: 403,
        data: { detail: 'Access denied: Cannot assign document to another user.' },
      },
    } as unknown as AxiosError;

    const result = normalizeApiError(mockAxiosError);
    expect(result.status).toBe(403);
    expect(result.message).toContain('Access denied');
  });

  it('handles 404 Not Found', () => {
    const mockAxiosError = {
      isAxiosError: true,
      response: {
        status: 404,
        data: { detail: 'Document with ID 999 not found.' },
      },
    } as unknown as AxiosError;

    const result = normalizeApiError(mockAxiosError);
    expect(result.status).toBe(404);
    expect(result.message).toContain('not found');
  });

  it('handles 422 Validation Error with array of errors', () => {
    const mockAxiosError = {
      isAxiosError: true,
      response: {
        status: 422,
        data: { detail: [{ msg: 'Field required: question' }] },
      },
    } as unknown as AxiosError;

    const result = normalizeApiError(mockAxiosError);
    expect(result.status).toBe(422);
    expect(result.message).toContain('Field required');
  });

  it('handles 429 Rate Limit exceeded', () => {
    const mockAxiosError = {
      isAxiosError: true,
      response: {
        status: 429,
        data: { detail: 'Rate limit exceeded: max 30 queries per minute.' },
      },
    } as unknown as AxiosError;

    const result = normalizeApiError(mockAxiosError);
    expect(result.status).toBe(429);
    expect(result.message).toContain('Rate limit');
  });

  it('handles 500 Server Error without leaking internal stack traces', () => {
    const mockAxiosError = {
      isAxiosError: true,
      response: {
        status: 500,
        data: { detail: 'Internal server error Traceback: File main.py line 40' },
      },
    } as unknown as AxiosError;

    const result = normalizeApiError(mockAxiosError);
    expect(result.status).toBe(500);
    expect(result.message).toBe('The clinical assistant service is temporarily unavailable. Please try again shortly.');
  });

  it('handles Network Failure / Offline backend cleanly', () => {
    const mockNetworkError = {
      isAxiosError: true,
      code: 'ECONNABORTED',
      message: 'Network Error',
    } as unknown as AxiosError;

    const result = normalizeApiError(mockNetworkError);
    expect(result.status).toBe(0);
    expect(result.message).toContain('Unable to connect to the backend server');
  });
});
