import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { ApiError } from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request Interceptor: Attach Bearer token
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('auth_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Helper to normalize errors into user-friendly messages
export function normalizeApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const err = error as AxiosError<{ detail?: string | Array<{ msg?: string }>; message?: string }>;
    const status = err.response?.status || 0;
    let detail = '';

    if (err.response?.data) {
      const data = err.response.data;
      if (typeof data.detail === 'string') {
        detail = data.detail;
      } else if (Array.isArray(data.detail)) {
        detail = data.detail.map((d) => d.msg || '').join(', ');
      } else if (data.message) {
        detail = data.message;
      }
    }

    if (status === 401) {
      return {
        status: 401,
        message: detail || 'Your session has expired. Please sign in again.',
        detail,
      };
    }
    if (status === 403) {
      return {
        status: 403,
        message: detail || 'Access denied. You do not have permission for this resource.',
        detail,
      };
    }
    if (status === 404) {
      return {
        status: 404,
        message: detail || 'The requested resource was not found.',
        detail,
      };
    }
    if (status === 422) {
      return {
        status: 422,
        message: detail || 'The submitted data was invalid. Please review your input.',
        detail,
      };
    }
    if (status === 429) {
      return {
        status: 429,
        message: detail || 'Too many requests. Please wait a moment before trying again.',
        detail,
      };
    }
    if (status >= 500) {
      return {
        status,
        message: 'The clinical assistant service is temporarily unavailable. Please try again shortly.',
        detail,
      };
    }

    // Network / timeout
    if (err.code === 'ECONNABORTED' || !err.response) {
      return {
        status: 0,
        message: 'Unable to connect to the backend server. Please verify your connection or service status.',
        detail: err.message,
      };
    }

    return {
      status,
      message: detail || err.message || 'An unexpected error occurred.',
      detail,
    };
  }

  return {
    status: 0,
    message: 'An unexpected application error occurred.',
    detail: String(error),
  };
}

// Response Interceptor: Normalize errors & handle 401
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('auth_user');
      window.dispatchEvent(new Event('auth_session_expired'));
    }
    return Promise.reject(normalizeApiError(error));
  }
);
