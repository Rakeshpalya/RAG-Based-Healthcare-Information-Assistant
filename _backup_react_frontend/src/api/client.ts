import axios from 'axios';

// Base URL resolution: falls back to relative root (utilizing Vite proxy in dev) or specified env
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000, // 20s timeout budget
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Standardize error messaging without leaking sensitive credentials or stack traces
    if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      return Promise.reject(new Error('The request timed out. The medical reasoning engine took too long to respond.'));
    }
    if (!error.response) {
      return Promise.reject(new Error('Unable to connect to the healthcare assistant. Please verify your connection or backend status.'));
    }
    const status = error.response.status;
    if (status === 422) {
      return Promise.reject(new Error('Invalid query format. Please enter a question between 3 and 1000 characters.'));
    }
    if (status === 503) {
      return Promise.reject(new Error('The healthcare reasoning service is currently unavailable. Please try again shortly.'));
    }
    const message = error.response.data?.detail || 'An unexpected error occurred while processing your request.';
    return Promise.reject(new Error(message));
  }
);
