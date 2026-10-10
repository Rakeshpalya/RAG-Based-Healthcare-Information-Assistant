import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { ProtectedRoute } from '../components/common/ProtectedRoute';
import { AuthProvider } from '../context/AuthContext';

vi.mock('../api/authApi', () => ({
  authApi: {
    login: vi.fn(),
    signup: vi.fn(),
    logout: vi.fn(),
    getMe: vi.fn().mockResolvedValue({
      user: { id: '1', email: 'doctor@hospital.org' },
    }),
  },
}));

describe('Protected Route Guard', () => {
  it('redirects to /login when user is unauthenticated', async () => {
    localStorage.clear();

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AuthProvider>
          <Routes>
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <div>Secret Clinical Dashboard</div>
                </ProtectedRoute>
              }
            />
            <Route path="/login" element={<div>Login Page Target</div>} />
          </Routes>
        </AuthProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Login Page Target')).toBeInTheDocument();
    });
    expect(screen.queryByText('Secret Clinical Dashboard')).not.toBeInTheDocument();
  });

  it('renders protected child when authenticated', async () => {
    localStorage.setItem('auth_token', 'valid-token-xyz');
    localStorage.setItem(
      'auth_user',
      JSON.stringify({ id: '1', email: 'doctor@hospital.org' })
    );

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AuthProvider>
          <Routes>
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <div>Secret Clinical Dashboard</div>
                </ProtectedRoute>
              }
            />
            <Route path="/login" element={<div>Login Page Target</div>} />
          </Routes>
        </AuthProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Secret Clinical Dashboard')).toBeInTheDocument();
    });
  });
});

