import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { User, AuthResponse } from '../types';
import { authApi } from '../api/authApi';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<AuthResponse>;
  signup: (email: string, password: string) => Promise<AuthResponse>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const clearAuth = useCallback(() => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    setToken(null);
    setUser(null);
  }, []);

  // Initialize from localStorage and verify
  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem('auth_token');
      const storedUser = localStorage.getItem('auth_user');

      if (storedToken && storedUser) {
        try {
          const parsedUser = JSON.parse(storedUser);
          setToken(storedToken);
          setUser(parsedUser);

          // Verify token is still valid with backend /auth/me
          try {
            const meRes = await authApi.getMe();
            if (meRes.user) {
              setUser(meRes.user);
              localStorage.setItem('auth_user', JSON.stringify(meRes.user));
            }
          } catch (err: any) {
            if (err.status === 401) {
              clearAuth();
            }
          }
        } catch {
          clearAuth();
        }
      }
      setIsLoading(false);
    };

    initAuth();

    const handleSessionExpired = () => {
      clearAuth();
    };

    window.addEventListener('auth_session_expired', handleSessionExpired);
    return () => {
      window.removeEventListener('auth_session_expired', handleSessionExpired);
    };
  }, [clearAuth]);

  const login = async (email: string, password: string): Promise<AuthResponse> => {
    const res = await authApi.login(email, password);
    if (res.session?.access_token && res.user) {
      localStorage.setItem('auth_token', res.session.access_token);
      localStorage.setItem('auth_user', JSON.stringify(res.user));
      setToken(res.session.access_token);
      setUser(res.user);
    }
    return res;
  };

  const signup = async (email: string, password: string): Promise<AuthResponse> => {
    const res = await authApi.signup(email, password);
    if (res.session?.access_token && res.user) {
      localStorage.setItem('auth_token', res.session.access_token);
      localStorage.setItem('auth_user', JSON.stringify(res.user));
      setToken(res.session.access_token);
      setUser(res.user);
    }
    return res;
  };

  const logout = async () => {
    try {
      await authApi.logout();
    } catch {
      // ignore network errors on logout
    } finally {
      clearAuth();
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token && !!user,
        isLoading,
        login,
        signup,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
