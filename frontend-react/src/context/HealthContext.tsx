import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { BackendHealthResponse, HealthConnectivityState } from '../types';
import { ragApi } from '../api/ragApi';

interface HealthContextType {
  health: BackendHealthResponse | null;
  status: HealthConnectivityState;
  lastChecked: Date | null;
  refreshHealth: () => Promise<void>;
}

const HealthContext = createContext<HealthContextType | undefined>(undefined);

export const HealthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [health, setHealth] = useState<BackendHealthResponse | null>(null);
  const [status, setStatus] = useState<HealthConnectivityState>('CHECKING');
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const data = await ragApi.checkHealth();
      setHealth(data);
      if (data.status === 'healthy') {
        setStatus('CONNECTED');
      } else {
        setStatus('DEGRADED');
      }
      setLastChecked(new Date());
    } catch {
      setStatus('OFFLINE');
      setLastChecked(new Date());
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    // Non-aggressive polling every 60 seconds
    const interval = setInterval(refreshHealth, 60000);
    return () => clearInterval(interval);
  }, [refreshHealth]);

  return (
    <HealthContext.Provider
      value={{
        health,
        status,
        lastChecked,
        refreshHealth,
      }}
    >
      {children}
    </HealthContext.Provider>
  );
};

export const useHealth = (): HealthContextType => {
  const context = useContext(HealthContext);
  if (!context) {
    throw new Error('useHealth must be used within a HealthProvider');
  }
  return context;
};
