import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { HealthStatusBadge } from '../components/common/HealthStatusBadge';
import { HealthProvider } from '../context/HealthContext';
import { ragApi } from '../api/ragApi';

vi.mock('../api/ragApi', () => ({
  ragApi: {
    checkHealth: vi.fn(),
  },
}));

describe('Backend Health & Connectivity UX', () => {
  it('displays "Connected" when backend returns healthy status', async () => {
    vi.mocked(ragApi.checkHealth).mockResolvedValueOnce({
      status: 'healthy',
      service: 'AI-Healthcare-Agent',
      environment: 'development',
    });

    render(
      <HealthProvider>
        <HealthStatusBadge />
      </HealthProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });

  it('displays "Offline" when backend health check rejects/fails', async () => {
    vi.mocked(ragApi.checkHealth).mockRejectedValueOnce(new Error('Network error'));

    render(
      <HealthProvider>
        <HealthStatusBadge />
      </HealthProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Offline')).toBeInTheDocument();
    });
  });
});
