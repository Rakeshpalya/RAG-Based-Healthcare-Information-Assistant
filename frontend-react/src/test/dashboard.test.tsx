import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { DashboardPage } from '../pages/DashboardPage';
import { conversationApi, normalizeConversationList } from '../api/conversationApi';
import { documentApi, normalizeDocumentList } from '../api/documentApi';
import { AuthProvider } from '../context/AuthContext';
import { HealthProvider } from '../context/HealthContext';
import { Conversation, DocumentRecord } from '../types';

vi.mock('../api/conversationApi', async () => {
  const actual = await vi.importActual<typeof import('../api/conversationApi')>(
    '../api/conversationApi'
  );
  return {
    ...actual,
    conversationApi: {
      listConversations: vi.fn(),
      createConversation: vi.fn(),
      getMessages: vi.fn(),
      appendMessage: vi.fn(),
      updateTitle: vi.fn(),
      deleteConversation: vi.fn(),
      clearAllConversations: vi.fn(),
    },
  };
});

vi.mock('../api/documentApi', async () => {
  const actual = await vi.importActual<typeof import('../api/documentApi')>(
    '../api/documentApi'
  );
  return {
    ...actual,
    documentApi: {
      listDocuments: vi.fn(),
      uploadDocument: vi.fn(),
      getDocument: vi.fn(),
      deleteDocument: vi.fn(),
    },
  };
});

vi.mock('../api/ragApi', () => ({
  ragApi: {
    checkHealth: vi.fn().mockResolvedValue({
      status: 'healthy',
      service: 'AI-Healthcare-Agent',
      vector_store: 'ready',
      embedding_service: 'ready',
    }),
  },
}));

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

describe('Dashboard Conversation & Document Normalization Unit Tests', () => {
  const sampleConvs: Conversation[] = [
    {
      id: 1,
      user_id: 1,
      title: 'Diabetes Mellitus Consultation',
      created_at: '2026-10-10T12:00:00Z',
      updated_at: '2026-10-10T12:30:00Z',
    },
  ];

  const sampleDocs: DocumentRecord[] = [
    {
      id: 101,
      filename: 'clinical_guidelines_2026.pdf',
      file_size_bytes: 1048576,
      num_pages: 5,
      num_chunks: 12,
      status: 'processed',
      created_at: '2026-10-10T10:00:00Z',
    },
  ];

  it('normalizes direct array response cleanly', () => {
    expect(normalizeConversationList(sampleConvs)).toEqual(sampleConvs);
    expect(normalizeDocumentList(sampleDocs)).toEqual(sampleDocs);
  });

  it('normalizes wrapped payload { conversations: [...] } or { documents: [...] }', () => {
    expect(normalizeConversationList({ conversations: sampleConvs })).toEqual(sampleConvs);
    expect(normalizeDocumentList({ documents: sampleDocs })).toEqual(sampleDocs);
  });

  it('normalizes wrapped payload { items: [...] }, { data: [...] }, or { results: [...] }', () => {
    expect(normalizeConversationList({ items: sampleConvs })).toEqual(sampleConvs);
    expect(normalizeConversationList({ data: sampleConvs })).toEqual(sampleConvs);
    expect(normalizeConversationList({ results: sampleConvs })).toEqual(sampleConvs);

    expect(normalizeDocumentList({ items: sampleDocs })).toEqual(sampleDocs);
    expect(normalizeDocumentList({ data: sampleDocs })).toEqual(sampleDocs);
    expect(normalizeDocumentList({ results: sampleDocs })).toEqual(sampleDocs);
  });

  it('normalizes empty array to empty array', () => {
    expect(normalizeConversationList([])).toEqual([]);
    expect(normalizeDocumentList([])).toEqual([]);
  });

  it('handles unexpected response types (HTML string, numbers, null, undefined) safely', () => {
    // The exact bug scenario: Vite/proxy returns index.html string
    const htmlString = '<!doctype html><html><head><title>App</title></head><body></body></html>';
    expect(normalizeConversationList(htmlString)).toEqual([]);
    expect(normalizeDocumentList(htmlString)).toEqual([]);

    expect(normalizeConversationList(null)).toEqual([]);
    expect(normalizeConversationList(undefined)).toEqual([]);
    expect(normalizeConversationList(12345)).toEqual([]);
    expect(normalizeConversationList({})).toEqual([]);
    expect(normalizeConversationList({ unexpected: 'shape' })).toEqual([]);

    expect(normalizeDocumentList(null)).toEqual([]);
    expect(normalizeDocumentList(undefined)).toEqual([]);
    expect(normalizeDocumentList(false)).toEqual([]);
    expect(normalizeDocumentList({})).toEqual([]);
  });

  it('filters out malformed or primitive items within an array', () => {
    const mixed = [null, undefined, 'corrupt', 42, sampleConvs[0]];
    const res = normalizeConversationList(mixed);
    expect(res).toHaveLength(1);
    expect(res[0].id).toBe(1);
  });
});

describe('DashboardPage Blank-Page Crash Fix & Resilience', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem('auth_token', 'test-token');
    localStorage.setItem(
      'auth_user',
      JSON.stringify({ id: '1', email: 'doctor@hospital.org' })
    );
  });

  const renderDashboard = () =>
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <AuthProvider>
          <HealthProvider>
            <DashboardPage />
          </HealthProvider>
        </AuthProvider>
      </MemoryRouter>
    );

  it('Scenario 1: Renders successfully with standard array response', async () => {
    const mockConvs: Conversation[] = [
      {
        id: 10,
        user_id: 1,
        title: 'Cardiovascular Risk Assessment',
        created_at: '2026-10-10T14:00:00Z',
        updated_at: '2026-10-10T14:20:00Z',
      },
    ];
    const mockDocs: DocumentRecord[] = [
      {
        id: 20,
        filename: 'echocardiogram_findings.pdf',
        file_size_bytes: 2048576,
        num_pages: 3,
        num_chunks: 8,
        status: 'processed',
        created_at: '2026-10-10T12:00:00Z',
      },
    ];

    vi.mocked(conversationApi.listConversations).mockResolvedValue(mockConvs);
    vi.mocked(documentApi.listDocuments).mockResolvedValue(mockDocs);

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Cardiovascular Risk Assessment')).toBeInTheDocument();
      expect(screen.getByText(/echocardiogram_findings/i)).toBeInTheDocument();
    });

    // Metric counts should display 1
    const cards = screen.getAllByText('1');
    expect(cards.length).toBeGreaterThanOrEqual(2);
  });

  it('Scenario 2: Renders successfully with wrapped object response shape', async () => {
    const wrappedConvs: any = {
      conversations: [
        {
          id: 15,
          user_id: 1,
          title: 'Pulmonary Hypertension Followup',
          created_at: '2026-10-10T15:00:00Z',
          updated_at: '2026-10-10T15:30:00Z',
        },
      ],
    };

    vi.mocked(conversationApi.listConversations).mockResolvedValue(wrappedConvs);
    vi.mocked(documentApi.listDocuments).mockResolvedValue([]);

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Pulmonary Hypertension Followup')).toBeInTheDocument();
    });
  });

  it('Scenario 3: Renders clean empty state when response is an empty array', async () => {
    vi.mocked(conversationApi.listConversations).mockResolvedValue([]);
    vi.mocked(documentApi.listDocuments).mockResolvedValue([]);

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByTestId('conversations-empty')).toBeInTheDocument();
      expect(screen.getByText(/No recent consultations found/i)).toBeInTheDocument();
      expect(screen.getByTestId('documents-empty')).toBeInTheDocument();
      expect(screen.getByText(/No documents uploaded yet/i)).toBeInTheDocument();
    });
  });

  it('Scenario 4: Handles failed request without blank-page crash and renders retry button', async () => {
    vi.mocked(conversationApi.listConversations).mockRejectedValue(
      new Error('Backend server 503 unavailable')
    );
    vi.mocked(documentApi.listDocuments).mockResolvedValue([]);

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByTestId('conversations-error')).toBeInTheDocument();
      expect(screen.getByText(/Backend server 503 unavailable/i)).toBeInTheDocument();
      expect(screen.getByTestId('retry-conversations-btn')).toBeInTheDocument();
    });

    // Rest of dashboard remains interactive and functional
    expect(screen.getByText(/Welcome back/i)).toBeInTheDocument();
  });

  it('Scenario 5: Prevents crash when API returns unexpected HTML string (the original bug)', async () => {
    // Exactly replicates the original uncaught TypeError: conversations.map is not a function
    const htmlResponse = '<!doctype html><html><body>Proxy Fallback</body></html>' as any;
    vi.mocked(conversationApi.listConversations).mockResolvedValue(htmlResponse);
    vi.mocked(documentApi.listDocuments).mockResolvedValue(htmlResponse);

    renderDashboard();

    // Must NOT crash, should render clean empty states
    await waitFor(() => {
      expect(screen.getByTestId('conversations-empty')).toBeInTheDocument();
      expect(screen.getByTestId('documents-empty')).toBeInTheDocument();
    });

    expect(screen.queryByText(/TypeError/i)).not.toBeInTheDocument();
  });

  it('Scenario 6: Retry action re-triggers data fetch upon user click', async () => {
    vi.mocked(conversationApi.listConversations)
      .mockRejectedValueOnce(new Error('Transient network glitch'))
      .mockResolvedValueOnce([
        {
          id: 99,
          user_id: 1,
          title: 'Resolved Consultation After Retry',
          created_at: '2026-10-10T16:00:00Z',
          updated_at: '2026-10-10T16:10:00Z',
        },
      ]);
    vi.mocked(documentApi.listDocuments).mockResolvedValue([]);

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByTestId('conversations-error')).toBeInTheDocument();
    });

    const retryBtn = screen.getByTestId('retry-conversations-btn');
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(screen.getByText('Resolved Consultation After Retry')).toBeInTheDocument();
    });
  });
});
