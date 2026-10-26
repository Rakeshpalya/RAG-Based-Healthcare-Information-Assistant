import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { FloatingChatbot } from '../components/chat/FloatingChatbot';
import { HealthLogo } from '../components/common/HealthLogo';
import { ragApi } from '../api/ragApi';

vi.mock('../api/ragApi', () => ({
  ragApi: {
    queryRAG: vi.fn(),
    checkHealth: vi.fn().mockResolvedValue({ status: 'healthy' }),
  },
}));

describe('HealthLogo White Background & Stethoscope Graphic', () => {
  it('renders white background (#FFFFFF) container with rounded corners and stethoscope vector graphic', () => {
    const { container } = render(<HealthLogo size="md" />);

    const logoContainer = screen.getByLabelText(/HealthAI Stethoscope Logo/i);
    expect(logoContainer).toBeInTheDocument();

    // Verify background is white
    expect(logoContainer).toHaveStyle({ backgroundColor: '#FFFFFF' });
    expect(logoContainer.className).toContain('bg-white');
    expect(logoContainer.className).toContain('border-slate-200');
    expect(logoContainer.className).toContain('rounded-xl');

    // Verify stethoscope SVG graphic exists (not emoji text)
    const svgIcon = container.querySelector('svg');
    expect(svgIcon).toBeInTheDocument();
    expect(logoContainer.textContent).not.toContain('🩺');
  });

  it('supports different sizes (sm, md, lg) with consistent white background', () => {
    const { unmount } = render(<HealthLogo size="sm" />);
    let logo = screen.getByLabelText(/HealthAI Stethoscope Logo/i);
    expect(logo.className).toContain('w-8 h-8');
    expect(logo).toHaveStyle({ backgroundColor: '#FFFFFF' });
    unmount();

    render(<HealthLogo size="lg" />);
    logo = screen.getByLabelText(/HealthAI Stethoscope Logo/i);
    expect(logo.className).toContain('w-12 h-12');
    expect(logo).toHaveStyle({ backgroundColor: '#FFFFFF' });
  });
});

describe('Floating Chatbot Widget Behavior & RAG Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders launcher button in closed state with Stethoscope graphic and HealthAI label', () => {
    render(<FloatingChatbot />);

    const launcher = screen.getByTestId('floating-chatbot-launcher');
    expect(launcher).toBeInTheDocument();
    expect(launcher.className).toContain('fixed');
    expect(launcher.className).toContain('bottom-6');
    expect(launcher.className).toContain('right-6');
    expect(screen.getByText('HealthAI')).toBeInTheDocument();

    // Popup window should not be open yet
    expect(screen.queryByTestId('floating-chatbot-popup')).not.toBeInTheDocument();
  });

  it('opens compact popup window when launcher button is clicked', () => {
    render(<FloatingChatbot />);

    fireEvent.click(screen.getByTestId('floating-chatbot-launcher'));

    const popup = screen.getByTestId('floating-chatbot-popup');
    expect(popup).toBeInTheDocument();
    expect(popup.className).toContain('fixed');
    expect(popup.className).toContain('bottom-6');
    expect(popup.className).toContain('right-6');

    // Title and initial greeting
    expect(screen.getByText('HealthAI Medical Assistant')).toBeInTheDocument();
    expect(screen.getByText('General Medical Q&A')).toBeInTheDocument();
    expect(screen.getByText(/Ask me general medical questions/i)).toBeInTheDocument();

    // Suggested question chips
    expect(screen.getByText('What is hypertension?')).toBeInTheDocument();
    expect(screen.getByText('What is asthma?')).toBeInTheDocument();
  });

  it('closes popup window when close (X) button is clicked', () => {
    render(<FloatingChatbot />);

    // Open
    fireEvent.click(screen.getByTestId('floating-chatbot-launcher'));
    expect(screen.getByTestId('floating-chatbot-popup')).toBeInTheDocument();

    // Close
    fireEvent.click(screen.getByTestId('floating-chatbot-close-btn'));
    expect(screen.queryByTestId('floating-chatbot-popup')).not.toBeInTheDocument();
    expect(screen.getByTestId('floating-chatbot-launcher')).toBeInTheDocument();
  });

  it('sends inquiry to RAG endpoint and displays answer with source citations', async () => {
    vi.mocked(ragApi.queryRAG).mockResolvedValue({
      question: 'What is hypertension?',
      answer: 'Hypertension is defined as chronically elevated systolic blood pressure >= 130 mmHg.',
      retrieval_status: 'success',
      sources: [
        {
          chunk_id: 'chunk-1',
          document_name: 'aha_hypertension_guidelines.pdf',
          page_number: 14,
          similarity_score: 0.92,
          text: 'Stage 1 hypertension definition...',
        },
      ],
      disclaimer: 'General medical information only.',
    });

    render(<FloatingChatbot />);
    fireEvent.click(screen.getByTestId('floating-chatbot-launcher'));

    const input = screen.getByPlaceholderText(/Ask about your health.../i);
    fireEvent.change(input, { target: { value: 'What is hypertension?' } });

    const sendBtn = screen.getByTestId('floating-chatbot-send-btn');
    fireEvent.click(sendBtn);

    await waitFor(() => {
      expect(ragApi.queryRAG).toHaveBeenCalledWith(
        expect.objectContaining({
          question: 'What is hypertension?',
          top_k: 5,
        })
      );
      expect(
        screen.getByText(/Hypertension is defined as chronically elevated/i)
      ).toBeInTheDocument();
      expect(screen.getByText(/aha_hypertension_guidelines/i)).toBeInTheDocument();
    });
  });

  it('renders EmergencyBanner when backend flags an emergency query', async () => {
    vi.mocked(ragApi.queryRAG).mockResolvedValue({
      question: 'I have severe chest pain',
      answer: 'Acute crushing chest pain radiating to the left arm requires urgent evaluation.',
      retrieval_status: 'safety_intercepted',
      sources: [],
      disclaimer: 'General medical information only.',
      timings: {
        safety_assessment: {
          category: 'emergency',
          risk_level: 'high',
          requires_escalation: true,
        },
      },
    });

    render(<FloatingChatbot />);
    fireEvent.click(screen.getByTestId('floating-chatbot-launcher'));

    const input = screen.getByPlaceholderText(/Ask about your health.../i);
    fireEvent.change(input, { target: { value: 'I have severe chest pain' } });

    fireEvent.click(screen.getByTestId('floating-chatbot-send-btn'));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(screen.getByText('EMERGENCY MEDICAL ADVISORY')).toBeInTheDocument();
      expect(
        screen.getByText(/Emergency symptoms or acute medical danger detected/i)
      ).toBeInTheDocument();
    });
  });

  it('clicking suggested question chip triggers immediate query', async () => {
    vi.mocked(ragApi.queryRAG).mockResolvedValue({
      question: 'What is asthma?',
      answer: 'Asthma is a chronic inflammatory disorder of the airways.',
      retrieval_status: 'success',
      sources: [],
      disclaimer: 'General medical information only.',
    });

    render(<FloatingChatbot />);
    fireEvent.click(screen.getByTestId('floating-chatbot-launcher'));

    const chip = screen.getByText('What is asthma?');
    fireEvent.click(chip);

    await waitFor(() => {
      expect(ragApi.queryRAG).toHaveBeenCalledWith(
        expect.objectContaining({
          question: 'What is asthma?',
        })
      );
      expect(
        screen.getByText(/Asthma is a chronic inflammatory disorder/i)
      ).toBeInTheDocument();
    });
  });
});
