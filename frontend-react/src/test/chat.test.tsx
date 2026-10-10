import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatArea } from '../components/chat/ChatArea';
import { ChatInput } from '../components/chat/ChatInput';
import { ChatMessageItem } from '../components/chat/ChatMessage';

describe('Clinical Chat UX & Multi-Turn Presentation', () => {
  it('renders clinical starter prompts when dialogue is empty', () => {
    const onSelectPrompt = vi.fn();

    render(
      <ChatArea
        messages={[]}
        isLoading={false}
        streamingToken=""
        onSelectPrompt={onSelectPrompt}
      />
    );

    expect(
      screen.getByText(/Clinical Intelligence & Consultation Assistant/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Suggested Clinical Prompts/i)
    ).toBeInTheDocument();

    const promptBtn = screen.getByText(/Type 2 Diabetes/i);
    expect(promptBtn).toBeInTheDocument();

    fireEvent.click(promptBtn);
    expect(onSelectPrompt).toHaveBeenCalledWith(
      expect.stringContaining('Type 2 Diabetes')
    );
  });

  it('renders message thread with request ID and multi-turn badge', () => {
    const messages: ChatMessageItem[] = [
      {
        id: 'user-1',
        sender: 'user',
        text: 'Patient has stage 3 CKD. What analgesic options are safe?',
      },
      {
        id: 'asst-1',
        sender: 'assistant',
        text: 'In stage 3 CKD, acetaminophen is generally preferred for mild-to-moderate pain. NSAIDs must be avoided.',
        requestId: 'req_clinical_audit_987',
        dialogueContext: {
          is_follow_up: true,
          turn_count: 2,
        },
      },
    ];

    render(
      <ChatArea
        messages={messages}
        isLoading={false}
        streamingToken=""
        onSelectPrompt={() => {}}
      />
    );

    expect(screen.getByText(/Patient has stage 3 CKD/i)).toBeInTheDocument();
    expect(screen.getByText(/acetaminophen is generally preferred/i)).toBeInTheDocument();
    expect(screen.getByText('req_clinical_audit_987')).toBeInTheDocument();
    expect(screen.getByText(/Multi-turn context preserved/i)).toBeInTheDocument();
  });

  it('handles input submission and dispatches message', () => {
    const onSendMessage = vi.fn();

    render(
      <ChatInput
        onSendMessage={onSendMessage}
        isLoading={false}
        isStreaming={false}
      />
    );

    const textarea = screen.getByPlaceholderText(/Ask a clinical research question/i);
    fireEvent.change(textarea, { target: { value: 'Explain hypertension staging criteria' } });

    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSendMessage).toHaveBeenCalledWith('Explain hypertension staging criteria');
  });
});
