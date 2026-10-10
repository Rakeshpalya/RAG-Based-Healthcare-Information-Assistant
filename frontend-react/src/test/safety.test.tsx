import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatMessage, ChatMessageItem } from '../components/chat/ChatMessage';
import { EmergencyBanner } from '../components/chat/EmergencyBanner';
import { ContraindicationBanner } from '../components/chat/ContraindicationBanner';

describe('Medical Safety & Contraindication UI', () => {
  it('renders emergency advisory banner with neutral emergency text', () => {
    render(
      <EmergencyBanner message="Acute chest pain with radiation to arm detected. Immediate clinical triage required." />
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/EMERGENCY MEDICAL ADVISORY/i)).toBeInTheDocument();
    expect(
      screen.getByText(
        /If this may be an emergency, seek immediate help from your local emergency services or a qualified healthcare professional\./i
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Acute chest pain with radiation to arm detected/i)
    ).toBeInTheDocument();

    // Verify 911 or 112 are NOT hardcoded
    const bannerText = screen.getByRole('alert').textContent || '';
    expect(bannerText).not.toContain('911');
    expect(bannerText).not.toContain('112');
  });

  it('renders contraindication banner with "Important safety consideration" and reasons', () => {
    const alerts = [
      {
        contraindication_id: 'RENAL_IMPAIRMENT_NSAID',
        severity: 'HIGH',
        reason:
          'NSAIDs are contraindicated in patients with chronic kidney disease due to the risk of acute renal failure.',
        patient_condition: 'chronic kidney disease',
        conflicting_entity: 'ibuprofen',
      },
    ];

    render(<ContraindicationBanner alerts={alerts} />);

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/Important safety consideration/i)).toBeInTheDocument();
    expect(
      screen.getByText(
        /NSAIDs are contraindicated in patients with chronic kidney disease/i
      )
    ).toBeInTheDocument();
    expect(screen.getByText(/Contraindication Alert/i)).toBeInTheDocument();
  });

  it('transforms assistant message into emergency mode when retrievalStatus is safety_intercepted', () => {
    const message: ChatMessageItem = {
      id: 'asst-emergency-1',
      sender: 'assistant',
      text: 'EMERGENCY: Immediate emergency care is advised for potential cardiac events.',
      retrievalStatus: 'safety_intercepted',
      requestId: 'req_emergency_audit_01',
    };

    render(<ChatMessage message={message} />);

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/EMERGENCY MEDICAL ADVISORY/i)).toBeInTheDocument();
    expect(screen.getByText(/req_emergency_audit_01/i)).toBeInTheDocument();
  });
});
