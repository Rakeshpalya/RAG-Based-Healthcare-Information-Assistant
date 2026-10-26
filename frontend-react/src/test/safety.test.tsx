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

  it('Priority 1: does not render asthma contraindication alert on unrelated queries when alerts are empty', () => {
    const message: ChatMessageItem = {
      id: 'asst-diabetes-1',
      sender: 'assistant',
      text: 'Type 2 Diabetes risk factors include obesity, physical inactivity, and family history [Source 1].',
      retrievalStatus: 'success',
      dialogueContext: {
        is_follow_up: false,
        turn_count: 1,
        contraindications: [],
      },
    };

    render(<ChatMessage message={message} />);

    expect(screen.queryByText(/Asthma Contraindication/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Important safety consideration/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Type 2 Diabetes risk factors include/i)).toBeInTheDocument();
  });

  it('Priority 2: renders emergency toxicology advisory for chemical ingestion and forbids vomiting', () => {
    const toxicMsg: ChatMessageItem = {
      id: 'asst-toxic-1',
      sender: 'assistant',
      text: 'URGENT TOXICOLOGY ADVISORY: If you suspect poisoning or accidental ingestion of a hazardous substance, contact Poison Control or local emergency services immediately. Do not induce vomiting.',
      retrievalStatus: 'safety_intercepted',
      requestId: 'req_toxic_001',
    };

    render(<ChatMessage message={toxicMsg} />);

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/EMERGENCY MEDICAL ADVISORY/i)).toBeInTheDocument();
    expect(screen.getByText(/Do not induce vomiting/i)).toBeInTheDocument();
  });

  it('Priority 4: renders properly formatted text without merged words', () => {
    const formattedMsg: ChatMessageItem = {
      id: 'asst-format-1',
      sender: 'assistant',
      text: 'Excess body weight is a primary risk factor [Source 1]. Regarding complications, retinopathy is common.',
      retrievalStatus: 'success',
    };

    render(<ChatMessage message={formattedMsg} />);

    expect(screen.getByText(/Excess body weight is a primary risk factor/i)).toBeInTheDocument();
    expect(screen.queryByText(/Excessbody/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/\[Source 1\]\.Regarding/i)).not.toBeInTheDocument();
  });

  it('Priority 5: renders clarifying question response cleanly when references cannot be resolved', () => {
    const clarifyMsg: ChatMessageItem = {
      id: 'asst-clarify-1',
      sender: 'assistant',
      text: 'Could you please specify which recommendation or clinical finding you would like me to explain in simpler language?',
      retrievalStatus: 'clarification_required',
    };

    render(<ChatMessage message={clarifyMsg} />);

    expect(screen.getByText(/Could you please specify which recommendation/i)).toBeInTheDocument();
  });
});
