import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CitationCard } from '../components/chat/CitationCard';
import { SourceCitation } from '../types';

describe('Grounded Citation Cards UX', () => {
  const sampleSource: SourceCitation = {
    chunk_id: 'DIABETES_GUIDELINE_CHUNK_12',
    document_name: 'clinical_guidelines_diabetes_2026.pdf',
    page_number: 4,
    similarity_score: 0.887,
    text: 'Metformin remains the preferred initial pharmacologic agent for the treatment of type 2 diabetes.',
  };

  it('renders citation card header with document name, page, and relevance score', () => {
    render(<CitationCard source={sampleSource} index={0} />);

    expect(screen.getByText('Source 1')).toBeInTheDocument();
    expect(screen.getByText(/clinical_guidelines_diabetes_2026/i)).toBeInTheDocument();
    expect(screen.getByText('p. 4')).toBeInTheDocument();
    expect(screen.getByText('89% relevance')).toBeInTheDocument();
  });

  it('toggles verified text excerpt on click and keyboard interaction', () => {
    render(<CitationCard source={sampleSource} index={0} />);

    const button = screen.getByRole('button');

    // Initially collapsed
    expect(
      screen.queryByText(/Metformin remains the preferred initial pharmacologic agent/i)
    ).not.toBeInTheDocument();

    // Click to expand
    fireEvent.click(button);
    expect(
      screen.getByText(/Metformin remains the preferred initial pharmacologic agent/i)
    ).toBeInTheDocument();
    expect(screen.getByText('DIABETES_GUIDELINE_CHUNK_12')).toBeInTheDocument();

    // Press Enter to collapse
    fireEvent.keyDown(button, { key: 'Enter', code: 'Enter' });
    expect(
      screen.queryByText(/Metformin remains the preferred initial pharmacologic agent/i)
    ).not.toBeInTheDocument();
  });
});
