import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DocumentCard } from '../components/documents/DocumentCard';
import { DocumentUploadModal } from '../components/documents/DocumentUploadModal';
import { DocumentRecord } from '../types';

describe('Document Management UI & Validation', () => {
  const mockDoc: DocumentRecord = {
    id: 42,
    filename: 'guidelines_cardiovascular_prevention.pdf',
    file_size_bytes: 2048576,
    num_pages: 18,
    num_chunks: 35,
    status: 'indexed',
    created_at: new Date().toISOString(),
  };

  it('renders document card with chunk count, size, and status', () => {
    const onDelete = vi.fn();
    render(<DocumentCard doc={mockDoc} onDelete={onDelete} />);

    expect(screen.getAllByText(/guidelines_cardiovascular_prevention/i)[0]).toBeInTheDocument();
    expect(screen.getByText(/35 Chunks indexed/i)).toBeInTheDocument();
    expect(screen.getByText(/2 MB/i)).toBeInTheDocument();
    expect(screen.getByText(/^indexed$/i)).toBeInTheDocument();

    const deleteBtn = screen.getByRole('button', { name: /Delete/i });
    fireEvent.click(deleteBtn);
    expect(onDelete).toHaveBeenCalledWith(42);
  });

  it('rejects non-PDF files in upload modal', async () => {
    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={() => {}}
        onUploadSuccess={() => {}}
      />
    );

    const file = new File(['dummy-content'], 'patient_notes.txt', { type: 'text/plain' });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(
        screen.getByText(/Only PDF documents \(\.pdf\) are supported/i)
      ).toBeInTheDocument();
    });
  });

  it('rejects files exceeding 10 MB in upload modal', async () => {
    render(
      <DocumentUploadModal
        isOpen={true}
        onClose={() => {}}
        onUploadSuccess={() => {}}
      />
    );

    // 11 MB dummy file
    const largeBlob = new Uint8Array(11 * 1024 * 1024);
    const file = new File([largeBlob], 'oversized_scan.pdf', { type: 'application/pdf' });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(
        screen.getByText(/exceeds maximum allowed limit of 10 MB/i)
      ).toBeInTheDocument();
    });
  });
});
