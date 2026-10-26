import { apiClient } from './client';
import { DocumentRecord, DocumentUploadResponse } from '../types';

export function normalizeDocumentList(data: unknown): DocumentRecord[] {
  if (Array.isArray(data)) {
    return data.filter(
      (item): item is DocumentRecord =>
        Boolean(item && typeof item === 'object' && ('id' in item || 'filename' in item))
    );
  }

  if (data && typeof data === 'object') {
    const candidate =
      (data as Record<string, unknown>).documents ??
      (data as Record<string, unknown>).items ??
      (data as Record<string, unknown>).data ??
      (data as Record<string, unknown>).results;

    if (Array.isArray(candidate)) {
      return candidate.filter(
        (item): item is DocumentRecord =>
          Boolean(item && typeof item === 'object' && ('id' in item || 'filename' in item))
      );
    }
  }

  return [];
}

export const documentApi = {
  uploadDocument: async (
    file: File,
    onUploadProgress?: (progressEvent: any) => void
  ): Promise<DocumentUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<DocumentUploadResponse>(
      '/documents/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress,
      }
    );
    return response.data;
  },

  listDocuments: async (skip: number = 0, limit: number = 100): Promise<DocumentRecord[]> => {
    const response = await apiClient.get<unknown>('/documents', {
      params: { skip, limit },
    });
    return normalizeDocumentList(response.data);
  },

  getDocument: async (documentId: number): Promise<DocumentRecord> => {
    const response = await apiClient.get<DocumentRecord>(`/documents/${documentId}`);
    return response.data;
  },

  deleteDocument: async (documentId: number): Promise<{ success: boolean; message: string }> => {
    const response = await apiClient.delete<{ success: boolean; message: string }>(
      `/documents/${documentId}`
    );
    return response.data;
  },
};
