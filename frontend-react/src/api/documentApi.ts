import { apiClient } from './client';
import { DocumentRecord, DocumentUploadResponse } from '../types';

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
    const response = await apiClient.get<DocumentRecord[]>('/documents', {
      params: { skip, limit },
    });
    return response.data;
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
