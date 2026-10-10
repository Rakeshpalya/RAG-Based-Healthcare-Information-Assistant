import { apiClient } from './client';
import { Conversation, ConversationMessage } from '../types';

export const conversationApi = {
  listConversations: async (skip: number = 0, limit: number = 100): Promise<Conversation[]> => {
    const response = await apiClient.get<Conversation[]>('/conversations', {
      params: { skip, limit },
    });
    return response.data;
  },

  createConversation: async (title: string = 'Healthcare Consultation'): Promise<Conversation> => {
    const response = await apiClient.post<Conversation>('/conversations', {
      title,
    });
    return response.data;
  },

  getMessages: async (
    conversationId: number,
    skip: number = 0,
    limit: number = 100
  ): Promise<ConversationMessage[]> => {
    const response = await apiClient.get<ConversationMessage[]>(
      `/conversations/${conversationId}/messages`,
      { params: { skip, limit } }
    );
    return response.data;
  },

  appendMessage: async (
    conversationId: number,
    sender: 'user' | 'assistant' | 'agent',
    text: string,
    citations?: any
  ): Promise<ConversationMessage> => {
    const response = await apiClient.post<ConversationMessage>(
      `/conversations/${conversationId}/messages`,
      {
        sender,
        text,
        citations,
      }
    );
    return response.data;
  },

  updateTitle: async (conversationId: number, title: string): Promise<Conversation> => {
    const response = await apiClient.patch<Conversation>(
      `/conversations/${conversationId}`,
      { title }
    );
    return response.data;
  },

  deleteConversation: async (
    conversationId: number
  ): Promise<{ success: boolean; message: string }> => {
    const response = await apiClient.delete<{ success: boolean; message: string }>(
      `/conversations/${conversationId}`
    );
    return response.data;
  },

  clearAllConversations: async (): Promise<{
    success: boolean;
    deleted_count: number;
    message: string;
  }> => {
    const response = await apiClient.delete<{
      success: boolean;
      deleted_count: number;
      message: string;
    }>('/conversations');
    return response.data;
  },
};
