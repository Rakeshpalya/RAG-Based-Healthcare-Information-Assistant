import { ChatSession, ChatMessage } from '../types';

const STORAGE_KEY = 'ai_healthcare_chat_sessions_v1';
const ACTIVE_SESSION_KEY = 'ai_healthcare_active_session_id_v1';

export function loadChatSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch (e) {
    console.error('Failed to load chat sessions from localStorage:', e);
    return [];
  }
}

export function saveChatSessions(sessions: ChatSession[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch (e) {
    console.error('Failed to save chat sessions to localStorage:', e);
  }
}

export function getActiveSessionId(): string | null {
  try {
    return localStorage.getItem(ACTIVE_SESSION_KEY);
  } catch {
    return null;
  }
}

export function setActiveSessionId(id: string): void {
  try {
    localStorage.setItem(ACTIVE_SESSION_KEY, id);
  } catch (e) {
    console.error('Failed to save active session ID:', e);
  }
}

export function createNewSession(initialMessage?: ChatMessage): ChatSession {
  const now = new Date().toISOString();
  const id = `session_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  const title = initialMessage 
    ? (initialMessage.content.slice(0, 30) + (initialMessage.content.length > 30 ? '...' : ''))
    : 'New Conversation';

  return {
    id,
    title,
    createdAt: now,
    updatedAt: now,
    messages: initialMessage ? [initialMessage] : [],
  };
}
