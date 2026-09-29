import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { ChatInput } from './components/ChatInput';
import { MedicalDisclaimer } from './components/MedicalDisclaimer';
import { ChatSession, ChatMessage, ReadinessResponse } from './types';
import { queryRAG, checkReadiness } from './api/ragApi';
import {
  loadChatSessions,
  saveChatSessions,
  getActiveSessionId,
  setActiveSessionId,
  createNewSession,
} from './utils/storage';

export const App: React.FC = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionIdState] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [isLoadingReadiness, setIsLoadingReadiness] = useState(true);

  // Initialize sessions from local storage
  useEffect(() => {
    const loaded = loadChatSessions();
    if (loaded.length > 0) {
      setSessions(loaded);
      const savedActiveId = getActiveSessionId();
      if (savedActiveId && loaded.some((s) => s.id === savedActiveId)) {
        setActiveSessionIdState(savedActiveId);
      } else {
        setActiveSessionIdState(loaded[0].id);
        setActiveSessionId(loaded[0].id);
      }
    } else {
      const initialSession = createNewSession();
      setSessions([initialSession]);
      setActiveSessionIdState(initialSession.id);
      saveChatSessions([initialSession]);
      setActiveSessionId(initialSession.id);
    }
  }, []);

  // Poll backend health readiness on mount
  const fetchHealthReadiness = useCallback(async () => {
    try {
      setIsLoadingReadiness(true);
      const res = await checkReadiness();
      setReadiness(res);
    } catch (e) {
      console.warn('Backend readiness check failed:', e);
      setReadiness({
        status: 'not_ready',
        service: 'AI-Healthcare-Agent',
        environment: 'unknown',
        checks: {},
      });
    } finally {
      setIsLoadingReadiness(false);
    }
  }, []);

  useEffect(() => {
    fetchHealthReadiness();
  }, [fetchHealthReadiness]);

  // Current Active Session
  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];
  const messages = activeSession ? activeSession.messages : [];

  const handleSelectSession = (id: string) => {
    setActiveSessionIdState(id);
    setActiveSessionId(id);
    setIsSidebarOpen(false);
  };

  const handleNewChat = () => {
    const newSession = createNewSession();
    const updated = [newSession, ...sessions];
    setSessions(updated);
    setActiveSessionIdState(newSession.id);
    setActiveSessionId(newSession.id);
    saveChatSessions(updated);
    setIsSidebarOpen(false);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const filtered = sessions.filter((s) => s.id !== id);
    if (filtered.length === 0) {
      const fallback = createNewSession();
      setSessions([fallback]);
      setActiveSessionIdState(fallback.id);
      setActiveSessionId(fallback.id);
      saveChatSessions([fallback]);
    } else {
      setSessions(filtered);
      saveChatSessions(filtered);
      if (activeSessionId === id) {
        setActiveSessionIdState(filtered[0].id);
        setActiveSessionId(filtered[0].id);
      }
    }
  };

  const handleSendMessage = async (question: string) => {
    if (!activeSession || isLoading) return;

    const userMessage: ChatMessage = {
      id: `msg_user_${Date.now()}`,
      role: 'user',
      content: question,
      timestamp: new Date().toISOString(),
    };

    // Update active session with user message
    const updatedMessagesWithUser = [...messages, userMessage];
    const updatedTitle =
      activeSession.messages.length === 0
        ? question.slice(0, 32) + (question.length > 32 ? '...' : '')
        : activeSession.title;

    const sessionWithUser: ChatSession = {
      ...activeSession,
      title: updatedTitle,
      updatedAt: new Date().toISOString(),
      messages: updatedMessagesWithUser,
    };

    const sessionsAfterUser = sessions.map((s) =>
      s.id === activeSession.id ? sessionWithUser : s
    );
    setSessions(sessionsAfterUser);
    saveChatSessions(sessionsAfterUser);
    setIsLoading(true);

    try {
      // Build lightweight conversation history for follow-up resolution
      const historyPayload = updatedMessagesWithUser
        .slice(-4)
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await queryRAG({
        question,
        conversation_history: historyPayload,
      });

      const assistantMessage: ChatMessage = {
        id: `msg_asst_${Date.now()}`,
        role: 'assistant',
        content: res.answer,
        timestamp: new Date().toISOString(),
        sources: res.sources,
        retrieval_status: res.retrieval_status,
        safety_assessment: res.timings?.safety_assessment,
        request_id: res.request_id,
      };

      const finalSession: ChatSession = {
        ...sessionWithUser,
        updatedAt: new Date().toISOString(),
        messages: [...updatedMessagesWithUser, assistantMessage],
      };

      const finalSessions = sessions.map((s) =>
        s.id === activeSession.id ? finalSession : s
      );
      setSessions(finalSessions);
      saveChatSessions(finalSessions);
    } catch (err: any) {
      const errorMessage: ChatMessage = {
        id: `msg_err_${Date.now()}`,
        role: 'assistant',
        content: err.message || 'An error occurred while connecting to the assistant. Please retry.',
        timestamp: new Date().toISOString(),
        isError: true,
      };

      const errorSession: ChatSession = {
        ...sessionWithUser,
        messages: [...updatedMessagesWithUser, errorMessage],
      };

      const errSessions = sessions.map((s) =>
        s.id === activeSession.id ? errorSession : s
      );
      setSessions(errSessions);
      saveChatSessions(errSessions);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetryLastMessage = () => {
    if (!messages || messages.length === 0) return;
    const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user');
    if (lastUserMsg) {
      handleSendMessage(lastUserMsg.content);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 overflow-hidden font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        isOpen={isSidebarOpen}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onCloseMobile={() => setIsSidebarOpen(false)}
      />

      {/* Main Workspace Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full">
        <Header
          readiness={readiness}
          isLoadingReadiness={isLoadingReadiness}
          onNewChat={handleNewChat}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        />

        <ChatArea
          messages={messages}
          isLoading={isLoading}
          onSelectPrompt={(text) => handleSendMessage(text)}
          onRetryMessage={handleRetryLastMessage}
        />

        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          disabled={readiness?.status === 'not_ready'}
        />

        <MedicalDisclaimer />
      </div>
    </div>
  );
};

export default App;
