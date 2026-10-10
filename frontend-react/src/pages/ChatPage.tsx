import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ragApi } from '../api/ragApi';
import { conversationApi } from '../api/conversationApi';
import { Conversation, RAGQueryResponse } from '../types';
import { ChatArea } from '../components/chat/ChatArea';
import { ChatInput } from '../components/chat/ChatInput';
import { ChatMessageItem } from '../components/chat/ChatMessage';
import { useHealth } from '../context/HealthContext';
import {
  MessageSquare,
  Plus,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Clock,
} from 'lucide-react';

export const ChatPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const { status: healthStatus } = useHealth();

  const conversationIdParam = searchParams.get('id');

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(
    conversationIdParam ? parseInt(conversationIdParam, 10) : null
  );

  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingToken, setStreamingToken] = useState('');
  const [streamingStatus, setStreamingStatus] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Load conversation list
  const loadConversations = useCallback(async () => {
    try {
      const data = await conversationApi.listConversations(0, 50);
      setConversations(data);
    } catch {
      // offline or unauthenticated fallback
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // Load messages when active conversation changes
  useEffect(() => {
    const loadConversationMessages = async (id: number) => {
      try {
        const msgs = await conversationApi.getMessages(id, 0, 100);
        const mapped: ChatMessageItem[] = msgs.map((m) => ({
          id: `msg-${m.id}`,
          sender: m.sender === 'user' ? 'user' : 'assistant',
          text: m.text,
          sources: m.citations?.sources || [],
          retrievalStatus: m.citations?.retrieval_status,
          requestId: m.citations?.request_id,
          timestamp: m.created_at,
          dialogueContext: m.citations?.dialogue_context,
        }));
        setMessages(mapped);
      } catch {
        // failed to fetch messages
      }
    };

    if (activeConversationId) {
      loadConversationMessages(activeConversationId);
    } else {
      setMessages([]);
    }
  }, [activeConversationId]);

  // Switch or create conversation
  const handleSelectConversation = (id: number) => {
    setActiveConversationId(id);
    setSearchParams({ id: id.toString() });
  };

  const handleNewConversation = async () => {
    setActiveConversationId(null);
    setMessages([]);
    setErrorMessage(null);
    setSearchParams({});
  };

  const handleDeleteConversation = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await conversationApi.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversationId === id) {
        handleNewConversation();
      }
    } catch (err: any) {
      alert(err.message || 'Failed to delete conversation.');
    }
  };

  // Stop current active streaming response
  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setIsLoading(false);
  };

  // Send message and execute RAG query
  const handleSendMessage = async (text: string) => {
    setErrorMessage(null);

    // 1. Append user message to UI immediately
    const userMsgId = `usr-${Date.now()}`;
    const userMsg: ChatMessageItem = {
      id: userMsgId,
      sender: 'user',
      text,
      timestamp: new Date().toISOString(),
    };

    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);

    // 2. Prepare conversation history for Phase 6.9 longitudinal context
    // Extracts recent turns (user & assistant) to maintain dialogue state & contraindications
    const historyPayload = nextMessages.slice(-6).map((m) => ({
      role: m.sender === 'user' ? ('user' as const) : ('assistant' as const),
      content: m.text,
    }));

    setIsLoading(true);
    setIsStreaming(true);
    setStreamingToken('');
    setStreamingStatus('Initializing clinical intelligence pipeline...');

    // 3. Check or create backend conversation record for persistence
    let currentConvId = activeConversationId;
    if (!currentConvId) {
      try {
        const titleSnippet = text.length > 45 ? `${text.slice(0, 42)}...` : text;
        const newConv = await conversationApi.createConversation(titleSnippet);
        currentConvId = newConv.id;
        setActiveConversationId(newConv.id);
        setSearchParams({ id: newConv.id.toString() });
        setConversations((prev) => [newConv, ...prev]);
      } catch {
        // Fall back to ephemeral conversation
      }
    }

    // Persist user turn if conversation exists
    if (currentConvId) {
      conversationApi.appendMessage(currentConvId, 'user', text).catch(() => {});
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    let accumulatedAnswer = '';
    let finalSources: any[] = [];
    let finalStatus = 'success';
    let finalReqId = '';
    let finalDialogueContext: any = undefined;

    try {
      // 4. Attempt streaming via POST /rag/stream
      await ragApi.streamRAG(
        {
          question: text,
          top_k: 5,
          similarity_threshold: 0.25,
          conversation_history: historyPayload,
        },
        {
          onStart: () => {
            setStreamingStatus('Retrieving evidence from FAISS index...');
          },
          onStatus: (data) => {
            if (data.message) {
              setStreamingStatus(data.message);
            }
          },
          onToken: (token) => {
            accumulatedAnswer += token;
            setStreamingToken(accumulatedAnswer);
          },
          onComplete: (resp: RAGQueryResponse) => {
            accumulatedAnswer = resp.answer || accumulatedAnswer;
            finalSources = resp.sources || [];
            finalStatus = resp.retrieval_status || 'success';
            finalReqId = resp.request_id || '';
            finalDialogueContext = resp.dialogue_context;
          },
          onError: async (err) => {
            // Safe fallback to synchronous /rag/query if SSE streaming drops
            console.warn('Streaming encountered an issue, falling back to /rag/query:', err.message);
            try {
              const syncResp = await ragApi.queryRAG({
                question: text,
                top_k: 5,
                similarity_threshold: 0.25,
                conversation_history: historyPayload,
              });
              accumulatedAnswer = syncResp.answer;
              finalSources = syncResp.sources || [];
              finalStatus = syncResp.retrieval_status || 'success';
              finalReqId = syncResp.request_id || '';
              finalDialogueContext = syncResp.dialogue_context;
            } catch (fallbackErr: any) {
              throw fallbackErr;
            }
          },
        },
        abortController.signal
      );

      // 5. Build final assistant message
      const assistantMsg: ChatMessageItem = {
        id: `asst-${Date.now()}`,
        sender: 'assistant',
        text: accumulatedAnswer || 'No grounded answer was returned.',
        sources: finalSources,
        retrievalStatus: finalStatus,
        requestId: finalReqId,
        timestamp: new Date().toISOString(),
        dialogueContext: finalDialogueContext,
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Persist assistant message in backend database
      if (currentConvId) {
        conversationApi
          .appendMessage(currentConvId, 'assistant', assistantMsg.text, {
            sources: finalSources,
            retrieval_status: finalStatus,
            request_id: finalReqId,
            dialogue_context: finalDialogueContext,
          })
          .catch(() => {});
      }
    } catch (err: any) {
      if (abortController.signal.aborted) {
        // Stream aborted by user
        if (accumulatedAnswer) {
          const partialMsg: ChatMessageItem = {
            id: `asst-aborted-${Date.now()}`,
            sender: 'assistant',
            text: accumulatedAnswer + '\n\n*(Generation stopped by user)*',
            sources: finalSources,
            timestamp: new Date().toISOString(),
          };
          setMessages((prev) => [...prev, partialMsg]);
        }
      } else {
        setErrorMessage(
          err.message || 'Failed to complete clinical inquiry. Please verify service connectivity.'
        );
      }
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      setStreamingToken('');
      setStreamingStatus('');
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="flex h-[calc(100vh-4rem-2.25rem)] overflow-hidden bg-slate-50">
      {/* Collapsible Left Sidebar for Conversations */}
      <div
        className={`${
          isSidebarOpen ? 'w-72 sm:w-80' : 'w-0'
        } shrink-0 bg-white border-r border-slate-200 transition-all duration-200 overflow-hidden flex flex-col`}
      >
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <button
            onClick={handleNewConversation}
            className="flex-1 mr-2 inline-flex items-center justify-center gap-2 px-3.5 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded-xl text-xs font-semibold shadow-xs transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>New Consultation</span>
          </button>
          <button
            onClick={() => setIsSidebarOpen(false)}
            className="p-2 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
            title="Collapse sidebar"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="px-3 py-1.5 text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5" />
            <span>Past Dialogue Sessions</span>
          </div>

          {conversations.length === 0 ? (
            <p className="px-3 py-4 text-xs text-slate-400 text-center">No past chats recorded</p>
          ) : (
            conversations.map((c) => (
              <div
                key={c.id}
                onClick={() => handleSelectConversation(c.id)}
                className={`w-full group flex items-center justify-between p-2.5 rounded-xl text-left text-xs cursor-pointer transition-colors ${
                  activeConversationId === c.id
                    ? 'bg-sky-50 text-sky-800 font-semibold'
                    : 'text-slate-700 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2 min-w-0 pr-1">
                  <MessageSquare className="w-3.5 h-3.5 shrink-0 text-slate-400 group-hover:text-sky-600" />
                  <span className="truncate" title={c.title}>
                    {c.title}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={(e) => handleDeleteConversation(c.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-rose-600 rounded transition-opacity"
                  title="Delete chat"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Panel */}
      <div className="flex-1 flex flex-col min-w-0 relative bg-slate-50">
        {/* Top Floating Toggle if Sidebar is Closed */}
        {!isSidebarOpen && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="absolute top-3 left-3 z-10 p-2 bg-white border border-slate-200 rounded-xl text-slate-600 hover:text-slate-900 shadow-xs transition-colors"
            title="Open conversations sidebar"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        )}

        {/* Scrollable Message Thread */}
        <ChatArea
          messages={messages}
          isLoading={isLoading}
          streamingToken={streamingToken}
          streamingStatus={streamingStatus}
          onSelectPrompt={(p) => handleSendMessage(p)}
          onRetryLast={() => {
            if (messages.length > 0) {
              const lastUser = [...messages].reverse().find((m) => m.sender === 'user');
              if (lastUser) handleSendMessage(lastUser.text);
            }
          }}
          errorMessage={errorMessage}
        />

        {/* Bottom Chat Input Form */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          isStreaming={isStreaming}
          onStopStreaming={handleStopStreaming}
          disabled={healthStatus === 'OFFLINE'}
        />
      </div>
    </div>
  );
};
