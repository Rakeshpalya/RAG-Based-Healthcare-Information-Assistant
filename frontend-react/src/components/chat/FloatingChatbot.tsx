import React, { useState, useRef, useEffect } from 'react';
import { ragApi } from '../../api/ragApi';
import { normalizeApiError } from '../../api/client';
import { HealthLogo } from '../common/HealthLogo';
import { CitationCard } from './CitationCard';
import { EmergencyBanner } from './EmergencyBanner';
import { ContraindicationBanner } from './ContraindicationBanner';
import { SourceCitation } from '../../types';
import { Send, X, Stethoscope, Loader2 } from 'lucide-react';

interface FloatingMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  sources?: SourceCitation[];
  isEmergency?: boolean;
  emergencyMessage?: string;
  contraindications?: any[];
  timestamp: string;
}

const INITIAL_GREETING: FloatingMessage = {
  id: 'msg-welcome',
  sender: 'assistant',
  text: `Hello! I am the HealthAI Medical Assistant. Ask me general medical questions such as:\n• What is hypertension?\n• What are the symptoms of diabetes?\n• What is asthma?`,
  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
};

const SUGGESTED_QUESTIONS = [
  'What is hypertension?',
  'What are the symptoms of diabetes?',
  'What is asthma?',
  'What causes high blood pressure?',
];

export const FloatingChatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<FloatingMessage[]>([INITIAL_GREETING]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen, isLoading]);

  const handleSendMessage = async (queryText?: string) => {
    const textToSend = (queryText || inputText).trim();
    if (!textToSend || isLoading) return;

    setInputText('');

    const userMsg: FloatingMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await ragApi.queryRAG({
        question: textToSend,
        top_k: 5,
        conversation_history: messages
          .filter((m) => m.id !== 'msg-welcome')
          .map((m) => ({
            role: m.sender,
            content: m.text,
          })),
      });

      const isEmergency =
        response.retrieval_status === 'safety_intercepted' ||
        Boolean(response.timings?.safety_assessment?.requires_escalation);

      const assistantMsg: FloatingMessage = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: response.answer || 'No specific medical information found for this inquiry.',
        sources: response.sources || [],
        isEmergency,
        emergencyMessage:
          response.retrieval_status === 'safety_intercepted'
            ? 'Emergency symptoms or acute medical danger detected. Please seek emergency medical care immediately.'
            : undefined,
        contraindications: response.dialogue_context?.contraindications || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: unknown) {
      const apiErr = normalizeApiError(err);
      let userFriendlyText = 'Sorry, I encountered an issue processing your inquiry. Please try again.';

      if (apiErr.status === 401) {
        userFriendlyText = 'Your session has expired. Please sign in to continue.';
      } else if (apiErr.status === 429) {
        userFriendlyText = 'The assistant is currently handling high request volume. Please wait a moment and try again.';
      } else if (apiErr.status === 503) {
        userFriendlyText = apiErr.message || 'The AI service is temporarily unavailable. Please try again in a few moments.';
      } else if (apiErr.status === 0 || apiErr.message?.toLowerCase().includes('network error') || apiErr.message?.toLowerCase().includes('failed to fetch')) {
        userFriendlyText = 'Unable to connect to the HealthAI backend service. Please verify the service is running at http://127.0.0.1:8000 and try again.';
      } else if (apiErr.message) {
        userFriendlyText = apiErr.message;
      }

      const errorMsg: FloatingMessage = {
        id: `assistant-err-${Date.now()}`,
        sender: 'assistant',
        text: userFriendlyText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage();
  };

  return (
    <>
      {/* =========================================================================
          CLOSED STATE: Floating Bottom-Right Launcher Button
          ========================================================================= */}
      {!isOpen && (
        <button
          type="button"
          onClick={() => setIsOpen(true)}
          data-testid="floating-chatbot-launcher"
          aria-label="Open HealthAI Medical Assistant for General Medical Q&A"
          title="Open HealthAI Medical Assistant for General Medical Q&A"
          className="fixed bottom-6 right-6 z-40 group inline-flex items-center gap-2.5 px-4 py-3 rounded-full bg-sky-600 text-white font-semibold text-sm shadow-xl hover:bg-sky-700 hover:shadow-2xl hover:scale-105 transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-sky-500/30"
        >
          <div className="w-7 h-7 rounded-full bg-white flex items-center justify-center text-sky-600 shadow-2xs">
            <Stethoscope className="w-4 h-4 text-sky-600" strokeWidth={2.4} aria-hidden="true" />
          </div>
          <span className="font-semibold tracking-tight">HealthAI</span>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-200 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
          </span>
        </button>
      )}

      {/* =========================================================================
          OPEN STATE: Compact Floating Chat Window (Bottom-Right)
          ========================================================================= */}
      {isOpen && (
        <div
          data-testid="floating-chatbot-popup"
          className="fixed bottom-6 right-6 z-40 w-96 max-w-[calc(100vw-2rem)] h-[540px] max-h-[calc(100vh-6rem)] bg-white rounded-2xl shadow-2xl border border-slate-200 flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-200"
        >
          {/* 1. Header Bar */}
          <div className="bg-white border-b border-slate-200 px-4 py-3 flex items-center justify-between shadow-2xs shrink-0">
            <div className="flex items-center gap-2.5 min-w-0">
              <HealthLogo size="sm" />
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <h3 className="text-xs font-bold text-slate-900 truncate">
                    HealthAI Medical Assistant
                  </h3>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
                </div>
                <p className="text-[11px] text-slate-500 font-medium">General Medical Q&A</p>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setIsOpen(false)}
              data-testid="floating-chatbot-close-btn"
              aria-label="Close chatbot"
              title="Close chatbot"
              className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* 2. Scrollable Messages History Area */}
          <div
            className="flex-1 overflow-y-auto p-3.5 space-y-3.5 bg-slate-50/50"
            data-testid="floating-chatbot-messages"
          >
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex flex-col ${
                  msg.sender === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div
                  className={`max-w-[88%] rounded-2xl p-3 text-xs leading-relaxed shadow-2xs ${
                    msg.sender === 'user'
                      ? 'bg-sky-600 text-white rounded-br-xs'
                      : 'bg-white text-slate-800 border border-slate-200/80 rounded-bl-xs'
                  }`}
                >
                  {msg.sender === 'assistant' && (
                    <div className="flex items-center justify-between text-[10px] font-semibold text-sky-700 pb-1 mb-1 border-b border-slate-100">
                      <span>🩺 HealthAI</span>
                      <span className="text-slate-400 font-normal">{msg.timestamp}</span>
                    </div>
                  )}

                  {/* Message Body */}
                  <div className="whitespace-pre-line">{msg.text}</div>

                  {/* Safety & Emergency Alerts */}
                  {msg.isEmergency && (
                    <div className="mt-2.5">
                      <EmergencyBanner message={msg.emergencyMessage} />
                    </div>
                  )}

                  {/* Contraindication Warnings */}
                  {msg.contraindications && msg.contraindications.length > 0 && (
                    <div className="mt-2.5">
                      <ContraindicationBanner alerts={msg.contraindications} />
                    </div>
                  )}

                  {/* Verified Citations */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2.5 pt-2 border-t border-slate-100 space-y-1.5">
                      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">
                        Verified Sources ({msg.sources.length})
                      </span>
                      {msg.sources.map((src, idx) => (
                        <CitationCard key={idx} source={src} index={idx} />
                      ))}
                    </div>
                  )}
                </div>

                {msg.sender === 'user' && (
                  <span className="text-[10px] text-slate-400 mt-0.5 px-1">
                    {msg.timestamp}
                  </span>
                )}
              </div>
            ))}

            {/* Typing Indicator */}
            {isLoading && (
              <div className="flex items-center gap-2 text-xs text-slate-500 p-2.5 bg-white border border-slate-200/80 rounded-2xl w-fit shadow-2xs">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-sky-600" />
                <span>HealthAI is thinking...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* 3. Suggested Question Chips (Displayed when conversation has <= 2 messages) */}
          {messages.length <= 2 && (
            <div className="px-3 py-2 bg-white border-t border-slate-100 shrink-0">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                💡 Suggested Questions
              </span>
              <div className="grid grid-cols-2 gap-1.5">
                {SUGGESTED_QUESTIONS.map((question, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSendMessage(question)}
                    disabled={isLoading}
                    className="text-left text-[11px] font-medium text-slate-700 bg-slate-50 hover:bg-sky-50 hover:text-sky-700 border border-slate-200 rounded-lg px-2 py-1.5 transition-colors truncate disabled:opacity-50"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 4. Input Row */}
          <form
            onSubmit={handleFormSubmit}
            className="p-3 bg-white border-t border-slate-200 flex items-center gap-2 shrink-0"
          >
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Ask about your health..."
              disabled={isLoading}
              className="flex-1 text-xs px-3 py-2 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 disabled:bg-slate-50 text-slate-900 placeholder:text-slate-400"
            />
            <button
              type="submit"
              disabled={isLoading || !inputText.trim()}
              data-testid="floating-chatbot-send-btn"
              aria-label="Send medical question"
              className="p-2 rounded-xl bg-sky-600 text-white hover:bg-sky-700 disabled:bg-slate-200 disabled:text-slate-400 transition-colors shrink-0 shadow-2xs"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </form>

          {/* 5. Safety Disclaimer Footer */}
          <div className="px-3 py-1.5 bg-slate-50 border-t border-slate-100 text-[10px] text-slate-400 text-center shrink-0">
            General educational information only. Does not diagnose or prescribe.
          </div>
        </div>
      )}
    </>
  );
};
