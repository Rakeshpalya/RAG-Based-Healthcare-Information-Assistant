import React, { useRef, useEffect } from 'react';
import { Activity, Sparkles, BookOpen } from 'lucide-react';
import { ChatMessage as ChatMessageType } from '../types';
import { ChatMessage } from './ChatMessage';

interface ChatAreaProps {
  messages: ChatMessageType[];
  isLoading: boolean;
  onSelectPrompt: (promptText: string) => void;
  onRetryMessage?: () => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  isLoading,
  onSelectPrompt,
  onRetryMessage,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const samplePrompts = [
    {
      title: 'Hypertension Guidelines',
      query: 'What are the recommended lifestyle measures for managing hypertension?',
      icon: BookOpen,
    },
    {
      title: 'Blood Pressure Classification',
      query: 'What criteria define Stage 1 vs Stage 2 hypertension in adults?',
      icon: Sparkles,
    },
    {
      title: 'Pharmacological Evidence',
      query: 'What does the clinical evidence state regarding initial first-line antihypertensive medications?',
      icon: Activity,
    },
  ];

  return (
    <main
      className="flex-1 overflow-y-auto bg-slate-50/50"
      role="log"
      aria-label="Medical consultation conversation log"
      aria-live="polite"
    >
      {messages.length === 0 ? (
        /* Empty State */
        <div className="max-w-2xl mx-auto px-4 py-12 text-center">
          <div className="w-14 h-14 bg-sky-100 text-sky-700 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-xs">
            <Activity className="w-7 h-7" />
          </div>

          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            How can I assist your medical research today?
          </h2>
          <p className="mt-2 text-sm text-slate-500 max-w-md mx-auto leading-relaxed">
            Inquire about clinical guidelines, disease definitions, or evidence-grounded treatment protocols. Responses are strictly grounded in our verified medical vector database.
          </p>

          <div className="mt-8 text-left">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 text-center">
              Suggested Clinical Inquiries
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {samplePrompts.map((item, i) => {
                const IconComponent = item.icon;
                return (
                  <button
                    key={i}
                    onClick={() => onSelectPrompt(item.query)}
                    className="p-3.5 bg-white border border-slate-200 hover:border-sky-300 hover:shadow-sm rounded-xl text-left transition-all group focus:outline-none focus:ring-2 focus:ring-sky-500"
                    aria-label={`Select prompt: ${item.title}`}
                  >
                    <div className="w-7 h-7 rounded-lg bg-slate-50 text-slate-600 group-hover:bg-sky-50 group-hover:text-sky-600 flex items-center justify-center mb-2 transition-colors">
                      <IconComponent className="w-4 h-4" />
                    </div>
                    <div className="font-semibold text-xs text-slate-800 group-hover:text-sky-900">
                      {item.title}
                    </div>
                    <div className="text-[11px] text-slate-500 line-clamp-2 mt-1">
                      {item.query}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        /* Message Stream */
        <div className="divide-y divide-slate-100">
          {messages.map((msg, idx) => (
            <ChatMessage
              key={msg.id || idx}
              message={msg}
              onRetry={msg.isError ? onRetryMessage : undefined}
            />
          ))}

          {/* Loading Indicator Bubble */}
          {isLoading && (
            <div className="py-5 px-4 sm:px-6 bg-slate-50/70 border-t border-slate-100">
              <div className="max-w-3xl mx-auto flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-sky-600 text-white flex items-center justify-center shrink-0 animate-pulse shadow-xs">
                  <Activity className="w-4 h-4" />
                </div>
                <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
                  <span className="w-2 h-2 rounded-full bg-sky-500 animate-ping" />
                  <span>Searching vector database & synthesizing grounded evidence...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} aria-hidden="true" />
        </div>
      )}
    </main>
  );
};
