import React, { useEffect, useRef } from 'react';
import { ChatMessage, ChatMessageItem } from './ChatMessage';
import { Sparkles, AlertCircle, RefreshCw } from 'lucide-react';

interface ChatAreaProps {
  messages: ChatMessageItem[];
  isLoading: boolean;
  streamingToken: string;
  streamingStatus?: string;
  onSelectPrompt: (prompt: string) => void;
  onRetryLast?: () => void;
  errorMessage?: string | null;
}

const STARTER_PROMPTS = [
  'What are the first-line pharmacotherapies and diagnostic criteria for Type 2 Diabetes?',
  'Explain clinical guidelines for hypertension staging and blood pressure goals.',
  'What contraindications and risks exist between chronic kidney disease and NSAIDs?',
  'What are the recognized warning signs of acute coronary syndrome versus stable angina?',
];

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  isLoading,
  streamingToken,
  streamingStatus,
  onSelectPrompt,
  onRetryLast,
  errorMessage,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingToken, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 space-y-4">
      {/* Empty State with Clinical Starter Prompts */}
      {messages.length === 0 && (
        <div className="max-w-2xl mx-auto py-8 sm:py-12 text-center space-y-6">
          <div className="w-16 h-16 rounded-2xl bg-sky-100 text-sky-700 mx-auto flex items-center justify-center text-3xl shadow-xs">
            🩺
          </div>

          <div className="space-y-2">
            <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              Clinical Intelligence & Consultation Assistant
            </h2>
            <p className="text-sm text-slate-600 leading-relaxed max-w-lg mx-auto">
              Ask questions grounded in clinical guidelines, peer-reviewed documents, and multi-turn longitudinal patient disclosures.
            </p>
          </div>

          <div className="space-y-2 text-left pt-2">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-sky-600" />
              Suggested Clinical Prompts
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {STARTER_PROMPTS.map((prompt, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => onSelectPrompt(prompt)}
                  className="p-3.5 rounded-xl border border-slate-200 bg-white hover:border-sky-300 hover:bg-sky-50/50 text-left text-xs sm:text-sm font-medium text-slate-700 transition-all shadow-2xs hover:shadow-xs group"
                >
                  <span className="group-hover:text-sky-800">{prompt}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Render Message Thread */}
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} />
      ))}

      {/* Active Streaming Token Bubble */}
      {isLoading && (
        <div className="flex gap-3 sm:gap-4 my-4 flex-row items-start">
          <div className="w-8 h-8 rounded-full bg-slate-800 text-white flex items-center justify-center shrink-0 text-xs font-semibold shadow-xs animate-pulse">
            🩺
          </div>
          <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-xs p-4 sm:p-5 shadow-xs max-w-[88%] sm:max-w-[80%] space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium text-sky-600">
              <span className="w-2 h-2 rounded-full bg-sky-600 animate-ping" />
              <span>{streamingStatus || 'Synthesizing verified medical response...'}</span>
            </div>
            {streamingToken ? (
              <div className="text-sm text-slate-800 leading-relaxed whitespace-pre-wrap font-sans">
                {streamingToken}
              </div>
            ) : (
              <div className="space-y-2 animate-pulse pt-1">
                <div className="h-3 bg-slate-200 rounded w-3/4" />
                <div className="h-3 bg-slate-200 rounded w-1/2" />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error Notice & Retry Option */}
      {errorMessage && (
        <div className="max-w-xl mx-auto my-4 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs sm:text-sm flex items-start justify-between gap-3 shadow-xs">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-rose-900">Communication Notice</p>
              <p className="text-rose-700 mt-0.5 leading-relaxed">{errorMessage}</p>
            </div>
          </div>
          {onRetryLast && (
            <button
              onClick={onRetryLast}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-rose-300 text-xs font-medium text-rose-800 hover:bg-rose-100 transition-colors shrink-0"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry
            </button>
          )}
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
};
