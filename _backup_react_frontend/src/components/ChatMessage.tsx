import React from 'react';
import { User, Activity, AlertCircle } from 'lucide-react';
import { ChatMessage as ChatMessageType } from '../types';
import { CitationCard } from './CitationCard';
import { EmergencyBanner } from './EmergencyBanner';
import { RequestIdBadge } from './RequestIdBadge';
import { formatTimestamp } from '../utils/formatters';

interface ChatMessageProps {
  message: ChatMessageType;
  onRetry?: () => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onRetry }) => {
  const isUser = message.role === 'user';
  const isEmergency = message.retrieval_status === 'safety_intercepted';
  const hasSources = Boolean(message.sources && message.sources.length > 0);

  return (
    <article
      className={`py-4 px-4 sm:px-6 transition-colors ${
        isUser ? 'bg-white' : 'bg-slate-50/70 border-y border-slate-100'
      }`}
      aria-label={`${isUser ? 'User question' : 'Assistant response'}`}
    >
      <div className="max-w-3xl mx-auto flex items-start gap-3.5">
        {/* Avatar */}
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5 shadow-xs ${
            isUser
              ? 'bg-slate-200 text-slate-700'
              : isEmergency
              ? 'bg-rose-600 text-white'
              : 'bg-sky-600 text-white'
          }`}
          aria-hidden="true"
        >
          {isUser ? (
            <User className="w-4 h-4" />
          ) : (
            <Activity className="w-4 h-4" />
          )}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0 space-y-2">
          {/* Header Row */}
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-slate-800">
              {isUser ? 'You' : 'Clinical Knowledge Assistant'}
            </span>
            <div className="flex items-center gap-2">
              <RequestIdBadge requestId={message.request_id} />
              <time
                dateTime={message.timestamp}
                className="text-[11px] text-slate-400"
              >
                {formatTimestamp(message.timestamp)}
              </time>
            </div>
          </div>

          {/* Emergency Advisory Banner */}
          {isEmergency && (
            <EmergencyBanner message={message.content} />
          )}

          {/* Normal Body / Error Body */}
          {message.isError ? (
            <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-medium">{message.content}</p>
                {onRetry && (
                  <button
                    onClick={onRetry}
                    className="mt-2 text-xs font-semibold text-rose-700 hover:text-rose-900 underline focus:outline-none"
                  >
                    Retry this query
                  </button>
                )}
              </div>
            </div>
          ) : (
            !isEmergency && (
              <div className="text-sm text-slate-800 leading-relaxed font-sans whitespace-pre-wrap break-words">
                {message.content}
              </div>
            )
          )}

          {/* Grounded Evidence Citations */}
          {hasSources && (
            <div className="mt-4 pt-3 border-t border-slate-200/80">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Verified Reference Sources ({message.sources!.length})
                </span>
                <span className="text-[11px] text-slate-400">
                  Grounded from Clinical Vector Store
                </span>
              </div>

              <div className="space-y-2">
                {message.sources!.map((source, idx) => (
                  <CitationCard key={source.chunk_id || idx} source={source} index={idx} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </article>
  );
};
