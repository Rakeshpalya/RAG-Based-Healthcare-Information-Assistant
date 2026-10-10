import React from 'react';
import { SourceCitation, DialogueContextPayload } from '../../types';
import { CitationCard } from './CitationCard';
import { EmergencyBanner } from './EmergencyBanner';
import { ContraindicationBanner } from './ContraindicationBanner';
import { RequestIdBadge } from './RequestIdBadge';
import { formatDate } from '../../utils/formatters';
import { User as UserIcon, Bot, Clock } from 'lucide-react';

export interface ChatMessageItem {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  sources?: SourceCitation[];
  retrievalStatus?: string;
  requestId?: string;
  timestamp?: string;
  dialogueContext?: DialogueContextPayload;
}

interface ChatMessageProps {
  message: ChatMessageItem;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.sender === 'user';
  const isEmergency = message.retrievalStatus === 'safety_intercepted';
  const contraindications = message.dialogueContext?.contraindications;

  return (
    <div
      className={`flex gap-3 sm:gap-4 my-4 ${
        isUser ? 'flex-row-reverse' : 'flex-row'
      }`}
    >
      {/* Sender Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-xs font-semibold shadow-xs ${
          isUser
            ? 'bg-sky-600 text-white'
            : isEmergency
            ? 'bg-rose-600 text-white'
            : 'bg-slate-800 text-white'
        }`}
      >
        {isUser ? <UserIcon className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Message Content Container */}
      <div
        className={`flex flex-col space-y-2 max-w-[88%] sm:max-w-[80%] ${
          isUser ? 'items-end' : 'items-start'
        }`}
      >
        {/* User Bubble */}
        {isUser ? (
          <div className="bg-sky-600 text-white rounded-2xl rounded-tr-xs px-4 py-3 shadow-xs text-sm leading-relaxed whitespace-pre-wrap">
            {message.text}
          </div>
        ) : (
          /* Assistant Bubble */
          <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-xs p-4 sm:p-5 shadow-xs w-full space-y-3">
            {/* 1. Emergency Banner if intercepted */}
            {isEmergency ? (
              <EmergencyBanner message={message.text} />
            ) : (
              <>
                {/* 2. Contraindication Banner if flagged */}
                {contraindications && contraindications.length > 0 && (
                  <ContraindicationBanner alerts={contraindications} />
                )}

                {/* 3. Assistant Answer Text */}
                <div className="text-sm text-slate-800 leading-relaxed whitespace-pre-wrap font-sans space-y-2">
                  {message.text}
                </div>

                {/* 4. Grounded Source Citations */}
                {message.sources && message.sources.length > 0 && (
                  <div className="pt-3 border-t border-slate-100 space-y-2">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-600">
                      <span>Verified Sources & Citations ({message.sources.length})</span>
                    </div>
                    <div className="space-y-1.5">
                      {message.sources.map((src, idx) => (
                        <CitationCard key={src.chunk_id || idx} source={src} index={idx} />
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Footer: Timestamp, Trace ID, Follow-up indicator */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-2 text-[11px] text-slate-400 border-t border-slate-50">
              <div className="flex items-center gap-1.5">
                <Clock className="w-3 h-3" />
                <span>{formatDate(message.timestamp)}</span>
                {message.dialogueContext?.is_follow_up && (
                  <span className="ml-1 px-1.5 py-0.2 bg-sky-50 text-sky-700 font-medium rounded text-[10px]">
                    Multi-turn context preserved
                  </span>
                )}
              </div>
              <RequestIdBadge requestId={message.requestId} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
