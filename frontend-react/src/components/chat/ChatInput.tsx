import React, { useState, useRef, useEffect } from 'react';
import { Send, Square } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  isStreaming: boolean;
  onStopStreaming?: () => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  isStreaming,
  onStopStreaming,
  disabled = false,
}) => {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!isLoading && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isLoading]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!text.trim() || isLoading || disabled) return;
    onSendMessage(text.trim());
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    // Auto-adjust height
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
  };

  return (
    <div className="bg-white border-t border-slate-200 px-4 py-3 sm:py-4 shadow-sm">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto space-y-2">
        <div className="relative flex items-end gap-2 border border-slate-300 rounded-xl bg-slate-50 focus-within:bg-white focus-within:border-sky-500 focus-within:ring-2 focus-within:ring-sky-100 transition-all p-2">
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={
              disabled
                ? 'Backend is offline. Reconnecting...'
                : 'Ask a clinical research question or follow-up inquiry (e.g., guidelines for hypertension, CKD contraindications)...'
            }
            disabled={disabled || isLoading}
            className="w-full resize-none bg-transparent px-2 py-1.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none min-h-[38px] max-h-[180px]"
          />

          {isStreaming ? (
            <button
              type="button"
              onClick={onStopStreaming}
              title="Stop generating response"
              className="inline-flex items-center justify-center p-2 rounded-lg bg-rose-600 text-white hover:bg-rose-700 transition-colors shrink-0 shadow-xs"
            >
              <Square className="w-4 h-4 fill-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={disabled || isLoading || !text.trim()}
              title="Send clinical inquiry (Enter)"
              className="inline-flex items-center justify-center p-2 rounded-lg bg-sky-600 text-white hover:bg-sky-700 disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed transition-colors shrink-0 shadow-xs"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-400 px-1">
          <span>Press <kbd className="font-sans px-1 py-0.5 bg-slate-100 border border-slate-200 rounded">Enter</kbd> to send, <kbd className="font-sans px-1 py-0.5 bg-slate-100 border border-slate-200 rounded">Shift+Enter</kbd> for new line</span>
          <span>Max 1,000 characters</span>
        </div>
      </form>
    </div>
  );
};
