import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (question: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  disabled = false,
}) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        140
      )}px`;
    }
  }, [input]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (isLoading || disabled) return;

    const trimmed = input.trim();
    if (trimmed.length < 3 || trimmed.length > 1000) return;

    onSendMessage(trimmed);
    setInput('');
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

  const isValidLength = input.trim().length >= 3 && input.trim().length <= 1000;
  const canSubmit = !isLoading && !disabled && isValidLength;

  return (
    <form
      onSubmit={handleSubmit}
      className="p-3 sm:p-4 bg-white border-t border-slate-200"
      aria-label="Ask a medical question form"
    >
      <div className="max-w-3xl mx-auto relative flex items-end gap-2 border border-slate-300 rounded-xl p-2 focus-within:ring-2 focus-within:ring-sky-500 focus-within:border-sky-500 bg-white shadow-xs transition-all">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a medical or clinical research question... (e.g., What are the lifestyle measures for hypertension?)"
          disabled={isLoading || disabled}
          rows={1}
          maxLength={1000}
          className="flex-1 max-h-36 resize-none p-1.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none bg-transparent disabled:opacity-50"
          aria-label="Clinical inquiry text input"
        />

        <div className="flex items-center gap-2 shrink-0 pb-0.5 pr-0.5">
          {input.length > 0 && (
            <span
              className={`text-[11px] font-mono ${
                input.trim().length < 3 || input.length > 950
                  ? 'text-amber-600'
                  : 'text-slate-400'
              }`}
            >
              {input.length}/1000
            </span>
          )}

          <button
            type="submit"
            disabled={!canSubmit}
            className="w-8 h-8 rounded-lg bg-sky-600 hover:bg-sky-700 disabled:bg-slate-200 disabled:text-slate-400 text-white flex items-center justify-center transition-colors focus:outline-none focus:ring-2 focus:ring-sky-500"
            aria-label={isLoading ? 'Generating medical response' : 'Send inquiry'}
            title="Send query (Enter)"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      <div className="max-w-3xl mx-auto flex items-center justify-between mt-1 px-1 text-[11px] text-slate-400">
        <span>Press <strong>Enter</strong> to submit, <strong>Shift + Enter</strong> for a new line</span>
        <span>Min 3 chars</span>
      </div>
    </form>
  );
};
