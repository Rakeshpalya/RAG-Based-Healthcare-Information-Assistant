import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface RequestIdBadgeProps {
  requestId?: string;
}

export const RequestIdBadge: React.FC<RequestIdBadgeProps> = ({ requestId }) => {
  const [copied, setCopied] = useState(false);

  if (!requestId) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(requestId);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Failed to copy request ID:', e);
    }
  };

  return (
    <div className="inline-flex items-center gap-1.5 text-[11px] text-slate-400 font-mono bg-slate-100/80 px-2 py-0.5 rounded border border-slate-200">
      <span>ID: {requestId}</span>
      <button
        onClick={handleCopy}
        className="p-0.5 text-slate-400 hover:text-slate-700 transition-colors rounded focus:outline-none"
        title="Copy request ID for audit or support"
        aria-label="Copy request ID"
      >
        {copied ? (
          <Check className="w-3 h-3 text-emerald-600" />
        ) : (
          <Copy className="w-3 h-3" />
        )}
      </button>
      {copied && (
        <span className="text-[10px] text-emerald-600 font-sans font-medium">Copied!</span>
      )}
    </div>
  );
};
