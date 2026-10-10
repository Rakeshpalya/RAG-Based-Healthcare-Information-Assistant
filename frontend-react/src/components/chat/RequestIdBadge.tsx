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
    } catch {
      // fallback
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      title="Click to copy audit trace ID"
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-mono text-slate-500 bg-slate-100 hover:bg-slate-200 transition-colors"
    >
      <span>{requestId}</span>
      {copied ? (
        <Check className="w-3 h-3 text-emerald-600" />
      ) : (
        <Copy className="w-3 h-3 text-slate-400" />
      )}
    </button>
  );
};
