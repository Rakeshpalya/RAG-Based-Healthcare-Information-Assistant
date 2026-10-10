import React, { useState } from 'react';
import { SourceCitation } from '../../types';
import { cleanDocumentName, formatRelevanceScore } from '../../utils/formatters';
import { ChevronDown, ChevronUp, FileText, CheckCircle2 } from 'lucide-react';

interface CitationCardProps {
  source: SourceCitation;
  index: number;
}

export const CitationCard: React.FC<CitationCardProps> = ({ source, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const docName = cleanDocumentName(source.document_name);
  const relevance = formatRelevanceScore(source.similarity_score);

  return (
    <div className="border border-slate-200 rounded-lg bg-white overflow-hidden shadow-xs transition-colors hover:border-slate-300">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setIsExpanded(!isExpanded);
          }
        }}
        aria-expanded={isExpanded}
        className="w-full px-3 py-2.5 flex items-center justify-between text-left focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-inset"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="inline-flex items-center justify-center px-2 py-0.5 rounded text-xs font-semibold bg-sky-100 text-sky-800">
            Source {index + 1}
          </span>
          <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <span className="text-xs font-medium text-slate-800 truncate" title={docName}>
            {docName}
          </span>
          {source.page_number && (
            <span className="text-xs text-slate-500 shrink-0">p. {source.page_number}</span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0 ml-2">
          {relevance && (
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded">
              <CheckCircle2 className="w-3 h-3 text-emerald-600" />
              {relevance}
            </span>
          )}
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="px-3.5 pb-3 pt-1 border-t border-slate-100 bg-slate-50 text-xs text-slate-700 leading-relaxed">
          <div className="text-[11px] font-medium text-slate-500 mb-1 flex items-center justify-between">
            <span>Verified Source Excerpt</span>
            {source.chunk_id && <span className="font-mono text-[10px]">{source.chunk_id}</span>}
          </div>
          <p className="whitespace-pre-wrap font-sans bg-white p-2.5 rounded border border-slate-200">
            {source.text}
          </p>
        </div>
      )}
    </div>
  );
};
