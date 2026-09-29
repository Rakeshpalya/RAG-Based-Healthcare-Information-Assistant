import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, CheckCircle2 } from 'lucide-react';
import { SourceCitation } from '../types';
import { formatSimilarityScore, cleanDocumentName } from '../utils/formatters';

interface CitationCardProps {
  source: SourceCitation;
  index: number;
}

export const CitationCard: React.FC<CitationCardProps> = ({ source, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const sourceNumber = index + 1;
  const docName = cleanDocumentName(source.document_name);
  const relevance = formatSimilarityScore(source.similarity_score);

  return (
    <div className="border border-slate-200 bg-white rounded-lg shadow-xs overflow-hidden transition-all hover:border-slate-300">
      {/* Header / Summary Bar */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3.5 py-2.5 flex items-center justify-between text-left hover:bg-slate-50/80 transition-colors focus:outline-none focus:bg-slate-50"
        aria-expanded={isExpanded}
        aria-label={`Toggle source ${sourceNumber}: ${docName}`}
      >
        <div className="flex items-center gap-2.5 truncate mr-2">
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-sky-100 text-sky-800 shrink-0">
            Source {sourceNumber}
          </span>
          <FileText className="w-4 h-4 text-slate-400 shrink-0" />
          <span className="text-xs font-medium text-slate-800 truncate" title={docName}>
            {docName}
          </span>
          {source.page_number !== undefined && (
            <span className="text-xs text-slate-400 shrink-0 hidden sm:inline">
              (Page {source.page_number})
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-100">
            <CheckCircle2 className="w-3 h-3" />
            <span>{relevance}</span>
          </span>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </button>

      {/* Expandable Evidence Excerpt */}
      {isExpanded && (
        <div className="px-3.5 pb-3 pt-1 border-t border-slate-100 bg-slate-50/50">
          <div className="text-[11px] font-semibold text-slate-500 mb-1 uppercase tracking-wider">
            Verified Reference Passage:
          </div>
          <blockquote className="text-xs text-slate-700 font-serif leading-relaxed italic bg-white p-2.5 rounded border border-slate-200/80">
            "{source.text}"
          </blockquote>
          {source.chunk_id && (
            <div className="mt-2 text-[10px] text-slate-400 font-mono">
              Chunk ID: {source.chunk_id}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
