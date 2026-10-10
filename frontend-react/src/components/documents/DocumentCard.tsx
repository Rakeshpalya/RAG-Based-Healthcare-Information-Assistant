import React from 'react';
import { DocumentRecord } from '../../types';
import { cleanDocumentName, formatFileSize, formatDate } from '../../utils/formatters';
import { FileText, Trash2, Layers, CheckCircle2, AlertTriangle } from 'lucide-react';

interface DocumentCardProps {
  doc: DocumentRecord;
  onDelete: (id: number) => void;
  isDeleting?: boolean;
}

export const DocumentCard: React.FC<DocumentCardProps> = ({ doc, onDelete, isDeleting = false }) => {
  const displayName = cleanDocumentName(doc.filename);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-2xs hover:shadow-xs transition-all flex flex-col justify-between">
      <div className="space-y-3">
        {/* Header: Icon, Name, and Status */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 min-w-0">
            <div className="p-2.5 bg-sky-50 rounded-xl text-sky-700 shrink-0">
              <FileText className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <h3 className="font-semibold text-sm text-slate-900 truncate" title={doc.filename}>
                {displayName}
              </h3>
              <p className="text-xs text-slate-500 font-mono truncate">{doc.filename}</p>
            </div>
          </div>

          <span
            className={`inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full shrink-0 ${
              doc.status === 'indexed' || doc.status === 'processed'
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : 'bg-amber-50 text-amber-700 border border-amber-200'
            }`}
          >
            {doc.status === 'indexed' || doc.status === 'processed' ? (
              <CheckCircle2 className="w-3 h-3 text-emerald-600" />
            ) : (
              <AlertTriangle className="w-3 h-3 text-amber-600" />
            )}
            <span className="capitalize">{doc.status}</span>
          </span>
        </div>

        {/* Metadata Details */}
        <div className="grid grid-cols-2 gap-2 text-xs text-slate-600 pt-2 border-t border-slate-100">
          <div className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-slate-400" />
            <span>{doc.num_chunks} Chunks indexed</span>
          </div>
          <div>
            <span className="text-slate-400">Size: </span>
            <span>{formatFileSize(doc.file_size_bytes)}</span>
          </div>
        </div>
      </div>

      {/* Footer: Date & Delete button */}
      <div className="flex items-center justify-between pt-3 mt-3 border-t border-slate-100 text-[11px] text-slate-400">
        <span>Uploaded {formatDate(doc.created_at)}</span>
        <button
          type="button"
          onClick={() => onDelete(doc.id)}
          disabled={isDeleting}
          className="inline-flex items-center gap-1 text-slate-500 hover:text-rose-600 p-1 rounded hover:bg-rose-50 transition-colors disabled:opacity-50"
          title="Delete document and remove indexed chunks"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Delete</span>
        </button>
      </div>
    </div>
  );
};
