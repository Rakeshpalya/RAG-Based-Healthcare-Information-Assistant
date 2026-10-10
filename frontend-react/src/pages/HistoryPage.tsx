import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { conversationApi } from '../api/conversationApi';
import { Conversation } from '../types';
import { formatDate } from '../utils/formatters';
import {
  History,
  MessageSquare,
  Trash2,
  ArrowRight,
  Clock,
  Loader2,
  AlertCircle,
  Plus,
} from 'lucide-react';

export const HistoryPage: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const navigate = useNavigate();

  const fetchConversations = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await conversationApi.listConversations(0, 100);
      setConversations(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load conversation history.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchConversations();
  }, []);

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (!window.confirm('Delete this clinical conversation session?')) {
      return;
    }

    setDeletingId(id);
    try {
      await conversationApi.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
    } catch (err: any) {
      alert(err.message || 'Failed to delete conversation.');
    } finally {
      setDeletingId(null);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm('Are you sure you want to delete ALL consultation history? This cannot be undone.')) {
      return;
    }

    try {
      await conversationApi.clearAllConversations();
      setConversations([]);
    } catch (err: any) {
      alert(err.message || 'Failed to clear conversations.');
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <History className="w-6 h-6 text-sky-600" />
            Clinical Consultation History
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Review prior multi-turn clinical inquiry dialogues, follow-up resolutions, and grounded citations.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          {conversations.length > 0 && (
            <button
              type="button"
              onClick={handleClearAll}
              className="px-3 py-2 text-xs font-medium text-rose-600 hover:text-rose-700 border border-rose-200 hover:bg-rose-50 rounded-xl transition-colors"
            >
              Clear All History
            </button>
          )}
          <button
            type="button"
            onClick={() => navigate('/chat')}
            className="inline-flex items-center gap-2 px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white text-xs sm:text-sm font-semibold rounded-xl shadow-xs transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>New Consultation</span>
          </button>
        </div>
      </div>

      {/* Error Notice */}
      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* List / Loading / Empty */}
      {isLoading ? (
        <div className="py-20 text-center space-y-3">
          <Loader2 className="w-8 h-8 animate-spin text-sky-600 mx-auto" />
          <p className="text-xs text-slate-500">Loading conversation history...</p>
        </div>
      ) : conversations.length === 0 ? (
        <div className="bg-white border-2 border-dashed border-slate-200 rounded-2xl p-12 text-center max-w-md mx-auto space-y-4">
          <div className="w-12 h-12 rounded-xl bg-sky-50 text-sky-600 mx-auto flex items-center justify-center">
            <MessageSquare className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-slate-900">No Past Consultations</h3>
            <p className="text-xs text-slate-500">
              Start a new dialogue session to consult grounded clinical knowledge.
            </p>
          </div>
          <Link
            to="/chat"
            className="inline-flex items-center gap-2 px-4 py-2 bg-sky-600 text-white rounded-xl text-xs font-semibold hover:bg-sky-700 transition-colors shadow-xs"
          >
            <Plus className="w-4 h-4" />
            <span>Start Clinical Chat</span>
          </Link>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-2xl divide-y divide-slate-100 shadow-2xs overflow-hidden">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className="p-4 sm:p-5 flex items-center justify-between hover:bg-slate-50/80 transition-colors group"
            >
              <Link
                to={`/chat?id=${conv.id}`}
                className="flex items-start gap-3.5 min-w-0 flex-1 pr-4"
              >
                <div className="p-2 bg-slate-100 text-slate-600 rounded-xl group-hover:bg-sky-100 group-hover:text-sky-700 transition-colors shrink-0">
                  <MessageSquare className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <h3 className="font-semibold text-sm text-slate-800 truncate group-hover:text-sky-700">
                    {conv.title}
                  </h3>
                  <div className="flex items-center gap-3 text-xs text-slate-400 mt-1">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      {formatDate(conv.updated_at || conv.created_at)}
                    </span>
                    <span>•</span>
                    <span className="font-mono text-[11px]">ID: #{conv.id}</span>
                  </div>
                </div>
              </Link>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={(e) => handleDelete(conv.id, e)}
                  disabled={deletingId === conv.id}
                  className="p-2 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition-colors"
                  title="Delete conversation"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
                <Link
                  to={`/chat?id=${conv.id}`}
                  className="p-2 text-slate-400 group-hover:text-sky-600 group-hover:translate-x-0.5 transition-all"
                  title="Open conversation"
                >
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
