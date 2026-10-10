import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useHealth } from '../context/HealthContext';
import { documentApi } from '../api/documentApi';
import { conversationApi } from '../api/conversationApi';
import { DocumentRecord, Conversation } from '../types';
import { DocumentUploadModal } from '../components/documents/DocumentUploadModal';
import { cleanDocumentName, formatDate } from '../utils/formatters';
import {
  MessageSquare,
  FileText,
  UploadCloud,
  ArrowRight,
  Activity,
  Clock,
  Loader2,
} from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const { status: healthStatus } = useHealth();

  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  const fetchDashboardData = async () => {
    setIsLoading(true);
    try {
      const [docsRes, convsRes] = await Promise.allSettled([
        documentApi.listDocuments(0, 5),
        conversationApi.listConversations(0, 5),
      ]);

      if (docsRes.status === 'fulfilled') {
        setDocuments(docsRes.value);
      }
      if (convsRes.status === 'fulfilled') {
        setConversations(convsRes.value);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Welcome Banner */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 shadow-2xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-sky-50 text-sky-700">
              Clinical Workspace
            </span>
            <span className="text-xs text-slate-400">•</span>
            <span className="text-xs text-slate-500">Status: {healthStatus}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
            Welcome back, {user?.email?.split('@')[0] || 'Clinician'}
          </h1>
          <p className="text-xs sm:text-sm text-slate-600">
            Search clinical references, explore multi-turn patient dialogue, and audit grounded evidence.
          </p>
        </div>

        {/* Quick Action Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => setIsUploadModalOpen(true)}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl border border-slate-300 bg-white text-slate-700 text-xs sm:text-sm font-semibold hover:bg-slate-50 transition-colors shadow-2xs"
          >
            <UploadCloud className="w-4 h-4 text-slate-500" />
            <span>Upload Document</span>
          </button>
          <Link
            to="/chat"
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-sky-600 text-white text-xs sm:text-sm font-semibold hover:bg-sky-700 transition-colors shadow-xs"
          >
            <MessageSquare className="w-4 h-4" />
            <span>Ask a Healthcare Question</span>
          </Link>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-2xs space-y-1">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Indexed Documents</span>
            <FileText className="w-4 h-4 text-sky-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900">
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin text-slate-400" /> : documents.length}
          </div>
          <p className="text-[11px] text-slate-400">Available in personal clinical vector space</p>
        </div>

        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-2xs space-y-1">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Consultation Sessions</span>
            <MessageSquare className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900">
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin text-slate-400" /> : conversations.length}
          </div>
          <p className="text-[11px] text-slate-400">Saved clinical dialogue histories</p>
        </div>

        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-2xs space-y-1">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>System Connectivity</span>
            <Activity className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900 flex items-center gap-2">
            <span
              className={`w-3 h-3 rounded-full ${
                healthStatus === 'CONNECTED'
                  ? 'bg-emerald-500'
                  : healthStatus === 'DEGRADED'
                  ? 'bg-amber-500'
                  : 'bg-rose-500'
              }`}
            />
            <span className="capitalize text-lg">{healthStatus}</span>
          </div>
          <p className="text-[11px] text-slate-400">Core RAG & FAISS vector store state</p>
        </div>
      </div>

      {/* Two Column Layout: Recent Documents & Recent Conversations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Documents */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 shadow-2xs space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-sky-600" />
              <h2 className="text-base font-bold text-slate-900">Recent Documents</h2>
            </div>
            <Link
              to="/documents"
              className="text-xs font-medium text-sky-600 hover:text-sky-700 flex items-center gap-1"
            >
              <span>View all</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {isLoading ? (
            <div className="py-8 text-center">
              <Loader2 className="w-6 h-6 animate-spin text-slate-400 mx-auto" />
            </div>
          ) : documents.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-500 space-y-2">
              <p>No documents uploaded yet.</p>
              <button
                onClick={() => setIsUploadModalOpen(true)}
                className="text-sky-600 font-semibold hover:underline"
              >
                Upload your first clinical PDF
              </button>
            </div>
          ) : (
            <div className="divide-y divide-slate-100">
              {documents.map((doc) => (
                <div key={doc.id} className="py-2.5 flex items-center justify-between text-xs">
                  <div className="min-w-0 pr-2">
                    <p className="font-semibold text-slate-800 truncate" title={doc.filename}>
                      {cleanDocumentName(doc.filename)}
                    </p>
                    <p className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                      <span>{doc.num_chunks} chunks</span>
                      <span>•</span>
                      <span>{formatDate(doc.created_at)}</span>
                    </p>
                  </div>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-50 text-emerald-700 shrink-0">
                    {doc.status}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recent Conversations */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 shadow-2xs space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-sky-600" />
              <h2 className="text-base font-bold text-slate-900">Recent Consultations</h2>
            </div>
            <Link
              to="/history"
              className="text-xs font-medium text-sky-600 hover:text-sky-700 flex items-center gap-1"
            >
              <span>View all</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {isLoading ? (
            <div className="py-8 text-center">
              <Loader2 className="w-6 h-6 animate-spin text-slate-400 mx-auto" />
            </div>
          ) : conversations.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-500 space-y-2">
              <p>No recent consultations found.</p>
              <Link to="/chat" className="text-sky-600 font-semibold hover:underline inline-block">
                Start a new clinical chat
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-slate-100">
              {conversations.map((conv) => (
                <Link
                  key={conv.id}
                  to={`/chat?id=${conv.id}`}
                  className="py-2.5 flex items-center justify-between text-xs hover:bg-slate-50 px-2 rounded-lg transition-colors group"
                >
                  <div className="min-w-0 pr-2">
                    <p className="font-semibold text-slate-800 truncate group-hover:text-sky-700">
                      {conv.title}
                    </p>
                    <p className="text-[11px] text-slate-400 flex items-center gap-1.5 mt-0.5">
                      <Clock className="w-3 h-3" />
                      <span>{formatDate(conv.updated_at || conv.created_at)}</span>
                    </p>
                  </div>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-300 group-hover:text-sky-600 shrink-0" />
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Document Upload Modal */}
      <DocumentUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUploadSuccess={fetchDashboardData}
      />
    </div>
  );
};
