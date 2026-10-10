import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  ShieldCheck,
  FileText,
  MessageSquare,
  Activity,
  ArrowRight,
  Database,
  Lock,
  Sparkles,
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  const { isAuthenticated } = useAuth();

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col justify-between bg-slate-50">
      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-20 text-center space-y-8">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-sky-100 text-sky-800 text-xs font-semibold shadow-2xs">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Grounded Clinical RAG & Longitudinal Context (Phase 7)</span>
        </div>

        <div className="space-y-4 max-w-3xl mx-auto">
          <h1 className="text-3xl sm:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Evidence-Grounded Clinical Intelligence for Healthcare Research
          </h1>
          <p className="text-base sm:text-lg text-slate-600 leading-relaxed">
            Delivering auditable, guideline-concordant medical answers with automated citation attribution,
            multi-turn longitudinal context tracking, and non-bypassable patient safety guardrails.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
          {isAuthenticated ? (
            <Link
              to="/chat"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-sky-600 text-white font-semibold hover:bg-sky-700 shadow-sm transition-all"
            >
              <MessageSquare className="w-5 h-5" />
              <span>Launch Clinical Chat</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </Link>
          ) : (
            <>
              <Link
                to="/register"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-sky-600 text-white font-semibold hover:bg-sky-700 shadow-sm transition-all"
              >
                <span>Get Started</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                to="/login"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-white border border-slate-300 text-slate-700 font-semibold hover:bg-slate-50 transition-all shadow-2xs"
              >
                <span>Sign In to Workspace</span>
              </Link>
            </>
          )}
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-12 text-left">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-2xs space-y-3">
            <div className="w-10 h-10 rounded-xl bg-sky-50 text-sky-700 flex items-center justify-center">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base">Longitudinal Safety & Context</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Maintains deterministic tracking of conditions, allergies, and medications across dialogue turns,
              proactively detecting contraindication alerts without non-deterministic LLM summarization.
            </p>
          </div>

          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-2xs space-y-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center">
              <FileText className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base">Verifiable Source Attribution</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Answers are bound to verified reference chunks indexed in FAISS with similarity thresholds,
              providing clickable inline citation cards for clinical provenance.
            </p>
          </div>

          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-2xs space-y-3">
            <div className="w-10 h-10 rounded-xl bg-rose-50 text-rose-700 flex items-center justify-center">
              <Activity className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base">Emergency Interception</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Autonomous safety gates intercept emergency symptoms, poisonings, and acute conditions with immediate,
              authoritative guidance to seek qualified healthcare services.
            </p>
          </div>
        </div>
      </div>

      {/* Trust & Architecture Banner */}
      <div className="bg-white border-t border-slate-200 py-6">
        <div className="max-w-6xl mx-auto px-4 flex flex-wrap items-center justify-center gap-8 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-slate-400" />
            <span>744 Production Vectors Indexed</span>
          </div>
          <div className="flex items-center gap-2">
            <Lock className="w-4 h-4 text-slate-400" />
            <span>Multi-Tenant Row-Level Security</span>
          </div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-slate-400" />
            <span>Zero-PHI Auditability Exposition</span>
          </div>
        </div>
      </div>
    </div>
  );
};
