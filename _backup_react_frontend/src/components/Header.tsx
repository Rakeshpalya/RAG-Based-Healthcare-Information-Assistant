import React from 'react';
import { Activity, Plus, PanelLeft } from 'lucide-react';
import { ReadinessResponse } from '../types';

interface HeaderProps {
  readiness: ReadinessResponse | null;
  isLoadingReadiness: boolean;
  onNewChat: () => void;
  onToggleSidebar: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  readiness,
  isLoadingReadiness,
  onNewChat,
  onToggleSidebar,
}) => {
  const isReady = readiness?.status === 'ready';

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-4 flex items-center justify-between shadow-sm z-10 sticky top-0">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg md:hidden transition-colors"
          aria-label="Toggle conversation history sidebar"
        >
          <PanelLeft className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-600 to-blue-700 flex items-center justify-center text-white shadow-sm">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-semibold text-slate-900 text-base leading-tight">
              AI Healthcare Assistant
            </h1>
            <p className="text-xs text-slate-500 hidden sm:block">
              Evidence-Grounded Clinical Decision Support
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* System Readiness Indicator */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
            isLoadingReadiness
              ? 'bg-slate-50 text-slate-500 border-slate-200'
              : isReady
              ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
              : 'bg-rose-50 text-rose-700 border-rose-200'
          }`}
          title={
            isReady
              ? `System Ready • 744 Vectors Active • Safety Engine Verified`
              : `System Degraded or Offline`
          }
          role="status"
          aria-live="polite"
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isLoadingReadiness
                ? 'bg-slate-400 animate-pulse'
                : isReady
                ? 'bg-emerald-500'
                : 'bg-rose-500'
            }`}
          />
          <span className="hidden sm:inline">
            {isLoadingReadiness
              ? 'Checking status...'
              : isReady
              ? 'System Ready'
              : 'System Unavailable'}
          </span>
        </div>

        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="flex items-center gap-1.5 bg-sky-600 hover:bg-sky-700 text-white px-3 py-1.5 rounded-lg text-xs font-medium transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-1"
          aria-label="Start new conversation"
        >
          <Plus className="w-4 h-4" />
          <span className="hidden sm:inline">New Chat</span>
        </button>
      </div>
    </header>
  );
};
