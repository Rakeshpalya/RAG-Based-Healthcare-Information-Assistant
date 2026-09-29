import React from 'react';
import { MessageSquare, Plus, Trash2, X, Clock } from 'lucide-react';
import { ChatSession } from '../types';

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  isOpen: boolean;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string, e: React.MouseEvent) => void;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  activeSessionId,
  isOpen,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onCloseMobile,
}) => {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/40 z-20 md:hidden backdrop-blur-sm transition-opacity"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Content */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-30 w-72 bg-white border-r border-slate-200 flex flex-col transform transition-transform duration-200 ease-in-out md:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        aria-label="Conversation History Sidebar"
      >
        {/* Top Actions */}
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <button
            onClick={onNewChat}
            className="flex-1 flex items-center justify-center gap-2 bg-slate-900 hover:bg-slate-800 text-white py-2 px-3 rounded-lg text-sm font-medium transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-900"
          >
            <Plus className="w-4 h-4" />
            <span>New Consultation</span>
          </button>
          <button
            onClick={onCloseMobile}
            className="md:hidden ml-2 p-2 text-slate-400 hover:text-slate-600 rounded-lg"
            aria-label="Close conversation history sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="px-3 py-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5" />
            <span>Recent Consultations</span>
          </div>

          {sessions.length === 0 ? (
            <div className="p-4 text-center text-xs text-slate-400 italic">
              No previous conversations. Start a new medical query above.
            </div>
          ) : (
            sessions.map((session) => {
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  className={`group relative flex items-center justify-between p-2.5 rounded-lg text-sm cursor-pointer transition-all ${
                    isActive
                      ? 'bg-sky-50 text-sky-900 font-medium border border-sky-200 shadow-xs'
                      : 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                  role="button"
                  tabIndex={0}
                  aria-selected={isActive}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelectSession(session.id);
                    }
                  }}
                >
                  <div className="flex items-center gap-2.5 truncate pr-6">
                    <MessageSquare
                      className={`w-4 h-4 shrink-0 ${
                        isActive ? 'text-sky-600' : 'text-slate-400'
                      }`}
                    />
                    <span className="truncate">{session.title}</span>
                  </div>

                  <button
                    onClick={(e) => onDeleteSession(session.id, e)}
                    className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 text-slate-400 hover:text-rose-600 rounded transition-all"
                    title="Delete consultation"
                    aria-label={`Delete consultation titled ${session.title}`}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Footer Info */}
        <div className="p-3 border-t border-slate-100 text-xs text-slate-400 text-center">
          Local Session Storage • Privacy Preserved
        </div>
      </aside>
    </>
  );
};
