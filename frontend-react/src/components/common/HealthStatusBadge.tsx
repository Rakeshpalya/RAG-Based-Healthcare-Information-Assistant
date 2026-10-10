import React from 'react';
import { useHealth } from '../../context/HealthContext';
import { RefreshCw } from 'lucide-react';

export const HealthStatusBadge: React.FC = () => {
  const { status, refreshHealth } = useHealth();

  const getStatusConfig = () => {
    switch (status) {
      case 'CONNECTED':
        return {
          bg: 'bg-emerald-50 text-emerald-700 border-emerald-200',
          dot: 'bg-emerald-500',
          label: 'Connected',
        };
      case 'DEGRADED':
        return {
          bg: 'bg-amber-50 text-amber-700 border-amber-200',
          dot: 'bg-amber-500',
          label: 'Degraded',
        };
      case 'OFFLINE':
        return {
          bg: 'bg-rose-50 text-rose-700 border-rose-200',
          dot: 'bg-rose-500',
          label: 'Offline',
        };
      case 'CHECKING':
      default:
        return {
          bg: 'bg-sky-50 text-sky-700 border-sky-200',
          dot: 'bg-sky-500 animate-pulse',
          label: 'Checking',
        };
    }
  };

  const config = getStatusConfig();

  return (
    <button
      onClick={() => refreshHealth()}
      title="Click to check backend status"
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors hover:opacity-80 ${config.bg}`}
    >
      <span className={`w-2 h-2 rounded-full ${config.dot}`} />
      <span>{config.label}</span>
      <RefreshCw className="w-3 h-3 text-slate-400 hover:text-slate-600" />
    </button>
  );
};
