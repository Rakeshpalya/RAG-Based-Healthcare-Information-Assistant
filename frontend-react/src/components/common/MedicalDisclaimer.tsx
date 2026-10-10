import React from 'react';
import { AlertCircle } from 'lucide-react';

export const MedicalDisclaimer: React.FC = () => {
  return (
    <aside
      aria-label="Clinical Disclaimer"
      className="bg-slate-100 border-t border-slate-200 px-4 py-2 text-xs text-slate-600 text-center flex items-center justify-center gap-2"
    >
      <AlertCircle className="w-3.5 h-3.5 text-slate-500 shrink-0" />
      <span>
        This application provides healthcare information and decision-support information for educational purposes.
        It is not a substitute for diagnosis, treatment, or advice from a qualified healthcare professional.
      </span>
    </aside>
  );
};
