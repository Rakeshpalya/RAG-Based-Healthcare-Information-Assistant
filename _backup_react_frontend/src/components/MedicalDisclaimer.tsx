import React from 'react';
import { ShieldCheck } from 'lucide-react';

export const MedicalDisclaimer: React.FC = () => {
  return (
    <aside
      className="py-2 px-4 bg-slate-100/90 border-t border-slate-200 text-center text-xs text-slate-500 flex items-center justify-center gap-1.5 shrink-0"
      aria-label="Medical Disclaimer"
    >
      <ShieldCheck className="w-3.5 h-3.5 text-slate-400 shrink-0" />
      <span>
        <strong>Educational information only.</strong> This assistant does not replace professional medical diagnosis, clinical judgment, or individualized treatment. Always consult a licensed healthcare practitioner.
      </span>
    </aside>
  );
};
