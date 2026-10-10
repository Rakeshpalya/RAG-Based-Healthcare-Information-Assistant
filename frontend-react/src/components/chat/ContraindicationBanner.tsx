import React from 'react';
import { ShieldAlert } from 'lucide-react';

interface ContraindicationAlert {
  contraindication_id?: string;
  severity?: string;
  reason?: string;
  patient_condition?: string;
  conflicting_entity?: string;
}

interface ContraindicationBannerProps {
  alerts?: ContraindicationAlert[];
}

export const ContraindicationBanner: React.FC<ContraindicationBannerProps> = ({ alerts }) => {
  if (!alerts || alerts.length === 0) return null;

  return (
    <div
      role="alert"
      className="bg-amber-50 border-2 border-amber-400 rounded-xl p-4 text-amber-950 mb-3 shadow-sm"
    >
      <div className="flex items-start gap-3">
        <div className="p-1.5 bg-amber-100 rounded-lg shrink-0 text-amber-700">
          <ShieldAlert className="w-5 h-5" />
        </div>
        <div className="space-y-2 flex-1">
          <div className="flex items-center gap-2">
            <h4 className="font-bold text-sm text-amber-900 tracking-tight">
              Important safety consideration
            </h4>
            <span className="text-xs font-semibold px-2 py-0.5 bg-amber-200 text-amber-900 rounded-full">
              Contraindication Alert
            </span>
          </div>

          <div className="space-y-1.5">
            {alerts.map((alert, idx) => (
              <div
                key={idx}
                className="bg-white/80 border border-amber-200 rounded-lg p-2.5 text-xs sm:text-sm text-amber-900 leading-relaxed"
              >
                {alert.reason || (
                  <span>
                    Potential clinical conflict between{' '}
                    <strong>{alert.patient_condition || 'reported condition'}</strong> and{' '}
                    <strong>{alert.conflicting_entity || 'medication'}</strong>.
                  </span>
                )}
              </div>
            ))}
          </div>
          <p className="text-xs text-amber-800">
            Please consult your prescribing physician or pharmacist before taking or modifying medications.
          </p>
        </div>
      </div>
    </div>
  );
};
