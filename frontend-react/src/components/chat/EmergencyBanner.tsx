import React from 'react';
import { AlertOctagon, PhoneCall } from 'lucide-react';

interface EmergencyBannerProps {
  message?: string;
}

export const EmergencyBanner: React.FC<EmergencyBannerProps> = ({ message }) => {
  return (
    <div
      role="alert"
      className="bg-rose-50 border-2 border-rose-400 rounded-xl p-4 sm:p-5 text-rose-950 shadow-sm"
    >
      <div className="flex items-start gap-3">
        <div className="p-2 bg-rose-100 rounded-lg shrink-0 text-rose-700">
          <AlertOctagon className="w-6 h-6" />
        </div>
        <div className="space-y-2 flex-1">
          <div className="flex items-center gap-2">
            <h4 className="font-bold text-base text-rose-900 tracking-tight">
              EMERGENCY MEDICAL ADVISORY
            </h4>
            <span className="text-xs font-semibold uppercase px-2 py-0.5 bg-rose-200 text-rose-900 rounded-full">
              Critical
            </span>
          </div>

          <p className="text-sm font-semibold text-rose-800 leading-relaxed">
            If this may be an emergency, seek immediate help from your local emergency services or a qualified healthcare professional.
          </p>

          {message && (
            <div className="bg-white/80 border border-rose-200 rounded-lg p-3 text-xs sm:text-sm text-rose-900 leading-relaxed">
              {message}
            </div>
          )}

          <div className="flex items-center gap-2 text-xs text-rose-700 pt-1">
            <PhoneCall className="w-3.5 h-3.5" />
            <span>Do not wait or rely on automated systems for acute symptoms. Contact medical services immediately.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
