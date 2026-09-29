import React from 'react';
import { AlertOctagon, PhoneCall } from 'lucide-react';

interface EmergencyBannerProps {
  message?: string;
}

export const EmergencyBanner: React.FC<EmergencyBannerProps> = ({ message }) => {
  return (
    <div
      className="bg-rose-50 border-2 border-rose-400 rounded-xl p-4 my-3 text-rose-950 shadow-sm"
      role="alert"
      aria-live="assertive"
    >
      <div className="flex items-start gap-3">
        <div className="p-2 bg-rose-600 text-white rounded-lg shrink-0 mt-0.5">
          <AlertOctagon className="w-6 h-6 animate-pulse" />
        </div>

        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h2 className="font-bold text-base text-rose-900 tracking-wide uppercase">
              IMPORTANT — SEEK IMMEDIATE EMERGENCY CARE
            </h2>
          </div>

          <p className="mt-1 text-sm font-medium text-rose-900 leading-snug">
            Your inquiry indicates acute or potentially life-threatening emergency symptoms.
          </p>

          {message ? (
            <div className="mt-2.5 p-3 bg-white/80 rounded-lg border border-rose-200 text-xs text-rose-950 leading-relaxed font-sans">
              {message}
            </div>
          ) : (
            <div className="mt-2 text-xs text-rose-800 leading-relaxed">
              If you or someone nearby is experiencing chest pain, difficulty breathing, slurred speech, sudden weakness, or poisoning, do not wait for an online response.
            </div>
          )}

          <div className="mt-3 flex items-center gap-2 text-xs font-semibold text-rose-900">
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-rose-600 text-white rounded-md font-bold shadow-xs">
              <PhoneCall className="w-3.5 h-3.5" />
              <span>Call 911 (or local emergency) Immediately</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
