import React from 'react';
import { Stethoscope } from 'lucide-react';

interface HealthLogoProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

/**
 * HealthAI Brand Logo.
 * Displays a crisp stethoscope graphic inside a white (#FFFFFF) container
 * with elegant rounded corners, subtle contrast border, and medical sky-blue stroke.
 */
export const HealthLogo: React.FC<HealthLogoProps> = ({ size = 'md', className = '' }) => {
  const sizeClasses = {
    sm: 'w-8 h-8 rounded-lg',
    md: 'w-9 h-9 rounded-xl',
    lg: 'w-12 h-12 rounded-2xl',
  };

  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  };

  return (
    <div
      className={`bg-white border border-slate-200/90 shadow-2xs flex items-center justify-center text-sky-600 shrink-0 transition-transform ${sizeClasses[size]} ${className}`}
      style={{ backgroundColor: '#FFFFFF' }}
      aria-label="HealthAI Stethoscope Logo"
    >
      <Stethoscope
        className={`${iconSizes[size]} text-sky-600`}
        strokeWidth={2.2}
        aria-hidden="true"
      />
    </div>
  );
};
