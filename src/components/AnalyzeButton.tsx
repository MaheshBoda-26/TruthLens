import React from 'react';
import { Search } from 'lucide-react';

interface AnalyzeButtonProps {
  onClick: () => void;
  disabled: boolean;
}

export function AnalyzeButton({ onClick, disabled }: AnalyzeButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full max-w-2xl mx-auto py-4 rounded-xl font-mono text-xl font-bold tracking-widest flex items-center justify-center gap-3 transition-all duration-300
        ${disabled 
          ? 'bg-surface text-gray-500 cursor-not-allowed border border-border-dim' 
          : 'bg-accent text-black hover:bg-accent/90 shadow-[0_0_20px_rgba(0,255,128,0.3)] hover:shadow-[0_0_30px_rgba(0,255,128,0.5)] active:scale-[0.98]'
        }`}
    >
      <Search className="w-6 h-6" />
      ANALYZE MEDIA
    </button>
  );
}
