import React from 'react';

interface VerdictBadgeProps {
  verdict: "LIKELY DEEPFAKE" | "POSSIBLY DEEPFAKE" | "AI EDITED" | "LIKELY AUTHENTIC" | "INCONCLUSIVE";
  confidence: number;
}

export function VerdictBadge({ verdict, confidence }: VerdictBadgeProps) {
  let colorClass = '';
  let bgClass = '';
  let shadowClass = '';

  switch (verdict) {
    case 'LIKELY DEEPFAKE':
      colorClass = 'text-danger';
      bgClass = 'bg-[#2a0a0a]';
      shadowClass = 'shadow-[0_0_30px_rgba(255,51,51,0.3)] border-danger/50';
      break;
    case 'POSSIBLY DEEPFAKE':
      colorClass = 'text-warning';
      bgClass = 'bg-[#2a1a00]';
      shadowClass = 'shadow-[0_0_30px_rgba(255,170,0,0.3)] border-warning/50';
      break;
    case 'AI EDITED':
      colorClass = 'text-purple-400';
      bgClass = 'bg-[#1a0a2a]';
      shadowClass = 'shadow-[0_0_30px_rgba(168,85,247,0.3)] border-purple-500/50';
      break;
    case 'LIKELY AUTHENTIC':
      colorClass = 'text-safe';
      bgClass = 'bg-[#0a1a0a]';
      shadowClass = 'shadow-[0_0_30px_rgba(0,204,68,0.3)] border-safe/50';
      break;
    case 'INCONCLUSIVE':
    default:
      colorClass = 'text-gray-400';
      bgClass = 'bg-[#1a1a1a]';
      shadowClass = 'shadow-[0_0_20px_rgba(136,136,136,0.2)] border-gray-600/50';
      break;
  }

  return (
    <div className={`w-full py-8 rounded-2xl border flex flex-col items-center justify-center ${bgClass} ${shadowClass} relative overflow-hidden`}>
      {/* Inner glow */}
      <div className={`absolute inset-0 opacity-20 bg-gradient-to-b from-transparent to-current ${colorClass}`}></div>
      
      <h2 className={`font-mono text-3xl md:text-5xl font-bold tracking-wider mb-3 z-10 ${colorClass}`}>
        {verdict}
      </h2>
      
      <div className="font-mono text-sm text-gray-300 z-10 flex items-center gap-2 bg-black/40 px-4 py-1.5 rounded-full border border-white/10">
        <span className="uppercase tracking-widest opacity-70">Confidence:</span>
        <span className="font-bold">{confidence}%</span>
      </div>
    </div>
  );
}
