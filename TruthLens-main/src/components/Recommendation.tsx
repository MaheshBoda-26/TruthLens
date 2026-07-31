import React from 'react';

interface RecommendationProps {
  recommendation: string;
  verdict: "LIKELY DEEPFAKE" | "POSSIBLY DEEPFAKE" | "AI EDITED" | "LIKELY AUTHENTIC" | "INCONCLUSIVE";
}

export function Recommendation({ recommendation, verdict }: RecommendationProps) {
  let bgClass = 'bg-surface border-border-dim';
  let icon = '⚪';

  switch (verdict) {
    case 'LIKELY DEEPFAKE':
      bgClass = 'bg-danger/10 border-danger/30';
      icon = '🔴';
      break;
    case 'POSSIBLY DEEPFAKE':
      bgClass = 'bg-warning/10 border-warning/30';
      icon = '🟡';
      break;
    case 'AI EDITED':
      bgClass = 'bg-purple-900/20 border-purple-500/30';
      icon = '🟣';
      break;
    case 'LIKELY AUTHENTIC':
      bgClass = 'bg-safe/10 border-safe/30';
      icon = '🟢';
      break;
    case 'INCONCLUSIVE':
    default:
      bgClass = 'bg-gray-800/50 border-gray-600/30';
      icon = '⚪';
      break;
  }

  return (
    <div className={`w-full p-6 rounded-xl border ${bgClass} flex flex-col gap-3`}>
      <h3 className="font-mono text-xs uppercase tracking-widest text-gray-500">
        RECOMMENDATION
      </h3>
      <div className="flex items-start gap-3">
        <span className="text-lg leading-none">{icon}</span>
        <p className="font-sans text-gray-200 leading-relaxed text-[15px]">
          {recommendation}
        </p>
      </div>
    </div>
  );
}
