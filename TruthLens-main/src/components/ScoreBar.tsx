import React, { useEffect, useState } from 'react';

interface ScoreBarProps {
  label: string;
  score: number;
}

export function ScoreBar({ label, score }: ScoreBarProps) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    // Animate width on mount
    const timer = setTimeout(() => setWidth(score), 100);
    return () => clearTimeout(timer);
  }, [score]);

  let colorClass = '';
  if (score > 70) {
    colorClass = 'bg-danger';
  } else if (score >= 40) {
    colorClass = 'bg-warning';
  } else {
    colorClass = 'bg-safe';
  }

  return (
    <div className="w-full flex flex-col gap-2 mb-4">
      <div className="flex justify-between items-end">
        <span className="font-mono text-xs uppercase text-accent/70 tracking-widest">{label.replace(/_/g, ' ')}</span>
        <span className={`font-mono text-sm font-bold ${colorClass.replace('bg-', 'text-')}`}>
          {score}/100
        </span>
      </div>
      <div className="w-full h-1.5 bg-black/50 rounded-full overflow-hidden border border-white/5">
        <div 
          className={`h-full ${colorClass} transition-all duration-1000 ease-out`}
          style={{ width: `${width}%` }}
        ></div>
      </div>
    </div>
  );
}
