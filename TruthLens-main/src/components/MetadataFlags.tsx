import React from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';

interface MetadataFlagsProps {
  flags?: string[];
}

export function MetadataFlags({ flags = [] }: MetadataFlagsProps) {
  const hasFlags = flags.length > 0;

  return (
    <div className={`w-full p-6 rounded-xl border ${hasFlags ? 'bg-warning/5 border-warning/20' : 'bg-surface border-border-dim'}`}>
      <h3 className="font-mono text-xs uppercase tracking-widest text-gray-500 mb-4">
        METADATA FLAGS
      </h3>
      
      {!hasFlags ? (
        <div className="flex items-center gap-3 text-safe font-mono text-sm">
          <CheckCircle2 className="w-4 h-4" />
          <span>No metadata anomalies detected</span>
        </div>
      ) : (
        <ul className="flex flex-col gap-3">
          {flags.map((flag, index) => (
            <li key={index} className="flex items-start gap-3 text-warning font-mono text-sm">
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              <span className="opacity-90 leading-relaxed">{flag}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
