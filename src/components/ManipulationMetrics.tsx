import React from 'react';

interface ManipulationMetricsProps {
  metrics?: {
    full_ai_generation_likelihood: number;
    partial_ai_edit_likelihood: number;
  };
}

export function ManipulationMetrics({ metrics }: ManipulationMetricsProps) {
  if (!metrics) return null;

  return (
    <div className="mb-8">
      <h3 className="font-mono text-sm text-gray-400 tracking-widest uppercase mb-6 border-b border-border-dim pb-2">
        MANIPULATION SCOPE
      </h3>
      <div className="flex flex-col gap-4">
        <div>
          <div className="flex justify-between text-xs font-mono mb-2">
            <span className="text-gray-300">Full AI Generation</span>
            <span className={metrics.full_ai_generation_likelihood > 50 ? 'text-danger' : 'text-safe'}>
              {metrics.full_ai_generation_likelihood}%
            </span>
          </div>
          <div className="h-2 bg-black rounded-full overflow-hidden">
            <div 
              className={`h-full transition-all duration-1000 ${metrics.full_ai_generation_likelihood > 50 ? 'bg-danger' : 'bg-safe'}`} 
              style={{ width: `${metrics.full_ai_generation_likelihood}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs font-mono mb-2">
            <span className="text-gray-300">Partial AI Edit (Inpainting/Generative Fill)</span>
            <span className={metrics.partial_ai_edit_likelihood > 50 ? 'text-purple-400' : 'text-safe'}>
              {metrics.partial_ai_edit_likelihood}%
            </span>
          </div>
          <div className="h-2 bg-black rounded-full overflow-hidden">
            <div 
              className={`h-full transition-all duration-1000 ${metrics.partial_ai_edit_likelihood > 50 ? 'bg-purple-500' : 'bg-safe'}`} 
              style={{ width: `${metrics.partial_ai_edit_likelihood}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
