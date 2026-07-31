import React from 'react';

interface SummaryBlockProps {
  summary: string;
}

export function SummaryBlock({ summary }: SummaryBlockProps) {
  return (
    <div className="w-full p-6 bg-surface border-l-4 border-l-accent border-y border-r border-border-dim rounded-r-xl">
      <h3 className="font-mono text-xs uppercase tracking-widest text-gray-500 mb-3">
        ANALYSIS SUMMARY
      </h3>
      <p className="font-sans text-gray-300 leading-relaxed text-[15px]">
        {summary}
      </p>
    </div>
  );
}
