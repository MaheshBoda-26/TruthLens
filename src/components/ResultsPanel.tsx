import React from 'react';
import { VerdictBadge } from './VerdictBadge';
import { ScoreBar } from './ScoreBar';
import { MetadataFlags } from './MetadataFlags';
import { SummaryBlock } from './SummaryBlock';
import { Recommendation } from './Recommendation';
import { AnalysisResult } from '../utils/analyzeWithGemini';
import { RefreshCw } from 'lucide-react';

interface ResultsPanelProps {
  result: AnalysisResult;
  onReset: () => void;
}

export function ResultsPanel({ result, onReset }: ResultsPanelProps) {
  return (
    <div className="w-full max-w-3xl mx-auto flex flex-col gap-8 pb-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <VerdictBadge verdict={result.verdict} confidence={result.confidence} />
      
      <div className="bg-surface border border-border-dim p-8 rounded-2xl shadow-lg">
        <div className="mb-10">
          <ScoreBar label="OVERALL RISK SCORE" score={result.risk_score} />
        </div>
        
        <div className="mb-8">
          <h3 className="font-mono text-sm text-gray-400 tracking-widest uppercase mb-6 border-b border-border-dim pb-2">
            SIGNAL BREAKDOWN
          </h3>
          <div className="flex flex-col gap-4">
            <ScoreBar label="FACE ANALYSIS" score={result.signals.face_analysis.score} />
            <ScoreBar label="LIGHTING & SHADOWS" score={result.signals.lighting_shadows.score} />
            <ScoreBar label="TEXTURE ARTIFACTS" score={result.signals.texture_artifacts.score} />
            <ScoreBar label="EDGE CONSISTENCY" score={result.signals.edge_consistency.score} />
            <ScoreBar label="BACKGROUND COHERENCE" score={result.signals.background_coherence.score} />
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <MetadataFlags flags={result.metadata_flags} />
          <SummaryBlock summary={result.summary} />
          <Recommendation recommendation={result.recommendation} verdict={result.verdict} />
        </div>
      </div>

      <button
        onClick={onReset}
        className="w-full py-4 rounded-xl font-mono text-lg font-bold tracking-widest flex items-center justify-center gap-3 bg-surface hover:bg-border-dim text-white border border-border-dim transition-all duration-300"
      >
        <RefreshCw className="w-5 h-5" />
        SCAN ANOTHER FILE
      </button>
    </div>
  );
}
