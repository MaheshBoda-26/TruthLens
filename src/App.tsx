import React, { useState } from 'react';
import { Header } from './components/Header';
import { UploadZone } from './components/UploadZone';
import { FilePreview } from './components/FilePreview';
import { AnalyzeButton } from './components/AnalyzeButton';
import { LoadingScreen } from './components/LoadingScreen';
import { ResultsPanel } from './components/ResultsPanel';
import { analyzeWithGemini, AnalysisResult } from './utils/analyzeWithGemini';
import { extractMetadata } from './utils/extractMetadata';
import { AlertCircle } from 'lucide-react';

const ERROR_MESSAGES: Record<string, string> = {
  UNSUPPORTED_FILE: "Unsupported format. Upload JPG, PNG, WEBP, MP4, or MOV.",
  FILE_TOO_LARGE:   "File exceeds 10MB. Please try a smaller file.",
  FILE_TOO_SMALL:   "Image too small for reliable analysis.",
  API_FAILED:       "Analysis failed. Check your connection and try again.",
  PARSE_FAILED:     "Couldn't read the AI response. Please try again.",
  NO_CONTENT:       "No response from AI. Please try again.",
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<'image' | 'video' | null>(null);
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'done' | 'error'>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelected = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    
    const type = selectedFile.type.startsWith('video/') ? 'video' : 'image';
    setMediaType(type);
    
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  };

  const handleRemoveFile = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setFile(null);
    setPreview(null);
    setMediaType(null);
    setError(null);
  };

  const handleError = (msg: string) => {
    setError(msg);
  };

  const handleAnalyze = async () => {
    if (!file || !mediaType) return;
    
    setStatus('analyzing');
    setError(null);
    
    try {
      const [aiResult, metaResult] = await Promise.all([
        analyzeWithGemini(file, mediaType),
        extractMetadata(file)
      ]);
      
      const unified: AnalysisResult = {
        ...aiResult,
        metadata_flags: metaResult.flags
      };
      
      setResult(unified);
      setStatus('done');
    } catch (err: any) {
      console.error("Analysis error:", err);
      setError(ERROR_MESSAGES[err.message] || ERROR_MESSAGES.API_FAILED);
      setStatus('error');
    }
  };

  const handleReset = () => {
    handleRemoveFile();
    setResult(null);
    setStatus('idle');
  };

  return (
    <div className="min-h-screen w-full px-4 py-8 md:py-12 flex flex-col items-center relative z-10">
      <Header />

      <main className="w-full max-w-4xl mx-auto flex-1 flex flex-col items-center">
        {status === 'analyzing' && <LoadingScreen />}

        {(status === 'idle' || status === 'error') && (
          <div className="w-full flex flex-col gap-8 animate-in fade-in duration-500">
            {!file ? (
              <UploadZone onFileSelected={handleFileSelected} onError={handleError} />
            ) : (
              <FilePreview 
                file={file} 
                preview={preview!} 
                mediaType={mediaType!} 
                onRemove={handleRemoveFile} 
              />
            )}

            {error && (
              <div className="w-full max-w-2xl mx-auto p-4 bg-danger/10 border border-danger/30 rounded-lg flex items-start gap-3 text-danger">
                <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                <p className="font-sans text-sm">{error}</p>
              </div>
            )}

            <div className="w-full max-w-2xl mx-auto mt-4">
              <AnalyzeButton 
                onClick={handleAnalyze} 
                disabled={!file || status === 'analyzing'} 
              />
            </div>
          </div>
        )}

        {status === 'done' && result && (
          <ResultsPanel result={result} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}
