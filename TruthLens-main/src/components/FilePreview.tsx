import React from 'react';
import { X } from 'lucide-react';

interface FilePreviewProps {
  file: File;
  preview: string;
  mediaType: 'image' | 'video';
  onRemove: () => void;
}

export function FilePreview({ file, preview, mediaType, onRemove }: FilePreviewProps) {
  const formatBytes = (bytes: number, decimals = 2) => {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-surface border border-border-dim rounded-xl p-6 flex flex-col items-center relative">
      <button 
        onClick={onRemove}
        className="absolute top-4 right-4 p-2 bg-black/50 hover:bg-danger/20 text-gray-400 hover:text-danger rounded-full transition-colors"
        title="Remove file"
      >
        <X className="w-5 h-5" />
      </button>

      <div className="w-full flex justify-center mb-4 bg-black/30 rounded-lg overflow-hidden">
        {mediaType === 'image' ? (
          <img src={preview} alt="Preview" className="max-h-[300px] object-contain" />
        ) : (
          <video src={preview} controls className="max-h-[300px] object-contain" />
        )}
      </div>

      <div className="w-full flex justify-between items-center px-2">
        <span className="font-mono text-sm text-gray-300 truncate max-w-[70%]">{file.name}</span>
        <span className="font-mono text-xs text-gray-500">{formatBytes(file.size)}</span>
      </div>
    </div>
  );
}
