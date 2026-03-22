import React, { useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  onError: (msg: string) => void;
}

export function UploadZone({ onFileSelected, onError }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const validateAndProcessFile = (file: File) => {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'video/mp4', 'video/quicktime'];
    if (!validTypes.includes(file.type)) {
      onError("Unsupported format. Upload JPG, PNG, WEBP, MP4, or MOV.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      onError("File exceeds 10MB. Please try a smaller file.");
      return;
    }

    onFileSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  return (
    <div 
      className={`w-full max-w-2xl mx-auto border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer transition-colors duration-300 ${isDragging ? 'border-accent bg-accent/5' : 'border-border-dim bg-surface hover:border-accent/50'}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleChange} 
        className="hidden" 
        accept="image/jpeg, image/png, image/webp, video/mp4, video/quicktime" 
      />
      <UploadCloud className={`w-16 h-16 mb-4 ${isDragging ? 'text-accent' : 'text-gray-500'}`} />
      <h3 className="text-xl font-mono text-white mb-2">Drag & Drop Media Here</h3>
      <p className="text-gray-400 font-sans text-sm text-center">or click to browse files</p>
      <p className="text-gray-500 font-mono text-xs mt-6 text-center">
        Supports JPG, PNG, WEBP, MP4, MOV — Max 10MB
      </p>
    </div>
  );
}
