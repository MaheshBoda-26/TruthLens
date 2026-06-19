import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, Film } from 'lucide-react';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  onError: (msg: string) => void;
}

export function UploadZone({ onFileSelected, onError }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadType, setUploadType] = useState<'photo' | 'video'>('photo');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const validateAndProcessFile = (file: File) => {
    const validPhotoTypes = ['image/jpeg', 'image/png', 'image/webp'];
    const validVideoTypes = ['video/mp4', 'video/quicktime'];
    const validTypes = uploadType === 'photo' ? validPhotoTypes : validVideoTypes;

    if (!validTypes.includes(file.type)) {
      onError(`Unsupported format. Please upload a valid ${uploadType === 'photo' ? 'image (JPG, PNG, WEBP)' : 'video (MP4, MOV)'}.`);
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
    <div className="w-full max-w-2xl mx-auto flex flex-col items-center">
      {/* Format Selection Toggle */}
      <div className="flex bg-surface border border-border-dim rounded-lg p-1 mb-6 w-full max-w-xs">
        <button
          onClick={() => setUploadType('photo')}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-mono rounded-md transition-all duration-300 ${
            uploadType === 'photo' 
              ? 'bg-accent text-black font-bold shadow-[0_0_10px_rgba(0,255,128,0.2)]' 
              : 'text-gray-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <ImageIcon className="w-4 h-4" />
          PHOTO
        </button>
        <button
          onClick={() => setUploadType('video')}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-mono rounded-md transition-all duration-300 ${
            uploadType === 'video' 
              ? 'bg-accent text-black font-bold shadow-[0_0_10px_rgba(0,255,128,0.2)]' 
              : 'text-gray-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <Film className="w-4 h-4" />
          VIDEO
        </button>
      </div>

      {/* Drop Zone */}
      <div 
        className={`w-full border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer transition-colors duration-300 ${
          isDragging ? 'border-accent bg-accent/5' : 'border-border-dim bg-surface hover:border-accent/50'
        }`}
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
          accept={uploadType === 'photo' ? "image/jpeg, image/png, image/webp" : "video/mp4, video/quicktime"} 
        />
        <UploadCloud className={`w-16 h-16 mb-4 ${isDragging ? 'text-accent' : 'text-gray-500'}`} />
        <h3 className="text-xl font-mono text-white mb-2">
          Drag & Drop {uploadType === 'photo' ? 'Photo' : 'Video'} Here
        </h3>
        <p className="text-gray-400 font-sans text-sm text-center">or click to browse files</p>
        <p className="text-gray-500 font-mono text-xs mt-6 text-center">
          Supports {uploadType === 'photo' ? 'JPG, PNG, WEBP' : 'MP4, MOV'} — Max 10MB
        </p>
      </div>
    </div>
  );
}
