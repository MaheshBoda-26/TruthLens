import React, { useState, useEffect } from 'react';

const MESSAGES = [
  "Analyzing facial geometry...",
  "Checking lighting consistency...",
  "Scanning texture artifacts...",
  "Extracting metadata signals...",
  "Generating forensic report..."
];

export function LoadingScreen() {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % MESSAGES.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 bg-bg-main/90 backdrop-blur-sm z-50 flex flex-col items-center justify-center">
      <div className="w-32 h-32 rounded-full border-2 border-accent/20 flex items-center justify-center mb-8 relative">
        <div className="w-24 h-24 rounded-full bg-accent/10 animate-pulse-green"></div>
        <div className="absolute inset-0 border-t-2 border-accent rounded-full animate-spin" style={{ animationDuration: '3s' }}></div>
      </div>
      
      <h2 className="font-mono text-2xl text-accent font-bold tracking-widest mb-4">
        SCANNING MEDIA...
      </h2>
      
      <p className="font-mono text-sm text-gray-400 h-6 transition-opacity duration-500">
        {MESSAGES[messageIndex]}
      </p>
    </div>
  );
}
