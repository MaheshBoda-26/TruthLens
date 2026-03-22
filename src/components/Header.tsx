import React from 'react';

export function Header() {
  return (
    <header className="w-full py-6 flex flex-col items-center justify-center border-b border-border-dim mb-8">
      <h1 className="font-mono text-4xl md:text-5xl text-accent font-bold tracking-wider">
        TruthLens
      </h1>
      <p className="font-sans text-sm text-gray-400 mt-2 tracking-widest uppercase">
        See Through The Fake
      </p>
      <div className="w-24 h-0.5 bg-accent mt-4 opacity-70"></div>
    </header>
  );
}
