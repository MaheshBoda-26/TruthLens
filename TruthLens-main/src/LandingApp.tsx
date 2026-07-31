import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Header as AppHeader } from './components/Header';
import { Home } from './pages/Home';
import { About } from './pages/About';
import { Features } from './pages/Features';
import { Navigation } from './components/Navigation';
import { Footer } from './components/Footer';
import { BackgroundEffects } from './components/BackgroundEffects';
import './index.css';

function LandingLayout() {
  const location = useLocation();

  return (
    <div className="min-h-screen w-full relative">
      <BackgroundEffects />
      <Navigation />
      <main className="relative z-10 pt-16 pb-20">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/features" element={<Features />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default function LandingApp() {
  return (
    <BrowserRouter>
      <LandingLayout />
    </BrowserRouter>
  );
}