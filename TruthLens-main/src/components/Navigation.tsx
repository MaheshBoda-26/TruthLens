import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Github, Linkedin, Twitter } from 'lucide-react';

const navLinks = [
  { path: '/', label: 'HOME' },
  { path: '/features', label: 'FEATURES' },
  { path: '/about', label: 'ABOUT' },
  { path: '/setup', label: 'SETUP' },
];

export function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location]);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-bg-main/95 backdrop-blur-sm border-b border-border-dim shadow-[0_0_30px_rgba(0,0,0,0.5)]'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 md:h-20">
          {/* Logo */}
          <Link
            to="/"
            className="font-mono text-xl md:text-2xl text-accent font-bold tracking-wider flex items-center gap-2 z-10"
            aria-label="TruthLens Home"
          >
            <span className="relative">
              TruthLens
              <span className="absolute bottom-0 left-0 w-full h-0.5 bg-accent opacity-70"></span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`font-mono text-xs tracking-widest uppercase transition-colors duration-300 relative ${
                  location.pathname === link.path
                    ? 'text-accent'
                    : 'text-gray-400 hover:text-accent'
                }`}
              >
                {link.label}
                {location.pathname === link.path && (
                  <span className="absolute bottom-[-4px] left-0 right-0 h-0.5 bg-accent animate-in expand" />
                )}
              </Link>
            ))}
            <a
              href="#"
              className="px-5 py-2 bg-accent text-black font-mono text-xs tracking-widest uppercase rounded-lg hover:bg-accent/90 transition-all duration-300 shadow-[0_0_20px_rgba(0,255,128,0.3)] hover:shadow-[0_0_30px_rgba(0,255,128,0.5)]"
            >
              LAUNCH APP
            </a>
          </div>

          {/* Social Links - Desktop */}
          <div className="hidden lg:flex items-center gap-4">
            <a
              href="#"
              className="text-gray-400 hover:text-accent transition-colors duration-300"
              aria-label="GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="#"
              className="text-gray-400 hover:text-accent transition-colors duration-300"
              aria-label="LinkedIn"
            >
              <Linkedin className="w-5 h-5" />
            </a>
            <a
              href="#"
              className="text-gray-400 hover:text-accent transition-colors duration-300"
              aria-label="Twitter"
            >
              <Twitter className="w-5 h-5" />
            </a>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2 text-gray-400 hover:text-accent transition-colors duration-300 z-10"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-expanded={isMobileMenuOpen}
            aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
          >
            {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden overflow-hidden bg-bg-main/98 backdrop-blur-sm border-t border-border-dim"
          >
            <div className="py-6 px-2 flex flex-col gap-4">
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`font-mono text-sm tracking-widest uppercase py-3 px-4 rounded-lg transition-all duration-300 ${
                    location.pathname === link.path
                      ? 'bg-accent/10 text-accent border border-accent/30'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {link.label}
                </Link>
              ))}
              <a
                href="#"
                className="text-center px-5 py-3 bg-accent text-black font-mono text-sm tracking-widest uppercase rounded-lg hover:bg-accent/90 transition-all duration-300 shadow-[0_0_20px_rgba(0,255,128,0.3)]"
              >
                LAUNCH APP
              </a>
              <div className="flex justify-center gap-6 pt-4 border-t border-border-dim">
                <a href="#" className="text-gray-400 hover:text-accent transition-colors" aria-label="GitHub">
                  <Github className="w-6 h-6" />
                </a>
                <a href="#" className="text-gray-400 hover:text-accent transition-colors" aria-label="LinkedIn">
                  <Linkedin className="w-6 h-6" />
                </a>
                <a href="#" className="text-gray-400 hover:text-accent transition-colors" aria-label="Twitter">
                  <Twitter className="w-6 h-6" />
                </a>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </nav>
  );
}