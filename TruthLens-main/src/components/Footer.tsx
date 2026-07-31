import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Linkedin, Mail, ArrowRight, Shield, Zap, Eye, Code } from 'lucide-react';

const socialLinks = [
  { href: 'https://github.com/MaheshBoda-26', icon: Github, label: 'GitHub', ariaLabel: 'GitHub' },
  { href: 'https://www.linkedin.com/in/maheshboda/', icon: Linkedin, label: 'LinkedIn', ariaLabel: 'LinkedIn' },
  { href: '#', icon: Mail, label: 'Email', ariaLabel: 'Email' },
];

const footerLinks = {
  product: [
    { label: 'Features', href: '/features' },
    { label: 'Documentation', href: '/about' },
    { label: 'Setup Guide', href: '/setup' },
    { label: 'Launch App', href: '#' },
  ],
  company: [
    { label: 'About', href: '/about' },
    { label: 'Blog', href: '#' },
    { label: 'Careers', href: '#' },
    { label: 'Contact', href: '#' },
  ],
  legal: [
    { label: 'Privacy Policy', href: '#' },
    { label: 'Terms of Service', href: '#' },
    { label: 'Security', href: '#' },
  ],
};

export function Footer() {
  return (
    <footer className="relative border-t border-border-dim bg-bg-main/50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-12">
          {/* Brand */}
          <div className="lg:col-span-1">
            <Link to="/" className="font-mono text-2xl text-accent font-bold tracking-wider inline-block mb-4">
              TruthLens
            </Link>
            <p className="font-sans text-sm text-gray-400 leading-relaxed mb-6">
              Forensic-grade deepfake detection powered by hybrid ML + LLM analysis.
              See through the fake with confidence.
            </p>
            <div className="flex gap-4">
              {socialLinks.map((social) => (
                <a
                  key={social.label}
                  href={social.href}
                  className="w-10 h-10 rounded-lg bg-surface border border-border-dim flex items-center justify-center text-gray-400 hover:text-accent hover:border-accent/50 transition-all duration-300"
                  aria-label={social.ariaLabel}
                >
                  <social.icon className="w-5 h-5" />
                </a>
              ))}
            </div>
          </div>

          {/* Product */}
          <nav className="lg:col-span-1">
            <h4 className="font-mono text-xs tracking-widest uppercase text-accent mb-4">PRODUCT</h4>
            <ul className="space-y-3">
              {footerLinks.product.map((link) => (
                <li key={link.label}>
                  <Link
                    to={link.href}
                    className="font-sans text-sm text-gray-400 hover:text-accent transition-colors duration-300 flex items-center gap-2"
                  >
                    {link.label}
                    <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          {/* Company */}
          <nav className="lg:col-span-1">
            <h4 className="font-mono text-xs tracking-widest uppercase text-accent mb-4">COMPANY</h4>
            <ul className="space-y-3">
              {footerLinks.company.map((link) => (
                <li key={link.label}>
                  <Link
                    to={link.href}
                    className="font-sans text-sm text-gray-400 hover:text-accent transition-colors duration-300"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          {/* Legal */}
          <nav className="lg:col-span-1">
            <h4 className="font-mono text-xs tracking-widest uppercase text-accent mb-4">LEGAL</h4>
            <ul className="space-y-3">
              {footerLinks.legal.map((link) => (
                <li key={link.label}>
                  <Link
                    to={link.href}
                    className="font-sans text-sm text-gray-400 hover:text-accent transition-colors duration-300"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          {/* Tech Stack */}
          <div className="lg:col-span-1">
            <h4 className="font-mono text-xs tracking-widest uppercase text-accent mb-4">TECHNOLOGY</h4>
            <div className="space-y-3">
              <div className="flex items-center gap-3 p-3 bg-surface border border-border-dim rounded-lg hover:border-accent/30 transition-colors">
                <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
                  <Shield className="w-5 h-5 text-accent" />
                </div>
                <div>
                  <p className="font-mono text-xs text-white">Hybrid Analysis</p>
                  <p className="font-sans text-[11px] text-gray-500">ML + LLM Pipeline</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-surface border border-border-dim rounded-lg hover:border-accent/30 transition-colors">
                <div className="w-10 h-10 rounded-lg bg-warning/10 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-warning" />
                </div>
                <div>
                  <p className="font-mono text-xs text-white">Real-time</p>
                  <p className="font-sans text-[11px] text-gray-500">Fast Detection</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-surface border border-border-dim rounded-lg hover:border-accent/30 transition-colors">
                <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
                  <Eye className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <p className="font-mono text-xs text-white">Forensic Detail</p>
                  <p className="font-sans text-[11px] text-gray-500">Metadata Analysis</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-surface border border-border-dim rounded-lg hover:border-accent/30 transition-colors">
                <div className="w-10 h-10 rounded-lg bg-safe/10 flex items-center justify-center">
                  <Code className="w-5 h-5 text-safe" />
                </div>
                <div>
                  <p className="font-mono text-xs text-white">Open Source</p>
                  <p className="font-sans text-[11px] text-gray-500">Transparent Code</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 pt-8 border-t border-border-dim flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="font-sans text-sm text-gray-500">
            © {new Date().getFullYear()} TruthLens. Built with precision for digital truth.
          </p>
          <div className="flex items-center gap-6 text-[11px] font-mono text-gray-500 uppercase tracking-widest">
            <span>v1.0.0</span>
            <span className="w-px h-4 bg-border-dim"></span>
            <span>MIT License</span>
            <span className="w-px h-4 bg-border-dim"></span>
            <a href="#" className="hover:text-accent transition-colors">Open Source</a>
          </div>
        </div>
      </div>
    </footer>
  );
}