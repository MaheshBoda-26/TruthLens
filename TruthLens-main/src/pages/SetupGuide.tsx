import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Copy, Check, Terminal, Github, FolderOpen, FileText, Settings, Play, Shield, Zap, Code, ArrowRight, ExternalLink, ChevronDown, ChevronUp, CheckCircle, Globe } from 'lucide-react';

const steps = [
  {
    number: '01',
    title: 'Clone Repository',
    description: 'Get the latest source code from GitHub',
    commands: [
      'git clone https://github.com/yourusername/truthlens.git',
      'cd truthlens/TruthLens-main',
    ],
    category: 'setup',
  },
  {
    number: '02',
    title: 'Install Dependencies',
    description: 'Install all npm packages for frontend and backend',
    commands: [
      'npm install',
    ],
    category: 'setup',
  },
  {
    number: '03',
    title: 'Configure Environment',
    description: 'Set up your configuration (no API keys required)',
    commands: [
      'cp .env.example .env.local',
      '# Edit .env.local if needed (no API keys required for local model)',
    ],
    category: 'config',
  },
  {
    number: '04',
    title: 'Start Development Servers',
    description: 'Launch both frontend (Vite) and backend (Express) together',
    commands: [
      'npm run dev:all',
    ],
    category: 'run',
    note: 'Runs frontend on http://localhost:3000 and API on http://localhost:3001',
  },
  {
    number: '05',
    title: 'Verify Installation',
    description: 'Confirm everything is working correctly',
    commands: [
      '# Open http://localhost:3000 in your browser',
      '# You should see the TruthLens analyzer interface',
      '# Try uploading a test image to verify the pipeline',
    ],
    category: 'verify',
  },
];

const envVars = [
  {
    key: 'PORT',
    required: false,
    description: 'Backend API server port (default: 3001)',
    example: '3001',
  },
  {
    key: 'VITE_API_URL',
    required: false,
    description: 'Frontend API endpoint (default: http://localhost:3001)',
    example: 'http://localhost:3001',
  },
];

const scripts = [
  { command: 'npm run dev', description: 'Start Vite dev server only (frontend on :3000)' },
  { command: 'npm run server', description: 'Start Express API only (backend on :3001)' },
  { command: 'npm run dev:all', description: 'Run both frontend and backend concurrently' },
  { command: 'npm run build', description: 'Build production frontend to dist/' },
  { command: 'npm run preview', description: 'Preview production build locally' },
  { command: 'npm run clean', description: 'Remove dist/ build artifacts' },
  { command: 'npm run lint', description: 'Type-check TypeScript (no emit)' },
];

const troubleshooting = [
  {
    issue: 'Port already in use',
    solution: 'Kill existing processes: lsof -ti:3000,3001 | xargs kill -9',
  },
  {
    issue: 'ML backend unavailable',
    solution: 'Ensure Python server runs on localhost:5001 for local heuristic analysis (optional)',
  },
  {
    issue: 'Module not found errors',
    solution: 'Delete node_modules and package-lock.json, then run npm install again',
  },
  {
    issue: 'TypeScript errors',
    solution: 'Run npm run lint to see full error details. Ensure Node 18+ is installed',
  },
  {
    issue: 'Video upload fails',
    solution: 'Ensure FFmpeg is installed for video frame extraction (brew install ffmpeg / apt install ffmpeg)',
  },
];

const production = [
  {
    title: 'Docker Deployment',
    description: 'Containerize the application for production',
    steps: [
      'Create Dockerfile for frontend (multi-stage with Nginx)',
      'Create Dockerfile for backend (Node.js Alpine)',
      'Use docker-compose.yml to orchestrate services',
      'Configure reverse proxy (Nginx/Traefik) for SSL termination',
    ],
  },
  {
    title: 'Vercel Deployment',
    description: 'Deploy frontend to Vercel with serverless functions',
    steps: [
      'Connect GitHub repository to Vercel',
      'Set environment variables in Vercel dashboard',
      'Configure vercel.json for API routes',
      'Deploy — automatic on push to main',
    ],
  },
  {
    title: 'Environment Hardening',
    description: 'Secure your production deployment',
    steps: [
      'Use strong, unique API keys per environment',
      'Enable CORS only for your domain',
      'Set up rate limiting on API endpoints',
      'Configure CSP headers for frontend',
      'Enable HTTPS only (HSTS)',
      'Monitor API usage and set alerts',
    ],
  },
];

export function SetupGuide() {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    troubleshooting: true,
    production: false,
  });

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const CommandBlock = ({ commands, key }: { commands: string[], key: string }) => (
    <div className="relative group">
      <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => copyToClipboard(commands.join('\n'))}
          className="p-2 bg-surface border border-border-dim rounded-lg hover:border-accent/50 hover:bg-accent/5 transition-all text-gray-400 hover:text-accent"
          aria-label="Copy commands"
        >
          <Copy className="w-4 h-4" />
        </button>
      </div>
      <pre className="bg-[#020804] border border-border-dim rounded-xl p-4 overflow-x-auto font-mono text-sm text-gray-300 leading-relaxed">
        {commands.map((cmd, i) => (
          <div key={i} className={cmd.startsWith('#') ? 'text-gray-500' : 'text-white'}>
            <span className="text-accent/60">$ </span>{cmd}
          </div>
        ))}
      </pre>
    </div>
  );

  const ToggleSection = ({ title, icon, defaultOpen, sectionKey, children }: { title: string, icon: React.ReactNode, defaultOpen: boolean, sectionKey: string, children: React.ReactNode }) => (
    <div className="bg-surface border border-border-dim rounded-2xl overflow-hidden">
      <button
        onClick={() => setExpandedSections(prev => ({ ...prev, [sectionKey]: !prev[sectionKey] }))}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-surface/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-accent/10 border border-accent/30 flex items-center justify-center text-accent">
            {icon}
          </div>
          <h3 className="font-mono text-lg font-bold text-white">{title}</h3>
        </div>
        <motion.div
          animate={{ rotate: expandedSections[sectionKey] ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-gray-400"
        >
          <ChevronDown className="w-5 h-5" />
        </motion.div>
      </button>
      <motion.div
        initial={{ height: 0, opacity: 0 }}
        animate={{ height: expandedSections[sectionKey] ? 'auto' : 0, opacity: expandedSections[sectionKey] ? 1 : 0 }}
        transition={{ duration: 0.3 }}
        className="overflow-hidden border-t border-border-dim px-6 pb-6"
      >
        <div className="pt-4 space-y-6" />
      </motion.div>
    </div>
  );

  return (
    <>
      {/* Page Header */}
      <section className="relative py-20 px-4">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-6"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase">
              SETUP GUIDE
            </span>
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="font-mono text-4xl md:text-6xl font-bold text-white mb-6"
          >
            Get <span className="text-accent">TruthLens</span> Running
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="font-sans text-lg md:text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed"
          >
            Complete installation guide for local development, production deployment, and troubleshooting.
            From zero to detecting deepfakes in under 5 minutes.
          </motion.p>
        </div>
      </section>

      {/* Quick Start */}
      <section className="py-20 px-4 relative">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-10"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-xl bg-accent/10 border border-accent/30 flex items-center justify-center text-accent">
                <Zap className="w-6 h-6" />
              </div>
              <h2 className="font-mono text-2xl font-bold text-white">Quick Start (TL;DR)</h2>
            </div>
            <CommandBlock commands={[
              'git clone https://github.com/yourusername/truthlens.git',
              'cd truthlens/TruthLens-main',
              'npm install',
              'cp .env.example .env.local',
              '# Add VITE_GEMINI_API_KEY to .env.local',
              'npm run dev:all',
            ]} key="quickstart" />
          </motion.div>

          {/* Prerequisites */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="mb-12 p-6 bg-surface border border-border-dim rounded-2xl"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-lg bg-warning/10 border border-warning/30 flex items-center justify-center text-warning">
                <Shield className="w-5 h-5" />
              </div>
              <h2 className="font-mono text-xl font-bold text-white">Prerequisites</h2>
            </div>
            <div className="grid sm:grid-cols-2 gap-4 font-sans text-sm text-gray-300">
              <div className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-accent shrink-0" /> Node.js 18+ (LTS recommended)</div>
              <div className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-accent shrink-0" /> npm 9+ or pnpm/yarn</div>
              <div className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-accent shrink-0" /> Git for version control</div>
              <div className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-warning shrink-0" /> FFmpeg (for video support)</div>
              <div className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-warning shrink-0" /> Python 3.10+ (optional, for local ML)</div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Step-by-Step Guide */}
      <section className="py-20 px-4 relative bg-bg-main/50">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              STEP BY STEP
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Detailed <span className="text-accent">Installation</span>
            </h2>
          </motion.div>

          <div className="space-y-8">
            {steps.map((step, index) => (
              <motion.div
                key={step.number}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="relative"
              >
                <div className="flex gap-6">
                  <div className="flex-shrink-0 relative">
                    <div className="w-16 h-16 rounded-2xl bg-surface border border-border-dim flex items-center justify-center font-mono text-2xl font-bold text-accent relative z-10">
                      {step.number}
                    </div>
                    {index < steps.length - 1 && (
                      <div className="absolute left-7 top-16 bottom-0 w-0.5 bg-gradient-to-b from-accent/30 to-transparent" />
                    )}
                  </div>
                  <div className="flex-1 pt-2">
                    <div className="flex items-center gap-3 mb-3">
                      <h3 className="font-mono text-xl font-bold text-white">{step.title}</h3>
                      <span className="px-2 py-0.5 text-[10px] font-mono tracking-widest uppercase rounded bg-accent/10 border border-accent/30 text-accent">{step.category.toUpperCase()}</span>
                    </div>
                    <p className="font-sans text-gray-400 mb-4">{step.description}</p>
                    <CommandBlock commands={step.commands} key={step.number} />
                    {step.note && (
                      <p className="mt-3 font-sans text-sm text-warning/80 flex items-center gap-2">
                        <Zap className="w-4 h-4" /> {step.note}
                      </p>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Environment Variables */}
      <section className="py-20 px-4 relative">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-10"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-xl bg-warning/10 border border-warning/30 flex items-center justify-center text-warning">
                <Settings className="w-6 h-6" />
              </div>
              <h2 className="font-mono text-2xl font-bold text-white">Environment Variables</h2>
            </div>
            <div className="bg-surface border border-border-dim rounded-2xl overflow-hidden">
              <div className="grid grid-cols-4 border-b border-border-dim bg-bg-main/50 px-4 py-3 font-mono text-xs tracking-widest uppercase text-gray-400">
                <div>VARIABLE</div>
                <div>REQUIRED</div>
                <div>DESCRIPTION</div>
                <div>EXAMPLE</div>
              </div>
              {envVars.map((env, index) => (
                <div key={env.key} className={`grid grid-cols-4 px-4 py-3 border-b border-border-dim/50 last:border-0 hover:bg-surface/50 transition-colors ${index % 2 === 0 ? 'bg-bg-main/30' : ''}`}>
                  <div className="font-mono text-sm text-white flex items-center gap-2">
                    <Code className="w-4 h-4 text-gray-500" />
                    {env.key}
                  </div>
                  <div className="flex items-center">
                    <span className={`px-2 py-0.5 text-[10px] font-mono tracking-widest uppercase rounded ${env.required ? 'bg-danger/20 text-danger border border-danger/30' : 'bg-warning/20 text-warning border border-warning/30'}`}>
                      {env.required ? 'REQUIRED' : 'OPTIONAL'}
                    </span>
                  </div>
                  <div className="font-sans text-sm text-gray-400">{env.description}</div>
                  <div className="font-mono text-xs text-gray-500 font-normal">{env.example}</div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Available Scripts */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="mb-10"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400">
                <Terminal className="w-6 h-6" />
              </div>
              <h2 className="font-mono text-2xl font-bold text-white">Available Scripts</h2>
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              {scripts.map((script) => (
                <div key={script.command} className="p-4 bg-surface border border-border-dim rounded-xl hover:border-accent/30 transition-all">
                  <div className="font-mono text-sm text-accent mb-1">{script.command}</div>
                  <div className="font-sans text-sm text-gray-400">{script.description}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Troubleshooting */}
      <section className="py-20 px-4 relative bg-bg-main/50">
        <div className="max-w-4xl mx-auto">
          <ToggleSection
            title="Troubleshooting"
            icon={<Shield className="w-5 h-5" />}
            defaultOpen={true}
            sectionKey="troubleshooting"
          >
            <div className="space-y-4">
              {troubleshooting.map((item, index) => (
                <motion.div
                  key={item.issue}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className="p-4 bg-bg-main/50 border border-border-dim rounded-xl"
                >
                  <div className="font-mono text-sm text-white mb-2 flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-accent" />
                    {item.issue}
                  </div>
                  <div className="font-sans text-sm text-gray-400 pl-6">
                    {item.solution}
                  </div>
                </motion.div>
              ))}
            </div>
          </ToggleSection>
        </div>
      </section>

      {/* Production Deployment */}
      <section className="py-20 px-4 relative">
        <div className="max-w-4xl mx-auto">
          <ToggleSection
            title="Production Deployment"
            icon={<Globe className="w-5 h-5" />}
            defaultOpen={false}
            sectionKey="production"
          >
            <div className="space-y-6">
              {production.map((item, index) => (
                <motion.div
                  key={item.title}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className="p-6 bg-surface border border-border-dim rounded-2xl"
                >
                  <div className="flex items-start gap-4 mb-4">
                    <div className="w-12 h-12 rounded-xl bg-accent/10 border border-accent/30 flex items-center justify-center text-accent shrink-0">
                      {index === 0 ? <FolderOpen className="w-6 h-6" /> : index === 1 ? <ExternalLink className="w-6 h-6" /> : <Shield className="w-6 h-6" />}
                    </div>
                    <div>
                      <h3 className="font-mono text-lg font-bold text-white">{item.title}</h3>
                      <p className="font-sans text-sm text-gray-400 mt-1">{item.description}</p>
                    </div>
                  </div>
                  <ol className="space-y-2 pl-4">
                    {item.steps.map((step, stepIndex) => (
                      <li key={stepIndex} className="font-sans text-sm text-gray-300 list-decimal marker:text-accent/60 relative pl-2">
                        {step}
                      </li>
                    ))}
                  </ol>
                </motion.div>
              ))}
            </div>
          </ToggleSection>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 relative">
        <div className="max-w-2xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="p-8 md:p-12 bg-surface border border-border-dim rounded-3xl relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-accent/5 via-transparent to-warning/5 opacity-50" />
            <div className="relative z-10">
              <h2 className="font-mono text-3xl md:text-4xl font-bold text-white mb-4">
                Ready to <span className="text-accent">Start Detecting</span>?
              </h2>
              <p className="font-sans text-lg text-gray-400 mb-8">
                You have the guide. Now build something great.
              </p>
              <a
                href="https://github.com/yourusername/truthlens"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 bg-accent text-black font-mono text-lg tracking-widest uppercase rounded-xl hover:bg-accent/90 transition-all duration-300 shadow-[0_0_30px_rgba(0,255,128,0.4)]"
              >
                <Github className="w-6 h-6" />
                VIEW ON GITHUB
                <ArrowRight className="w-5 h-5" />
              </a>
            </div>
          </motion.div>
        </div>
      </section>
    </>
  );
}