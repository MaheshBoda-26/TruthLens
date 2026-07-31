import React from 'react';
import { motion } from 'motion/react';
import { Shield, Zap, Eye, Code, Lock, Globe, Cpu, Database, Layers, Terminal, CheckCircle, ArrowRight, ExternalLink, Search, Film, Image, FileText, Settings, Sliders, BarChart, AlertTriangle, Clock, Microscope, Network, Key, Github } from 'lucide-react';

const detectionFeatures = [
  {
    category: 'Image Analysis',
    icon: Image,
    color: 'text-accent',
    bgColor: 'bg-accent/10',
    borderColor: 'border-accent/30',
    features: [
      { title: 'Frequency Domain Analysis', desc: 'DCT/FFT-based detection of GAN artifacts and upsampling traces in spectral domain' },
      { title: 'Noise Fingerprinting (PRNU)', desc: 'Camera sensor pattern noise analysis to detect device inconsistencies and splicing' },
      { title: 'Compression Artifact Detection', desc: 'Quantization table analysis and double-compression detection for JPEG/WEBP' },
      { title: 'Facial Landmark Consistency', desc: '68-point landmark tracking with geometric constraint validation across frames' },
      { title: 'Eye Blink & Micro-expressions', desc: 'Temporal analysis of natural blink rates and involuntary facial micro-movements' },
      { title: 'Skin Texture & Lighting', desc: 'Subsurface scattering simulation consistency and physically-based lighting validation' },
      { title: 'Edge & Boundary Analysis', desc: 'Laplacian pyramid decomposition for splice boundary and clone detection' },
      { title: 'Color Space Anomalies', desc: 'Color constancy violations and illuminant estimation inconsistencies' },
    ],
  },
  {
    category: 'Video Analysis',
    icon: Film,
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/30',
    features: [
      { title: 'Frame Extraction Pipeline', desc: 'FFmpeg-based keyframe sampling with configurable interval and scene detection' },
      { title: 'Temporal Consistency', desc: 'Optical flow analysis for motion coherence and frame interpolation artifacts' },
      { title: 'Flicker & Jitter Detection', desc: 'High-frequency temporal artifacts from GAN frame generation inconsistencies' },
      { title: 'Audio-Visual Sync', desc: 'Lip-sync verification and acoustic-visual correlation analysis' },
      { title: 'Scene Transition Analysis', desc: 'Cut detection and transition artifact examination for manipulation boundaries' },
      { title: 'Resolution Consistency', desc: 'Per-frame resolution and quality metric tracking for upscaling artifacts' },
    ],
  },
  {
    category: 'Metadata Forensics',
    icon: FileText,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
    features: [
      { title: 'Full EXIF/XMP/IPTC Parsing', desc: 'Complete metadata extraction with MakerNote deep inspection for vendor tags' },
      { title: 'Software Signature Database', desc: '1000+ editing tool signatures (Photoshop, GIMP, AI generators, mobile apps)' },
      { title: 'GPS & Timestamp Validation', desc: 'Geolocation consistency checks and temporal plausibility verification' },
      { title: 'Thumbnail Consistency', desc: 'Embedded thumbnail vs. main image content hash comparison' },
      { title: 'Quantization Table Analysis', desc: 'JPEG quantization matrix fingerprinting for source device identification' },
      { title: 'Color Profile Forensics', desc: 'ICC profile analysis for color management software detection' },
      { title: 'Hex Dump Visualization', desc: 'Raw byte-level inspection with annotated structure parsing' },
    ],
  },
];

const llmFeatures = [
  {
    title: 'Structured Forensic Prompts',
    desc: 'Chain-of-thought reasoning with domain-specific forensic methodology embedded in prompts',
    icon: Search,
  },
  {
    title: 'Multi-Modal Analysis',
    desc: 'Simultaneous image + text + ML prediction input for holistic Gemini 1.5 Pro reasoning',
    icon: Layers,
  },
  {
    title: 'Confidence Calibration',
    desc: 'Temperature-scaled probability outputs with uncertainty quantification per verdict category',
    icon: BarChart,
  },
  {
    title: 'Adversarial Hardening',
    desc: 'Prompt injection defenses, output format enforcement, and hallucination detection checks',
    icon: Shield,
  },
  {
    title: 'Explanation Generation',
    desc: 'Human-readable forensic reports with specific evidence citations and confidence intervals',
    icon: FileText,
  },
  {
    title: 'Fallback Provider Support',
    desc: 'OpenRouter integration for automatic failover to Claude, GPT-4, or open-source models',
    icon: Network,
  },
];

const architectureFeatures = [
  {
    title: 'React 19 + TypeScript Strict',
    desc: 'Latest React with concurrent features, strict type checking, and zero `any` tolerance',
    icon: Code,
  },
  {
    title: 'Tailwind CSS v4 (CSS-First)',
    desc: 'Native CSS cascade layers, `@theme` design tokens, zero-config JIT compilation',
    icon: Settings,
  },
  {
    title: 'Vite 6 Build System',
    desc: 'Instant HMR, optimized production builds, SWC transpilation, WASM-ready',
    icon: Zap,
  },
  {
    title: 'Express.js API Backend',
    desc: 'RESTful endpoints with middleware pipeline, rate limiting, and request validation',
    icon: Terminal,
  },
  {
    title: 'Local-First Architecture',
    desc: 'Core ML heuristics run client-side; cloud LLM is optional enhancement only',
    icon: Lock,
  },
  {
    title: 'WebSocket Real-Time Updates',
    desc: 'Live analysis progress streaming with progress events and incremental results',
    icon: Globe,
  },
  {
    title: 'Docker-Ready Deployment',
    desc: 'Multi-stage Dockerfiles, docker-compose orchestration, health checks included',
    icon: Database,
  },
  {
    title: 'Accessibility (WCAG 2.1 AA)',
    desc: 'Semantic HTML, ARIA labels, keyboard navigation, focus management, color contrast',
    icon: Key,
  },
];

const verdictDetails = [
  {
    verdict: 'LIKELY DEEPFAKE',
    range: 'Score > 70%',
    color: 'text-danger',
    bgColor: 'bg-[#2a0a0a]',
    borderColor: 'border-danger/50',
    description: 'Strong evidence of AI generation or manipulation. Multiple detection vectors triggered.',
    indicators: ['Frequency anomalies', 'Inconsistent noise patterns', 'Facial geometry violations', 'Metadata mismatches'],
  },
  {
    verdict: 'POSSIBLY DEEPFAKE',
    range: 'Score 40-70%',
    color: 'text-warning',
    bgColor: 'bg-[#2a1a00]',
    borderColor: 'border-warning/50',
    description: 'Suspicious indicators present but not conclusive. Manual review recommended.',
    indicators: ['Minor frequency artifacts', 'Partial metadata inconsistency', 'Subtle lighting anomalies'],
  },
  {
    verdict: 'AI EDITED',
    range: 'Score 30-60%',
    color: 'text-purple-400',
    bgColor: 'bg-[#1a0a2a]',
    borderColor: 'border-purple-500/50',
    description: 'Authentic base media with localized AI modifications (inpainting, outpainting, filters).',
    indicators: ['Localized manipulation', 'Boundary artifacts', 'Tool signature detected'],
  },
  {
    verdict: 'LIKELY AUTHENTIC',
    range: 'Score < 40%',
    color: 'text-safe',
    bgColor: 'bg-[#0a1a0a]',
    borderColor: 'border-safe/50',
    description: 'No significant manipulation detected. Media appears consistent with natural capture.',
    indicators: ['Natural noise patterns', 'Consistent metadata', 'Valid device signatures'],
  },
  {
    verdict: 'INCONCLUSIVE',
    range: 'Insufficient data',
    color: 'text-gray-400',
    bgColor: 'bg-[#1a1a1a]',
    borderColor: 'border-gray-600/50',
    description: 'Analysis could not reach confident determination. Low quality, unsupported format, or conflicting signals.',
    indicators: ['Low resolution', 'Heavy compression', 'Conflicting detector outputs'],
  },
];

export function Features() {
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
              FEATURES & TECHNOLOGY
            </span>
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="font-mono text-4xl md:text-6xl font-bold text-white mb-6"
          >
            Complete <span className="text-accent">Capability Matrix</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="font-sans text-lg md:text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed"
          >
            Every detection vector, analysis mode, and architectural decision documented.
            Built for researchers who need to know exactly how the verdict is reached.
          </motion.p>
        </div>
      </section>

      {/* Detection Capabilities */}
      <section className="py-20 px-4 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              DETECTION ENGINE
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Multi-Vector <span className="text-accent">Analysis</span>
            </h2>
          </motion.div>

          <div className="space-y-12">
            {detectionFeatures.map((category, catIndex) => (
              <motion.div
                key={category.category}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: catIndex * 0.1 }}
              >
                <div className="flex items-center gap-4 mb-8">
                  <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${category.bgColor} ${category.borderColor}`}>
                    <category.icon className={`w-7 h-7 ${category.color}`} />
                  </div>
                  <div>
                    <h3 className="font-mono text-2xl font-bold text-white">{category.category}</h3>
                    <p className="font-sans text-gray-400 mt-1">{category.features.length} detection vectors</p>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  {category.features.map((feature, featIndex) => (
                    <motion.div
                      key={feature.title}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.4, delay: catIndex * 0.1 + featIndex * 0.05 }}
                      className="p-5 bg-surface border border-border-dim rounded-xl hover:border-accent/30 hover:bg-surface/50 transition-all duration-300 group"
                    >
                      <div className="flex items-start gap-3">
                        <CheckCircle className={`w-5 h-5 shrink-0 mt-0.5 ${category.color}`} />
                        <div>
                          <h4 className="font-mono text-sm font-bold text-white group-hover:text-accent transition-colors">{feature.title}</h4>
                          <p className="font-sans text-sm text-gray-400 mt-1">{feature.desc}</p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* LLM Pipeline */}
      <section className="py-20 px-4 relative bg-bg-main/50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 font-mono text-xs tracking-widest uppercase mb-4">
              LLM REASONING
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Gemini <span className="text-purple-400">Forensic</span> Pipeline
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {llmFeatures.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="p-6 bg-surface border border-border-dim rounded-2xl hover:border-purple-500/30 hover:bg-purple-500/5 transition-all duration-500"
              >
                <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center mb-4 text-purple-400">
                  <feature.icon className="w-6 h-6" />
                </div>
                <h3 className="font-mono text-lg font-bold text-white mb-2">{feature.title}</h3>
                <p className="font-sans text-sm text-gray-400 leading-relaxed">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section className="py-20 px-4 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              ARCHITECTURE
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Technical <span className="text-accent">Stack</span>
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {architectureFeatures.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.05 }}
                className="p-5 bg-surface border border-border-dim rounded-xl hover:border-accent/30 hover:bg-surface/50 transition-all duration-300 group"
              >
                <div className="w-10 h-10 rounded-lg bg-accent/10 border border-accent/30 flex items-center justify-center mb-3 text-accent group-hover:scale-110 transition-transform">
                  <feature.icon className="w-5 h-5" />
                </div>
                <h4 className="font-mono text-sm font-bold text-white mb-1 group-hover:text-accent transition-colors">{feature.title}</h4>
                <p className="font-sans text-[12px] text-gray-400">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Verdict System */}
      <section className="py-20 px-4 relative bg-bg-main/50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-warning/10 border border-warning/30 text-warning font-mono text-xs tracking-widest uppercase mb-4">
              VERDICT SYSTEM
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Five-Tier <span className="text-warning">Classification</span>
            </h2>
          </motion.div>

          <div className="space-y-6">
            {verdictDetails.map((verdict, index) => (
              <motion.div
                key={verdict.verdict}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className={`p-6 bg-surface border rounded-2xl flex flex-col md:flex-row md:items-center gap-6 ${verdict.bgColor} ${verdict.borderColor}`}
              >
                <div className="flex items-center gap-4 md:w-48 md:flex-none">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-mono text-xl font-bold ${verdict.color} ${verdict.bgColor} ${verdict.borderColor}`}>
                    {index + 1}
                  </div>
                  <div>
                    <div className={`font-mono text-lg font-bold ${verdict.color}`}>{verdict.verdict}</div>
                    <div className="font-sans text-xs text-gray-400">{verdict.range}</div>
                  </div>
                </div>
                <div className="flex-1 font-sans text-gray-300 leading-relaxed">{verdict.description}</div>
                <div className="flex flex-wrap gap-2 md:w-64 md:flex-none">
                  {verdict.indicators.map((indicator, i) => (
                    <span key={i} className="px-3 py-1 text-[11px] font-mono tracking-widest uppercase rounded-full bg-bg-main/50 border border-border-dim text-gray-400">
                      {indicator}
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Supported Formats */}
      <section className="py-20 px-4 relative">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              SUPPORTED FORMATS
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Input <span className="text-accent">Compatibility</span>
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="p-8 bg-surface border border-border-dim rounded-2xl"
            >
              <div className="flex items-center gap-4 mb-6">
                <div className="w-14 h-14 rounded-xl bg-accent/10 border border-accent/30 flex items-center justify-center text-accent">
                  <Image className="w-7 h-7" />
                </div>
                <div>
                  <h3 className="font-mono text-xl font-bold text-white">Images</h3>
                  <p className="font-sans text-sm text-gray-400">Static image analysis</p>
                </div>
              </div>
              <ul className="space-y-3">
                {['JPEG (.jpg, .jpeg)', 'PNG (.png)', 'WEBP (.webp)'].map((format) => (
                  <li key={format} className="flex items-center gap-3 font-sans text-sm text-gray-300">
                    <CheckCircle className="w-4 h-4 text-accent shrink-0" />
                    <span>{format}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-6 p-4 bg-bg-main/50 border border-border-dim rounded-xl font-sans text-sm text-gray-400">
                <strong className="text-white">Max file size:</strong> 10MB<br />
                <strong className="text-white">Min dimensions:</strong> 64x64px<br />
                <strong className="text-white">Color spaces:</strong> sRGB, Adobe RGB, P3
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="p-8 bg-surface border border-border-dim rounded-2xl"
            >
              <div className="flex items-center gap-4 mb-6">
                <div className="w-14 h-14 rounded-xl bg-warning/10 border border-warning/30 flex items-center justify-center text-warning">
                  <Film className="w-7 h-7" />
                </div>
                <div>
                  <h3 className="font-mono text-xl font-bold text-white">Videos</h3>
                  <p className="font-sans text-sm text-gray-400">Temporal frame analysis</p>
                </div>
              </div>
              <ul className="space-y-3">
                {['MP4 (.mp4)', 'QuickTime (.mov)'].map((format) => (
                  <li key={format} className="flex items-center gap-3 font-sans text-sm text-gray-300">
                    <CheckCircle className="w-4 h-4 text-warning shrink-0" />
                    <span>{format}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-6 p-4 bg-bg-main/50 border border-border-dim rounded-xl font-sans text-sm text-gray-400">
                <strong className="text-white">Max file size:</strong> 10MB<br />
                <strong className="text-white">Max duration:</strong> 60 seconds<br />
                <strong className="text-white">Frame sampling:</strong> Configurable (default: 1fps)<br />
                <strong className="text-white">Requires:</strong> FFmpeg installed
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* API Reference */}
      <section className="py-20 px-4 relative bg-bg-main/50">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              API REFERENCE
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              REST <span className="text-accent">Endpoints</span>
            </h2>
          </motion.div>

          <div className="space-y-4">
            {[
              { method: 'POST', path: '/api/analyze', desc: 'Analyze image/video for deepfake indicators. Accepts base64-encoded media.', auth: false },
              { method: 'GET', path: '/api/health', desc: 'Health check endpoint for load balancers and monitoring.', auth: false },
              { method: 'WS', path: '/api/ws/analyze', desc: 'WebSocket for real-time analysis progress streaming.', auth: false },
            ].map((endpoint) => (
              <motion.div
                key={endpoint.path}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="p-5 bg-surface border border-border-dim rounded-xl hover:border-accent/30 transition-all"
              >
                <div className="flex items-center gap-4 mb-3 flex-wrap">
                  <span className={`px-3 py-1 rounded-lg font-mono text-xs font-bold ${endpoint.method === 'POST' ? 'bg-accent/20 text-accent' : endpoint.method === 'GET' ? 'bg-safe/20 text-safe' : 'bg-purple-500/20 text-purple-400'}`}>
                    {endpoint.method}
                  </span>
                  <code className="font-mono text-sm text-white bg-bg-main/50 px-3 py-1 rounded">{endpoint.path}</code>
                  {endpoint.auth && <span className="px-2 py-0.5 text-[10px] font-mono tracking-widest uppercase bg-warning/20 text-warning rounded">AUTH REQUIRED</span>}
                </div>
                <p className="font-sans text-sm text-gray-400">{endpoint.desc}</p>
              </motion.div>
            ))}
          </div>
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
            <div className="absolute inset-0 bg-gradient-to-r from-accent/5 via-transparent to-purple-500/5 opacity-50" />
            <div className="relative z-10">
              <h2 className="font-mono text-3xl md:text-4xl font-bold text-white mb-4">
                Explore the <span className="text-accent">Source Code</span>
              </h2>
              <p className="font-sans text-lg text-gray-400 mb-8">
                Every algorithm, every threshold, every prompt — open for inspection and contribution.
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