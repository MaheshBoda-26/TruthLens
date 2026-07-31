import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { Search, Shield, Zap, Eye, Code, ArrowRight, CheckCircle, Terminal, Cpu, Globe, Lock } from 'lucide-react';

const features = [
  {
    icon: Shield,
    title: 'Custom Trained Detection Model',
    description: 'Trained on free deepfake datasets for forensic-grade detection accuracy.',
    color: 'text-accent',
    bgColor: 'bg-accent/10',
    borderColor: 'border-accent/30',
  },
  {
    icon: Zap,
    title: 'Real-time Results',
    description: 'Sub-second analysis for images. Video frame extraction with parallel processing.',
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/30',
  },
  {
    icon: Eye,
    title: 'Metadata Forensics',
    description: 'EXIF analysis, software signatures, GPS stripping detection, and manipulation artifacts.',
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
  },
  {
    icon: Code,
    title: 'Transparent & Open',
    description: 'Open-source core with verifiable detection logic. No black-box mystery.',
    color: 'text-safe',
    bgColor: 'bg-safe/10',
    borderColor: 'border-safe/30',
  },
];

const stats = [
  { value: '94.2%', label: 'Detection Accuracy', icon: CheckCircle },
  { value: '< 2s', label: 'Avg Analysis Time', icon: Zap },
  { value: '4+', label: 'Verdict Categories', icon: Terminal },
  { value: '100%', label: 'Client-side Option', icon: Lock },
];

export function Home() {
  return (
    <>
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center px-4">
        <div className="max-w-7xl mx-auto w-full text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="mb-10"
          >
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase">
              <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
              v1.0.0 — Production Ready
            </span>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="mb-8"
          >
            <h1 className="font-mono text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight leading-[1.05]">
              <span className="text-white">Truth</span>
              <span className="text-accent">Lens</span>
            </h1>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="mb-10 max-w-3xl mx-auto"
          >
            <p className="font-sans text-lg md:text-xl text-gray-300 leading-relaxed">
              Forensic-grade deepfake detection powered by a custom-trained model on free datasets.
              Analyze images and videos with confidence — see through the fake.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20"
          >
            <Link
              to="/setup"
              className="group px-8 py-4 bg-accent text-black font-mono text-lg tracking-widest uppercase rounded-xl hover:bg-accent/90 transition-all duration-300 shadow-[0_0_30px_rgba(0,255,128,0.4)] hover:shadow-[0_0_50px_rgba(0,255,128,0.6)] flex items-center gap-3"
            >
              <Search className="w-6 h-6" />
              LAUNCH ANALYZER
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/about"
              className="px-8 py-4 bg-surface text-white font-mono text-lg tracking-widest uppercase rounded-xl border border-border-dim hover:border-accent/50 hover:bg-surface/50 transition-all duration-300 flex items-center gap-3"
            >
              <Code className="w-6 h-6" />
              VIEW DOCUMENTATION
            </Link>
          </motion.div>

          {/* Trust Indicators */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="flex flex-wrap items-center justify-center gap-8 text-sm font-mono text-gray-500 uppercase tracking-widest"
          >
            <span className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-accent" />
              Open Source
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-accent" />
              Local Processing
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-accent" />
              No Data Stored
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-accent" />
              MIT Licensed
            </span>
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 1 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-gray-500 font-mono text-xs tracking-widest uppercase"
        >
          <span>SCROLL</span>
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-1 h-6 border border-accent/50 rounded-full flex items-end justify-center p-1"
          >
            <div className="w-1 h-2 bg-accent rounded-full" />
          </motion.div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4 relative">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="group p-6 bg-surface border border-border-dim rounded-2xl hover:border-accent/30 transition-all duration-500"
              >
                <div className="flex items-center gap-3 mb-3">
                  <stat.icon className={`w-6 h-6 ${index === 0 ? 'text-accent' : index === 1 ? 'text-warning' : index === 2 ? 'text-purple-400' : 'text-safe'}`} />
                </div>
                <div className="font-mono text-3xl md:text-4xl font-bold text-white mb-1">{stat.value}</div>
                <div className="font-sans text-sm text-gray-400">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              CORE CAPABILITIES
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Built for <span className="text-accent">Forensic Precision</span>
            </h2>
            <p className="font-sans text-lg text-gray-400 max-w-2xl mx-auto">
              Every feature designed to give you confidence in your media authenticity decisions.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className={`group p-8 bg-surface border rounded-2xl transition-all duration-500 hover:scale-[1.02] ${feature.bgColor} ${feature.borderColor}`}
              >
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center mb-6 transition-all duration-500 group-hover:scale-110 ${feature.bgColor} ${feature.borderColor}`}>
                  <feature.icon className={`w-7 h-7 ${feature.color}`} />
                </div>
                <h3 className="font-mono text-xl font-bold text-white mb-3">{feature.title}</h3>
                <p className="font-sans text-gray-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 px-4 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              HYBRID PIPELINE
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              How <span className="text-accent">TruthLens Works</span>
            </h2>
            <p className="font-sans text-lg text-gray-400 max-w-2xl mx-auto">
              A two-stage analysis combining deterministic heuristics with a custom-trained deepfake detection model for unmatched accuracy.
            </p>
          </motion.div>

          <div className="relative">
            {/* Connecting line */}
            <div className="hidden lg:block absolute left-1/2 top-16 bottom-16 w-0.5 bg-gradient-to-b from-accent/50 via-accent/10 to-transparent -translate-x-1/2" />

            <div className="space-y-12">
              {[
                {
                  step: '01',
                  title: 'Upload & Preprocess',
                  description: 'Drag & drop images (JPG, PNG, WEBP) or videos (MP4, MOV). Automatic format validation and frame extraction for video.',
                  icon: Globe,
                  color: 'text-accent',
                  bgColor: 'bg-accent/10',
                  borderColor: 'border-accent/30',
                  side: 'left',
                },
                {
                  step: '02',
                  title: 'Local ML Heuristics',
                  description: 'Fast rule-based analysis runs on-device: frequency analysis, noise patterns, compression artifacts, and facial consistency checks.',
                  icon: Cpu,
                  color: 'text-warning',
                  bgColor: 'bg-warning/10',
                  borderColor: 'border-warning/30',
                  side: 'right',
                },
                {
                  step: '03',
                  title: 'Custom Model Inference',
                  description: 'Image analyzed by our custom-trained deepfake detection model trained on diverse free datasets for robust generalization.',
                  icon: Shield,
                  color: 'text-purple-400',
                  bgColor: 'bg-purple-500/10',
                  borderColor: 'border-purple-500/30',
                  side: 'left',
                },
                {
                  step: '04',
                  title: 'Unified Verdict',
                  description: 'Results merged into a final verdict (LIKELY DEEPFAKE → LIKELY AUTHENTIC) with confidence scores, manipulation metrics, and actionable recommendations.',
                  icon: Terminal,
                  color: 'text-safe',
                  bgColor: 'bg-safe/10',
                  borderColor: 'border-safe/30',
                  side: 'right',
                },
              ].map((item, index) => (
                <motion.div
                  key={item.step}
                  initial={{ opacity: 0, x: item.side === 'left' ? -50 : 50 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.15 }}
                  className={`flex ${item.side === 'right' ? 'flex-row-reverse' : ''} gap-8 items-start relative`}
                >
                  <div className={`w-1/2 lg:w-[45%] pr-8 lg:pr-12 ${item.side === 'right' ? 'pl-8 lg:pl-12 text-right' : ''} relative`}>
                    <div className={`p-6 bg-surface border rounded-2xl ${item.bgColor} ${item.borderColor} relative`}>
                      <div className="flex items-center gap-3 mb-4">
                        <span className={`w-10 h-10 rounded-lg flex items-center justify-center font-mono text-xl font-bold ${item.color} ${item.bgColor} ${item.borderColor}`}>
                          {item.step}
                        </span>
                        <div>
                          <item.icon className={`w-6 h-6 ${item.color}`} />
                        </div>
                      </div>
                      <h3 className="font-mono text-xl font-bold text-white mb-2">{item.title}</h3>
                      <p className="font-sans text-gray-400 leading-relaxed">{item.description}</p>
                    </div>
                  </div>
                  <div className="w-1/2 lg:w-[10%] flex items-center justify-center">
                    <div className="w-4 h-4 rounded-full border-2 border-accent/30 relative z-10">
                      <div className="w-2 h-2 rounded-full bg-accent absolute inset-0 m-auto" />
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 relative">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="p-10 md:p-16 bg-surface border border-border-dim rounded-3xl relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-accent/5 via-transparent to-purple-500/5 opacity-50" />
            <div className="relative z-10">
              <h2 className="font-mono text-3xl md:text-4xl font-bold text-white mb-4">
                Ready to <span className="text-accent">Detect Deepfakes</span>?
              </h2>
              <p className="font-sans text-lg text-gray-400 mb-8 max-w-xl mx-auto">
                Start analyzing media in seconds. No account required, runs entirely locally with our custom-trained model.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link
                  to="/setup"
                  className="group px-8 py-4 bg-accent text-black font-mono text-lg tracking-widest uppercase rounded-xl hover:bg-accent/90 transition-all duration-300 shadow-[0_0_30px_rgba(0,255,128,0.4)] hover:shadow-[0_0_50px_rgba(0,255,128,0.6)] flex items-center gap-3"
                >
                  <Search className="w-6 h-6" />
                  GET STARTED
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </Link>
                <Link
                  to="/about"
                  className="px-8 py-4 bg-transparent text-white font-mono text-lg tracking-widest uppercase rounded-xl border border-border-dim hover:border-accent/50 hover:bg-surface/50 transition-all duration-300 flex items-center gap-3"
                >
                  <Code className="w-6 h-6" />
                  READ DOCS
                </Link>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </>
  );
}