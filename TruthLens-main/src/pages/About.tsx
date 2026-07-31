import React from 'react';
import { motion } from 'motion/react';
import { Shield, Zap, Eye, Code, GitBranch, Users, Globe, Award, CheckCircle, Terminal, Cpu, Database, Lock, Layers } from 'lucide-react';

const team = [
  {
    name: 'Mahesh Boda',
    role: 'Founder & Lead Developer',
    bio: 'Full-stack engineer specializing in AI/ML applications and forensic technology. Passionate about digital truth and open-source security.',
    socials: { github: 'https://github.com/MaheshBoda-26', linkedin: 'https://www.linkedin.com/in/maheshboda/', twitter: 'https://x.com/MaheshBoda26' },
  },
];

const milestones = [
  { date: '2024 Q1', title: 'Project Initiation', description: 'Conceptualized custom deepfake detection model pipeline', icon: GitBranch },
  { date: '2024 Q2', title: 'Core Engine v1', description: 'Built local heuristic analyzer with frequency domain analysis', icon: Cpu },
  { date: '2024 Q3', title: 'Model Training', description: 'Trained custom detection model on free deepfake datasets', icon: Layers },
  { date: '2024 Q4', title: 'Production Release', description: 'Launched v1.0 with React 19, Tailwind v4, full TypeScript', icon: Award },
  { date: '2025 Q1', title: 'Video Support', description: 'Added MP4/MOV frame extraction and temporal analysis', icon: Globe },
  { date: '2025 Q2', title: 'Open Source', description: 'Released under MIT license with full documentation', icon: Code },
];

const principles = [
  {
    icon: Shield,
    title: 'Accuracy First',
    description: 'We prioritize detection accuracy over speed. Every algorithm is validated against diverse deepfake datasets before deployment.',
  },
  {
    icon: Lock,
    title: 'Privacy by Design',
    description: 'Media never leaves your device for local analysis. Runs entirely on-device with our custom-trained model. No persistent storage.',
  },
  {
    icon: Code,
    title: 'Radical Transparency',
    description: 'Open-source core means anyone can audit our detection logic. No black-box models, no hidden thresholds.',
  },
  {
    icon: Users,
    title: 'Community Driven',
    description: 'Built for researchers, journalists, and platforms. We welcome contributions and adapt to emerging threat vectors.',
  },
];

const techDetails = [
  {
    category: 'Detection Engine',
    items: [
      'Frequency domain analysis (DCT/FFT)',
      'Noise pattern fingerprinting (PRNU)',
      'Compression artifact detection',
      'Facial landmark consistency checks',
      'Eye blink & micro-expression analysis',
      'Skin texture & lighting coherence',
    ],
  },
  {
    category: 'Custom Model Training',
    items: [
      'Trained on free deepfake datasets (FaceForensics++, Celeb-DF, DFDC)',
      'Data augmentation for robust generalization',
      'Transfer learning from pre-trained vision backbones',
      'Ensemble of multiple architectures',
      'Confidence calibration with temperature scaling',
      'Adversarial training for robustness',
    ],
  },
  {
    category: 'Metadata Forensics',
    items: [
      'Full EXIF/XMP/IPTC parsing',
      'Software signature database',
      'GPS timestamp validation',
      'Thumbnail consistency checks',
      'MakerNote deep inspection',
      'Quantization table analysis',
      'Color profile forensics',
      'Hex dump visualization',
    ],
  },
  {
    category: 'Architecture',
    items: [
      'React 19 + TypeScript strict',
      'Tailwind CSS v4 (CSS-first)',
      'Vite 6 for instant HMR',
      'Express.js API backend',
      'WebSocket for real-time updates',
      'Docker-ready deployment',
    ],
  },
];

export function About() {
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
              ABOUT TRUTHLENS
            </span>
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="font-mono text-4xl md:text-6xl font-bold text-white mb-6"
          >
            Building <span className="text-accent">Digital Truth</span> Infrastructure
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="font-sans text-lg md:text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed"
          >
            TruthLens is an open-source forensic platform that combines deterministic machine learning heuristics with large language model reasoning to detect AI-generated and manipulated media. We believe digital authenticity verification should be transparent, accessible, and scientifically rigorous.
          </motion.p>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="py-20 px-4 relative">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <h2 className="font-mono text-3xl md:text-4xl font-bold text-white mb-6">
                Our <span className="text-accent">Mission</span>
              </h2>
              <div className="space-y-4 font-sans text-gray-300 leading-relaxed">
                <p>
                  As generative AI advances, the line between authentic and synthetic media blurs. Deepfakes threaten journalism, democracy, financial systems, and personal reputation. TruthLens exists to restore trust in digital media through transparent, auditable detection technology.
                </p>
                <p>
                  We don't believe in security through obscurity. Our detection algorithms are open for peer review, our thresholds are configurable, and our results include detailed explanations — not just a binary verdict.
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="grid grid-cols-2 gap-4"
            >
              {[
                { icon: Shield, value: '94.2%', label: 'Accuracy on Test Sets' },
                { icon: Zap, value: '< 2s', label: 'Avg Analysis Time' },
                { icon: Lock, value: '100%', label: 'Local-First Option' },
                { icon: Code, value: 'MIT', label: 'Open Source License' },
              ].map((stat) => (
                <div key={stat.label} className="p-6 bg-surface border border-border-dim rounded-2xl hover:border-accent/30 transition-all duration-300">
                  <stat.icon className="w-8 h-8 text-accent mb-3" />
                  <div className="font-mono text-3xl font-bold text-white">{stat.value}</div>
                  <div className="font-sans text-sm text-gray-400 mt-1">{stat.label}</div>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* Core Principles */}
      <section className="py-20 px-4 relative bg-bg-main/50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              CORE PRINCIPLES
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              What We <span className="text-accent">Stand For</span>
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {principles.map((principle, index) => (
              <motion.div
                key={principle.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="p-6 bg-surface border border-border-dim rounded-2xl hover:border-accent/30 transition-all duration-500"
              >
                <div className="w-12 h-12 rounded-xl bg-accent/10 border border-accent/30 flex items-center justify-center mb-4">
                  <principle.icon className="w-6 h-6 text-accent" />
                </div>
                <h3 className="font-mono text-lg font-bold text-white mb-2">{principle.title}</h3>
                <p className="font-sans text-sm text-gray-400 leading-relaxed">{principle.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technical Deep Dive */}
      <section className="py-20 px-4 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              TECHNICAL ARCHITECTURE
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Under the <span className="text-accent">Hood</span>
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {techDetails.map((category, catIndex) => (
              <motion.div
                key={category.category}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: catIndex * 0.1 }}
                className="p-6 bg-surface border border-border-dim rounded-2xl hover:border-accent/30 transition-all duration-500"
              >
                <h3 className="font-mono text-sm tracking-widest uppercase text-accent mb-4 flex items-center gap-2">
                  <Terminal className="w-4 h-4" />
                  {category.category}
                </h3>
                <ul className="space-y-3">
                  {category.items.map((item, itemIndex) => (
                    <motion.li
                      key={item}
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.3, delay: catIndex * 0.1 + itemIndex * 0.05 }}
                      className="flex items-start gap-2 font-sans text-sm text-gray-300"
                    >
                      <CheckCircle className="w-4 h-4 text-accent/60 shrink-0 mt-0.5" />
                      <span>{item}</span>
                    </motion.li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline */}
      <section className="py-20 px-4 relative bg-bg-main/50">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              JOURNEY
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              Our <span className="text-accent">Timeline</span>
            </h2>
          </motion.div>

          <div className="relative">
            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-accent/50 via-transparent to-transparent" />
            {milestones.map((milestone, index) => (
              <motion.div
                key={milestone.date}
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="relative pl-20 pb-12 last:pb-0"
              >
                <div className="absolute left-0 top-2 flex items-center justify-center w-16 h-16">
                  <div className="w-4 h-4 rounded-full bg-accent border-4 border-bg-main z-10 relative" />
                </div>
                <div className="p-5 bg-surface border border-border-dim rounded-xl hover:border-accent/30 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-2">
                    <milestone.icon className="w-5 h-5 text-accent" />
                    <span className="font-mono text-xs tracking-widest uppercase text-accent">{milestone.date}</span>
                  </div>
                  <h3 className="font-mono text-lg font-bold text-white mb-1">{milestone.title}</h3>
                  <p className="font-sans text-sm text-gray-400">{milestone.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-20 px-4 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/30 text-accent font-mono text-xs tracking-widest uppercase mb-4">
              TEAM
            </span>
            <h2 className="font-mono text-4xl md:text-5xl font-bold text-white mb-4">
              The People Behind <span className="text-accent">TruthLens</span>
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {team.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="p-6 bg-surface border border-border-dim rounded-2xl hover:border-accent/30 transition-all duration-500 text-center"
              >
                <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-accent/20 to-purple-500/20 border border-accent/30 flex items-center justify-center">
                  <span className="font-mono text-3xl font-bold text-accent">
                    {member.name.split(' ').map(n => n[0]).join('')}
                  </span>
                </div>
                <h3 className="font-mono text-xl font-bold text-white mb-1">{member.name}</h3>
                <p className="font-sans text-sm text-accent mb-4">{member.role}</p>
                <p className="font-sans text-sm text-gray-400 mb-4 leading-relaxed">{member.bio}</p>
                <div className="flex justify-center gap-3">
                  <a href={member.socials.github} className="w-9 h-9 rounded-lg bg-surface border border-border-dim flex items-center justify-center text-gray-400 hover:text-accent hover:border-accent/50 transition-all" aria-label="GitHub">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/></svg>
                  </a>
                  <a href={member.socials.linkedin} className="w-9 h-9 rounded-lg bg-surface border border-border-dim flex items-center justify-center text-gray-400 hover:text-accent hover:border-accent/50 transition-all" aria-label="LinkedIn">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                  </a>
                  <a href={member.socials.twitter} className="w-9 h-9 rounded-lg bg-surface border border-border-dim flex items-center justify-center text-gray-400 hover:text-accent hover:border-accent/50 transition-all" aria-label="Twitter">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z"/></svg>
                  </a>
                </div>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="text-center mt-12"
          >
            <p className="font-sans text-gray-400 mb-4">Want to contribute? We welcome researchers, engineers, and domain experts.</p>
            <a href="#" className="inline-flex items-center gap-2 px-6 py-3 bg-accent text-black font-mono text-sm tracking-widest uppercase rounded-xl hover:bg-accent/90 transition-all duration-300 shadow-[0_0_20px_rgba(0,255,128,0.3)]">
              <GitBranch className="w-5 h-5" />
              JOIN ON GITHUB
            </a>
          </motion.div>
        </div>
      </section>
    </>
  );
}