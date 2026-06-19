<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# TruthLens AI

TruthLens AI is a local deepfake and media authenticity analysis tool built with React, Vite, and a small Express analysis API.

## Run Locally

**Prerequisites:** Node.js

Run these commands from the app directory:

```bash
cd TruthLens/TruthLens-main
```

1. Install dependencies:
   ```bash
   npm install
   ```
2. Create a local environment file:
   ```bash
   cp .env.example .env.local
   ```
3. Add your OpenRouter key to `.env.local`:
   ```bash
   VITE_GEMINI_API_KEY="sk-or-v1-..."
   ```
4. Start the frontend only:
   ```bash
   npm run dev
   ```
5. Start the frontend and analysis API together:
   ```bash
   npm run dev:all
   ```

The Vite app runs on `http://localhost:3000`. The Express analysis API runs on `http://localhost:3001` when using `npm run dev:all`.
