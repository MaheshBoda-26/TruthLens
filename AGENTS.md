# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

---

## Common Development Commands

- **Install dependencies**: `npm install`
- **Run the frontend (Vite dev server)**: `npm run dev`
- **Run the frontend **and** the analysis API together**: `npm run dev:all`
- **Start the Express analysis API only**: `npm run server`
- **Build the production frontend**: `npm run build`
- **Preview a production build**: `npm run preview`
- **Clean built assets**: `npm run clean`
- **Type‑check / lint the TypeScript code**: `npm run lint`

> There are currently no test scripts defined in `package.json`. If tests are added later, follow the standard `npm test` convention.

---

## High‑Level Architecture

- **Frontend** – A React application bundled with Vite (`src/`). The entry point is `src/main.tsx` and the UI is composed of reusable components under `src/components/` (e.g., `UploadZone`, `AnalyzeButton`, `ResultsPanel`). The app communicates with the backend via HTTP calls to `/api/*` endpoints.

- **Backend API** – An Express server written in TypeScript (`server.ts`). It provides two primary routes:
  1. `POST /api/analyze` – Accepts a base‑64‑encoded image, runs a heuristic deep‑fake detection algorithm (`analyzeImage`) and returns a JSON verdict.
  2. `GET /api/health` – Simple health‑check.

- **Hybrid Analysis Pipeline** – The frontend’s `analyzeWithGemini` utility performs a two‑step analysis:
  1. **Local ML model** – Calls the Express `/api/analyze` endpoint for a fast, rule‑based prediction.
  2. **Gemini LLM** – Sends the image (as a data URL) and the ML prediction to Gemini via OpenRouter for a detailed forensic explanation.
  The results are merged (`mergeWithMLPrediction`) to produce the final `AnalysisResult` displayed to the user.

- **Utilities** – Helper functions for video frame extraction (`extractVideoFrame.ts`) and metadata extraction (`extractMetadata.ts`).

- **Configuration** – Environment variables are loaded from `.env.local`. The critical variable is `VITE_GEMINI_API_KEY`, which must be set for Gemini calls.

- **Build & Deploy** – In production the Express server serves the compiled Vite assets from `dist/`. The `build` script creates these assets, and `preview` can be used to locally test the production bundle.

---

## Important Project Files

- `README.md` – Overview, prerequisites, and quick‑start instructions.
- `package.json` – Scripts, dependencies, and lint command.
- `src/` – Frontend source code.
- `server.ts` – Backend API implementation and deep‑fake detection heuristics.
- `src/utils/analyzeWithGemini.ts` – Orchestrates the hybrid ML + Gemini workflow.

---

## Cursor / Copilot Rules (if any)

No `.cursor/` or `.github/copilot‑instructions.md` files were found, so there are no custom Copilot or Cursor rules to surface.

---

## Tips for Codex

- Prefer using the **npm scripts** listed above for any development task; they encapsulate the required environment settings.
- When inspecting the detection logic, focus on `server.ts` (heuristic analysis) and `src/utils/analyzeWithGemini.ts` (LLM orchestration).
- The frontend expects the backend to be reachable at `http://localhost:3001` when running `npm run dev:all`.
- Remember to keep the `VITE_GEMINI_API_KEY` secret; it is read from `.env.local`.
