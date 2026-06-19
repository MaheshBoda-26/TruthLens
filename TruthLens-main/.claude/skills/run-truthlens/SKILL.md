---
name: run-truthlens
description: run truthlens app, start dev servers, drive with puppeteer, screenshot
---

# TruthLens Run Skill

This skill launches the TruthLens web application (frontend + backend) in development mode and drives it with a Puppeteer script to take a screenshot of the main UI.

## Prerequisites

- Node.js (v20+ recommended)
- The project's dependencies installed:
  ```bash
  npm install
  ```
- Puppeteer is installed (the skill's driver script installs it if missing):
  ```bash
  npm install puppeteer@22.8.0
  ```

## Build

No build step is required for this skill; the app is run in development mode using Vite.

## Run (agent path)

To launch the app and capture a screenshot, run the driver script:
```bash
node .claude/skills/run-truthlens/driver.mjs [optional-image-path]
```
- The optional argument is a local path to an image file that will be uploaded via the UI before the screenshot. If omitted, the driver just loads the page.
- The screenshot is saved as `truthlens-screenshot.png` in the project root.

## Run (human path)

If you prefer to run the app manually, you can start the frontend and backend:
```bash
npm run dev:all
```
Then open <http://localhost:3000> in a browser.

## Gotchas

- The development server must be running on ports **3000** (frontend) and **3001** (backend). The driver will start them automatically if they are not already running.
- Puppeteer downloads a recent Chromium binary on first run; this may take a minute and requires about 150 MiB of disk space.
- On headless Linux containers you may need additional system libraries for Chromium. The driver installs the necessary `apt` packages (`libnss3`, `libatk-bridge2.0-0`, `libx11-6`, `libxcomposite1`, `libxrandr2`, `libasound2`) if they are missing.

## Troubleshooting

- **Chromium fails to launch** – ensure the required system libraries are installed. The driver attempts to install them automatically; you may need to run the script with sudo if your user cannot install packages.
- **Port 3000 or 3001 already in use** – stop any existing process or change the ports in `package.json` scripts.
- **Screenshot is blank** – verify the dev servers are up (you should see log lines `VITE ...` and `TruthLens ML server running on port 3001`). If they are not, run `npm run dev:all` manually and retry.
