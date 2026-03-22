import { extractVideoFrame } from './extractVideoFrame';

export interface AnalysisResult {
  verdict: "LIKELY DEEPFAKE" | "POSSIBLY DEEPFAKE" | "AI EDITED" | "LIKELY AUTHENTIC" | "INCONCLUSIVE";
  confidence: number;
  risk_score: number;
  manipulation_metrics?: {
    full_ai_generation_likelihood: number;
    partial_ai_edit_likelihood: number;
  };
  summary: string;
  signals: {
    face_analysis: { score: number; findings: string[] };
    lighting_shadows: { score: number; findings: string[] };
    texture_artifacts: { score: number; findings: string[] };
    edge_consistency: { score: number; findings: string[] };
    background_coherence: { score: number; findings: string[] };
  };
  recommendation: string;
  metadata_flags?: string[];
}

export async function analyzeWithGemini(file: File, mediaType: 'image' | 'video'): Promise<AnalysisResult> {
  let base64Image = '';

  if (mediaType === 'image') {
    base64Image = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  } else if (mediaType === 'video') {
    base64Image = await extractVideoFrame(file);
  }

  const SYSTEM_PROMPT = `
You are an expert deepfake and AI-generated media forensics analyst.
Analyze the provided image carefully to differentiate between fully AI-generated images, partial AI edits (such as generative fill, inpainting, outpainting, or AI filters), and authentic images.
Look specifically for blending errors, mismatched lighting, localized texture blurring, or unnatural boundaries that indicate partial AI edits.
For fully AI-generated images, look for global inconsistencies, structural impossibilities, and uniform AI textures.
You MUST return ONLY a valid JSON object. No explanation, no markdown, no code fences.
Use exactly this structure:
{
  "verdict": "LIKELY DEEPFAKE" or "POSSIBLY DEEPFAKE" or "AI EDITED" or "LIKELY AUTHENTIC" or "INCONCLUSIVE",
  "confidence": <integer 0-100>,
  "risk_score": <integer 0-100>,
  "manipulation_metrics": {
    "full_ai_generation_likelihood": <integer 0-100>,
    "partial_ai_edit_likelihood": <integer 0-100>
  },
  "summary": "<2-3 sentence plain English explanation. Explicitly state if the image is fully generated or partially edited, and point out the specific edited regions if applicable.>",
  "signals": {
    "face_analysis": { "score": <0-100>, "findings": ["<finding>", "<finding>"] },
    "lighting_shadows": { "score": <0-100>, "findings": ["<finding>", "<finding>"] },
    "texture_artifacts": { "score": <0-100>, "findings": ["<finding>", "<finding>"] },
    "edge_consistency": { "score": <0-100>, "findings": ["<finding>", "<finding>"] },
    "background_coherence": { "score": <0-100>, "findings": ["<finding>", "<finding>"] }
  },
  "recommendation": "<one sentence on what the user should do>"
}
Score meaning: 0 = no suspicion, 100 = highly suspicious.
If no face is present, still analyze for AI generation artifacts in textures, edges, and background.
If the image is fully AI-generated, use "LIKELY DEEPFAKE" or "POSSIBLY DEEPFAKE" and set full_ai_generation_likelihood high.
If the image appears mostly real but contains localized AI edits (like generative fill), use the "AI EDITED" verdict and set partial_ai_edit_likelihood high.
Be honest — if the image looks authentic, say so.
`;

  // Use the API key from process.env (injected by Vite define)
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("API_FAILED");
  }

  const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

  const body = {
    contents: [{
      parts: [
        {
          inline_data: {
            mime_type: "image/jpeg",
            data: base64Image
          }
        },
        {
          text: SYSTEM_PROMPT
        }
      ]
    }],
    generationConfig: {
      temperature: 0.1,
      maxOutputTokens: 1000
    }
  };

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      throw new Error("API_FAILED");
    }

    const data = await response.json();
    
    if (!data.candidates || data.candidates.length === 0) {
      throw new Error("NO_CONTENT");
    }

    const text = data.candidates[0].content.parts[0].text;
    const cleanedText = text.replace(/\`\`\`json|\`\`\`/g, "").trim();
    
    try {
      const parsed = JSON.parse(cleanedText);
      return parsed as AnalysisResult;
    } catch (e) {
      throw new Error("PARSE_FAILED");
    }
  } catch (err: any) {
    if (err.message === "PARSE_FAILED" || err.message === "NO_CONTENT") {
      throw err;
    }
    throw new Error("API_FAILED");
  }
}
