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

interface MLPrediction {
  is_fake: boolean;
  confidence: number;
  probability_fake: number;
  verdict: string;
  risk_score: number;
  threshold_used?: number;
}

function getMimeType(file: File): string {
  const type = file.type.toLowerCase();
  if (type === 'image/png') return 'image/png';
  if (type === 'image/webp') return 'image/webp';
  if (type === 'image/gif') return 'image/gif';
  return 'image/jpeg';
}

/**
 * Step 1: Call the local trained ML model for classification
 */
async function getMLPrediction(base64Image: string): Promise<MLPrediction | null> {
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: base64Image }),
    });
    
    if (!response.ok) {
      console.warn("ML model server not available, falling back to Gemini-only");
      return null;
    }
    
    return await response.json();
  } catch (err) {
    console.warn("ML model server not reachable:", err);
    return null;
  }
}

/**
 * Step 2: Call Gemini via OpenRouter for detailed forensic explanation
 */
async function getGeminiAnalysis(
  base64Image: string,
  mimeType: string,
  mlPrediction: MLPrediction | null
): Promise<AnalysisResult> {
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("API_KEY_MISSING");
  }

  // Build context from ML prediction
  const mlContext = mlPrediction
    ? `\n\nIMPORTANT CONTEXT FROM TRAINED ML MODEL:
A dedicated deepfake detection neural network (ResNet18 fine-tuned on real and fake images) has analyzed this image and returned:
- Verdict: ${mlPrediction.verdict}
- Confidence: ${mlPrediction.confidence}%
- Risk Score: ${mlPrediction.risk_score}%
- Is Fake: ${mlPrediction.is_fake}
- Decision Threshold: ${mlPrediction.threshold_used ?? 'n/a'}%

You MUST treat this ML model prediction as the primary classification result. Your job is to explain the likely forensic reasons behind that result, not to contradict it unless the response would otherwise be invalid JSON.`
    : '';

  const SYSTEM_MESSAGE = `You are TruthLens, an advanced deepfake forensic detection system backed by a trained neural network.${mlContext}

YOUR DETECTION CAPABILITIES:
1. You combine trained ML model predictions with visual forensic analysis
2. The ML model was fine-tuned on a real/fake image dataset and should be treated as the source of truth for classification
3. Your role is to provide supporting forensic explanation and highlight uncertainty where appropriate

BEHAVIORAL RULES:
- If the ML model flags an image as fake, your verdict must stay fake
- If the ML model flags an image as real, you may downgrade to "INCONCLUSIVE" but not upgrade to a fake verdict unless the ML context is unavailable
- When in doubt, lean toward "INCONCLUSIVE" rather than contradicting the ML result`;

  const USER_PROMPT = `Analyze this image for deepfake manipulation or AI generation.

ANALYSIS PROTOCOL:
1. Note what the trained ML model predicted (if available in context above)
2. Perform your own visual forensic analysis:
   □ EYE REGION: Asymmetric reflections, pupil mismatches, iris uniformity, lifeless quality
   □ MOUTH/JAW: Lip boundary issues, teeth anomalies, jaw-neck blending
   □ SKIN TEXTURE: Unnaturally smooth/plastic vs natural pores and blemishes
   □ HAIR: Strand merging, impossible physics, background blending
   □ NOISE: Consistent sensor noise (real) vs smooth/patterned (AI)
   □ LIGHTING: Shadow consistency, reflection accuracy
   □ BACKGROUND: Warped geometry, melting objects, impossible architecture
   □ EDGES: Halos, blending artifacts between subject and background
3. Combine ML prediction + visual analysis for final verdict

Return ONLY valid JSON, no markdown, no code fences:
{
  "verdict": "LIKELY DEEPFAKE" or "POSSIBLY DEEPFAKE" or "AI EDITED" or "LIKELY AUTHENTIC" or "INCONCLUSIVE",
  "confidence": <integer 0-100>,
  "risk_score": <integer 0-100>,
  "manipulation_metrics": {
    "full_ai_generation_likelihood": <integer 0-100>,
    "partial_ai_edit_likelihood": <integer 0-100>
  },
  "summary": "<2-3 sentences citing ML model prediction AND your visual findings>",
  "signals": {
    "face_analysis": { "score": <0-100>, "findings": ["<observation>", "<observation>"] },
    "lighting_shadows": { "score": <0-100>, "findings": ["<observation>", "<observation>"] },
    "texture_artifacts": { "score": <0-100>, "findings": ["<observation>", "<observation>"] },
    "edge_consistency": { "score": <0-100>, "findings": ["<observation>", "<observation>"] },
    "background_coherence": { "score": <0-100>, "findings": ["<observation>", "<observation>"] }
  },
  "recommendation": "<one actionable sentence>"
}

SCORING RULES:
- If ML model says fake → verdict must be "LIKELY DEEPFAKE" or "POSSIBLY DEEPFAKE"
- If ML model says fake with >70% confidence → risk_score must be at least the ML risk score
- "LIKELY AUTHENTIC" requires ML model agreeing (if available) AND low visual risk`;

  const API_URL = "https://openrouter.ai/api/v1/chat/completions";

  const body = {
    model: "google/gemini-2.5-flash",
    messages: [
      { role: "system", content: SYSTEM_MESSAGE },
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: `data:${mimeType};base64,${base64Image}` }
          },
          { type: "text", text: USER_PROMPT }
        ]
      }
    ],
    temperature: 0.3,
    max_tokens: 3000
  };

  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'X-Title': 'TruthLens AI'
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    const errorBody = await response.text();
    console.error("OpenRouter API error:", response.status, errorBody);
    if (response.status === 429) throw new Error("QUOTA_EXCEEDED");
    throw new Error("API_FAILED");
  }

  const data = await response.json();
  if (!data.choices || data.choices.length === 0) throw new Error("NO_CONTENT");

  let text = data.choices[0].message.content;
  console.log("Raw Gemini response:", text);

  // Clean response
  text = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  text = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    console.error("No JSON in response:", text);
    throw new Error("PARSE_FAILED");
  }

  return JSON.parse(jsonMatch[0]) as AnalysisResult;
}


/**
 * Main analysis function — hybrid ML model + Gemini
 */
export async function analyzeWithGemini(file: File, mediaType: 'image' | 'video'): Promise<AnalysisResult> {
  let base64Image = '';
  let mimeType = 'image/jpeg';

  if (mediaType === 'image') {
    mimeType = getMimeType(file);
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
    mimeType = 'image/jpeg';
  }

  try {
    // Step 1: Get ML model prediction (fast, trained classifier)
    console.log("🔬 Querying trained ML model...");
    const mlPrediction = await getMLPrediction(base64Image);
    if (mlPrediction) {
      console.log("✅ ML Model prediction:", mlPrediction);
    } else {
      console.log("❌ ML model not available");
      throw new Error("ML_BACKEND_UNAVAILABLE");
    }

    // Step 2: Get Gemini analysis (detailed forensic explanation, informed by ML prediction)
    console.log("🤖 Getting Gemini forensic analysis...");
    const geminiResult = await getGeminiAnalysis(base64Image, mimeType, mlPrediction);

    return mergeWithMLPrediction(geminiResult, mlPrediction);
  } catch (err: any) {
    if (err.message === "PARSE_FAILED" || err.message === "NO_CONTENT" || err.message === "QUOTA_EXCEEDED") {
      throw err;
    }
    throw new Error("API_FAILED");
  }
}
function mergeWithMLPrediction(
  geminiResult: AnalysisResult,
  mlPrediction: MLPrediction | null
): AnalysisResult {
  if (!mlPrediction) {
    return geminiResult;
  }

  const merged = { ...geminiResult };
  const modelPrefix = `[ML Model: ${mlPrediction.verdict} at ${mlPrediction.confidence}% confidence]`;

  if (mlPrediction.is_fake) {
    merged.verdict = mlPrediction.confidence >= 85 ? "LIKELY DEEPFAKE" : "POSSIBLY DEEPFAKE";
    merged.risk_score = Math.max(geminiResult.risk_score, mlPrediction.risk_score);
    merged.confidence = Math.max(geminiResult.confidence, mlPrediction.confidence);
  } else {
    merged.verdict = mlPrediction.confidence >= 75 ? "LIKELY AUTHENTIC" : "INCONCLUSIVE";
    merged.risk_score = Math.min(geminiResult.risk_score, mlPrediction.risk_score);
    merged.confidence = Math.max(geminiResult.confidence, mlPrediction.confidence);
  }

  merged.summary = `${modelPrefix} ${geminiResult.summary}`;
  return merged;
}
