"""
TruthLens ML Inference Server
==============================
Flask API serving the trained deepfake detection model.
"""

import io
import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from pathlib import Path

from model_def import create_model

MODEL_PATH = Path("model/deepfake_detector.pth")
PORT = 5001

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    img_size = checkpoint.get('image_size', 128)
    threshold = checkpoint.get('best_threshold', 0.5)

    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    acc = checkpoint.get('best_accuracy', 0)
    arch = checkpoint.get('architecture', 'unknown')
    print(f"✅ Model loaded (architecture: {arch}, F1: {acc:.1f}%, image_size: {img_size}, threshold: {threshold:.2f})")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform, threshold


model = None
transform = None
threshold = 0.5


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    global model, transform, threshold
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get image
    if 'image' in request.files:
        image = Image.open(request.files['image'].stream)
    elif request.is_json and 'image_base64' in request.json:
        import base64
        image = Image.open(io.BytesIO(base64.b64decode(request.json['image_base64'])))
    else:
        return jsonify({"error": "No image provided"}), 400
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor).squeeze()
        prob = torch.sigmoid(output).item()

    is_fake = prob >= threshold
    conf = prob if is_fake else (1 - prob)
    
    if is_fake:
        verdict = "LIKELY DEEPFAKE" if conf > 0.8 else "POSSIBLY DEEPFAKE" if conf > 0.6 else "INCONCLUSIVE"
    else:
        verdict = "LIKELY AUTHENTIC" if conf > 0.8 else "LIKELY AUTHENTIC" if conf > 0.6 else "INCONCLUSIVE"
    
    return jsonify({
        "is_fake": is_fake,
        "confidence": round(conf * 100, 1),
        "probability_fake": round(prob * 100, 1),
        "threshold_used": round(threshold * 100, 1),
        "verdict": verdict,
        "risk_score": round(prob * 100, 1),
    })


if __name__ == '__main__':
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}. Run: python3 train_model.py")
        exit(1)
    model, transform, threshold = load_model()
    print(f"🚀 Server running on http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
