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
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    arch = checkpoint.get('architecture', 'resnet18')
    img_size = checkpoint.get('image_size', 224 if 'efficientnet' in arch.lower() else 128)
    threshold = checkpoint.get('best_threshold', 0.5)
    temperature = checkpoint.get('temperature', 1.0)
    calibration_method = checkpoint.get('calibration_method', 'none')

    model = create_model(arch=arch)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    acc = checkpoint.get('best_accuracy', 0)
    f1 = checkpoint.get('best_f1', 0)
    print(f"✅ Model loaded (architecture: {arch}, Accuracy: {acc:.1f}%, F1: {f1:.1f}%, image_size: {img_size}, threshold: {threshold:.2f}, calibration: {calibration_method}, temperature: {temperature:.4f})")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform, threshold, temperature, calibration_method


model = None
transform = None
threshold = 0.5
temperature = 1.0
calibration_method = 'none'


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    global model, transform, threshold, temperature, calibration_method
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
        logit = output.item() if output.ndim == 0 else output[0].item()

        # Apply temperature scaling for calibrated probabilities
        if calibration_method == 'temperature_scaling' and temperature != 1.0:
            calibrated_logit = logit / temperature
            prob = torch.sigmoid(torch.tensor(calibrated_logit)).item()
        else:
            prob = torch.sigmoid(torch.tensor(logit)).item()

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
        print(f"❌ Model not found at {MODEL_PATH}.")
        exit(1)
    model, transform, threshold, temperature, calibration_method = load_model()
    print(f"🚀 Server running on http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
