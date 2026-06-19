"""
TruthLens Grad-CAM Attention Analysis
======================================
Visualizes which image regions drive the model's fake/real decision
using Grad-CAM on the last convolutional layer of ResNet-18.

Detects whether the model is learning genuine forensic features
(eyes, skin, edges) or shortcut artifacts (backgrounds, corners).

Usage:
    ./venv/bin/python gradcam_analysis.py
    ./venv/bin/python gradcam_analysis.py --quick --threshold 0.68
"""

import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model_def import create_model

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("model/deepfake_detector.pth")
DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
OUT = Path("fp_investigation/gradcam")
IMG_SIZE = 224

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class GradCAM:
    """Grad-CAM for the last conv layer of ResNet-18."""

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # Hook into layer4 (last residual block)
        target = model.backbone.layer4[-1].conv2
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """Returns (heatmap_7x7, probability)."""
        self.model.zero_grad()
        output = self.model(input_tensor).squeeze()
        prob = torch.sigmoid(output).item()

        # Backward pass
        output.backward()

        # Grad-CAM computation
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP over spatial
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, prob


def overlay_heatmap(img_bgr: np.ndarray, cam: np.ndarray, alpha=0.5) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original image."""
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


def compute_attention_regions(cam: np.ndarray) -> dict:
    """Classify where attention falls: center (face) vs periphery (background)."""
    h, w = cam.shape
    cy, cx = h // 2, w // 2
    rh, rw = max(h // 3, 1), max(w // 3, 1)

    center = cam[cy-rh:cy+rh, cx-rw:cx+rw]
    total_energy = cam.sum() + 1e-8
    center_energy = center.sum()

    # Corner energy
    cs = max(h // 4, 1)
    corners = np.concatenate([
        cam[:cs, :cs].ravel(), cam[:cs, -cs:].ravel(),
        cam[-cs:, :cs].ravel(), cam[-cs:, -cs:].ravel()
    ])
    corner_energy = corners.sum()

    return {
        "center_pct": round(float(center_energy / total_energy * 100), 1),
        "corner_pct": round(float(corner_energy / total_energy * 100), 1),
        "max_activation": round(float(cam.max()), 4),
        "mean_activation": round(float(cam.mean()), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--split", default="Test")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "false_positives").mkdir(exist_ok=True)
    (OUT / "true_negatives").mkdir(exist_ok=True)

    print("=" * 60)
    print("  TruthLens Grad-CAM Attention Analysis")
    print("=" * 60)

    # Load model (need gradients, so no eval freeze)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

    gcam = GradCAM(model)

    # Collect real image paths
    real_dir = DATASET_ROOT / args.split / "Real"
    real_files = sorted(f for f in real_dir.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png'})
    if args.quick:
        real_files = real_files[:500]
    print(f"  Scanning {len(real_files)} real images...")

    # Classify all real images
    fp_items, tn_items = [], []
    for path in tqdm(real_files, desc="Classifying"):
        try:
            img = Image.open(path).convert('RGB')
            tensor = val_tf(img).unsqueeze(0).to(DEVICE)

            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor).squeeze()).item()

            if prob >= args.threshold:
                fp_items.append((path, prob))
            else:
                tn_items.append((path, prob))
        except Exception:
            continue

    # Sort FPs by confidence (worst first)
    fp_items.sort(key=lambda x: -x[1])
    tn_items.sort(key=lambda x: x[1])

    print(f"\n  FP: {len(fp_items)}  TN: {len(tn_items)}")
    n = min(args.n_samples, len(fp_items), len(tn_items))
    print(f"  Generating Grad-CAM for top {n} FPs and {n} TNs...")

    # Generate Grad-CAM overlays
    fp_regions, tn_regions = [], []

    for label, items, regions, folder in [
        ("FP", fp_items[:n], fp_regions, "false_positives"),
        ("TN", tn_items[:n], tn_regions, "true_negatives"),
    ]:
        for i, (path, _) in enumerate(tqdm(items, desc=f"Grad-CAM {label}")):
            try:
                img_pil = Image.open(path).convert('RGB')
                tensor = val_tf(img_pil).unsqueeze(0).to(DEVICE)
                tensor.requires_grad_(True)

                model.train()  # Enable gradient tracking
                cam, prob = gcam.generate(tensor)

                # Analyze attention regions
                region = compute_attention_regions(cam)
                region["prob"] = round(prob, 4)
                region["file"] = path.name
                regions.append(region)

                # Save overlay
                img_bgr = cv2.imread(str(path))
                img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
                overlay = overlay_heatmap(img_bgr, cam)

                # Side by side: original | overlay
                combined = np.hstack([img_bgr, overlay])
                fname = f"{label}_{i:03d}_p{prob:.3f}_{path.name}"
                cv2.imwrite(str(OUT / folder / fname), combined)

            except Exception as e:
                continue

    # ── Statistical comparison of attention regions ──
    print(f"\n{'═'*60}")
    print("  ATTENTION REGION ANALYSIS")
    print(f"{'═'*60}")

    if fp_regions and tn_regions:
        fp_center = np.mean([r["center_pct"] for r in fp_regions])
        tn_center = np.mean([r["center_pct"] for r in tn_regions])
        fp_corner = np.mean([r["corner_pct"] for r in fp_regions])
        tn_corner = np.mean([r["corner_pct"] for r in tn_regions])

        print(f"  {'Metric':<25} {'FP':>10} {'TN':>10} {'Delta':>10}")
        print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
        print(f"  {'Center attention %':<25} {fp_center:>10.1f} {tn_center:>10.1f} {fp_center-tn_center:>+10.1f}")
        print(f"  {'Corner attention %':<25} {fp_corner:>10.1f} {tn_corner:>10.1f} {fp_corner-tn_corner:>+10.1f}")

        # Shortcut detection
        if fp_corner > tn_corner + 5:
            print("\n  ⚠️  SHORTCUT DETECTED: FP images have significantly more")
            print("     corner/background attention. The model may be learning")
            print("     background texture or compression artifacts instead of")
            print("     facial forensic features.")
        elif fp_center < tn_center - 10:
            print("\n  ⚠️  SHORTCUT DETECTED: FP images have less center/face")
            print("     attention. The model may struggle with face alignment.")
        else:
            print("\n  ✅ Attention patterns are similar between FP and TN.")
            print("     Model appears to use similar facial regions for both.")

    # Save report
    report = {
        "threshold": args.threshold,
        "n_fp": len(fp_items), "n_tn": len(tn_items),
        "n_gradcam": n,
        "fp_attention": fp_regions,
        "tn_attention": tn_regions,
        "summary": {
            "fp_center_mean": round(float(np.mean([r["center_pct"] for r in fp_regions])), 1) if fp_regions else 0,
            "tn_center_mean": round(float(np.mean([r["center_pct"] for r in tn_regions])), 1) if tn_regions else 0,
            "fp_corner_mean": round(float(np.mean([r["corner_pct"] for r in fp_regions])), 1) if fp_regions else 0,
            "tn_corner_mean": round(float(np.mean([r["corner_pct"] for r in tn_regions])), 1) if tn_regions else 0,
        }
    }
    with open(OUT / "gradcam_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  📁 Grad-CAM results saved to {OUT}/")
    print(f"     false_positives/  — {n} FP overlays (original | heatmap)")
    print(f"     true_negatives/   — {n} TN overlays")
    print(f"     gradcam_report.json")


if __name__ == "__main__":
    main()
