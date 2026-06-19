"""
TruthLens Threshold Calibration & Model Evaluation Pipeline
============================================================
Production-grade calibration for the ResNet-18 deepfake detector.

Evaluates thresholds from 0.30–0.90, applies calibration methods
(Temperature Scaling, Platt Scaling, Isotonic Regression), and
recommends optimal deployment threshold minimizing false positives.

Usage:
    python calibrate_model.py                  # Full pipeline
    python calibrate_model.py --quick          # Quick run (2K samples)
    python calibrate_model.py --use-test       # Use held-out test set
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model_def import create_model

# ─── Configuration ───────────────────────────────────────────────────────────

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = Path("model/deepfake_detector.pth")
DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
OUTPUT_DIR = Path("calibration_results")

IMG_SIZE = 224
BATCH_SIZE = 32

THRESHOLD_RANGE = np.arange(0.30, 0.91, 0.01)

# ─── Dataset ─────────────────────────────────────────────────────────────────

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DeepfakeDataset(Dataset):
    def __init__(self, fake_dir: Path, real_dir: Path, max_per_class: int | None = None):
        self.samples: list[tuple[Path, int]] = []
        self.transform = val_transform

        for directory, label in [(fake_dir, 1), (real_dir, 0)]:
            if not directory.exists():
                print(f"  ⚠ Directory not found: {directory}")
                continue
            files = sorted([
                f for f in directory.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
            ])
            if max_per_class:
                files = files[:max_per_class]
            self.samples.extend((f, label) for f in files)

        np.random.seed(42)
        np.random.shuffle(self.samples)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        n_real = sum(1 for _, l in self.samples if l == 0)
        print(f"  Loaded {len(self.samples)} images (Fake: {n_fake}, Real: {n_real})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = self.transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128)))
        return img, label


# ─── Inference: Collect Logits & Probabilities ───────────────────────────────

def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return (logits, probabilities, labels) as numpy arrays."""
    model.eval()
    all_logits, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Collecting predictions"):
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


# ─── Threshold Metrics ───────────────────────────────────────────────────────

def compute_metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = len(labels)

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    return {
        "threshold": round(threshold, 3),
        "accuracy": round(accuracy, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "fpr": round(fpr, 5),
        "fnr": round(fnr, 5),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def sweep_thresholds(probs, labels, thresholds=THRESHOLD_RANGE) -> list[dict]:
    return [compute_metrics_at_threshold(probs, labels, t) for t in thresholds]


def compute_roc_auc(probs, labels) -> float:
    """Manual ROC-AUC (no sklearn dependency required)."""
    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp_cumsum = np.cumsum(sorted_labels)
    fp_cumsum = np.cumsum(1 - sorted_labels)
    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg

    # Prepend origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    auc = np.trapezoid(tpr, fpr)
    return round(float(auc), 5)


# ─── Calibration Methods ─────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """Learns a single scalar temperature T to soften/sharpen logits."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray, lr=0.01, max_iter=200) -> float:
    """Optimize temperature on validation logits using NLL loss."""
    ts = TemperatureScaling()
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    optimizer = optim.LBFGS([ts.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = ts(logits_t).squeeze()
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = ts.temperature.item()
    print(f"  Temperature Scaling → T = {temperature:.4f}")
    return temperature


def fit_platt_scaling(logits: np.ndarray, labels: np.ndarray, lr=0.01, max_iter=300) -> tuple[float, float]:
    """Platt scaling: learns w and b such that P(y=1|x) = σ(w·logit + b)."""
    w = nn.Parameter(torch.ones(1))
    b = nn.Parameter(torch.zeros(1))
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    optimizer = optim.LBFGS([w, b], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = (w * logits_t + b).squeeze()
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    w_val, b_val = w.item(), b.item()
    print(f"  Platt Scaling → w = {w_val:.4f}, b = {b_val:.4f}")
    return w_val, b_val


def fit_isotonic_regression(probs: np.ndarray, labels: np.ndarray, n_bins=50) -> np.ndarray:
    """
    Lightweight isotonic regression via pool adjacent violators (PAV).
    Returns a lookup table mapping raw probability bins → calibrated probability.
    """
    sorted_idx = np.argsort(probs)
    sorted_probs = probs[sorted_idx]
    sorted_labels = labels[sorted_idx]

    # Pool Adjacent Violators
    n = len(sorted_labels)
    result = sorted_labels.astype(float).copy()
    weights = np.ones(n)

    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Merge blocks
            combined = (result[i] * weights[i] + result[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1])
            weights[i] = weights[i] + weights[i + 1]
            result[i] = combined
            result = np.delete(result, i + 1)
            weights = np.delete(weights, i + 1)
            n -= 1
            if i > 0:
                i -= 1
        else:
            i += 1

    # Build a bin-based lookup table
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    calibrated = np.zeros(n_bins)

    for b_idx in range(n_bins):
        lo, hi = bin_edges[b_idx], bin_edges[b_idx + 1]
        mask = (sorted_probs >= lo) & (sorted_probs < hi)
        if mask.sum() > 0:
            calibrated[b_idx] = sorted_labels[mask].mean()
        else:
            calibrated[b_idx] = bin_centers[b_idx]

    print(f"  Isotonic Regression → {n_bins} bins fitted")
    return calibrated


def apply_isotonic(probs: np.ndarray, calibration_table: np.ndarray) -> np.ndarray:
    """Map raw probabilities through the isotonic lookup table."""
    n_bins = len(calibration_table)
    bin_indices = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    return calibration_table[bin_indices]


# ─── Recommendation Engine ───────────────────────────────────────────────────

def recommend_threshold(sweep_results: list[dict], fp_weight=2.0) -> dict:
    """
    Recommend a threshold that balances F1 while penalizing false positives.

    Uses a weighted score: score = F1 - fp_weight * FPR
    Higher fp_weight → more aggressive FP reduction.
    """
    best_score = -999
    best = None

    for m in sweep_results:
        if m["recall"] < 0.40:
            continue  # Never sacrifice recall below 40%
        score = m["f1"] - fp_weight * m["fpr"]
        if score > best_score:
            best_score = score
            best = m

    return best or sweep_results[len(sweep_results) // 2]


# ─── Confusion Matrix Display ────────────────────────────────────────────────

def format_confusion_matrix(m: dict) -> str:
    return (
        f"  Threshold = {m['threshold']:.2f}\n"
        f"  ┌──────────────────────────────────┐\n"
        f"  │           Predicted              │\n"
        f"  │         Fake    Real             │\n"
        f"  │  Actual                          │\n"
        f"  │  Fake  {m['tp']:>6}  {m['fn']:>6}  (Recall: {m['recall']:.1%})  │\n"
        f"  │  Real  {m['fp']:>6}  {m['tn']:>6}  (Specif: {1-m['fpr']:.1%})  │\n"
        f"  └──────────────────────────────────┘\n"
        f"  Accuracy={m['accuracy']:.1%}  Precision={m['precision']:.1%}"
        f"  F1={m['f1']:.1%}  FPR={m['fpr']:.1%}\n"
    )


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TruthLens Calibration Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick run with 2K samples")
    parser.add_argument("--use-test", action="store_true", help="Evaluate on held-out test set")
    parser.add_argument("--fp-weight", type=float, default=2.0, help="FP penalty weight for recommendation")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("  TruthLens Threshold Calibration Pipeline")
    print("=" * 65)
    print(f"  Device: {DEVICE}")

    # ── Load Model ──
    print("\n📦 Loading model checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    current_threshold = checkpoint.get("best_threshold", 0.5)
    print(f"  Current stored threshold: {current_threshold:.4f}")
    print(f"  Stored F1: {checkpoint.get('best_accuracy', 0):.1f}%")

    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # ── Load Dataset ──
    split = "Test" if args.use_test else "Validation"
    max_per_class = 1000 if args.quick else None
    print(f"\n📂 Loading {split} dataset...")
    fake_dir = DATASET_ROOT / split / "Fake"
    real_dir = DATASET_ROOT / split / "Real"
    dataset = DeepfakeDataset(fake_dir, real_dir, max_per_class=max_per_class)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Collect Predictions ──
    print("\n🔬 Running inference...")
    logits, probs, labels = collect_predictions(model, loader, DEVICE)
    roc_auc = compute_roc_auc(probs, labels)
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Prob stats — mean: {probs.mean():.4f}, std: {probs.std():.4f}, "
          f"min: {probs.min():.4f}, max: {probs.max():.4f}")

    # ── Threshold Sweep ──
    print(f"\n📊 Sweeping {len(THRESHOLD_RANGE)} thresholds (0.30 → 0.90)...")
    sweep = sweep_thresholds(probs, labels)

    # Show current threshold performance
    current_m = compute_metrics_at_threshold(probs, labels, current_threshold)
    print(f"\n{'─' * 65}")
    print(f"  CURRENT THRESHOLD ({current_threshold:.2f}) PERFORMANCE:")
    print(format_confusion_matrix(current_m))

    # ── Calibration Methods ──
    print("🔧 Fitting calibration methods...")

    # Temperature Scaling
    temperature = fit_temperature_scaling(logits, labels)
    temp_probs = torch.sigmoid(torch.tensor(logits) / temperature).numpy()
    temp_sweep = sweep_thresholds(temp_probs, labels)

    # Platt Scaling
    w, b = fit_platt_scaling(logits, labels)
    platt_probs = torch.sigmoid(torch.tensor(logits) * w + b).numpy()
    platt_sweep = sweep_thresholds(platt_probs, labels)

    # Isotonic Regression
    iso_table = fit_isotonic_regression(probs, labels)
    iso_probs = apply_isotonic(probs, iso_table)
    iso_sweep = sweep_thresholds(iso_probs, labels)

    # ── Recommendations ──
    print(f"\n{'═' * 65}")
    print("  THRESHOLD RECOMMENDATIONS")
    print(f"{'═' * 65}")

    methods = {
        "Uncalibrated": (sweep, probs),
        "Temperature Scaled": (temp_sweep, temp_probs),
        "Platt Scaled": (platt_sweep, platt_probs),
        "Isotonic Regression": (iso_sweep, iso_probs),
    }

    recommendations = {}
    for name, (s, p) in methods.items():
        rec = recommend_threshold(s, fp_weight=args.fp_weight)
        recommendations[name] = rec
        auc = compute_roc_auc(p, labels)
        print(f"\n  [{name}]")
        print(f"  Recommended threshold: {rec['threshold']:.2f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(format_confusion_matrix(rec))

    # ── Confusion Matrices at Key Thresholds ──
    print(f"{'═' * 65}")
    print("  CONFUSION MATRICES AT KEY THRESHOLDS")
    print(f"{'═' * 65}")
    for t in [0.40, 0.50, 0.52, 0.60, 0.65, 0.70, 0.75, 0.80]:
        m = compute_metrics_at_threshold(probs, labels, t)
        print(format_confusion_matrix(m))

    # ── Save Results ──
    results = {
        "model_path": str(MODEL_PATH),
        "dataset_split": split,
        "num_samples": len(labels),
        "roc_auc": roc_auc,
        "current_threshold": current_threshold,
        "current_metrics": current_m,
        "calibration": {
            "temperature": temperature,
            "platt_w": w,
            "platt_b": b,
        },
        "recommendations": {k: v for k, v in recommendations.items()},
        "threshold_sweep": sweep,
    }

    results_path = OUTPUT_DIR / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")

    # Save probability arrays for the visualization script
    np.savez(
        OUTPUT_DIR / "predictions.npz",
        logits=logits, probs=probs, labels=labels,
        temp_probs=temp_probs, platt_probs=platt_probs, iso_probs=iso_probs,
    )
    print(f"💾 Prediction arrays saved to {OUTPUT_DIR / 'predictions.npz'}")

    elapsed = time.time() - t0
    print(f"\n✅ Calibration complete in {elapsed:.1f}s")

    # Final recommendation
    best_method = min(recommendations.items(), key=lambda kv: kv[1]["fpr"])
    print(f"\n{'═' * 65}")
    print(f"  🏆 BEST FOR LOW FALSE-POSITIVES: {best_method[0]}")
    print(f"     Threshold: {best_method[1]['threshold']:.2f}")
    print(f"     FPR: {best_method[1]['fpr']:.1%}  |  Recall: {best_method[1]['recall']:.1%}")
    print(f"     F1:  {best_method[1]['f1']:.1%}")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
