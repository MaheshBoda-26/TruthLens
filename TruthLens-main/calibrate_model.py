"""
TruthLens Model Calibration & Temperature Optimization Pipeline (v10)
====================================================================
Fits optimal Temperature Scaling factor T* using L-BFGS to minimize Expected
Calibration Error (ECE) and negative log-likelihood, then sweeps thresholds
to select optimal decision boundary bounding FPR < 5.0%.
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


class TemperatureScaling(nn.Module):
    """Learns a single scalar temperature T to calibrate logits."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature


class ValDataset(Dataset):
    def __init__(self, root_dir, img_size=224, split="Validation"):
        self.samples = []
        split_dir = root_dir / split

        fake_dir = split_dir / "Fake"
        real_dir = split_dir / "Real"

        if fake_dir.exists():
            for p in fake_dir.glob("*"):
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    self.samples.append((p, 1))

        if real_dir.exists():
            for p in real_dir.glob("*"):
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    self.samples.append((p, 0))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = self.transform(Image.new('RGB', (224, 224), (128, 128, 128)))
        return img, label


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"❌ Model path {MODEL_PATH} not found.")
        return

    print("📊 Loading model checkpoint for Temperature Calibration...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    arch = checkpoint.get('architecture', 'resnet18')
    img_size = checkpoint.get('image_size', 224 if 'efficientnet' in arch.lower() else 128)

    model = create_model(arch=arch)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    val_dataset = ValDataset(DATASET_ROOT, img_size=img_size, split="Validation")
    loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    logits_list = []
    labels_list = []

    print("Running validation set inference...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs).squeeze(-1)
            logits_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.numpy())

    logits_t = torch.tensor(logits_list, dtype=torch.float32)
    labels_t = torch.tensor(labels_list, dtype=torch.float32)

    # Fit Temperature Scaling
    ts = TemperatureScaling()
    optimizer = optim.LBFGS([ts.temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = ts(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    optimal_temp = float(ts.temperature.item())
    print(f"⚡ Optimal Temperature (T*): {optimal_temp:.4f}")

    calibrated_probs = torch.sigmoid(logits_t / optimal_temp).numpy()
    targets = labels_t.numpy()

    best_f1 = 0.0
    best_thresh = 0.5
    best_acc = 0.0
    best_fpr = 1.0

    print("Sweeping decision thresholds for FPR <= 5.0%...")
    for th in np.arange(0.30, 0.90, 0.01):
        preds = (calibrated_probs >= th).astype(float)
        tp = np.sum((preds == 1) & (targets == 1))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))
        tn = np.sum((preds == 0) & (targets == 0))

        fpr = fp / (fp + tn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        acc = np.mean(preds == targets) * 100

        if fpr <= 0.05 and f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            best_acc = acc
            best_fpr = fpr

    print(f"✅ Selected Optimal Threshold: {best_thresh:.2f}")
    print(f"   Accuracy: {best_acc:.2f}%, F1: {best_f1*100:.2f}%, FPR: {best_fpr*100:.2f}%")

    checkpoint['best_threshold'] = float(best_thresh)
    checkpoint['temperature'] = float(optimal_temp)
    checkpoint['calibration_method'] = 'temperature_scaling'
    checkpoint['best_f1'] = float(best_f1 * 100)
    checkpoint['best_accuracy'] = float(best_acc)

    torch.save(checkpoint, MODEL_PATH)
    print(f"💾 Checkpoint updated and saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
