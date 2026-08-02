"""
TruthLens ML Model Training (v10) — High Accuracy & Low-FPR Pipeline
====================================================================
Targets >95% F1 Accuracy and <5% False Positive Rate across compressed images.

Key Features:
1. EfficientNet-B2 backbone + Frequency Stream (Laplacian High-Pass Filter)
2. Heavy Realistic JPEG Degradation Augmentations (quality QF 10-95, blur, noise)
3. Focal Loss with Label Smoothing + MixUp regularization
4. Two-Phase Transfer Learning (Phase 1: freeze backbone, Phase 2: unfreeze top layers)
5. Automatic Checkpoint Metadata Export (architecture, image_size, threshold, f1)
"""

import io
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

from model_def import create_model

# ─── Environment & Device ───────────────────────────────────────────────────

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Device: {DEVICE}")

# ─── Configuration ─────────────────────────────────────────────────────────

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "15"))
LR_HEAD = float(os.getenv("LR_HEAD", "1e-3"))
LR_BACKBONE = float(os.getenv("LR_BACKBONE", "1e-5"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", "2.0"))
FOCAL_ALPHA = float(os.getenv("FOCAL_ALPHA", "0.25"))
USE_MIXUP = os.getenv("USE_MIXUP", "1") == "1"
MIXUP_ALPHA = float(os.getenv("MIXUP_ALPHA", "0.4"))

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "deepfake_detector_v10.pth"

DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"

# ─── Robust JPEG & Compression Augmentation Transforms ───────────────────────

class JPEGCompressionTransform:
    """Simulates social media JPEG compression block artifacts."""
    def __init__(self, quality_range=(15, 90), p=0.6):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            return Image.open(buf).convert('RGB')
        return img


class RandomBlurTransform:
    """Simulates camera motion / social media downsampling blur."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(0.5, 1.8)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


# ─── Dataset Loader ─────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        self.split_dir = root_dir / split
        self.transform = transform
        self.samples = []

        fake_dir = self.split_dir / "Fake"
        real_dir = self.split_dir / "Real"

        if fake_dir.exists():
            for p in fake_dir.glob("*"):
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    self.samples.append((p, 1.0))

        if real_dir.exists():
            for p in real_dir.glob("*"):
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    self.samples.append((p, 0.0))

        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} images for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback for corrupt images
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# ─── Focal Loss Function ────────────────────────────────────────────────────

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


# ─── Mixup Helper ───────────────────────────────────────────────────────────

def apply_mixup(x, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ─── Main Training Routine ──────────────────────────────────────────────────

def main():
    print(f"--- TruthLens v10 Model Training Pipeline ---")
    print(f"Data Root: {DATASET_ROOT}")

    # Transforms
    train_transform = transforms.Compose([
        JPEGCompressionTransform(quality_range=(15, 90), p=0.6),
        RandomBlurTransform(p=0.3),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets & Loaders
    train_dataset = DeepfakeDataset(DATASET_ROOT, split="Train", transform=train_transform)
    val_dataset = DeepfakeDataset(DATASET_ROOT, split="Validation", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate EfficientNet-B2 Detector
    model = create_model(arch="efficientnet_b2_v10", device=DEVICE, dropout_rate=0.4, use_freq_stream=True)

    criterion = FocalLossWithSmoothing(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING)

    # Optimizer with differential learning rates
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_threshold = 0.5

    print("\nStarting Training Loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # Phase 1 vs Phase 2 fine-tuning logic
        if epoch <= 2:
            # Freeze backbone for first 2 epochs
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if USE_MIXUP and random.random() < 0.5:
                mixed_imgs, targets_a, targets_b, lam = apply_mixup(imgs, labels)
                outputs = model(mixed_imgs).squeeze(-1)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(imgs).squeeze(-1)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        train_acc = (train_correct / train_total) * 100
        avg_train_loss = train_loss / train_total

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_logits = []
        val_targets = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

                val_logits.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_logits = np.array(val_logits)
        val_targets = np.array(val_targets)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        # Sweep threshold for optimal F1
        best_e_f1 = 0.0
        best_e_thresh = 0.5
        best_e_acc = 0.0
        for th in np.arange(0.3, 0.8, 0.05):
            preds = (val_probs >= th).astype(float)
            tp = np.sum((preds == 1) & (val_targets == 1))
            fp = np.sum((preds == 1) & (val_targets == 0))
            fn = np.sum((preds == 0) & (val_targets == 1))
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            acc = np.mean(preds == val_targets) * 100
            if f1 > best_e_f1:
                best_e_f1 = f1
                best_e_thresh = th
                best_e_acc = acc

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Acc: {best_e_acc:.2f}% F1: {best_e_f1*100:.2f}% (Thresh: {best_e_thresh:.2f}) | Time: {elapsed:.0f}s")

        # Save checkpoint if F1 improved
        if best_e_f1 > best_val_f1:
            best_val_f1 = best_e_f1
            best_val_acc = best_e_acc
            best_threshold = best_e_thresh

            print(f"  ⭐ Saving new best model to {CHECKPOINT_PATH} (F1: {best_val_f1*100:.2f}%)")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'architecture': 'efficientnet_b2_v10',
                'image_size': IMG_SIZE,
                'best_accuracy': best_val_acc,
                'best_f1': best_val_f1 * 100,
                'best_threshold': float(best_threshold),
                'temperature': 1.0,
                'calibration_method': 'none'
            }
            torch.save(checkpoint, CHECKPOINT_PATH)

    print("\n✅ Training complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%, Best F1: {best_val_f1*100:.2f}% (Threshold: {best_threshold:.2f})")


if __name__ == "__main__":
    main()
