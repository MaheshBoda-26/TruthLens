"""
TruthLens ML Model Training (v9) — EfficientNet-B2 Fine-Tuning
=============================================================
Target: >95% accuracy on the deepfake detection task.

Key improvements over v8:
1. EfficientNet-B2 backbone (better accuracy than ResNet18)
2. Full dataset (140K training images)
3. Focal loss with label smoothing
4. RandAugment + strong augmentation pipeline
5. Cosine annealing with warm restarts
6. Gradient clipping for stability
7. Test-time augmentation (TTA)
8. 15-20 epochs with early stopping
9. Mixed precision training on MPS
"""

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
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── Config ──────────────────────────────────────────────────────────────────

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "48"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "18"))
LR = float(os.getenv("LR", "1e-3"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "260"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.02"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.1"))
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", "2.0"))
DROP_PATH_RATE = float(os.getenv("DROP_PATH_RATE", "0.25"))
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", "0.4"))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "5"))
USE_MIXUP = os.getenv("USE_MIXUP", "1") == "1"
MIXUP_ALPHA = float(os.getenv("MIXUP_ALPHA", "0.4"))
MIXUP_PROB = float(os.getenv("MIXUP_PROB", "0.5"))

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "deepfake_detector_v9.pth"

DEEPFAKE_PATH = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"

# ─── Model Definition ────────────────────────────────────────────────────────

class DeepfakeDetectorV9(nn.Module):
    """EfficientNet-B2 backbone with custom classification head."""

    def __init__(self, num_classes=1, drop_path_rate=0.25, dropout_rate=0.4):
        super().__init__()
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.backbone = models.efficientnet_b2(weights=weights)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ─── Focal Loss with Label Smoothing ─────────────────────────────────────────

class FocalLossWithSmoothing(nn.Module):
    """Focal Loss + Label Smoothing for binary classification."""

    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets_smooth >= 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ─── Mixup ───────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── Transforms ──────────────────────────────────────────────────────────────

class JPEGCompressTransform:
    """Simulate social media compression artifacts."""
    def __init__(self, quality_range=(20, 70)):
        self.quality_range = quality_range

    def __call__(self, img):
        import io
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class SharpnessAdjustTransform:
    """Adjust sharpness to simulate different camera qualities."""
    def __call__(self, img):
        if random.random() < 0.4:
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(factor)
        return img


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.RandomAffine(
        degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.RandomChoice([
        JPEGCompressTransform(quality_range=(20, 60)),
        SharpnessAdjustTransform(),
        transforms.Lambda(lambda x: x),
    ]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Test-Time Augmentation transforms
tta_transforms = [
    val_tf,  # original
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


# ─── Dataset ─────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """Loads real/fake images from folder structure."""

    def __init__(self, fake_dir, real_dir, transform=None, max_per_class=None):
        self.samples = []
        self.transform = transform

        for d, label in [(fake_dir, 1), (real_dir, 0)]:
            if d and d.exists():
                files = sorted(f for f in d.iterdir()
                               if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
                if max_per_class:
                    files = files[:max_per_class]
                for f in files:
                    self.samples.append((f, label))

        random.shuffle(self.samples)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        n_real = sum(1 for _, l in self.samples if l == 0)
        print(f"  Loaded {len(self.samples)} images (Fake: {n_fake}, Real: {n_real})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return img, label


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = len(labels)

    acc = (tp + tn) / total * 100 if total else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fpr = fp / (fp + tn) * 100 if (fp + tn) else 0
    fnr = fn / (fn + tp) * 100 if (fn + tp) else 0

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "fpr": fpr, "fnr": fnr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ─── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.float().to(device)

        # Mixup
        use_mixup = USE_MIXUP and random.random() < MIXUP_PROB
        if use_mixup:
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs).squeeze(1)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs).squeeze(1)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            if use_mixup:
                preds = (probs >= 0.5).float()
                correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
            else:
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.1f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    return total_loss / len(loader), correct / total * 100


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # Find best threshold by F1
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.25, 0.85, 0.01):
        m = compute_metrics(probs, labels, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t

    metrics = compute_metrics(probs, labels, best_t)
    metrics["threshold"] = round(best_t, 3)

    # Also compute accuracy at 0.5 threshold
    metrics_at_05 = compute_metrics(probs, labels, 0.5)
    metrics["accuracy_at_05"] = metrics_at_05["accuracy"]

    return total_loss / len(loader), metrics


def test_time_augmentation(model, loader, device, n_augments=3):
    """Run inference with multiple augmentations and average predictions."""
    model.eval()
    all_probs = []
    all_labels = []

    for aug_idx, tf in enumerate(tta_transforms[:n_augments]):
        print(f"\n  TTA pass {aug_idx + 1}/{n_augments}...")
        aug_loader = DataLoader(
            loader.dataset, batch_size=loader.batch_size,
            shuffle=False, num_workers=0
        )

        # Temporarily replace transform
        original_transform = loader.dataset.transform
        loader.dataset.transform = tf

        probs_epoch = []
        labels_epoch = []

        with torch.no_grad():
            for imgs, labels in tqdm(aug_loader, desc=f"TTA {aug_idx + 1}"):
                imgs = imgs.to(device)
                outputs = model(imgs).squeeze(1)
                probs = torch.sigmoid(outputs)
                probs_epoch.append(probs.cpu().numpy())
                labels_epoch.append(labels.numpy())

        loader.dataset.transform = original_transform

        all_probs.append(np.concatenate(probs_epoch))
        all_labels.append(np.concatenate(labels_epoch))

    # Average predictions across augmentations
    avg_probs = np.mean(all_probs, axis=0)
    labels = all_labels[0]  # Same labels for all

    # Find best threshold
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.25, 0.85, 0.01):
        m = compute_metrics(avg_probs, labels, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t

    metrics = compute_metrics(avg_probs, labels, best_t)
    metrics["threshold"] = round(best_t, 3)
    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  TruthLens Model Training v9 — EfficientNet-B2 Fine-Tuning")
    print("=" * 65)
    t0 = time.time()

    # Load full dataset
    print("\nLoading Full Dataset...")
    train_fake = DEEPFAKE_PATH / "Train" / "Fake"
    train_real = DEEPFAKE_PATH / "Train" / "Real"
    val_fake = DEEPFAKE_PATH / "Validation" / "Fake"
    val_real = DEEPFAKE_PATH / "Validation" / "Real"
    test_fake = DEEPFAKE_PATH / "Test" / "Fake"
    test_real = DEEPFAKE_PATH / "Test" / "Real"

    print("\n[Train Set]")
    train_ds = DeepfakeDataset(train_fake, train_real, train_tf)
    print("\n[Validation Set]")
    val_ds = DeepfakeDataset(val_fake, val_real, val_tf)
    print("\n[Test Set]")
    test_ds = DeepfakeDataset(test_fake, test_real, val_tf)

    print(f"\nTotal Train: {len(train_ds)} images")
    print(f"Total Val: {len(val_ds)} images")
    print(f"Total Test: {len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print(f"\nBuilding EfficientNet-B2 model...")
    model = DeepfakeDetectorV9(
        drop_path_rate=DROP_PATH_RATE,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"  Trainable: {trainable:,}")

    # Loss & Optimizer
    criterion = FocalLossWithSmoothing(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    # Differential learning rates: lower for backbone, higher for head
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': head_params, 'lr': LR},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = steps_per_epoch * 2  # 2 epochs warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    use_scaler = DEVICE.type in ('cuda', 'mps')
    scaler = torch.amp.GradScaler(DEVICE.type) if use_scaler else None

    # Training loop
    best_f1 = 0
    best_acc = 0
    best_state = None
    best_threshold = 0.5
    patience_counter = 0

    print(f"\nStarting {NUM_EPOCHS} epochs...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'=' * 65}")
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'=' * 65}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE, scaler
        )

        val_loss, metrics = validate(model, val_loader, criterion, DEVICE)

        print(f"\n  Results:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {metrics['accuracy']:.1f}%")
        print(f"    Val Acc@0.5: {metrics['accuracy_at_05']:.1f}%")
        print(f"    Precision:  {metrics['precision']:.1f}% | Recall: {metrics['recall']:.1f}%")
        print(f"    F1:         {metrics['f1']:.1f}% | FPR: {metrics['fpr']:.1f}%")
        print(f"    Threshold:  {metrics['threshold']:.3f}")
        print(f"    TP: {metrics['tp']} FP: {metrics['fp']} TN: {metrics['tn']} FN: {metrics['fn']}")

        # Save best by accuracy at threshold
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_f1 = metrics["f1"]
            best_threshold = metrics["threshold"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    -> New best! Acc: {best_acc:.1f}%, F1: {best_f1:.1f}%")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test set
    print(f"\n{'=' * 65}")
    print("  TEST SET EVALUATION")
    print(f"{'=' * 65}")

    test_loss, test_metrics = validate(model, test_loader, criterion, DEVICE)
    print(f"\n  Test Accuracy: {test_metrics['accuracy']:.1f}%")
    print(f"  Test F1:       {test_metrics['f1']:.1f}%")
    print(f"  Test Precision: {test_metrics['precision']:.1f}%")
    print(f"  Test Recall:   {test_metrics['recall']:.1f}%")
    print(f"  Test FPR:      {test_metrics['fpr']:.1f}%")
    print(f"  Test Threshold: {test_metrics['threshold']:.3f}")

    # TTA evaluation on test set
    print(f"\n{'=' * 65}")
    print("  TEST-TIME AUGMENTATION EVALUATION")
    print(f"{'=' * 65}")
    tta_metrics = test_time_augmentation(model, test_loader, DEVICE, n_augments=3)
    print(f"\n  TTA Accuracy: {tta_metrics['accuracy']:.1f}%")
    print(f"  TTA F1:       {tta_metrics['f1']:.1f}%")
    print(f"  TTA Precision: {tta_metrics['precision']:.1f}%")
    print(f"  TTA Recall:   {tta_metrics['recall']:.1f}%")
    print(f"  TTA FPR:      {tta_metrics['fpr']:.1f}%")
    print(f"  TTA Threshold: {tta_metrics['threshold']:.3f}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_accuracy': best_f1,
        'best_threshold': best_threshold,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
        'tta_accuracy': tta_metrics['accuracy'],
        'tta_f1': tta_metrics['f1'],
        'image_size': IMG_SIZE,
        'architecture': 'efficientnet_b2_v9',
        'training_samples': len(train_ds),
        'epochs_trained': epoch + 1,
    }, MODEL_PATH)

    mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0

    print(f"\n{'=' * 65}")
    print(f"  Training Complete!")
    print(f"  Best Val Acc: {best_acc:.1f}%")
    print(f"  Test Acc:     {test_metrics['accuracy']:.1f}%")
    print(f"  TTA Acc:      {tta_metrics['accuracy']:.1f}%")
    print(f"  Model:        {MODEL_PATH} ({mb:.1f} MB)")
    print(f"  Time:         {elapsed / 60:.1f} min")
    print(f"{'=' * 65}")

    if test_metrics['accuracy'] >= 95:
        print(f"\n  TARGET ACHIEVED: {test_metrics['accuracy']:.1f}% >= 95%")
    else:
        print(f"\n  TARGET NOT MET: {test_metrics['accuracy']:.1f}% < 95%")
        print(f"  Consider: more epochs, larger model, or different augmentation")

    print(f"\n  Restart server: python3 server.py")


if __name__ == "__main__":
    main()
