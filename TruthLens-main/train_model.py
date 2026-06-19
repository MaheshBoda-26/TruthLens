"""
TruthLens ML Model Training (v8) — ResNet18 Fine-Tuning
=======================================================
Uses:
1. Deepfake Dataset (manjilkarki) — 70K+ real + fake images from Kaggle

Training: 20K images per class
Validation: 4K images per class
"""

import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model_def import create_model

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")

# Config
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "6"))
LR = float(os.getenv("LR", "3e-4"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
MAX_TRAIN = int(os.getenv("MAX_TRAIN", "20000"))
MAX_VAL = int(os.getenv("MAX_VAL", "4000"))
FREEZE_BACKBONE = os.getenv("FREEZE_BACKBONE", "1" if DEVICE.type == "cpu" else "0") == "1"
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "deepfake_detector.pth"

# Dataset path
DEEPFAKE_PATH = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"

# Transforms
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.03),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DeepfakeDataset(Dataset):
    """Dataset that loads real/fake images from folder structure."""
    def __init__(self, fake_dir, real_dir, transform=None, max_per_class=None):
        self.samples = []
        self.transform = transform

        # Load fake images
        if fake_dir and fake_dir.exists():
            fake_files = list(fake_dir.glob("*"))
            if max_per_class:
                fake_files = fake_files[:max_per_class]
            for f in fake_files:
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((f, 1))  # 1 = fake

        # Load real images
        if real_dir and real_dir.exists():
            real_files = list(real_dir.glob("*"))
            if max_per_class:
                real_files = real_files[:max_per_class]
            for f in real_files:
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((f, 0))  # 0 = real

        random.shuffle(self.samples)
        print(f"  Loaded {len(self.samples)} images (Fake: {sum(1 for _,l in self.samples if l==1)}, Real: {sum(1 for _,l in self.samples if l==0)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            # Return a blank image if loading fails
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return img, label

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total*100:.1f}%'})

    return total_loss / len(loader), correct / total * 100


def configure_trainable_layers(model):
    if not FREEZE_BACKBONE:
        return

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.backbone.fc.parameters():
        param.requires_grad = True


def compute_metrics(probs, labels, threshold):
    preds = (probs >= threshold).float()
    correct = (preds == labels).sum().item()
    total = labels.numel()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    accuracy = correct / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


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
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    best_threshold = 0.5
    best_metrics = compute_metrics(probs, labels, threshold=0.5)
    for threshold in [x / 100 for x in range(20, 81, 2)]:
        metrics = compute_metrics(probs, labels, threshold=threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_threshold = threshold

    return total_loss / len(loader), best_threshold, best_metrics


def main():
    print("="*60)
    print("🔬 TruthLens Model Training v8 — ResNet18 Fine-Tuning")
    print("="*60)
    t0 = time.time()

    # Load Deepfake Dataset (primary)
    print("\n📦 Loading Deepfake Dataset...")
    train_fake = DEEPFAKE_PATH / "Train" / "Fake"
    train_real = DEEPFAKE_PATH / "Train" / "Real"
    val_fake = DEEPFAKE_PATH / "Validation" / "Fake"
    val_real = DEEPFAKE_PATH / "Validation" / "Real"

    train_ds = DeepfakeDataset(train_fake, train_real, train_tf, max_per_class=MAX_TRAIN)
    val_ds = DeepfakeDataset(val_fake, val_real, val_tf, max_per_class=MAX_VAL)

    print(f"\n📊 Total Training: {len(train_ds)} images")
    print(f"📊 Total Validation: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("\n🏗️  ResNet18 fine-tuning")
    model = create_model(DEVICE)
    configure_trainable_layers(model)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Freeze Backbone: {FREEZE_BACKBONE}")

    # Training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_f1 = 0
    best_state = None
    best_threshold = 0.5

    print(f"\n🚀 Training {NUM_EPOCHS} epochs...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, threshold, metrics = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(metrics["f1"])

        print(f"\n📈 Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {metrics['accuracy']:.1f}%")
        print(f"  Precision:  {metrics['precision']:.1f}% | Recall: {metrics['recall']:.1f}% | F1: {metrics['f1']:.1f}%")
        print(f"  Threshold:  {threshold:.2f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✅ New best F1: {best_f1:.1f}%")

    # Save best model
    if best_state:
        model.load_state_dict(best_state)

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_accuracy': best_f1,
        'best_threshold': best_threshold,
        'image_size': IMG_SIZE,
        'architecture': 'resnet18_binary_v8',
        'training_samples': len(train_ds),
    }, MODEL_PATH)

    mb = MODEL_PATH.stat().st_size / (1024*1024)
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"  Best F1 Score: {best_f1:.1f}%")
    print(f"  Model: {MODEL_PATH} ({mb:.1f} MB)")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")
    print(f"\n▶ Restart server: python3 server.py")


if __name__ == "__main__":
    main()
