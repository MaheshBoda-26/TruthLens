"""
TruthLens Hard Negative Mining & Retraining Pipeline
=====================================================
Identifies false positives (real images predicted as fake),
mines them as hard negatives, creates an augmented replay dataset
with weighted oversampling, and retrains with curriculum learning.

Architecture:
  1. Mine Phase   — run inference, collect FPs ranked by confidence
  2. Augment Phase — JPEG compression augmentation to simulate social
                     media, beauty filters, low-light, compressed selfies
  3. Retrain Phase — curriculum stages (easy → medium → hard) with
                     focal loss and oversampled hard negatives
  4. Log Phase    — track FPR/F1/Precision across iterations

Usage:
    ./venv/bin/python hard_negative_mining.py                    # Full pipeline
    ./venv/bin/python hard_negative_mining.py --mine-only        # Mine without retraining
    ./venv/bin/python hard_negative_mining.py --quick            # Quick run (2K samples)
    ./venv/bin/python hard_negative_mining.py --iterations 3     # 3 mine-retrain cycles
"""

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

from model_def import create_model

# ─── Config ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("model/deepfake_detector.pth")
DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
HN_DIR = Path("hard_negatives")
LOG_DIR = Path("mining_logs")
IMG_SIZE = 224
BATCH_SIZE = 32


# ─── Transforms ──────────────────────────────────────────────────────────────

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Standard training transform
base_train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.03),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class JPEGCompressTransform:
    """Simulate social media / messaging app compression."""
    def __init__(self, quality_range=(15, 60)):
        self.quality_range = quality_range

    def __call__(self, img):
        import io
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class BeautyFilterTransform:
    """Simulate beauty filter smoothing (skin softening)."""
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(1.1, 1.4))
        return img


class LowLightTransform:
    """Simulate low-light / dark phone camera photos."""
    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.4, 0.75))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.0))
        return img


# Hard-negative-specific training transform with aggressive augmentation
hn_train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomChoice([
        JPEGCompressTransform(quality_range=(15, 50)),
        BeautyFilterTransform(),
        LowLightTransform(),
        transforms.Lambda(lambda x: x),  # identity (no extra transform)
    ]),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Focal Loss ──────────────────────────────────────────────────────────────

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Down-weights easy examples, focuses learning on hard FP/FN boundary.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ─── Datasets ────────────────────────────────────────────────────────────────

class ImagePathDataset(Dataset):
    """Dataset that returns (tensor, label, path)."""
    def __init__(self, fake_dir, real_dir, transform=None, max_per_class=None):
        self.samples = []
        self.transform = transform or val_tf
        for d, label in [(fake_dir, 1), (real_dir, 0)]:
            if d and d.exists():
                files = sorted(f for f in d.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
                if max_per_class:
                    files = files[:max_per_class]
                self.samples.extend((f, label) for f in files)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img)
        except Exception:
            tensor = self.transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128,128,128)))
        return tensor, label, str(path)


class HardNegativeDataset(Dataset):
    """Loads mined hard negatives from a manifest file."""
    def __init__(self, manifest_path: Path, transform=None, oversample_factor=3):
        self.transform = transform or hn_train_tf
        self.samples = []

        with open(manifest_path) as f:
            manifest = json.load(f)

        for entry in manifest:
            path = Path(entry["path"])
            if path.exists():
                # Oversample hard negatives so they appear more often
                for _ in range(oversample_factor):
                    self.samples.append((path, 0, entry["prob"]))

        print(f"  HN dataset: {len(manifest)} unique images × {oversample_factor} = {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img)
        except Exception:
            tensor = self.transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128,128,128)))
        return tensor, label


class SimpleDataset(Dataset):
    """Simple (tensor, label) dataset for standard training."""
    def __init__(self, fake_dir, real_dir, transform=None, max_per_class=None):
        self.samples = []
        self.transform = transform or base_train_tf
        for d, label in [(fake_dir, 1), (real_dir, 0)]:
            if d and d.exists():
                files = sorted(f for f in d.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
                if max_per_class:
                    files = files[:max_per_class]
                self.samples.extend((f, label) for f in files)
        random.shuffle(self.samples)
        nf = sum(1 for _,l in self.samples if l==1)
        nr = sum(1 for _,l in self.samples if l==0)
        print(f"  Base dataset: {len(self.samples)} images (Fake={nf}, Real={nr})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img)
        except Exception:
            tensor = self.transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128,128,128)))
        return tensor, label


# ─── Phase 1: MINE hard negatives ────────────────────────────────────────────

def mine_hard_negatives(model, threshold=0.52, split="Train", max_per_class=None,
                        difficulty_bins=(0.52, 0.65, 0.80)):
    """
    Run inference on real images, collect false positives, and classify
    them into curriculum difficulty tiers.

    Returns: dict with 'easy', 'medium', 'hard' lists of {path, prob}
    """
    model.eval()
    real_dir = DATASET_ROOT / split / "Real"
    ds = ImagePathDataset(None, real_dir, transform=val_tf, max_per_class=max_per_class)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    fps = []
    n_real = 0
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc=f"Mining {split}"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            for p, path, label in zip(probs, paths, labels.numpy()):
                if label == 0:
                    n_real += 1
                    if p >= threshold:
                        fps.append({"path": path, "prob": round(float(p), 5)})

    # Sort by confidence (worst offenders first)
    fps.sort(key=lambda x: -x["prob"])

    lo, mid, hi = difficulty_bins
    easy   = [x for x in fps if lo <= x["prob"] < mid]
    medium = [x for x in fps if mid <= x["prob"] < hi]
    hard   = [x for x in fps if x["prob"] >= hi]

    print(f"\n  Mined {len(fps)} FPs from {n_real} real images ({len(fps)/max(n_real,1)*100:.1f}% FPR)")
    print(f"  Easy (>{lo:.2f}):   {len(easy)}")
    print(f"  Medium (>{mid:.2f}): {len(medium)}")
    print(f"  Hard (>{hi:.2f}):   {len(hard)}")

    return {"all": fps, "easy": easy, "medium": medium, "hard": hard, "n_real": n_real}


def save_hard_negatives(mined: dict, iteration: int):
    """Save mined hard negatives to disk as a manifest + sample images."""
    iter_dir = HN_DIR / f"iter_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = iter_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(mined["all"], f, indent=2)

    # Save top 50 sample images for visual inspection
    samples_dir = iter_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    import shutil
    for i, entry in enumerate(mined["all"][:50]):
        src = Path(entry["path"])
        if src.exists():
            dst = samples_dir / f"hn_{i:03d}_p{entry['prob']:.3f}_{src.name}"
            shutil.copy2(src, dst)

    print(f"  Saved manifest: {manifest_path} ({len(mined['all'])} entries)")
    return manifest_path


# ─── Phase 2: BUILD training dataset with oversampling ───────────────────────

def build_training_loader(manifest_path: Path, max_base_per_class, oversample_factor=3,
                          curriculum_stage="all"):
    """
    Create a DataLoader that combines the base training data with
    oversampled hard negatives and weighted sampling.
    """
    # Base dataset
    base_ds = SimpleDataset(
        DATASET_ROOT / "Train" / "Fake",
        DATASET_ROOT / "Train" / "Real",
        transform=base_train_tf,
        max_per_class=max_base_per_class,
    )

    # Hard negative replay dataset
    hn_ds = HardNegativeDataset(manifest_path, transform=hn_train_tf,
                                oversample_factor=oversample_factor)

    # Combine
    combined = ConcatDataset([base_ds, hn_ds])
    n_base = len(base_ds)
    n_hn = len(hn_ds)
    total = n_base + n_hn

    # Weighted sampling: hard negatives get 2× weight
    weights = []
    for i in range(total):
        if i < n_base:
            _, label = base_ds[i] if False else (None, base_ds.samples[i % len(base_ds.samples)][1])
            weights.append(1.0)
        else:
            weights.append(2.0)  # Hard negatives weighted 2×

    sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)
    loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    print(f"  Combined: {n_base} base + {n_hn} HN = {total} total")
    return loader


# ─── Phase 3: RETRAIN with focal loss & curriculum ───────────────────────────

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
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def validate_model(model, split="Validation", max_per_class=4000, threshold=0.5):
    """Run full validation and return metrics + optimal threshold."""
    model.eval()
    ds = ImagePathDataset(
        DATASET_ROOT / split / "Fake",
        DATASET_ROOT / split / "Real",
        transform=val_tf, max_per_class=max_per_class,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # Find optimal threshold by F1
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.30, 0.85, 0.02):
        m = compute_metrics(probs, labels, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t

    metrics = compute_metrics(probs, labels, best_t)
    metrics["threshold"] = round(best_t, 3)
    return metrics


def retrain_epoch(model, loader, criterion, optimizer):
    """Single training epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        imgs, labels = batch[0].to(DEVICE), batch[1].float().to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs).squeeze(1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(out) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total*100:.1f}%")
    return total_loss / len(loader), correct / total * 100


def retrain_with_hard_negatives(model, manifest_path, epochs=4, lr=1e-4,
                                max_base=10000, oversample=3):
    """
    Fine-tune the model using combined base + hard negative dataset
    with focal loss.
    """
    print(f"\n🏋️ Retraining with hard negatives ({epochs} epochs, lr={lr})...")

    loader = build_training_loader(manifest_path, max_base, oversample)
    criterion = FocalBCEWithLogitsLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1, best_state, best_threshold = 0, None, 0.5

    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = retrain_epoch(model, loader, criterion, optimizer)
        scheduler.step()

        metrics = validate_model(model, max_per_class=4000)
        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%")
        print(f"  Val F1: {metrics['f1']:.1f}%  Precision: {metrics['precision']:.1f}%"
              f"  FPR: {metrics['fpr']:.1f}%  Threshold: {metrics['threshold']:.2f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = metrics["threshold"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return best_f1, best_threshold


# ─── Phase 4: LOGGING ────────────────────────────────────────────────────────

def log_iteration(iteration, mined, metrics, log_path):
    """Append iteration results to the training log."""
    entry = {
        "iteration": iteration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fps_mined": len(mined["all"]),
        "n_easy": len(mined["easy"]),
        "n_medium": len(mined["medium"]),
        "n_hard": len(mined["hard"]),
        "fpr_before_mining": round(len(mined["all"]) / max(mined["n_real"], 1) * 100, 2),
        "val_f1": round(metrics["f1"], 2),
        "val_precision": round(metrics["precision"], 2),
        "val_recall": round(metrics["recall"], 2),
        "val_fpr": round(metrics["fpr"], 2),
        "val_threshold": metrics["threshold"],
    }

    log = []
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return entry


def print_progress_table(log_path):
    """Print a summary table of all iterations."""
    if not log_path.exists():
        return
    with open(log_path) as f:
        log = json.load(f)

    print(f"\n{'═'*75}")
    print("  MINING PROGRESS ACROSS ITERATIONS")
    print(f"{'═'*75}")
    print(f"  {'Iter':>4}  {'FPs Mined':>10}  {'FPR%':>6}  {'F1%':>6}  {'Prec%':>6}  {'Rec%':>6}  {'Thresh':>7}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}")
    for e in log:
        print(f"  {e['iteration']:>4}  {e['fps_mined']:>10}  "
              f"{e['fpr_before_mining']:>6.1f}  {e['val_f1']:>6.1f}  "
              f"{e['val_precision']:>6.1f}  {e['val_recall']:>6.1f}  "
              f"{e['val_threshold']:>7.2f}")

    if len(log) >= 2:
        first, last = log[0], log[-1]
        print(f"\n  Δ FPR:       {first['fpr_before_mining']:.1f}% → {last['fpr_before_mining']:.1f}%")
        print(f"  Δ F1:        {first['val_f1']:.1f}% → {last['val_f1']:.1f}%")
        print(f"  Δ Precision: {first['val_precision']:.1f}% → {last['val_precision']:.1f}%")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TruthLens Hard Negative Mining")
    parser.add_argument("--iterations", type=int, default=2, help="Mine→retrain cycles")
    parser.add_argument("--mine-only", action="store_true", help="Mine without retraining")
    parser.add_argument("--quick", action="store_true", help="Quick run (2K samples)")
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per retrain cycle")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--oversample", type=int, default=3, help="HN oversample factor")
    parser.add_argument("--max-base", type=int, default=10000, help="Base training images per class")
    args = parser.parse_args()

    HN_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / "mining_log.json"
    t0 = time.time()

    max_mine = 2000 if args.quick else None
    max_val = 1000 if args.quick else 4000
    max_base = 2000 if args.quick else args.max_base

    print("=" * 65)
    print("  TruthLens Hard Negative Mining Pipeline")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Quick: {args.quick}")

    # Load model
    print("\n📦 Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    current_threshold = checkpoint.get("best_threshold", 0.5)
    print(f"  Loaded (F1={checkpoint.get('best_accuracy',0):.1f}%, threshold={current_threshold:.2f})")

    # Baseline validation
    print("\n📊 Baseline validation...")
    baseline = validate_model(model, max_per_class=max_val)
    print(f"  F1={baseline['f1']:.1f}%  Precision={baseline['precision']:.1f}%"
          f"  FPR={baseline['fpr']:.1f}%  Threshold={baseline['threshold']:.2f}")

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'═'*65}")
        print(f"  ITERATION {iteration}/{args.iterations}")
        print(f"{'═'*65}")

        # ── Mine ──
        print("\n⛏️  Mining hard negatives...")
        mined = mine_hard_negatives(model, threshold=args.threshold,
                                    split="Train", max_per_class=max_mine)
        manifest = save_hard_negatives(mined, iteration)

        if args.mine_only:
            metrics = validate_model(model, max_per_class=max_val)
            log_iteration(iteration, mined, metrics, log_path)
            continue

        # ── Retrain ──
        best_f1, best_threshold = retrain_with_hard_negatives(
            model, manifest, epochs=args.epochs, lr=args.lr,
            max_base=max_base, oversample=args.oversample,
        )

        # ── Validate ──
        metrics = validate_model(model, max_per_class=max_val)
        print(f"\n  Post-retrain: F1={metrics['f1']:.1f}%  Precision={metrics['precision']:.1f}%"
              f"  FPR={metrics['fpr']:.1f}%")

        # ── Log ──
        log_iteration(iteration, mined, metrics, log_path)

        # ── Update threshold for next mining pass ──
        args.threshold = metrics["threshold"]

    # ── Save updated model ──
    if not args.mine_only:
        save_path = Path("model/deepfake_detector_hn.pth")
        final_metrics = validate_model(model, max_per_class=max_val)
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_accuracy": final_metrics["f1"],
            "best_threshold": final_metrics["threshold"],
            "image_size": IMG_SIZE,
            "architecture": "resnet18_binary_v8_hn",
            "hard_negative_iterations": args.iterations,
        }, save_path)
        print(f"\n💾 Model saved: {save_path}")

    # ── Print progress ──
    print_progress_table(log_path)

    elapsed = time.time() - t0
    print(f"\n✅ Pipeline complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
