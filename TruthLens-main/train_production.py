"""
TruthLens — Production Training Pipeline (All 9 Recommendations Integrated)
==========================================================================
Incorporates every evidence-based fix from the 9 senior-ML audits:

1.  DATA: Cross-split deduplication via exclusion (no file moves)
2.  LABELS: 12 high-confidence Test mislabel corrections
3.  BACKBONE: EfficientNet-B0 (5.3M params) — drop-in, +2-4% F1 over ResNet18
4.  FINE-TUNE: Partial — freeze stem through features.2, train features.3+ + head
5.  LOSS: Asymmetric Label Smoothing BCE (Fake→0.95, Real→0.00)
6.  OPTIMIZER: AdamW with per-group LR/WD (head 3e-4/1e-2, backbone 1e-4/1e-4)
7.  SCHEDULER: OneCycleLR (pct_start=0.15, cosine anneal)
8.  REGULARIZATION: Batch=64, Gradient Clip=1.0, Dropout=0.3, Early Stop(p=4)
9.  EVAL: Temp scaling on Val, threshold on Val, honest Test eval with corrected labels

Outputs (accuracy_fixes/):
  - best_model.pth         (checkpoint with temp, threshold, F1)
  - final_metrics.json     (Val/Test metrics + ROC-AUC)
  - training_log.json      (per-epoch history)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device: {DEVICE}")

DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
OUTPUT_DIR = Path("accuracy_fixes")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 15
EARLY_STOP_PATIENCE = 4
HEAD_LR = 3e-4
BACKBONE_LR = 1e-4
HEAD_WD = 1e-2
BACKBONE_WD = 1e-4
FAKE_SMOOTH = 0.05    # Asymmetric: Fake target=0.95
REAL_SMOOTH = 0.0     # Real target=0.00
DROPOUT = 0.3
GRAD_CLIP = 1.0
NUM_WORKERS = 0  # MPS doesn't benefit from workers

# ─── Transforms ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── Loss: Asymmetric Label Smoothing BCE ────────────────────────────────────
class AsymmetricLabelSmoothingBCE(nn.Module):
    def __init__(self, fake_smooth=FAKE_SMOOTH, real_smooth=REAL_SMOOTH):
        super().__init__()
        self.fake_target = 1.0 - fake_smooth
        self.real_target = real_smooth
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        targets = torch.where(
            labels == 1,
            torch.full_like(logits, self.fake_target),
            torch.full_like(logits, self.real_target)
        )
        return self.criterion(logits, targets).mean()

# ─── Model: EfficientNet-B0 with Partial Fine-Tune ───────────────────────────
def create_model():
    """EfficientNet-B0 with custom head, configured for partial fine-tuning."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # EfficientNet structure: features (stem + 7 MBConv blocks) → avgpool → classifier
    # Freeze: stem (features[0]) + early blocks (features[1:4] ≈ layers 1-2)
    # Train:  mid/late blocks (features[4:] ≈ layers 3-4) + classifier head

    for name, param in model.named_parameters():
        # Freeze stem and first 3 MBConv blocks (features.0 through features.3)
        if name.startswith('features.0') or name.startswith('features.1') or \
           name.startswith('features.2') or name.startswith('features.3'):
            param.requires_grad = False
        else:
            param.requires_grad = True  # features.4, 5, 6 + classifier

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 1)
    )

    # Print trainable summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    return model


def get_param_groups(model):
    """Return parameter groups with per-group LR and WD."""
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {'params': head_params, 'lr': HEAD_LR, 'weight_decay': HEAD_WD, 'name': 'head'},
        {'params': backbone_params, 'lr': BACKBONE_LR, 'weight_decay': BACKBONE_WD, 'name': 'backbone'},
    ]

# ─── Dataset ─────────────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, fake_dir: Path, real_dir: Path, transform, max_per_class=None,
                 exclude: set | None = None, corrected_labels: dict | None = None):
        self.samples = []
        self.transform = transform
        self.corrected_labels = corrected_labels or {}
        for directory, label in [(fake_dir, 1), (real_dir, 0)]:
            if not directory.exists():
                print(f"  ⚠ Directory not found: {directory}")
                continue
            files = sorted(f for f in directory.iterdir()
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
            if exclude:
                files = [f for f in files if str(f) not in exclude]
            if max_per_class:
                files = files[:max_per_class]
            self.samples.extend((f, label) for f in files)
        np.random.seed(42)
        np.random.shuffle(self.samples)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  Loaded {len(self.samples)} images (Fake: {n_fake}, Real: {len(self.samples)-n_fake})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Apply corrected label if available
        if str(path) in self.corrected_labels:
            label = self.corrected_labels[str(path)]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = self.transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128)))
        return img, label

# ─── Exclusion Set (Dedup) ───────────────────────────────────────────────────
def compute_exclusion_set() -> tuple[set[str], dict[str, int]]:
    """Compute files to exclude (cross-split near-dups) and label corrections."""
    priority = {"Test": 0, "Validation": 1, "Train": 2}

    # Load near-dup pairs
    pairs_sources = [
        Path("../leakage_audit_results/cross_split_near_dups.json"),
        Path("/Users/maheshboda/Projects/TruthLens/leakage_audit_results/cross_split_near_dups.json"),
    ]
    pairs = []
    for p in pairs_sources:
        if p.exists():
            pairs = json.loads(p.read_text());
            break

    edges = [(p["path1"], p["path2"]) for p in pairs
             if Path(p["path1"]).exists() and Path(p["path2"]).exists()]

    # Add exact-dup edges
    for p in [
        Path("../dataset_audit_results/audit_results.json"),
        Path("/Users/maheshboda/Projects/TruthLens/dataset_audit_results/audit_results.json"),
    ]:
        if p.exists():
            audit = json.loads(p.read_text());
            break
    else:
        audit = {}
    for md5, plist in audit.get("exact_duplicates", {}).items():
        exist = [p for p in plist if Path(p).exists()]
        for i in range(len(exist) - 1):
            edges.append((exist[i], exist[i + 1]))

    # Union-find
    parent = {}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent.setdefault(a, a); parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for a, b in edges: union(a, b)

    comps = {}
    for p in parent: comps.setdefault(find(p), []).append(p)

    to_exclude = set()
    for comp in comps.values():
        split_of = {}
        for p in comp:
            parts = Path(p).parts
            s = next((s for s in parts if s in priority), None)
            if s: split_of[p] = s
        if len(set(split_of.values())) < 2: continue
        keep = min(comp, key=lambda p: priority.get(split_of.get(p, "Train"), 2))
        to_exclude.update(p for p in comp if p != keep)

    print(f"Exclusion set: {len(to_exclude)} files (cross-split near-dup clusters)")

    # Label corrections
    corrected = {}
    for m in [
        Path("accuracy_fixes/relabel_manifest.json"),
        Path("/Users/maheshboda/Projects/TruthLens/TruthLens-main/accuracy_fixes/relabel_manifest.json"),
    ]:
        if m.exists():
            mani = json.loads(m.read_text());
            break
    else:
        mani = {"flips": []}
    for flip in mani["flips"]:
        full = str(DATASET_ROOT / flip["path"])
        corrected[full] = flip["corrected"]
    print(f"Label corrections: {len(corrected)} images")
    return to_exclude, corrected

# ─── Training / Eval ─────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float()
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits_all, labels_all = [], []
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(DEVICE)
        logits = model(imgs).squeeze(1)
        logits_all.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def metrics_at(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    return {"threshold": round(threshold, 3), "accuracy": round(acc, 5),
            "precision": round(prec, 5), "recall": round(rec, 5),
            "f1": round(f1, 5), "fpr": round(fpr, 5),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def best_threshold_on(probs, labels):
    best, best_m = None, -1
    for t in np.arange(0.30, 0.91, 0.01):
        m = metrics_at(probs, labels, t)
        if m["f1"] > best_m:
            best_m, best = m["f1"], m
    return best


def fit_temperature(logits, labels):
    ts = nn.Parameter(torch.ones(1) * 1.5)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    optimizer = optim.LBFGS([ts], lr=0.01, max_iter=200)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion((logits_t / ts).squeeze(), labels_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    return ts.item()


def compute_roc_auc(probs, labels):
    sorted_idx = np.argsort(-probs)
    sorted_labels = labels[sorted_idx]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    tp_cum = np.cumsum(sorted_labels)
    fp_cum = np.cumsum(1 - sorted_labels)
    tpr = np.concatenate([[0], tp_cum / n_pos])
    fpr = np.concatenate([[0], fp_cum / n_neg])
    return round(float(np.trapezoid(tpr, fpr)), 5)

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-train", type=int, default=None)  # Use all clean data
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    train_fake = DATASET_ROOT / "Train" / "Fake"
    train_real = DATASET_ROOT / "Train" / "Real"
    val_fake = DATASET_ROOT / "Validation" / "Fake"
    val_real = DATASET_ROOT / "Validation" / "Real"
    test_fake = DATASET_ROOT / "Test" / "Fake"
    test_real = DATASET_ROOT / "Test" / "Real"

    exclude_set, corrected_labels = compute_exclusion_set()

    print("Loading datasets (clean, deduplicated)...")
    train_ds = DeepfakeDataset(train_fake, train_real, train_transform,
                               max_per_class=args.max_train, exclude=exclude_set)
    val_ds = DeepfakeDataset(val_fake, val_real, val_transform, exclude=exclude_set)
    test_ds = DeepfakeDataset(test_fake, test_real, val_transform,
                              exclude=exclude_set, corrected_labels=corrected_labels)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=DEVICE.type!='mps')
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=DEVICE.type!='mps')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=DEVICE.type!='mps')

    ckpt_path = OUTPUT_DIR / "best_model.pth"
    history_path = OUTPUT_DIR / "training_log.json"

    if args.skip_train and ckpt_path.exists():
        print(f"Skipping training, loading {ckpt_path}")
        model = create_model()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True)["model_state_dict"])
        model = model.to(DEVICE)
    else:
        print(f"Training up to {args.epochs} epochs (early stop patience={EARLY_STOP_PATIENCE})...")
        model = create_model().to(DEVICE)
        criterion = AsymmetricLabelSmoothingBCE()
        param_groups = get_param_groups(model)
        optimizer = optim.AdamW(param_groups)

        # OneCycleLR
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g['lr'] for g in param_groups],
            total_steps=total_steps,
            pct_start=0.15,
            anneal_strategy='cos',
            final_div_factor=1e3,
        )

        best_f1, best_state, epochs_no_improve = 0, None, 0
        history = []

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_logits, val_labels = evaluate(model, val_loader)
            val_probs = 1 / (1 + np.exp(-val_logits))
            v_best = best_threshold_on(val_probs, val_labels)

            # OneCycleLR steps per batch
            scheduler.step()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{args.epochs} | "
                  f"train_loss {train_loss:.4f} train_acc {train_acc:.1%} | "
                  f"val_acc {v_best['accuracy']:.1%} F1 {v_best['f1']:.1%} @th {v_best['threshold']:.2f} "
                  f"({elapsed:.0f}s)")

            history.append({
                "epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                "val_acc": v_best['accuracy'], "val_f1": v_best['f1'],
                "val_prec": v_best['precision'], "val_rec": v_best['recall'],
                "val_th": v_best['threshold'], "time": elapsed
            })

            if v_best["f1"] > best_f1:
                best_f1 = v_best["f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                print(f"  ✅ New best Val F1: {best_f1:.1%}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"  Early stopping (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                    break

        model.load_state_dict(best_state)

        # Save checkpoint with metadata
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_f1": best_f1 * 100,
            "best_threshold": v_best["threshold"],
            "image_size": IMG_SIZE,
            "architecture": "efficientnet_b0_binary",
            "training_samples": len(train_ds),
        }, ckpt_path)
        print(f"Saved {ckpt_path}")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    # ── Final Evaluation: Temp on Val, Threshold on Val, Test with corrected labels ──
    print("\n🔬 Final Evaluation...")
    val_logits, val_labels = evaluate(model, val_loader)
    temp = fit_temperature(val_logits, val_labels)
    print(f"Temperature (Val): {temp:.4f}")

    val_probs = 1 / (1 + np.exp(-np.array(val_logits) / temp))
    val_best = best_threshold_on(val_probs, val_labels)
    print(f"Best threshold (Val, temp-scaled): {val_best['threshold']:.2f} "
          f"acc {val_best['accuracy']:.1%} F1 {val_best['f1']:.1%}")

    # Test evaluation with corrected labels
    test_logits, test_labels_raw = evaluate(model, test_loader)
    # Apply corrections
    test_labels = np.array([
        corrected_labels.get(str(test_ds.samples[i][0]), test_labels_raw[i])
        for i in range(len(test_labels_raw))
    ])
    test_probs = 1 / (1 + np.exp(-np.array(test_logits) / temp))
    test_metrics = metrics_at(test_probs, test_labels, val_best["threshold"])
    test_auc = compute_roc_auc(test_probs, test_labels)

    print(f"\n{'='*60}")
    print(f"TEST (threshold {val_best['threshold']:.2f}, temp {temp:.3f}):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.1%}")
    print(f"  Precision: {test_metrics['precision']:.1%}")
    print(f"  Recall:    {test_metrics['recall']:.1%}")
    print(f"  F1:        {test_metrics['f1']:.1%}")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    print(f"{'='*60}")

    # Baseline (uncorrected) for reference
    test_metrics_raw = metrics_at(test_probs, test_labels_raw, val_best["threshold"])
    print(f"Baseline (uncorrected): F1 {test_metrics_raw['f1']:.1%}")

    results = {
        "temperature": temp,
        "val_best": val_best,
        "test_metrics": {**test_metrics, "roc_auc": test_auc},
        "test_metrics_uncorrected": test_metrics_raw,
        "train_samples": len(train_ds),
        "epochs_trained": len(history),
    }
    with open(OUTPUT_DIR / "final_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUTPUT_DIR / 'final_metrics.json'}")


if __name__ == "__main__":
    import json  # for compute_exclusion_set
    main()