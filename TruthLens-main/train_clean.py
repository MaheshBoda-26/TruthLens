"""
TruthLens — Clean-Data Retrain + Honest Eval
============================================
Trains ResNet18 v8 on the deduplicated (leakage-free) Train/Validation split
and evaluates on the (deduplicated, relabeled) Test split.

Why a separate script:
  - train_model.py saves to model/deepfake_detector.pth (the served checkpoint).
    We must not clobber the baseline before we have a better model to replace it.
  - We fit temperature scaling on VALIDATION and pick the threshold on
    VALIDATION, then report metrics on TEST — no eval contamination.

Outputs (all in accuracy_fixes/):
  - clean_train_model.pth   (the retrained checkpoint)
  - clean_metrics.json      (validation metrics + chosen threshold + test metrics)
  - honest_eval.txt         (human-readable comparison)
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
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model_def import create_model

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device: {DEVICE}")

DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
OUTPUT_DIR = Path("accuracy_fixes")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 6
LR = 3e-4
FREEZE_BACKBONE = False  # fine-tune the whole network for best accuracy

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.03),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DeepfakeDataset(Dataset):
    def __init__(self, fake_dir: Path, real_dir: Path, transform, max_per_class=None,
                 exclude: set | None = None):
        self.samples = []
        self.transform = transform
        for directory, label in [(fake_dir, 1), (real_dir, 0)]:
            if not directory.exists():
                print(f"  ⚠ Directory not found: {directory}")
                continue
            files = sorted(f for f in directory.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
            if exclude:
                files = [f for f in files if str(f) not in exclude]
            if max_per_class:
                files = files[:max_per_class]
            self.samples.extend((f, label) for f in files)
        np.random.seed(42)
        np.random.shuffle(self.samples)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  Loaded {len(self.samples)} images (Fake: {n_fake}, Real: {len(self.samples) - n_fake})")

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


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float()
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
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
    return {"threshold": round(threshold, 3), "accuracy": round(acc, 5), "precision": round(prec, 5),
            "recall": round(rec, 5), "f1": round(f1, 5), "fpr": round(fpr, 5),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def best_threshold_on(probs, labels):
    """Pick threshold maximizing F1 on this (validation) set."""
    best, best_m = None, -1
    for t in np.arange(0.30, 0.91, 0.01):
        m = metrics_at(probs, labels, t)
        if m["f1"] > best_m:
            best_m, best = m["f1"], m
    return best


def fit_temperature(logits, labels, lr=0.01, max_iter=200):
    ts = nn.Parameter(torch.ones(1) * 1.5)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    optimizer = optim.LBFGS([ts], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion((logits_t / ts).squeeze(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return ts.item()


def compute_exclusion_set() -> tuple[set[str], dict[str, str]]:
    """
    Compute which files to exclude to make Train / Validation / Test pairwise
    disjoint, and which labels to correct.

    Returns:
        exclude_paths: set of absolute-path strings to exclude.
        corrected_labels: {path: corrected_label_int} for high-confidence mislabels.
    """
    # Strategy: load near-dup + exact-dup edges, run union-find, keep the
    # highest-priority file per component (keep Test > Validation > Train).
    priority = {"Test": 0, "Validation": 1, "Train": 2}

    # Load near-dup pairs
    pairs_sources = [
        Path("../leakage_audit_results/cross_split_near_dups.json"),
        Path("/Users/maheshboda/Projects/TruthLens/leakage_audit_results/cross_split_near_dups.json"),
    ]
    pairs = []
    for p in pairs_sources:
        if p.exists(): pairs = json.loads(p.read_text()); break

    edges = [(p["path1"], p["path2"]) for p in pairs if Path(p["path1"]).exists() and Path(p["path2"]).exists()]

    # Add exact-dup edges from prior audit
    for p in [
        Path("../dataset_audit_results/audit_results.json"),
        Path("/Users/maheshboda/Projects/TruthLens/dataset_audit_results/audit_results.json"),
    ]:
        if p.exists(): audit = json.loads(p.read_text()); break
    else:
        audit = {}
    for md5, plist in audit.get("exact_duplicates", {}).items():
        exist = [p for p in plist if Path(p).exists()]
        for i in range(len(exist) - 1):
            edges.append((exist[i], exist[i + 1]))

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
        if len(set(split_of.values())) < 2:
            continue
        keep = min(comp, key=lambda p: priority.get(split_of.get(p, "Train"), 2))
        to_exclude.update(p for p in comp if p != keep)

    print(f"Exclusion set: {len(to_exclude)} files (cross-split near-dup clusters)")

    # Corrected labels manifest
    corrected = {}
    mani_paths = [
        Path("accuracy_fixes/relabel_manifest.json"),
        Path("/Users/maheshboda/Projects/TruthLens/TruthLens-main/accuracy_fixes/relabel_manifest.json"),
    ]
    for m in mani_paths:
        if m.exists(): mani = json.loads(m.read_text()); break
    else:
        mani = {"flips": []}
    for flip in mani["flips"]:
        full = str(DATASET_ROOT / flip["path"])
        corrected[full] = flip["corrected"]
    print(f"Label corrections: {len(corrected)} images")
    print(f"  → Test will use corrected labels for eval")

    return to_exclude, corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--skip-train", action="store_true", help="skip training, just eval existing ckpt")
    args = parser.parse_args()

    train_fake = DATASET_ROOT / "Train" / "Fake"
    train_real = DATASET_ROOT / "Train" / "Real"
    val_fake = DATASET_ROOT / "Validation" / "Fake"
    val_real = DATASET_ROOT / "Validation" / "Real"
    test_fake = DATASET_ROOT / "Test" / "Fake"
    test_real = DATASET_ROOT / "Test" / "Real"

    print("Loading datasets (deduplicated dirs)...")
    exclude_set, corrected_labels = compute_exclusion_set()
    train_ds = DeepfakeDataset(train_fake, train_real, train_transform,
                               max_per_class=args.max_train, exclude=exclude_set)
    val_ds = DeepfakeDataset(val_fake, val_real, val_transform, exclude=exclude_set)
    test_ds = DeepfakeDataset(test_fake, test_real, val_transform, exclude=exclude_set)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ckpt_path = OUTPUT_DIR / "clean_train_model.pth"
    if args.skip_train and ckpt_path.exists():
        print(f"Skipping training, loading {ckpt_path}")
        model = create_model()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True)["model_state_dict"])
        model = model.to(DEVICE)
    else:
        print(f"Training {args.epochs} epochs...")
        model = create_model(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
        best_f1, best_state = 0, None
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_logits, val_labels = evaluate(model, val_loader)
            val_probs = 1 / (1 + np.exp(-val_logits))
            v_best = best_threshold_on(val_probs, val_labels)
            scheduler.step(v_best["f1"])
            print(f"  Epoch {epoch+1}/{args.epochs} | train_loss {train_loss:.4f} train_acc {train_acc:.1%} "
                  f"| val acc {v_best['accuracy']:.1%} F1 {v_best['f1']:.1%} @th {v_best['threshold']:.2f}")
            if v_best["f1"] > best_f1:
                best_f1 = v_best["f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_accuracy": best_f1 * 100,
            "best_threshold": v_best["threshold"],
            "image_size": IMG_SIZE,
            "architecture": "resnet18_binary_v8_clean",
            "training_samples": len(train_ds),
        }, ckpt_path)
        print(f"Saved {ckpt_path}")

    # ── Fit temperature on VALIDATION, pick threshold on VALIDATION ──────
    val_logits, val_labels = evaluate(model, val_loader)
    temp = fit_temperature(val_logits, val_labels)
    print(f"Temperature (fit on Val): {temp:.4f}")
    val_probs = 1 / (1 + np.exp(-np.array(val_logits) / temp))
    val_best = best_threshold_on(val_probs, val_labels)
    print(f"Best threshold on Val (temp-scaled): {val_best['threshold']:.2f} "
          f"acc {val_best['accuracy']:.1%} F1 {val_best['f1']:.1%}")

    # ── Honest test evaluation with the chosen threshold ────────────────
    test_logits_dir, test_labels_dir = evaluate(model, test_loader)
    # Apply corrected labels for Test (high-confidence mislabels)
    test_labels_clean = np.array([
        corrected_labels.get(str(test_ds.samples[i][0]), test_labels_dir[i])
        for i in range(len(test_labels_dir))
    ])
    test_probs = 1 / (1 + np.exp(-np.array(test_logits_dir) / temp))
    test_metrics = metrics_at(test_probs, test_labels_clean, val_best["threshold"])
    test_auc = compute_roc_auc(test_probs, test_labels_clean)
    print(f"\nTEST (threshold {val_best['threshold']}): acc {test_metrics['accuracy']:.1%} "
          f"prec {test_metrics['precision']:.1%} rec {test_metrics['recall']:.1%} F1 {test_metrics['f1']:.1%} "
          f"AUC {test_auc:.4f}")

    # Also report baseline metrics (uncorrected labels) for comparison
    test_metrics_raw = metrics_at(test_probs, test_labels_dir, val_best["threshold"])
    print(f"  Baseline (uncorrected labels): acc {test_metrics_raw['accuracy']:.1%} "
          f"F1 {test_metrics_raw['f1']:.1%}")

    results = {
        "temperature": temp,
        "val_best": val_best,
        "test_metrics": {**test_metrics, "roc_auc": test_auc},
        "train_samples": len(train_ds),
        "epochs": args.epochs,
    }
    with open(OUTPUT_DIR / "clean_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUTPUT_DIR / 'clean_metrics.json'}")


def compute_roc_auc(probs, labels):
    """Manual ROC-AUC."""
    sorted_idx = np.argsort(-probs)
    sorted_labels = labels[sorted_idx]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp_cum = np.cumsum(sorted_labels)
    fp_cum = np.cumsum(1 - sorted_labels)
    tpr = np.concatenate([[0], tp_cum / n_pos])
    fpr = np.concatenate([[0], fp_cum / n_neg])
    return round(float(np.trapezoid(tpr, fpr)), 5)


if __name__ == "__main__":
    main()
