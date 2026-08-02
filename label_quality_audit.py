"""
TruthLens Label Quality Audit
==============================
Audits label quality of the deepfake detection dataset using:
1. Mechanical checks - cross-class exact/near duplicates, cross-split label conflicts
2. Model inference - find high-confidence mislabel candidates and ambiguous cases
3. Class boundary analysis - probability distribution, boundary width analysis

The audit REPORT only. It does NOT modify any labels.

Run: ./TruthLens-main/.venv/bin/python label_quality_audit.py [--limit N]
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, "TruthLens-main")
from model_def import create_model

# ─── Config ────────────────────────────────────────────────────────────────
DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
MODEL_BASE = Path("TruthLens-main/model/deepfake_detector.pth")
MODEL_HN = Path("TruthLens-main/model/deepfake_detector_hn.pth")
OUTPUT_DIR = Path("label_quality_results")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Ambiguity band: predictions inside this band are "ambiguous" (annotators would disagree)
AMBIGUOUS_BAND = (0.35, 0.65)
# High-confidence mislabel candidates: predicted opposite class with high confidence
MISLABEL_CONF = 0.9

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── Dataset ────────────────────────────────────────────────────────────────

class PathDataset(Dataset):
    def __init__(self, fake_dir, real_dir, max_per_class=None):
        self.samples = []
        for d, label in [(fake_dir, 1), (real_dir, 0)]:
            if not d.exists():
                continue
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
            if max_per_class:
                files = files[:max_per_class]
            self.samples.extend((f, label) for f in files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor = val_tf(img)
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            tensor = val_tf(img)
        return tensor, label, str(path)


def predict(model, loader, device):
    """Return (probs, labels, paths) for a dataloader."""
    model.eval()
    all_probs, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Inference", leave=False):
            imgs = imgs.to(device)
            out = model(imgs).squeeze(1)
            probs = torch.sigmoid(out)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)
    return np.concatenate(all_probs), np.concatenate(all_labels), all_paths


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model, ckpt


def dhash_of_path(path, hash_size=8):
    """Perceptual hash for near-dup detection."""
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    diff = []
    for row in range(hash_size):
        rs = row * (hash_size + 1)
        for col in range(hash_size):
            diff.append(pixels[rs + col] > pixels[rs + col + 1])
    hv = 0
    for bit in diff:
        hv = (hv << 1) | bit
    return hv


def hamming(a, b):
    return bin(a ^ b).count("1")


# ─── Main ──────────────────────────────────────────────────────────────────

def main(limit=None):
    print("=" * 70)
    print("TruthLens Label Quality Audit")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    report = {}
    splits_to_check = ["Test"] if not limit else ["Test"]
    all_issues = []

    # ── 1. Mechanical label conflict checks ─────────────────────────────
    print("\n[1/4] Mechanical label conflict checks...")
    audit_data = json.load(open("dataset_audit_results/audit_results.json"))
    exact_dups = audit_data["exact_duplicates"]
    near_dups = audit_data["near_duplicates"]

    def get_meta(p):
        parts = p.split("/")
        split, cls = None, None
        for i, seg in enumerate(parts):
            if seg in ("Train", "Validation", "Test"):
                split = seg
            if seg in ("Fake", "Real"):
                cls = seg
        return split, cls

    # Cross-class exact duplicates
    cross_exact = []
    for h, paths in exact_dups.items():
        classes = {get_meta(p)[1] for p in paths}
        if len(classes) > 1:
            cross_exact.append({"hash": h, "paths": paths, "classes": list(classes)})
    report["cross_class_exact_duplicates"] = len(cross_exact)
    print(f"  Cross-class EXACT duplicates (same bytes, different label): {len(cross_exact)}")

    # Cross-class near duplicates - recompute PROPERLY (original audit only compared within same class)
    print("  Recomputing cross-class near-duplicates (original audit missed these)...")
    # Build dhash for a sample of both classes in Test (or Train subset if limited)
    sample_limit = limit or 5000
    fake_paths = sorted(p for p in (DATASET_ROOT / "Test" / "Fake").glob("*"))[:sample_limit]
    real_paths = sorted(p for p in (DATASET_ROOT / "Test" / "Real").glob("*"))[:sample_limit]
    fake_hashes = {p: dhash_of_path(p) for p in tqdm(fake_paths, desc="dhash Fake")}
    real_hashes = {p: dhash_of_path(p) for p in tqdm(real_paths, desc="dhash Real")}
    cross_near = []
    for p1, h1 in fake_hashes.items():
        for p2, h2 in real_hashes.items():
            d = hamming(h1, h2)
            if d <= 5:
                cross_near.append({"path_fake": str(p1), "path_real": str(p2), "distance": d})
    report["cross_class_near_duplicates_sampled"] = len(cross_near)
    report["cross_class_near_dup_sample_size"] = (len(fake_paths), len(real_paths))
    print(f"  Cross-class NEAR-duplicates (dhash<=5) sampled: {len(cross_near)}")
    for cn in cross_near[:10]:
        print(f"    dist={cn['distance']}: {cn['path_fake'].split('/')[-1]} <-> {cn['path_real'].split('/')[-1]}")

    # Cross-split label conflicts
    split_conflicts = 0
    for h, paths in exact_dups.items():
        classes = {get_meta(p)[1] for p in paths}
        splits = {get_meta(p)[0] for p in paths}
        if len(splits) > 1 and len(classes) > 1:
            split_conflicts += 1
    report["cross_split_label_conflicts"] = split_conflicts
    print(f"  Cross-split label conflicts: {split_conflicts}")

    # ── 2. Model inference for mislabel/ambiguous detection ─────────────
    print("\n[2/4] Model inference (base model)...")
    model_base, ckpt_base = load_model(MODEL_BASE)
    print(f"  Base model: threshold={ckpt_base.get('best_threshold')}, F1={ckpt_base.get('best_accuracy')}")

    model_hn, ckpt_hn = load_model(MODEL_HN)
    print(f"  HN model: threshold={ckpt_hn.get('best_threshold')}, F1={ckpt_hn.get('best_accuracy')}")

    for split in ["Test"]:
        ds = PathDataset(DATASET_ROOT / split / "Fake", DATASET_ROOT / split / "Real")
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        probs_base, labels, paths = predict(model_base, loader, DEVICE)
        probs_hn, _, _ = predict(model_hn, loader, DEVICE)

        # Combine: average of the two models for robustness
        probs_ens = (probs_base + probs_hn) / 2

        results = []
        for i, (p, l, path) in enumerate(zip(probs_ens, labels, paths)):
            results.append({
                "path": path,
                "label": int(l),
                "label_name": "Fake" if l == 1 else "Real",
                "prob_ens": float(p),
                "prob_base": float(probs_base[i]),
                "prob_hn": float(probs_hn[i]),
            })

        # High-confidence mislabel candidates
        mislabel_candidates = [
            r for r in results
            if (r["label_name"] == "Fake" and r["prob_ens"] < (1 - MISLABEL_CONF)) or
               (r["label_name"] == "Real" and r["prob_ens"] > MISLABEL_CONF)
        ]
        # Sort by how strongly the model disagrees
        mislabel_candidates.sort(key=lambda r: abs(r["prob_ens"] - (0 if r["label_name"] == "Real" else 1)))

        # Ambiguous cases (near the decision boundary)
        ambiguous = [r for r in results if AMBIGUOUS_BAND[0] <= r["prob_ens"] <= AMBIGUOUS_BAND[1]]

        # Store all per-split results
        report[f"{split}_results"] = results

        print(f"\n  {split} split: {len(results)} images")
        print(f"    High-confidence mislabel candidates (prob<{1-MISLABEL_CONF:.0%} or prob>{MISLABEL_CONF:.0%}): {len(mislabel_candidates)}")
        print(f"    Ambiguous cases (prob in {AMBIGUOUS_BAND}): {len(ambiguous)} ({len(ambiguous)/len(results)*100:.1f}%)")

        # Breakdown of mislabel candidates
        mc_by_label = Counter(r["label_name"] for r in mislabel_candidates)
        print(f"    Mislabel candidates by label: {dict(mc_by_label)}")

        # Save top candidates
        report[f"{split}_mislabel_candidates"] = mislabel_candidates
        report[f"{split}_ambiguous"] = ambiguous

        # Print top 15
        print("\n    TOP mislabel candidates (model confidently disagrees with label):")
        for r in mislabel_candidates[:15]:
            print(f"      {r['label_name']} label, model says {1-r['prob_ens']:.0%} other class: {r['path'].split('/')[-1]}")

        print("\n    SAMPLE ambiguous cases:")
        for r in ambiguous[:15]:
            print(f"      prob={r['prob_ens']:.3f} ({r['label_name']}): {r['path'].split('/')[-1]}")

        # ── 3. Class boundary analysis ─────────────────────────────────
        print(f"\n[3/4] Class boundary analysis for {split}...")
        fake_probs = [r["prob_ens"] for r in results if r["label_name"] == "Fake"]
        real_probs = [r["prob_ens"] for r in results if r["label_name"] == "Real"]

        # Boundary width: fraction of each class inside the ambiguous band
        fake_amb = sum(1 for p in fake_probs if AMBIGUOUS_BAND[0] <= p <= AMBIGUOUS_BAND[1])
        real_amb = sum(1 for p in real_probs if AMBIGUOUS_BAND[0] <= p <= AMBIGUOUS_BAND[1])
        print(f"    Fake ambiguous in band: {fake_amb}/{len(fake_probs)} ({fake_amb/len(fake_probs)*100:.1f}%)")
        print(f"    Real ambiguous in band: {real_amb}/{len(real_probs)} ({real_amb/len(real_probs)*100:.1f}%)")

        # Overlap: how much do the two probability distributions overlap
        # KS-style overlap measure
        hist_f, edges = np.histogram(fake_probs, bins=20, range=(0, 1), density=True)
        hist_r, _ = np.histogram(real_probs, bins=20, range=(0, 1), density=True)
        overlap = np.sum(np.minimum(hist_f, hist_r)) * (edges[1] - edges[0])
        print(f"    Distribution overlap (0=separated, 1=identical): {overlap:.3f}")

        report[f"{split}_boundary"] = {
            "ambiguous_band": list(AMBIGUOUS_BAND),
            "fake_ambiguous": fake_amb,
            "fake_total": len(fake_probs),
            "real_ambiguous": real_amb,
            "real_total": len(real_probs),
            "overlap": float(overlap),
        }

    # ── 4. Save ────────────────────────────────────────────────────────
    print("\n[4/4] Saving report...")
    # Trim: don't store full results for all images in report (huge). Store counts + top lists.
    slim = {}
    for k, v in report.items():
        if k.endswith("_results"):
            continue  # skip the full dump
        if k.endswith("_mislabel_candidates") or k.endswith("_ambiguous"):
            # Keep top 100 only
            slim[k] = v[:100]
        else:
            slim[k] = v

    with open(OUTPUT_DIR / "label_quality_report.json", "w") as f:
        json.dump(slim, f, indent=2)
    print(f"  ✅ Saved to {OUTPUT_DIR}/label_quality_report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Limit samples per class")
    args = ap.parse_args()
    main(args.limit)
