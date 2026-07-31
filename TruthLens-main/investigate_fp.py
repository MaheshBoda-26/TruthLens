"""
TruthLens False Positive Investigation Pipeline
================================================
Collects misclassified real images, analyzes image-level artifacts
(compression, blur, smoothness, brightness, contrast, edge density,
color saturation, noise), ranks by confidence, generates comparison
grids, and produces a statistical report.

Usage:
    ./venv/bin/python investigate_fp.py                # Full run
    ./venv/bin/python investigate_fp.py --quick         # 2K samples
    ./venv/bin/python investigate_fp.py --threshold 0.68
"""

import argparse, json, os, time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model_def import create_model

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("model/deepfake_detector.pth")
DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
OUT = Path("fp_investigation")
IMG_SIZE = 224

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Dataset that also returns file paths ──────────────────────────────────

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
        np.random.seed(42)
        np.random.shuffle(self.samples)
        nf = sum(1 for _,l in self.samples if l==1)
        nr = sum(1 for _,l in self.samples if l==0)
        print(f"  Loaded {len(self.samples)} images (Fake={nf}, Real={nr})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor = val_tf(img)
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128,128,128))
            tensor = val_tf(img)
        return tensor, label, str(path)


# ── Image-level artifact analysis ─────────────────────────────────────────

def analyze_image_artifacts(path: str) -> dict:
    """Compute low-level image quality metrics using OpenCV."""
    try:
        img = cv2.imread(path)
        if img is None:
            return {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Laplacian variance → blur detection (lower = blurrier)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # JPEG quality estimate via DCT energy in 8x8 blocks
        dct_energy = 0
        n_blocks = 0
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = np.float32(gray[y:y+8, x:x+8])
                dct = cv2.dct(block)
                dct_energy += np.sum(np.abs(dct[1:, 1:]))
                n_blocks += 1
        avg_dct = dct_energy / max(n_blocks, 1)

        # Skin smoothness: std dev inside center crop
        cy, cx = h//2, w//2
        crop = gray[cy-h//6:cy+h//6, cx-w//6:cx+w//6]
        skin_std = float(crop.std()) if crop.size > 0 else 0

        # Edge density (Canny edges as % of total pixels)
        edges = cv2.Canny(gray, 50, 150)
        edge_pct = float(edges.sum() / 255) / (h * w) * 100

        # Brightness and contrast
        brightness = float(gray.mean())
        contrast = float(gray.std())

        # Color saturation (mean saturation in HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:,:,1].mean())

        # Noise estimate: high-pass filter energy
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        noise = float(np.abs(gray.astype(float) - blur.astype(float)).mean())

        # File size as quality proxy
        fsize = os.path.getsize(path) / 1024  # KB

        return {
            "laplacian_var": round(lap_var, 2),
            "avg_dct_energy": round(avg_dct, 2),
            "skin_smoothness_std": round(skin_std, 2),
            "edge_density_pct": round(edge_pct, 3),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "noise_level": round(noise, 3),
            "file_size_kb": round(fsize, 1),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Comparison grid builder ───────────────────────────────────────────────

def build_grid(image_paths: list, labels: list, cols=5, cell_size=160) -> np.ndarray:
    """Build a labeled image grid for visual comparison."""
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * (cell_size + 25), cols * cell_size, 3), dtype=np.uint8)

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        r, c = divmod(i, cols)
        y0 = r * (cell_size + 25)
        x0 = c * cell_size
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, (cell_size, cell_size))
            grid[y0:y0+cell_size, x0:x0+cell_size] = img
        except Exception:
            pass
        # Label
        color = (0, 0, 255) if "FP" in label else (0, 200, 0)
        cv2.putText(grid, label, (x0+4, y0+cell_size+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return grid


# ── Main pipeline ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--split", default="Test", choices=["Test", "Validation"])
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    (OUT / "fp_samples").mkdir(exist_ok=True)
    (OUT / "tp_samples").mkdir(exist_ok=True)
    (OUT / "grids").mkdir(exist_ok=True)

    t0 = time.time()
    threshold = args.threshold
    print("=" * 60)
    print("  TruthLens False Positive Investigation")
    print("=" * 60)
    print(f"  Device: {DEVICE}  |  Threshold: {threshold}")

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()

    # Load data
    max_pc = 1000 if args.quick else None
    ds = PathDataset(DATASET_ROOT / args.split / "Fake",
                     DATASET_ROOT / args.split / "Real", max_per_class=max_pc)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # ── Collect predictions with paths ──
    print("\n🔬 Running inference...")
    all_probs, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Inference"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(int)

    # ── Identify FP and TP-real ──
    fp_mask = (preds == 1) & (labels == 0)  # Real images flagged as fake
    tn_mask = (preds == 0) & (labels == 0)  # Real images correctly classified

    fp_indices = np.where(fp_mask)[0]
    tn_indices = np.where(tn_mask)[0]

    print(f"\n📊 Classification Summary (threshold={threshold}):")
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(fp_mask.sum())
    tn = int(tn_mask.sum())
    fn = int(((preds==0)&(labels==1)).sum())
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  FPR = {fp/(fp+tn)*100:.1f}%  ({fp} real images misclassified)")

    # ── Rank FPs by confidence (highest prob = worst offenders) ──
    fp_probs = probs[fp_indices]
    ranked = np.argsort(-fp_probs)
    fp_ranked_indices = fp_indices[ranked]

    print(f"\n🔝 Top 20 worst false positives (highest fake probability):")
    top_fps = []
    for rank, idx in enumerate(fp_ranked_indices[:20]):
        p = probs[idx]
        path = all_paths[idx]
        print(f"  #{rank+1:>2}  P(fake)={p:.4f}  {Path(path).name}")
        top_fps.append({"rank": rank+1, "prob": round(float(p),4), "path": path})

    # ── Save FP and TN sample images (top 100 each) ──
    print(f"\n💾 Saving sample images...")
    for i, idx in enumerate(fp_ranked_indices[:100]):
        src = all_paths[idx]
        dst = OUT / "fp_samples" / f"fp_{i:03d}_p{probs[idx]:.3f}_{Path(src).name}"
        try:
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(str(dst), img)
        except Exception:
            pass

    tn_sample_idx = tn_indices[np.random.choice(len(tn_indices), min(100, len(tn_indices)), replace=False)]
    for i, idx in enumerate(tn_sample_idx[:100]):
        src = all_paths[idx]
        dst = OUT / "tp_samples" / f"tn_{i:03d}_p{probs[idx]:.3f}_{Path(src).name}"
        try:
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(str(dst), img)
        except Exception:
            pass

    # ── Build comparison grids ──
    print("📐 Building comparison grids...")
    # Top 25 FPs
    grid_paths = [all_paths[i] for i in fp_ranked_indices[:25]]
    grid_labels = [f"FP {probs[i]:.2f}" for i in fp_ranked_indices[:25]]
    grid = build_grid(grid_paths, grid_labels, cols=5)
    cv2.imwrite(str(OUT / "grids" / "top25_false_positives.png"), grid)

    # Top 25 TNs for comparison
    tn_sample = tn_indices[np.argsort(probs[tn_indices])[:25]]
    grid_paths2 = [all_paths[i] for i in tn_sample]
    grid_labels2 = [f"TN {probs[i]:.2f}" for i in tn_sample]
    grid2 = build_grid(grid_paths2, grid_labels2, cols=5)
    cv2.imwrite(str(OUT / "grids" / "top25_true_negatives.png"), grid2)

    # ── Artifact analysis on FPs vs TNs ──
    print("\n🔍 Analyzing image artifacts (FPs vs TNs)...")
    n_analyze = min(200, len(fp_ranked_indices), len(tn_indices))

    fp_artifacts = []
    for idx in tqdm(fp_ranked_indices[:n_analyze], desc="Analyzing FPs"):
        a = analyze_image_artifacts(all_paths[idx])
        if a and "error" not in a:
            a["prob"] = float(probs[idx])
            a["path"] = all_paths[idx]
            fp_artifacts.append(a)

    tn_sample_for_analysis = tn_indices[np.random.choice(len(tn_indices), n_analyze, replace=False)]
    tn_artifacts = []
    for idx in tqdm(tn_sample_for_analysis, desc="Analyzing TNs"):
        a = analyze_image_artifacts(all_paths[idx])
        if a and "error" not in a:
            a["prob"] = float(probs[idx])
            a["path"] = all_paths[idx]
            tn_artifacts.append(a)

    # ── Statistical comparison ──
    metrics = ["laplacian_var", "avg_dct_energy", "skin_smoothness_std",
               "edge_density_pct", "brightness", "contrast", "saturation",
               "noise_level", "file_size_kb"]

    print(f"\n{'═'*60}")
    print("  ARTIFACT FEATURE COMPARISON: FP vs TN (real images)")
    print(f"{'═'*60}")
    print(f"  {'Metric':<24} {'FP Mean':>10} {'TN Mean':>10} {'Delta':>10} {'Flag':>6}")
    print(f"  {'─'*24} {'─'*10} {'─'*10} {'─'*10} {'─'*6}")

    comparison = {}
    for m in metrics:
        fp_vals = [a[m] for a in fp_artifacts if m in a]
        tn_vals = [a[m] for a in tn_artifacts if m in a]
        if not fp_vals or not tn_vals:
            continue
        fp_mean = np.mean(fp_vals)
        tn_mean = np.mean(tn_vals)
        delta = fp_mean - tn_mean
        pct = abs(delta) / max(abs(tn_mean), 1e-6) * 100
        flag = "⚠️" if pct > 15 else ""
        print(f"  {m:<24} {fp_mean:>10.2f} {tn_mean:>10.2f} {delta:>+10.2f} {flag:>6}")
        comparison[m] = {
            "fp_mean": round(float(fp_mean), 3), "fp_std": round(float(np.std(fp_vals)), 3),
            "tn_mean": round(float(tn_mean), 3), "tn_std": round(float(np.std(tn_vals)), 3),
            "delta": round(float(delta), 3), "pct_diff": round(float(pct), 1),
        }

    # ── Confidence distribution ──
    print(f"\n{'═'*60}")
    print("  CONFIDENCE DISTRIBUTION OF FALSE POSITIVES")
    print(f"{'═'*60}")
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
            (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.0)]
    conf_dist = {}
    for lo, hi in bins:
        count = int(((fp_probs >= lo) & (fp_probs < hi)).sum())
        bar = "█" * (count // max(1, len(fp_probs) // 40))
        print(f"  [{lo:.2f}-{hi:.2f})  {count:>5}  {bar}")
        conf_dist[f"{lo:.2f}-{hi:.2f}"] = count

    # ── Failure pattern classification ──
    print(f"\n{'═'*60}")
    print("  FAILURE PATTERN CLASSIFICATION")
    print(f"{'═'*60}")

    patterns = {"blur": 0, "smooth_skin": 0, "low_contrast": 0,
                "high_compression": 0, "low_edge": 0, "low_noise": 0,
                "high_saturation": 0, "dark_image": 0, "small_file": 0}

    for a in fp_artifacts:
        if a.get("laplacian_var", 999) < 50:
            patterns["blur"] += 1
        if a.get("skin_smoothness_std", 999) < 15:
            patterns["smooth_skin"] += 1
        if a.get("contrast", 999) < 35:
            patterns["low_contrast"] += 1
        if a.get("avg_dct_energy", 999) < 500:
            patterns["high_compression"] += 1
        if a.get("edge_density_pct", 999) < 3:
            patterns["low_edge"] += 1
        if a.get("noise_level", 999) < 3:
            patterns["low_noise"] += 1
        if a.get("saturation", 0) > 120:
            patterns["high_saturation"] += 1
        if a.get("brightness", 999) < 60:
            patterns["dark_image"] += 1
        if a.get("file_size_kb", 999) < 20:
            patterns["small_file"] += 1

    n_fp = max(len(fp_artifacts), 1)
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 40 // n_fp)
        print(f"  {pattern:<20} {count:>4} ({count/n_fp*100:>5.1f}%)  {bar}")

    # ── Generate distribution plots ──
    print("\n📊 Generating distribution plots...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.facecolor": "#0a0a0a", "axes.facecolor": "#111",
            "axes.edgecolor": "#333", "axes.labelcolor": "#ccc",
            "text.color": "#ccc", "xtick.color": "#999", "ytick.color": "#999",
            "grid.color": "#222", "font.family": "monospace", "font.size": 9,
        })

        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle("FP Investigation — Feature Distributions (FP=Red, TN=Green)",
                     fontsize=14, fontweight="bold", color="#00ff80")

        for i, m in enumerate(metrics):
            ax = axes[i // 3][i % 3]
            fp_v = [a[m] for a in fp_artifacts if m in a]
            tn_v = [a[m] for a in tn_artifacts if m in a]
            if fp_v and tn_v:
                lo = min(min(fp_v), min(tn_v))
                hi = max(max(fp_v), max(tn_v))
                b = np.linspace(lo, hi, 40)
                ax.hist(tn_v, bins=b, alpha=0.6, color="#00ff80", label="TN (correct)", density=True)
                ax.hist(fp_v, bins=b, alpha=0.6, color="#ff3333", label="FP (wrong)", density=True)
                ax.set_title(m, color="#aaa", fontsize=10)
                ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333")
                ax.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(OUT / "feature_distributions.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {OUT / 'feature_distributions.png'}")

        # Confidence histogram
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor("#0a0a0a")
        ax2.set_facecolor("#111")
        real_p = probs[labels == 0]
        fake_p = probs[labels == 1]
        b = np.linspace(0, 1, 80)
        ax2.hist(real_p, bins=b, alpha=0.6, color="#00ff80", label=f"Real (n={len(real_p)})", density=True)
        ax2.hist(fake_p, bins=b, alpha=0.6, color="#ff3333", label=f"Fake (n={len(fake_p)})", density=True)
        ax2.axvline(threshold, color="#ffaa00", ls="--", lw=2, label=f"Threshold={threshold}")
        ax2.set_title("Probability Distribution by Class", color="#00ff80", fontsize=13)
        ax2.set_xlabel("P(Fake)")
        ax2.legend(facecolor="#1a1a1a", edgecolor="#333")
        ax2.grid(True, alpha=0.2)
        fig2.savefig(str(OUT / "probability_distribution.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {OUT / 'probability_distribution.png'}")
        plt.close("all")
    except Exception as e:
        print(f"  ⚠ Plot generation failed: {e}")

    # ── Save full report ──
    report = {
        "threshold": threshold,
        "split": args.split,
        "total_samples": len(labels),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "fpr": round(fp / (fp + tn) * 100, 2),
        "top_false_positives": top_fps,
        "feature_comparison": comparison,
        "confidence_distribution": conf_dist,
        "failure_patterns": patterns,
        "n_fp_analyzed": len(fp_artifacts),
        "n_tn_analyzed": len(tn_artifacts),
    }
    with open(OUT / "fp_report.json", "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  ✅ Investigation complete in {elapsed:.1f}s")
    print(f"  📁 Results: {OUT}/")
    print(f"     fp_samples/       — {min(100,fp)} worst FP images")
    print(f"     tp_samples/       — {min(100,tn)} correct TN images")
    print(f"     grids/            — side-by-side comparison grids")
    print(f"     fp_report.json    — full JSON report")
    print(f"     feature_distributions.png")
    print(f"     probability_distribution.png")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
