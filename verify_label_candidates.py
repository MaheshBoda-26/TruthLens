"""
Verify label-quality candidates visually.
Builds comparison grids of:
1. Cross-class near-duplicates (near-identical images labeled Fake vs Real)
2. High-confidence mislabel candidates
3. A sample of ambiguous cases
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
OUT = Path("label_quality_results")


def dhash_of_path(path, hash_size=8):
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


def build_grid(pairs, cols=4, cell=220, out_name="grid.png"):
    """pairs: list of (img1_path, img2_path, caption1, caption2, title)"""
    rows = (len(pairs) + cols - 1) // cols
    W, H = cols * cell * 2, rows * (cell + 30)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        small = font

    for idx, (p1, p2, cap1, cap2, title) in enumerate(pairs):
        r, c = divmod(idx, cols)
        x0 = c * cell * 2
        y0 = r * (cell + 30)
        try:
            img1 = Image.open(p1).convert("RGB").resize((cell, cell))
            img2 = Image.open(p2).convert("RGB").resize((cell, cell))
        except Exception as e:
            print(f"  skip {p1} {e}")
            continue
        canvas.paste(img1, (x0, y0))
        canvas.paste(img2, (x0 + cell, y0))
        draw.text((x0 + 5, y0 + cell), cap1, fill=(255, 0, 0) if "Fake" in cap1 else (0, 128, 0), font=small)
        draw.text((x0 + cell + 5, y0 + cell), cap2, fill=(255, 0, 0) if "Fake" in cap2 else (0, 128, 0), font=small)
        draw.text((x0 + 5, y0 + cell + 14), title, fill=(0, 0, 0), font=small)

    canvas.save(OUT / out_name)
    print(f"  Saved {OUT / out_name}")


# ── 1. Cross-class near-duplicates ─────────────────────────────────────────
print("Computing cross-class near-duplicates...")
fake_paths = sorted(p for p in (DATASET_ROOT / "Test" / "Fake").glob("*"))
real_paths = sorted(p for p in (DATASET_ROOT / "Test" / "Real").glob("*"))
fake_hashes = {p: dhash_of_path(p) for p in fake_paths}
real_hashes = {p: dhash_of_path(p) for p in real_paths}

cross_near = []
for p1, h1 in fake_hashes.items():
    for p2, h2 in real_hashes.items():
        d = hamming(h1, h2)
        if d <= 5:
            cross_near.append((p1, p2, d))

print(f"Cross-class near-dups (dhash<=5): {len(cross_near)}")
cross_near.sort(key=lambda x: x[2])

# Save full list
with open(OUT / "cross_class_near_dups.json", "w") as f:
    json.dump([{"fake": str(a), "real": str(b), "distance": d} for a, b, d in cross_near], f, indent=2)
print("  Saved cross_class_near_dups.json")

# Build grid of all pairs (up to 40)
pairs = [(a, b, f"Fake: {a.name}", f"Real: {b.name}", f"d={d}") for a, b, d in cross_near[:40]]
build_grid(pairs, cols=5, cell=200, out_name="cross_class_near_dups_grid.png")
print(f"  Grid with {len(pairs)} pairs")

# ── 2. Mislabel candidates from report ─────────────────────────────────────
with open(OUT / "label_quality_report.json") as f:
    report = json.load(f)
mc = report["Test_mislabel_candidates"]
print(f"\nMislabel candidates in report: {len(mc)}")

# Build grid: show each candidate with its image
mc_pairs = []
for r in mc[:12]:
    p = Path(r["path"])
    prob = r["prob_ens"]
    cap = f"{r['label_name']} / model {prob:.2f}"
    mc_pairs.append((p, p, cap, "", f"p={prob:.3f}"))
build_grid(mc_pairs, cols=3, cell=200, out_name="mislabel_candidates_grid.png")

# ── 3. Ambiguous sample ────────────────────────────────────────────────────
amb = report["Test_ambiguous"]
print(f"Ambiguous in report: {len(amb)}")
# Show a spread across the band
import random
random.seed(42)
amb_sample = random.sample(amb, min(24, len(amb)))
amb_pairs = []
for r in amb_sample:
    p = Path(r["path"])
    prob = r["prob_ens"]
    amb_pairs.append((p, p, f"{r['label_name']} p={prob:.3f}", "", ""))
build_grid(amb_pairs, cols=4, cell=180, out_name="ambiguous_sample_grid.png")

print("\nDone.")
