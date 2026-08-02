"""
TruthLens Dataset Audit Script
==============================
Comprehensive audit of the deepfake detection dataset.
Reports on class balance, duplicates, corruption, resolution, blur, etc.
Generates histograms for class distribution, image size, brightness, aspect ratio.
"""

import os
import json
import hashlib
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── Configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
OUTPUT_DIR = Path("dataset_audit_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SPLITS = ["Train", "Validation", "Test"]
CLASSES = ["Fake", "Real"]

# Thresholds
MIN_RESOLUTION = (64, 64)  # Below this = low-res
BLUR_THRESHOLD = 50  # Laplacian variance below this = blurry
DUPLICATE_HASH_SIZE = 8  # For perceptual hashing (dhash)
NEAR_DUP_THRESHOLD = 5  # Hamming distance for near-duplicates
BLANK_THRESHOLD = 1.0  # Std dev below this = blank/constant image
MAX_WORKERS = os.cpu_count() or 4

# ─── Helper Functions ──────────────────────────────────────────────────────────

def compute_dhash(image, hash_size=8):
    """Compute difference hash (dhash) for near-duplicate detection."""
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    diff = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            diff.append(left > right)
    # Convert to integer hash
    hash_val = 0
    for bit in diff:
        hash_val = (hash_val << 1) | bit
    return hash_val

def hamming_distance(a, b):
    """Hamming distance between two integer hashes."""
    return bin(a ^ b).count('1')

def compute_blur_score(image):
    """Compute Laplacian variance as blur metric (higher = sharper)."""
    gray = np.array(image.convert('L'), dtype=np.float32)
    # Simple Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Convolve
    from scipy import ndimage
    laplacian = ndimage.convolve(gray, kernel, mode='constant')
    return np.var(laplacian)

def compute_brightness(image):
    """Compute mean brightness (0-255)."""
    return np.mean(np.array(image.convert('L')))

def process_image(args):
    """Process a single image file - returns dict with metrics or error."""
    path, split, class_name = args
    try:
        with Image.open(path) as img:
            img.load()  # Force load to catch corruption
            img = img.convert('RGB')

            width, height = img.size
            aspect_ratio = width / height
            file_size = path.stat().st_size

            # Brightness
            brightness = compute_brightness(img)

            # Blur (downsample for speed)
            small = img.resize((224, 224), Image.Resampling.LANCZOS)
            blur_score = compute_blur_score(small)

            # Perceptual hash for dedup
            dhash = compute_dhash(img, DUPLICATE_HASH_SIZE)

            # File hash for exact duplicates
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            # Blank/empty image detection (very low variance = solid color)
            img_array = np.array(img)
            pixel_std = np.std(img_array)
            is_blank = pixel_std < BLANK_THRESHOLD

            return {
                "path": str(path),
                "split": split,
                "class": class_name,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "file_size": file_size,
                "brightness": brightness,
                "blur_score": blur_score,
                "dhash": dhash,
                "file_hash": file_hash,
                "pixel_std": float(pixel_std),
                "is_low_res": width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1],
                "is_blurry": blur_score < BLUR_THRESHOLD,
                "is_blank": is_blank,
                "is_corrupt": False,
                "error": None
            }
    except Exception as e:
        return {
            "path": str(path),
            "split": split,
            "class": class_name,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "file_size": path.stat().st_size if path.exists() else 0,
            "brightness": None,
            "blur_score": None,
            "dhash": None,
            "file_hash": None,
            "is_low_res": False,
            "is_blurry": False,
            "is_corrupt": True,
            "error": str(e)
        }

# ─── Main Audit ────────────────────────────────────────────────────────────────

def collect_all_image_paths():
    """Collect all image paths with their split and class labels."""
    all_paths = []
    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = DATASET_ROOT / split / class_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff']:
                    for p in class_dir.glob(ext):
                        all_paths.append((p, split, class_name))
    return all_paths

def run_audit():
    print("=" * 70)
    print("TruthLens Dataset Audit")
    print("=" * 70)

    # 1. Collect paths
    print("\n📂 Collecting image paths...")
    all_paths = collect_all_image_paths()
    print(f"  Total images found: {len(all_paths)}")

    # Count by split/class
    counts = Counter()
    for _, split, class_name in all_paths:
        counts[f"{split}/{class_name}"] += 1

    for key in sorted(counts.keys()):
        print(f"  {key}: {counts[key]}")

    # 2. Process images in parallel
    print(f"\n🔍 Processing {len(all_paths)} images (this may take a while)...")
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_image, args): args for args in all_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
            results.append(future.result())

    # 3. Analyze results
    print("\n📊 Analyzing results...")

    # Separate valid vs corrupt
    valid_results = [r for r in results if not r["is_corrupt"]]
    corrupt_results = [r for r in results if r["is_corrupt"]]

    print(f"\n  Valid images: {len(valid_results)}")
    print(f"  Corrupt/unreadable: {len(corrupt_results)}")

    if corrupt_results:
        print("\n  ⚠️  Corrupt files:")
        for r in corrupt_results[:10]:
            print(f"    {r['path']}: {r['error']}")
        if len(corrupt_results) > 10:
            print(f"    ... and {len(corrupt_results) - 10} more")

    # 4. Class distribution
    class_dist = Counter()
    split_dist = Counter()
    for r in valid_results:
        class_dist[r["class"]] += 1
        split_dist[r["split"]] += 1

    print("\n📈 Class Distribution:")
    for cls, count in class_dist.most_common():
        print(f"  {cls}: {count} ({count/len(valid_results)*100:.1f}%)")

    print("\n📈 Split Distribution:")
    for split, count in split_dist.most_common():
        print(f"  {split}: {count} ({count/len(valid_results)*100:.1f}%)")

    # 5. Exact duplicates (by file hash)
    print("\n🔍 Checking for exact duplicates...")
    hash_to_paths = defaultdict(list)
    for r in valid_results:
        if r["file_hash"]:
            hash_to_paths[r["file_hash"]].append(r["path"])

    exact_duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    print(f"  Exact duplicate groups: {len(exact_duplicates)}")
    total_dup_files = sum(len(paths) - 1 for paths in exact_duplicates.values())
    print(f"  Total duplicate files: {total_dup_files}")
    if exact_duplicates:
        print("  Example groups:")
        for h, paths in list(exact_duplicates.items())[:5]:
            print(f"    Hash {h[:8]}...: {len(paths)} copies")
            for p in paths[:3]:
                print(f"      {p}")
            if len(paths) > 3:
                print(f"      ... and {len(paths) - 3} more")

    # 6. Near-duplicates (by dhash)
    print("\n🔍 Checking for near-duplicates (perceptual hash)...")
    hashes = [(r["dhash"], r["path"], r["split"], r["class"]) for r in valid_results if r["dhash"] is not None]
    near_dup_groups = []

    # Simple O(n^2) check within each split/class for near-duplicates
    # For large datasets, we'd use a more efficient approach, but this is manageable
    for split in SPLITS:
        for class_name in CLASSES:
            subset = [(h, p) for h, p, s, c in hashes if s == split and c == class_name]
            if len(subset) < 2:
                continue

            # Group by hash buckets for efficiency
            buckets = defaultdict(list)
            for h, p in subset:
                # Use top 16 bits as bucket key
                bucket = h >> 48
                buckets[bucket].append((h, p))

            for bucket, items in buckets.items():
                for i, (h1, p1) in enumerate(items):
                    for h2, p2 in items[i+1:]:
                        dist = hamming_distance(h1, h2)
                        if dist <= NEAR_DUP_THRESHOLD:
                            near_dup_groups.append((p1, p2, dist))

    print(f"  Near-duplicate pairs (distance ≤ {NEAR_DUP_THRESHOLD}): {len(near_dup_groups)}")
    if near_dup_groups:
        print("  Examples:")
        for p1, p2, dist in near_dup_groups[:5]:
            print(f"    Distance {dist}: {Path(p1).name} ↔ {Path(p2).name}")

    # 7. Resolution analysis
    print("\n📐 Resolution Analysis:")
    widths = [r["width"] for r in valid_results]
    heights = [r["height"] for r in valid_results]
    low_res = [r for r in valid_results if r["is_low_res"]]

    print(f"  Min resolution: {min(widths)}×{min(heights)}")
    print(f"  Max resolution: {max(widths)}×{max(heights)}")
    print(f"  Median resolution: {int(np.median(widths))}×{int(np.median(heights))}")
    print(f"  Mean resolution: {int(np.mean(widths))}×{int(np.mean(heights))}")
    print(f"  Low-res (< {MIN_RESOLUTION[0]}×{MIN_RESOLUTION[1]}): {len(low_res)} ({len(low_res)/len(valid_results)*100:.1f}%)")

    # 8. Aspect ratio analysis
    print("\n📏 Aspect Ratio Analysis:")
    aspects = [r["aspect_ratio"] for r in valid_results]
    print(f"  Min: {min(aspects):.3f}")
    print(f"  Max: {max(aspects):.3f}")
    print(f"  Median: {np.median(aspects):.3f}")
    print(f"  Mean: {np.mean(aspects):.3f}")
    print(f"  Std: {np.std(aspects):.3f}")

    # Count non-square
    non_square = sum(1 for a in aspects if abs(a - 1.0) > 0.05)
    print(f"  Non-square (±5%): {non_square} ({non_square/len(valid_results)*100:.1f}%)")

    # 9. Brightness analysis
    print("\n💡 Brightness Analysis:")
    brightnesses = [r["brightness"] for r in valid_results]
    print(f"  Min: {min(brightnesses):.1f}")
    print(f"  Max: {max(brightnesses):.1f}")
    print(f"  Median: {np.median(brightnesses):.1f}")
    print(f"  Mean: {np.mean(brightnesses):.1f}")
    print(f"  Std: {np.std(brightnesses):.1f}")

    # Very dark / very bright
    very_dark = sum(1 for b in brightnesses if b < 30)
    very_bright = sum(1 for b in brightnesses if b > 225)
    print(f"  Very dark (<30): {very_dark} ({very_dark/len(valid_results)*100:.1f}%)")
    print(f"  Very bright (>225): {very_bright} ({very_bright/len(valid_results)*100:.1f}%)")

    # 10. Blur analysis
    print("\n🔍 Blur Analysis (Laplacian variance):")
    blur_scores = [r["blur_score"] for r in valid_results]
    blurry = [r for r in valid_results if r["is_blurry"]]
    print(f"  Min: {min(blur_scores):.1f}")
    print(f"  Max: {max(blur_scores):.1f}")
    print(f"  Median: {np.median(blur_scores):.1f}")
    print(f"  Mean: {np.mean(blur_scores):.1f}")
    print(f"  Std: {np.std(blur_scores):.1f}")
    print(f"  Blurry (< {BLUR_THRESHOLD}): {len(blurry)} ({len(blurry)/len(valid_results)*100:.1f}%)")

    # 11. Blank/empty image analysis
    print("\n⬜ Blank/Empty Image Analysis:")
    blank_images = [r for r in valid_results if r["is_blank"]]
    pixel_stds = [r["pixel_std"] for r in valid_results]
    print(f"  Min pixel std: {min(pixel_stds):.4f}")
    print(f"  Max pixel std: {max(pixel_stds):.4f}")
    print(f"  Median pixel std: {np.median(pixel_stds):.4f}")
    print(f"  Mean pixel std: {np.mean(pixel_stds):.4f}")
    print(f"  Blank images (std < {BLANK_THRESHOLD}): {len(blank_images)} ({len(blank_images)/len(valid_results)*100:.2f}%)")
    if blank_images:
        print("  Examples:")
        for r in blank_images[:5]:
            print(f"    {r['path']} (std={r['pixel_std']:.4f}, brightness={r['brightness']:.1f})")

    # 12. File size analysis
    print("\n💾 File Size Analysis:")
    sizes = [r["file_size"] for r in valid_results]
    print(f"  Min: {min(sizes)/1024:.1f} KB")
    print(f"  Max: {max(sizes)/1024:.1f} KB")
    print(f"  Median: {np.median(sizes)/1024:.1f} KB")
    print(f"  Mean: {np.mean(sizes)/1024:.1f} KB")

    # 12. Cross-split leakage check
    print("\n🔒 Cross-split Leakage Check:")
    # Check if same file hash appears in multiple splits
    split_hashes = defaultdict(set)
    for r in valid_results:
        if r["file_hash"]:
            split_hashes[r["split"]].add(r["file_hash"])

    leakage_found = False
    for i, split1 in enumerate(SPLITS):
        for split2 in SPLITS[i+1:]:
            common = split_hashes[split1] & split_hashes[split2]
            if common:
                leakage_found = True
                print(f"  ⚠️  {len(common)} images shared between {split1} and {split2}")
                for h in list(common)[:3]:
                    # Find paths
                    paths = [r["path"] for r in valid_results if r["file_hash"] == h]
                    for p in paths:
                        print(f"    {p}")

    if not leakage_found:
        print("  ✅ No exact duplicates across splits")

    # 13. Class balance per split
    print("\n⚖️  Class Balance Per Split:")
    for split in SPLITS:
        fake_count = sum(1 for r in valid_results if r["split"] == split and r["class"] == "Fake")
        real_count = sum(1 for r in valid_results if r["split"] == split and r["class"] == "Real")
        total = fake_count + real_count
        if total > 0:
            imbalance = abs(fake_count - real_count) / total * 100
            print(f"  {split}: Fake={fake_count}, Real={real_count}, Imbalance={imbalance:.1f}%")

    # 14. Generate histograms
    print("\n📊 Generating histograms...")
    generate_histograms(valid_results)

    # 15. Summary ranking
    print("\n" + "=" * 70)
    print("ISSUES RANKED BY TRAINING IMPACT")
    print("=" * 70)

    issues = []

    # High impact
    if len(corrupt_results) > 0:
        issues.append(("HIGH", f"Corrupt/unreadable files: {len(corrupt_results)}",
                       "Silently skipped or replaced with gray images during training, wastes compute"))

    if total_dup_files > 0:
        issues.append(("HIGH", f"Exact duplicates: {total_dup_files} files in {len(exact_duplicates)} groups",
                       "Overfitting risk, inflates validation metrics, wastes training capacity"))

    if leakage_found:
        issues.append(("CRITICAL", "Cross-split data leakage detected",
                       "Invalidates validation/test results, overfitting detection impossible"))

    if len(near_dup_groups) > 100:
        issues.append(("HIGH", f"Near-duplicates: {len(near_dup_groups)} pairs",
                       "Reduces effective dataset diversity, memorization risk"))
    elif len(near_dup_groups) > 0:
        issues.append(("MEDIUM", f"Near-duplicates: {len(near_dup_groups)} pairs",
                       "Some redundancy, monitor for overfitting"))

    if len(low_res) > len(valid_results) * 0.01:
        issues.append(("MEDIUM", f"Low-resolution images: {len(low_res)} ({len(low_res)/len(valid_results)*100:.1f}%)",
                       "May not contain discriminative features, adds noise"))
    elif len(low_res) > 0:
        issues.append(("LOW", f"Low-resolution images: {len(low_res)}",
                       "Minor, but consider filtering"))

    if len(blurry) > len(valid_results) * 0.05:
        issues.append(("MEDIUM", f"Blurry images: {len(blurry)} ({len(blurry)/len(valid_results)*100:.1f}%)",
                       "Degraded features, harder to learn discriminative patterns"))
    elif len(blurry) > 0:
        issues.append(("LOW", f"Blurry images: {len(blurry)}",
                       "Minor quality issue"))

    # Blank images
    if len(blank_images) > 0:
        issues.append(("HIGH", f"Blank/empty images: {len(blank_images)}",
                       "Zero information content, wastes training compute, may cause NaN gradients"))

    # Class imbalance
    for split in SPLITS:
        fake_count = sum(1 for r in valid_results if r["split"] == split and r["class"] == "Fake")
        real_count = sum(1 for r in valid_results if r["split"] == split and r["class"] == "Real")
        total = fake_count + real_count
        if total > 0:
            imbalance = abs(fake_count - real_count) / total * 100
            if imbalance > 10:
                issues.append(("HIGH", f"{split} class imbalance: {imbalance:.1f}%",
                               "Biased predictions, poor minority class recall"))
            elif imbalance > 5:
                issues.append(("MEDIUM", f"{split} class imbalance: {imbalance:.1f}%",
                               "Some bias, consider class weighting"))

    # Brightness extremes
    if very_dark > len(valid_results) * 0.02:
        issues.append(("LOW", f"Very dark images: {very_dark}",
                       "May lose detail in shadows"))
    if very_bright > len(valid_results) * 0.02:
        issues.append(("LOW", f"Very bright images: {very_bright}",
                       "May lose detail in highlights"))

    # Non-square
    if non_square > len(valid_results) * 0.1:
        issues.append(("LOW", f"Non-square images: {non_square} ({non_square/len(valid_results)*100:.1f}%)",
                       "Resizing distorts aspect ratio, consider letterboxing"))

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    issues.sort(key=lambda x: severity_order.get(x[0], 4))

    for severity, desc, impact in issues:
        print(f"\n  [{severity}] {desc}")
        print(f"       Impact: {impact}")

    if not issues:
        print("\n  ✅ No significant issues found!")

    # Add notes about limitations
    print("\n" + "=" * 70)
    print("AUDIT LIMITATIONS (NOT CHECKED)")
    print("=" * 70)
    print("""
  [INFO] Mislabeled images: NOT DETECTED
       Requires model inference or manual review. Current labels assumed correct.

  [INFO] Cropped/occluded subjects: NOT DETECTED
       Requires face/object detection (e.g., MTCNN, YOLO) to verify subject visibility.

  [INFO] Deepfake generation artifacts: NOT DETECTED
       Requires frequency analysis or specialized forensic tools.

  [INFO] Social media compression artifacts: NOT DETECTED
       Could be simulated but not verified against source.
""")

    # Save detailed results
    output_data = {
        "summary": {
            "total_images": len(all_paths),
            "valid_images": len(valid_results),
            "corrupt_images": len(corrupt_results),
            "class_distribution": dict(class_dist),
            "split_distribution": dict(split_dist),
            "exact_duplicate_groups": len(exact_duplicates),
            "total_duplicate_files": total_dup_files,
            "near_duplicate_pairs": len(near_dup_groups),
            "low_resolution_count": len(low_res),
            "blurry_count": len(blurry),
            "blank_count": len(blank_images),
            "very_dark_count": very_dark,
            "very_bright_count": very_bright,
            "non_square_count": non_square,
            "cross_split_leakage": leakage_found,
        },
        "issues": [{"severity": s, "description": d, "impact": i} for s, d, i in issues],
        "corrupt_files": [{"path": r["path"], "error": r["error"]} for r in corrupt_results],
        "exact_duplicates": {h: paths for h, paths in exact_duplicates.items()},
        "near_duplicates": [{"path1": p1, "path2": p2, "distance": d} for p1, p2, d in near_dup_groups],
        "low_res_files": [r["path"] for r in low_res],
        "blurry_files": [r["path"] for r in blurry],
        "blank_files": [r["path"] for r in blank_images],
    }

    with open(OUTPUT_DIR / "audit_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Detailed results saved to {OUTPUT_DIR}/audit_results.json")
    print(f"📊 Histograms saved to {OUTPUT_DIR}/")

    return output_data

def generate_histograms(results):
    """Generate all requested histograms."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

    # 1. Class distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Class distribution overall
    classes = [r["class"] for r in results]
    class_counts = Counter(classes)
    axes[0, 0].bar(class_counts.keys(), class_counts.values(), color=['#ff6b6b', '#4ecdc4'])
    axes[0, 0].set_title("Class Distribution (Overall)")
    axes[0, 0].set_ylabel("Count")
    for i, (cls, count) in enumerate(class_counts.items()):
        axes[0, 0].text(i, count + max(class_counts.values())*0.01, str(count), ha='center')

    # Class distribution by split
    split_class = defaultdict(Counter)
    for r in results:
        split_class[r["split"]][r["class"]] += 1

    x = np.arange(len(SPLITS))
    width = 0.35
    fake_counts = [split_class[s]["Fake"] for s in SPLITS]
    real_counts = [split_class[s]["Real"] for s in SPLITS]
    axes[0, 1].bar(x - width/2, fake_counts, width, label='Fake', color='#ff6b6b')
    axes[0, 1].bar(x + width/2, real_counts, width, label='Real', color='#4ecdc4')
    axes[0, 1].set_title("Class Distribution by Split")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(SPLITS)
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # Image size distribution (width)
    widths = [r["width"] for r in results]
    axes[1, 0].hist(widths, bins=50, color='#45b7d1', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Image Width Distribution")
    axes[1, 0].set_xlabel("Width (pixels)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].axvline(np.median(widths), color='red', linestyle='--', label=f'Median: {np.median(widths):.0f}')
    axes[1, 0].legend()

    # Image size distribution (height)
    heights = [r["height"] for r in results]
    axes[1, 1].hist(heights, bins=50, color='#96ceb4', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Image Height Distribution")
    axes[1, 1].set_xlabel("Height (pixels)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].axvline(np.median(heights), color='red', linestyle='--', label=f'Median: {np.median(heights):.0f}')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_and_size_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Brightness and Aspect Ratio distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Brightness overall
    brightnesses = [r["brightness"] for r in results]
    axes[0, 0].hist(brightnesses, bins=50, color='#ffeaa7', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("Brightness Distribution (Overall)")
    axes[0, 0].set_xlabel("Mean Brightness (0-255)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(np.median(brightnesses), color='red', linestyle='--', label=f'Median: {np.median(brightnesses):.1f}')
    axes[0, 0].legend()

    # Brightness by class
    fake_bright = [r["brightness"] for r in results if r["class"] == "Fake"]
    real_bright = [r["brightness"] for r in results if r["class"] == "Real"]
    axes[0, 1].hist(fake_bright, bins=50, alpha=0.5, label='Fake', color='#ff6b6b', density=True)
    axes[0, 1].hist(real_bright, bins=50, alpha=0.5, label='Real', color='#4ecdc4', density=True)
    axes[0, 1].set_title("Brightness Distribution by Class")
    axes[0, 1].set_xlabel("Mean Brightness (0-255)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()

    # Aspect ratio overall
    aspects = [r["aspect_ratio"] for r in results]
    axes[1, 0].hist(aspects, bins=50, color='#dda0dd', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Aspect Ratio Distribution (Overall)")
    axes[1, 0].set_xlabel("Width / Height")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Square (1.0)')
    axes[1, 0].axvline(np.median(aspects), color='orange', linestyle='--', label=f'Median: {np.median(aspects):.3f}')
    axes[1, 0].legend()

    # Aspect ratio by class
    fake_aspect = [r["aspect_ratio"] for r in results if r["class"] == "Fake"]
    real_aspect = [r["aspect_ratio"] for r in results if r["class"] == "Real"]
    axes[1, 1].hist(fake_aspect, bins=50, alpha=0.5, label='Fake', color='#ff6b6b', density=True)
    axes[1, 1].hist(real_aspect, bins=50, alpha=0.5, label='Real', color='#4ecdc4', density=True)
    axes[1, 1].set_title("Aspect Ratio Distribution by Class")
    axes[1, 1].set_xlabel("Width / Height")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].axvline(1.0, color='red', linestyle='--', label='Square (1.0)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "brightness_and_aspect_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Blur score distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    blur_scores = [r["blur_score"] for r in results]
    axes[0].hist(blur_scores, bins=50, color='#fab1a0', edgecolor='black', alpha=0.7)
    axes[0].set_title("Blur Score Distribution (Laplacian Variance)")
    axes[0].set_xlabel("Blur Score (higher = sharper)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(BLUR_THRESHOLD, color='red', linestyle='--', label=f'Blurry threshold: {BLUR_THRESHOLD}')
    axes[0].axvline(np.median(blur_scores), color='orange', linestyle='--', label=f'Median: {np.median(blur_scores):.1f}')
    axes[0].legend()
    axes[0].set_xlim(0, min(500, max(blur_scores)))

    # Blur by class
    fake_blur = [r["blur_score"] for r in results if r["class"] == "Fake"]
    real_blur = [r["blur_score"] for r in results if r["class"] == "Real"]
    axes[1].hist(fake_blur, bins=50, alpha=0.5, label='Fake', color='#ff6b6b', density=True)
    axes[1].hist(real_blur, bins=50, alpha=0.5, label='Real', color='#4ecdc4', density=True)
    axes[1].set_title("Blur Score Distribution by Class")
    axes[1].set_xlabel("Blur Score (higher = sharper)")
    axes[1].set_ylabel("Density")
    axes[1].axvline(BLUR_THRESHOLD, color='red', linestyle='--', label=f'Blurry threshold: {BLUR_THRESHOLD}')
    axes[1].legend()
    axes[1].set_xlim(0, min(500, max(blur_scores)))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "blur_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. File size distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes_kb = [r["file_size"] / 1024 for r in results]
    ax.hist(sizes_kb, bins=50, color='#a29bfe', edgecolor='black', alpha=0.7)
    ax.set_title("File Size Distribution")
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Frequency")
    ax.axvline(np.median(sizes_kb), color='red', linestyle='--', label=f'Median: {np.median(sizes_kb):.1f} KB')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "file_size_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✅ Histograms generated:")
    print(f"    - {OUTPUT_DIR}/class_and_size_distribution.png")
    print(f"    - {OUTPUT_DIR}/brightness_and_aspect_distribution.png")
    print(f"    - {OUTPUT_DIR}/blur_distribution.png")
    print(f"    - {OUTPUT_DIR}/file_size_distribution.png")

if __name__ == "__main__":
    # Check for scipy
    try:
        import scipy
    except ImportError:
        print("Installing scipy for blur detection...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "--break-system-packages"], check=True)
        import scipy

    run_audit()