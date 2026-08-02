"""
TruthLens Cross-Split Leakage Audit
===================================
Checks Train / Validation / Test for leakage:

1. Exact duplicate images across splits  (byte-identical content)
2. Near-duplicate images across splits   (perceptual dhash, Hamming <= 5)
3. Same subject / scene across splits    (near-dups are the proxy; also report
   same-scene pairs separately by class agreement)
4. Filename patterns that suggest shared source images (ID-stem overlap
   across splits for content that is NOT byte-identical)

Output: leakage_audit_results/leakage_report.json + printed summary.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
SPLITS = ("Train", "Validation", "Test")
CLASSES = ("Fake", "Real")
NEAR_DUP_THRESHOLD = 5  # hamming distance on 64-bit dhash
HASH_SIZE = 8  # 8x8 dhash -> 64 bits
OUT = Path("leakage_audit_results")
OUT.mkdir(exist_ok=True)


def dhash(path, hash_size=HASH_SIZE):
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    hv = 0
    for row in range(hash_size):
        rs = row * (hash_size + 1)
        for col in range(hash_size):
            hv = (hv << 1) | (pixels[rs + col] > pixels[rs + col + 1])
    return hv


def hamming(a, b):
    return bin(a ^ b).count("1")


def meta(path):
    """Return (split, class) from an absolute path."""
    parts = path.parts
    split = cls = None
    for seg in parts:
        if seg in SPLITS:
            split = seg
        elif seg in CLASSES:
            cls = seg
    return split, cls


def main():
    t0 = time.time()
    report = {}

    # ── 0. Enumerate all images and dhash them once ───────────────────────
    print("Hashing all images across Train / Validation / Test...")
    paths = []          # absolute Path
    hashes = {}         # path -> 64-bit int
    id_stems = defaultdict(list)  # stem ID -> list of paths (for filename check)
    for split in SPLITS:
        for cls in CLASSES:
            d = DATASET_ROOT / split / cls
            if not d.exists():
                continue
            for f in sorted(d.glob("*.jpg")):
                paths.append(f)
                try:
                    hashes[f] = dhash(f)
                except Exception as e:
                    print(f"  hash fail {f}: {e}")
                # strip extension -> numeric id (e.g. fake_42117 / real_14224)
                id_stems[f.name].append(f)
    print(f"  Total images: {len(paths)}  ({time.time()-t0:.0f}s)")

    # ── 1. Exact duplicates across splits ─────────────────────────────────
    print("\n[1/4] Exact duplicate images across splits...")
    exact_by_content = defaultdict(list)   # (size, mtime-independent) -> use md5? We don't have it here.
    # The prior audit stored md5 groups. We re-check via byte hash of files that share name-stem,
    # but for exact-dup across splits we recompute md5 on all files sharing a stem id.
    # Simpler + authoritative: use the prior audit's exact_duplicates groups directly.
    prior = json.load(open("dataset_audit_results/audit_results.json"))
    cross_split_exact = {}
    for h, plist in prior["exact_duplicates"].items():
        sl = sorted({meta(Path(p))[0] for p in plist})
        if len(sl) > 1:
            cross_split_exact[h] = (plist, sl)
    n_cross_exact = len(cross_split_exact)
    cross_exact_files = sum(len(v[0]) for v in cross_split_exact.values())
    # Split-pair breakdown
    pair_ct = Counter()
    for plist, sl in cross_split_exact.values():
        pair_ct[tuple(sl)] += 1
    # Class agreement within cross-split exact dups
    agree = 0
    for plist, sl in cross_split_exact.values():
        cls = {meta(Path(p))[1] for p in plist}
        if len(cls) == 1:
            agree += 1
    print(f"  Cross-split exact-dup groups: {n_cross_exact}  ({cross_exact_files} files)")
    print(f"    by split set: {dict(pair_ct)}")
    print(f"    groups with CONSISTENT label across splits: {agree} / {n_cross_exact}")
    print(f"    groups with CONFLICTING label across splits: {n_cross_exact - agree}")

    # ── 2. Near-duplicate images across splits (recomputed properly) ─────
    print("\n[2/4] Near-duplicate images across splits (dhash hamming<=5)...")
    # Index: group hashes by every combination of 3 bytes of the 8-byte hash.
    # Two 64-bit hashes at hamming <=5 share >=3 identical bytes (pigeonhole),
    # so every near-dup pair lands together in at least one byte-combination group.
    def byte_groups(h):
        b = [(h >> (8 * i)) & 0xFF for i in range(8)]
        for i in range(8):
            for j in range(i + 1, 8):
                for k in range(j + 1, 8):
                    yield (i, j, k, b[i] << 16 | b[j] << 8 | b[k])

    buckets = defaultdict(list)
    for p, h in hashes.items():
        for key in byte_groups(h):
            buckets[key].append(p)

    cross_split_near = []       # list of (path1, path2, distance)
    seen_pairs = set()
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        # compare all pairs within the bucket
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                p1, p2 = members[a], members[b]
                if (p2, p1) in seen_pairs or (p1, p2) in seen_pairs:
                    continue
                s1, _ = meta(p1)
                s2, _ = meta(p2)
                if s1 == s2:
                    continue
                d = hamming(hashes[p1], hashes[p2])
                if d <= NEAR_DUP_THRESHOLD:
                    seen_pairs.add((p1, p2))
                    cross_split_near.append((str(p1), str(p2), d))
    # dedupe
    cross_split_near.sort(key=lambda x: x[2])
    print(f"  Cross-split near-dup pairs: {len(cross_split_near)}")

    near_pair_ct = Counter()
    near_label_agree = 0
    near_label_conflict = 0
    for p1s, p2s, d in cross_split_near:
        p1, p2 = Path(p1s), Path(p2s)
        s1, c1 = meta(p1)
        s2, c2 = meta(p2)
        near_pair_ct[frozenset({s1, s2})] += 1
        if c1 == c2:
            near_label_agree += 1
        else:
            near_label_conflict += 1
    print(f"    by split pair: { {str(k): v for k, v in near_pair_ct.items()} }")
    print(f"    consistent label: {near_label_agree} | conflicting label: {near_label_conflict}")

    # distance histogram
    dist_ct = Counter(d for _, _, d in cross_split_near)
    print(f"    distance distribution: {dict(sorted(dist_ct.items()))}")

    # ── 3. Same subject / scene across splits ────────────────────────────
    print("\n[3/4] Same subject/scene across splits...")
    # Near-dup pairs already capture same-subject images (compression/scale changes).
    # Report the top-scene pairs and count unique "subjects" approximated by dhash buckets
    # that span splits.
    subj_span = defaultdict(set)  # (split,class) -> set of (subj hash, source)
    # group images by their full 64-bit hash: same hash bucket = same subject/scene copy
    full_bucket = defaultdict(list)
    for p, h in hashes.items():
        full_bucket[h].append(p)
    cross_split_scene = 0
    for h, members in full_bucket.items():
        sl = {meta(p)[0] for p in members}
        if len(sl) > 1:
            cross_split_scene += 1
    print(f"  Exact-hash buckets spanning splits (same scene, possibly re-encoded): {cross_split_scene}")
    # near-dups beyond exact: these are the subject repeats we found above
    print(f"  Near-dup pairs (same scene, slightly altered) spanning splits: {len(cross_split_near)}")

    # ── 4. Filename patterns suggesting shared source images ─────────────
    print("\n[4/4] Filename patterns (shared ID stems across splits)...")
    # stem id: fake_42117 / real_14224. Same stem in 2+ splits with DIFFERENT content => reused ID.
    stem_splits = defaultdict(set)
    for name, plist in id_stems.items():
        # strip leading class prefix to get numeric id
        for p in plist:
            stem_splits[name].add(meta(p)[0])
    reused_ids = {n: s for n, s in stem_splits.items() if len(s) > 1}
    print(f"  Unique filename stems: {len(id_stems)}")
    print(f"  Stems appearing in 2+ splits: {len(reused_ids)}")
    by_pair = Counter()
    for n, sl in reused_ids.items():
        by_pair[frozenset(sl)] += 1
    print(f"    by split pair: { {str(k): v for k, v in by_pair.items()} }")

    # For reused-id stems, check whether content differs (not caught by exact dup) => shared source
    examples = []
    for name, sl in list(reused_ids.items())[:20]:
        plist = id_stems[name]
        dists = []
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                dists.append(hamming(hashes[plist[i]], hashes[plist[j]]))
        examples.append({"stem": name, "splits": sorted(sl), "min_hamming": min(dists) if dists else None, "paths": [str(p) for p in plist]})
    print("    sample reused-ID stems:")
    for e in examples:
        print(f"      {e['stem']}: splits={e['splits']} min_hamming={e['min_hamming']}")

    # ── Save ─────────────────────────────────────────────────────────────
    report = {
        "total_images": len(paths),
        "exact_dup_cross_split_groups": n_cross_exact,
        "exact_dup_cross_split_files": cross_exact_files,
        "exact_dup_by_split_pair": {str(k): v for k, v in pair_ct.items()},
        "exact_dup_label_consistent": agree,
        "exact_dup_label_conflicting": n_cross_exact - agree,
        "near_dup_cross_split_pairs": len(cross_split_near),
        "near_dup_by_split_pair": {str(k): v for k, v in near_pair_ct.items()},
        "near_dup_label_consistent": near_label_agree,
        "near_dup_label_conflicting": near_label_conflict,
        "near_dup_distance_hist": {str(k): v for k, v in sorted(dist_ct.items())},
        "exact_hash_scene_buckets_spanning_splits": cross_split_scene,
        "filename_stem_reused_across_splits": len(reused_ids),
        "filename_stem_by_split_pair": {str(k): v for k, v in by_pair.items()},
        "sample_reused_id_stems": examples,
    }
    with open(OUT / "leakage_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {OUT}/leakage_report.json  ({time.time()-t0:.0f}s)")

    # Persist ALL cross-split near-dup pairs for visual verification
    pairs_out = []
    for p1s, p2s, d in cross_split_near:
        p1, p2 = Path(p1s), Path(p2s)
        pairs_out.append({
            "path1": str(p1), "path2": str(p2), "distance": d,
            "split1": meta(p1)[0], "split2": meta(p2)[0],
            "class1": meta(p1)[1], "class2": meta(p2)[1],
            "label_conflict": meta(p1)[1] != meta(p2)[1],
        })
    with open(OUT / "cross_split_near_dups.json", "w") as f:
        json.dump(pairs_out, f, indent=2)
    print(f"Saved {len(pairs_out)} cross-split near-dup pairs to {OUT}/cross_split_near_dups.json")


if __name__ == "__main__":
    main()
