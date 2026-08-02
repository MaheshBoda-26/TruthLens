"""
TruthLens — Cross-Split Leakage Deduplication
=============================================
Removes cross-split duplicate/near-duplicate images so that Train / Validation /
Test become disjoint. Uses the cross-split near-dup pairs computed by
leakage_audit.py (leakage_audit_results/cross_split_near_dups.json, 13,390 pairs).

Strategy
--------
1. Build a graph of near-dup pairs (undirected), union into connected components.
2. Within each component that spans >1 split, keep ONE representative and
   quarantine the rest. Representative priority preserves the eval splits:
       keep priority: Test > Validation > Train
   i.e. we remove Train copies whose near-twin lives in Validation/Test, and
   remove Validation copies whose near-twin lives in Test. This keeps the Test
   split fully intact and Validation nearly intact while making the splits
   pairwise disjoint.
3. Moved files are quarantined (moved, not deleted) into
   leaked_duplicates/ so nothing is destroyed and the process is reversible.

Safety
------
- Exact duplicates (byte-identical) are NOT in the pair list from the dhash
  audit, so we also remove any path that shares a md5 group spanning splits
  (from dataset_audit_results/audit_results.json).
- Files are moved, never deleted.
"""
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

DATASET_ROOT = Path.home() / ".cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset"
SPLITS = ("Train", "Validation", "Test")
CLASSES = ("Fake", "Real")
QUARANTINE = Path("leaked_duplicates")

# priority for "keep": Test and Validation are the eval/decision surfaces — keep
# them intact so the metrics are computed on the full original sets. Train is
# where we clean most aggressively (a Train copy whose near-twin sits in Val/Test
# is pure contamination; its unique signal is limited anyway).
#   Test(0) > Validation(1) > Train(2)   → we REMOVE Train and Validation copies
#   that share a component with a kept copy.
KEEP_PRIORITY = {"Test": 0, "Validation": 1, "Train": 2}


def meta(path: Path):
    parts = path.parts
    split = cls = None
    for seg in parts:
        if seg in SPLITS:
            split = seg
        elif seg in CLASSES:
            cls = seg
    return split, cls


def union_find(edges):
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    return parent, find


def main():
    t0 = time.time()
    QUARANTINE.mkdir(exist_ok=True)

    # ── 1. Load near-dup pairs ──────────────────────────────────────────────
    pairs_file = Path("../leakage_audit_results/cross_split_near_dups.json")
    if not pairs_file.exists():
        pairs_file = Path("/Users/maheshboda/Projects/TruthLens/leakage_audit_results/cross_split_near_dups.json")
    near_pairs = json.load(open(pairs_file))
    print(f"Loaded {len(near_pairs)} near-dup pairs")

    edges = []
    for p in near_pairs:
        p1, p2 = Path(p["path1"]), Path(p["path2"])
        if p1.exists() and p2.exists():
            edges.append((str(p1), str(p2)))
    print(f"Usable pairs (both files exist): {len(edges)}")

    # ── 2. Add exact-duplicate cross-split groups (md5) ────────────────────
    prior_audit = Path("../dataset_audit_results/audit_results.json")
    if not prior_audit.exists():
        prior_audit = Path("/Users/maheshboda/Projects/TruthLens/dataset_audit_results/audit_results.json")
    if prior_audit.exists():
        audit = json.load(open(prior_audit))
        exact_groups = audit.get("exact_duplicates", {})
        n_exact_edges = 0
        for md5, plist in exact_groups.items():
            # only keep groups that span splits (cross-split leakage)
            splits = {meta(Path(p))[0] for p in plist}
            if len(splits) > 1:
                plist = [p for p in plist if Path(p).exists()]
                # chain-connect the group
                for i in range(len(plist) - 1):
                    edges.append((plist[i], plist[i + 1]))
                    n_exact_edges += 1
        print(f"Added {n_exact_edges} exact-dup edges from {len(exact_groups)} md5 groups")

    # ── 3. Union-find → components ─────────────────────────────────────────
    parent, find = union_find(edges)
    comps = defaultdict(list)
    for p in parent:
        comps[find(p)].append(p)
    print(f"\nComponents spanning splits: {len(comps)}")

    # ── 4. Decide which files to quarantine ────────────────────────────────
    to_remove = []
    stats = defaultdict(lambda: [0, 0])  # (split, cls) -> [removed, total]

    # count totals per (split, cls) first
    for split in SPLITS:
        for cls in CLASSES:
            d = DATASET_ROOT / split / cls
            if d.exists():
                stats[(split, cls)][1] = len(list(d.glob("*.jpg")))

    for comp in comps.values():
        # split of each member
        split_of = {}
        for p in comp:
            s, _ = meta(Path(p))
            split_of[p] = s
        distinct_splits = set(split_of.values())
        if len(distinct_splits) < 2:
            continue  # within-split dup, not leakage

        # keep the member with the highest keep-priority (lowest index)
        keep = min(comp, key=lambda p: KEEP_PRIORITY[split_of[p]])
        for p in comp:
            if p != keep:
                to_remove.append((p, split_of[p], meta(Path(p))[1]))

    print(f"To quarantine: {len(to_remove)} files across {len(comps)} components")

    # ── 5. Quarantine ──────────────────────────────────────────────────────
    n_moved = 0
    errors = 0
    for p, split, cls in to_remove:
        src = Path(p)
        if not src.exists():
            continue
        dest_dir = QUARANTINE / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dest_dir / src.name))
            stats[(split, cls)][0] += 1
            n_moved += 1
        except Exception as e:
            errors += 1
            print(f"  move error {src}: {e}")

    # ── 6. Report ──────────────────────────────────────────────────────────
    print(f"\nMoved {n_moved} files to {QUARANTINE}/  ({errors} errors)")
    print("\nRemoved counts by split/class:")
    for key in sorted(stats):
        removed, total = stats[key]
        print(f"  {key[0]:>10}/{key[1]:<5}: {removed:>6} removed of {total}")

    summary = {
        "pairs_loaded": len(near_pairs),
        "edges_used": len(edges),
        "components_spanning_splits": len(comps),
        "files_quarantined": n_moved,
        "errors": errors,
        "by_split_class": {f"{k[0]}/{k[1]}": {"removed": v[0], "total": v[1]} for k, v in stats.items()},
    }
    with open("dedup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to dedup_summary.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
