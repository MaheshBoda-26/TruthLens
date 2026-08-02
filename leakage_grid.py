"""
Build a visual grid of cross-split leakage examples for manual verification.
Reads the top cross-split near-dup pairs persisted by leakage_audit.py.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PAIRS = json.load(open("leakage_audit_results/cross_split_near_dups.json"))
OUT = Path("leakage_audit_results")


def build_grid(pairs, cols=4, cell=200, out_name="grid.png", show_title=True):
    rows = (len(pairs) + cols - 1) // cols
    W, H = cols * cell * 2, rows * (cell + 34)
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = small = ImageFont.load_default()

    if not pairs:
        print(f"  no pairs for {out_name} — skipping")
        return
    for idx, p in enumerate(pairs):
        r, c = divmod(idx, cols)
        x0 = c * cell * 2
        y0 = r * (cell + 34)
        try:
            img1 = Image.open(p["path1"]).convert("RGB").resize((cell, cell))
            img2 = Image.open(p["path2"]).convert("RGB").resize((cell, cell))
        except Exception as e:
            print(f"  skip: {e}")
            continue
        canvas.paste(img1, (x0, y0))
        canvas.paste(img2, (x0 + cell, y0))

        def color(cls):
            return (200, 0, 0) if cls == "Fake" else (0, 128, 0)

        cap1 = f"{p['split1'][0]}.{p['class1']}"
        cap2 = f"{p['split2'][0]}.{p['class2']}"
        draw.text((x0 + 5, y0 + cell), cap1, fill=color(p["class1"]), font=small)
        draw.text((x0 + cell + 5, y0 + cell), cap2, fill=color(p["class2"]), font=small)
        if show_title:
            conflict = "CONFLICT" if p["label_conflict"] else "consistent"
            draw.text((x0 + 5, y0 + cell + 16),
                      f"d={p['distance']} {conflict}", fill=(60, 60, 60), font=small)

    canvas.save(OUT / out_name)
    print(f"  Saved {OUT / out_name}")


# 1. All label-conflict pairs (Fake in one split, Real in another)
conflicts = [p for p in PAIRS if p["label_conflict"]]
print(f"Label-conflict pairs: {len(conflicts)}")
build_grid(conflicts[:40], cols=4, cell=200, out_name="leakage_conflict_grid.png")

# 2. Sample of consistent-label cross-split pairs (leakage, same class)
consistent = [p for p in PAIRS if not p["label_conflict"]]
print(f"Consistent-label pairs: {len(consistent)}")
build_grid(consistent[:40], cols=4, cell=200, out_name="leakage_consistent_grid.png")

# 3. Test-vs-Train pairs specifically (test contamination is the worst kind)
test_pairs = [p for p in PAIRS if "Test" in {p["split1"], p["split2"]}]
print(f"Test-involved pairs: {len(test_pairs)}")
build_grid(test_pairs[:40], cols=4, cell=200, out_name="leakage_test_grid.png")
