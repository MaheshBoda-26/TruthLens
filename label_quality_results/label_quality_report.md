# Label Quality Audit — TruthLens Deepfake Dataset

**Scope:** `manjilkarki/deepfake-and-real-images` (Test split, 10,905 images — 5,492 Fake / 5,413 Real)
**Method:** mechanical checks (exact / perceptual near-duplicates, cross-split conflicts) + ensemble inference (ResNet18 base + hard-negative model, `probs_ens = (base + hn)/2`), then human visual verification of every flagged set.
**Date:** 2026-08-01

---

## 1. Labels That Look Wrong or Inconsistent

### 1a. Cross-class near-duplicates — the strongest signal (41–46 pairs)
Near-identical images (dhash 8×8, Hamming ≤5) sitting in **different classes**: one labeled `Fake`, its near-copy labeled `Real`. 46 total pairs across the Test split (41 unique pairs at the sampled threshold; 0 exact-byte duplicates across classes).

- Verified visually in `cross_class_near_dups_grid.png` (40 pairs). Pairs range from near-identical crops to same-scene/different-compression variants — i.e., images a human would label the same, but the dataset labels oppositely.
- This is **training data, not just test data**: the same fabrication pipeline contaminates Train/Validation (prior audit: 5,334 cross-split leakage pairs, 17,135 exact duplicates, 19,113 near-dups). At least one member of each conflicting pair is wrong by construction.

### 1b. High-confidence mislabel candidates (12)
Ensemble predicts the *opposite* class with ≥90% confidence. 11 of 12 are `Fake`-labeled images the model says are `Real` (p_ens 0.060–0.099); 1 is `Real`-labeled the model says is `Fake` (p_ens 0.906).

```
Fake  p=0.099  fake_3601.jpg      Fake  p=0.093  fake_4467.jpg
Fake  p=0.094  fake_4719.jpg      Fake  p=0.090  fake_19.jpg
Fake  p=0.094  fake_4384.jpg      Fake  p=0.087  fake_4151.jpg
Fake  p=0.083  fake_4848.jpg      Fake  p=0.083  fake_4264.jpg
Fake  p=0.070  fake_3859.jpg      Fake  p=0.064  fake_4649.jpg
Fake  p=0.060  fake_4030.jpg      Real  p=0.906  real_267.jpg
```

Verified visually in `mislabel_candidates_grid.png`. None overlap the near-dup set — an independent signal. These are the *most* defensible relabel candidates: two independently-trained models agree, and at these confidences the probability of a correct-but-hard example is low.

**Impact on metrics:** these 12 sit in the Test set the current model is scored against. At a 0.7 threshold, a Fake-labeled image the model calls Real is a **false negative** (hurts recall/F1); a Real-labeled image the model calls Fake is a **false positive** (hurts precision/FPR). Correcting them lifts measured F1 *and* makes the metric honest.

---

## 2. Ambiguous Cases — Where Two Annotators Would Disagree

**3,186 of 10,905 Test images (29.2%) fall inside the `prob_ens ∈ (0.35, 0.65)` ambiguity band** — the model cannot commit, and by extension the ground truth itself is soft on these.

| Class | In band | Total | % ambiguous |
|---|---|---|---|
| Fake | 2,033 | 5,492 | **37.0%** |
| Real | 1,153 | 5,413 | **21.3%** |
| **All** | 3,186 | 10,905 | **29.2%** |

- Verified visually in `ambiguous_sample_grid.png` (24 spread across the band) — genuinely borderline content (compression artifacts, low-quality faces, ambiguous texture), not model pathology.
- **The asymmetry is the story:** Fake-labeled images are 1.7× more likely to be ambiguous than Real-labeled ones. That is exactly the signature of a dataset where "fake" is a *family of generators with varying quality* — many fakes are clean enough to look real, so the Fake label is intrinsically soft. Real is a harder, better-defined class.

---

## 3. Classes With Unclear Boundaries

- **Distribution overlap = 0.512** (histogram-based, 0 = perfectly separated, 1 = identical). Substantial — the two probability distributions share half their mass. This is a dataset-boundary problem, not just a model-capacity problem: the current model is already an ensemble and still straddles.
- **Boundary asymmetry** (above): the Fake→Real edge is 1.7× wider than the Real→Fake edge. The classes do not have a symmetric boundary; "fake" degrades into "real" on a continuum of generation quality.
- Consistent with measured performance: base model F1 54.1%, hard-negative F1 59.0%, FPR dominated by high-compression / small-file / low-noise inputs (prior `fp_report.json`). The FP cluster (187/200 high-compression) is *label-adjacent* — heavily compressed fakes are borderline-by-content, which the labels do not acknowledge.

---

## 4. Recommendations — One Fix Per Issue, With Expected Impact

### Fix 1 — Cross-class near-duplicates → deterministic dedup + manual review
**Rec for:** the 41–46 Test pairs (small count). **Scale to:** dedup the *entire* dataset (Train/Val/Test) since the same fabrication produces thousands of in-class + cross-class dups.

1. **Automated pass (deterministic):** remove the lower-priority copy of each near-duplicate pair, keeping cross-split canonical versions separate so leakage doesn't silently re-enter Train. This is cheap and fully safe — no judgment involved.
2. **Manual review (small count):** hand-examine the ≤46 residual cross-class pairs and flip the label of the wrong member. At this count (tens, not thousands), human review is the correct tool — confidence-based relabeling would just re-introduce model bias.

**Expected accuracy impact: MEDIUM.** Every corrected Test pair is a guaranteed denoising of the eval set (removes at least 1 guaranteed error per pair, up to 46); deduping Train removes contradictory gradient signal that currently forces the model to memorize one copy against the other. F1 gain is bounded (≈0.2–0.5 pts from the Test correction) but the *train-side* consistency gain compounds with Fix 4.

### Fix 2 — High-confidence mislabel candidates → confidence-based relabeling, then manual sign-off
**Rec for:** 12 candidates (small count, but model evidence is strong). 

Take the ensemble's predicted label where confidence ≥0.9 and flip the stored label — **but only after a human signs off on the flipped set** (12 images is a 5-minute job). Use *both* models' agreement (already baked into `probs_ens`) so you're never relabeling on a single weak learner. Do **not** flip on confidence alone at scale without the Fix-3 loop.

**Expected accuracy impact: LOW→MEDIUM.** 12 images is <0.2% of Test — the direct metric gain is small. The value is **honesty**: these are the exact points where the current eval set is lying about the model, and each one currently corrupts precision *or* recall. Label impact LOW; trustworthiness impact worth doing regardless.

### Fix 3 — 3,186 ambiguous cases (large count) → active learning loop
**Rec for:** the large band (29% of the split). Manual review of 3K+ images is the wrong tool.

Feed the ambiguity band into an **active learning loop**: the two-model disagreement score (`|p_base − p_hn|` plus proximity to 0.5) ranks images by annotation value; send the top-ranked batch to human annotators *with the model's disagreement shown as a hint*; retrain; re-rank. The Fake-class 37% share means the loop is also a **data-collection signal** — it tells you the Fake class needs *cleaner* (higher-quality-generation) positives and explicit "degraded" negatives.

**Expected accuracy impact: HIGH — but only over retraining.** This is the single largest lever: 29% of the split sits on the boundary, so relabeling or re-curating even a quarter of that band removes the dominant source of gradient noise. The impact is delivered through the next training run, not the current checkpoint, and it directly attacks the 0.512 overlap.

### Fix 4 — Soft/overlapping class boundaries → label smoothing on the Fake class
**Rec for:** the intrinsic Fake→Real continuum (overlap 0.512, asymmetric band). Boundaries here are genuinely soft — a heavily compressed fake *is* visually indistinguishable from real, and no binary label can express that.

Apply **asymmetric label smoothing only to the Fake class** (e.g., targets `1 → 0.95` for Fake, keep `0 → 0` for Real), or equivalently **soft-target the ambiguity band** with the ensemble probability. Do this *only* if Fix 3 confirms the boundary is content-soft rather than annotation-sloppy (which our evidence says it is). Do **not** smooth the Real class — it is well-defined (21.3% ambiguity) and smoothing it would blur a clean signal. Pair with a **calibration step** (the checkpoint already carries temperature scaling) so the softened outputs stay decision-calibrated.

**Expected accuracy impact: MEDIUM.** Won't raise the ceiling (it can't create signal that isn't there) but stabilizes training, prevents the current F1 oscillation seen across the hard-negative-mining iterations (75.4→68.8→59.0), and gives the model a *calibrated* probability output — which is what the deployed UI actually shows users.

---

## 5. Summary Table

| # | Issue | Count | Fix | Expected accuracy impact |
|---|---|---|---|---|
| 1 | Cross-class near-dups (contradictory labels) | 41–46 pairs | Dedup + manual review | **MEDIUM** |
| 2 | High-confidence mislabel candidates | 12 | Confidence relabel + human sign-off | **LOW→MEDIUM** |
| 3 | Ambiguous band (annotators would disagree) | 3,186 (29.2%) | Active learning loop | **HIGH** (via retrain) |
| 4 | Soft class boundary (overlap 0.512, Fake-skewed) | — | Asymmetric label smoothing + calibration | **MEDIUM** |

**Prioritization:** Fix 3 is the highest-value work but requires a human-annotation pipeline. Fix 1 is the cheapest certainty (deterministic, no judgment). Fix 2 is the fastest honesty win. Fix 4 is only valid *after* Fix 3 confirms the boundary is content-soft. Do 1 → 2 immediately, run 3 as the main effort, and gate 4 on 3's findings.
