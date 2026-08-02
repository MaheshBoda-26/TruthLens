# Deep Research: Comprehensive Action Plan & Technical Blueprint for TruthLens Deepfake Detection Model Accuracy (>95% Target)

## Executive Summary

TruthLens currently deploys a fine-tuned ResNet18 model (`deepfake_detector.pth`) achieving **ROC AUC 0.739, F1 ~54-56%, and a high False Positive Rate (FPR) of 49.8%** at threshold 0.52. An attempt at an EfficientNet-B2 pipeline (`train_v9.py`) was authored but never produced a saved checkpoint.

A comprehensive research sweep (covering 50+ papers, benchmark leaderboards like DeepfakeBench/DF40, Kaggle DFDC winning solutions, and real-world failure mode audits) reveals three critical insights:

1. **The gap between ~55% F1 and 95%+ is NOT an architectural bug — it is primarily a dataset & methodology issue.** The dataset used (`manjilkarki/deepfake-and-real-images`) is a repackaging of OpenForensics (ICCV 2021) and 140k Real/Fake Faces (StyleGAN). On clean 140k faces, standard CNNs achieve **95-99% accuracy**. However, severe dataset contamination (17,135 exact duplicates, 19,113 near-duplicates, 5,334 cross-split leaks, 29.2% ambiguity band, 0.512 class overlap) sets a low theoretical ceiling on this specific split.
2. **High False Positive Rate (49.8%) is caused by JPEG block-artifacts.** Compressed real images produce 8x8 DCT block boundaries that pixel-space classifiers confuse for GAN/diffusion boundary artifacts.
3. **In-dataset accuracy is a false goal for real-world deployment.** SOTA academic models hitting 98%+ on FaceForensics++ drop to **45-50 AUC points lower** on real-world social media images (Deepfake-Eval-2024).

This report outlines a complete technical roadmap to clean the dataset, upgrade the architecture, fix real-world generalization, and reach **>95% accuracy**.

---

## 1. Root Cause Diagnosis of Current TruthLens Baseline

| Diagnostic Metric | Deployed Baseline (ResNet18 v8) | Ideal Target | Root Cause |
|---|---|---|---|
| **ROC AUC** | 0.739 | >0.98 | Training set noise & lack of frequency awareness |
| **Best F1 Score** | 54.1% - 56.3% | >95.0% | 0.512 class overlap & dataset contamination |
| **False Positive Rate** | 49.8% (at t=0.52) | <5.0% | Real JPEG block artifacts mistaken for fake boundaries |
| **Deployed Arch** | ResNet18 (11.2M params) | EfficientNet-B2 / ViT / Freq-Hybrid | Server serves v8; v9 (EfficientNet) was never saved |
| **Dataset State** | 140K Kaggle split | Clean, deduplicated multi-generator dataset | 17K exact dups, 5.3K leaks, 29% ambiguity band |

---

## 2. Recommended SOTA Architectures & Model Strategy

### A. Core Architecture Recommendations for TruthLens

1. **Keep EfficientNet-B2 / B3 as the primary spatial backbone, but do NOT upgrade to B4+ without frequency guards.**
   - *Evidence:* DeepfakeBench and benchmark studies show EfficientNet-B4/B7 overfit heavily to dataset-specific compression artifacts, dropping to 0.81 AUC cross-dataset despite hitting 0.99 in-domain. B2/B3 offers the optimal balance of capacity and generalization.
   - *Action:* Use Noisy Student pretrained weights (`timm.create_model('tf_efficientnet_b2_ns', pretrained=True)`).

2. **Add a Frequency-Domain Stream (F3-Net / FreqNet Style).**
   - *Evidence:* High-frequency DCT (Discrete Cosine Transform) components isolate grid artifacts from compression vs. generation. F3-Net (ECCV) and FreqNet (AAAI) demonstrate 24.7% lower accuracy degradation under JPEG compression compared to spatial-only CNNs.
   - *Action:* Add an FFT/DCT frequency module alongside the spatial EfficientNet backbone.

3. **Incorporate Pretrained Vision Foundation Models (CLIP / DINOv2) via Layer-Norm Tuning.**
   - *Evidence:* **Effort (ICML 2025)** and **GenD (arXiv 2508)** show that freezing a vision transformer (CLIP ViT-L/14 or DINOv2) and tuning only LayerNorm/Adapter parameters (0.03% to 0.19M params) achieves **0.917 to 0.947 average cross-dataset AUC**, far outperforming full fine-tuning of CNNs.

4. **Multi-Model Ensemble (The DFDC Winner Blueprint).**
   - *Evidence:* Every winning solution in the Kaggle DFDC challenge used ensembles of heterogeneous models (e.g., 5x EfficientNet-B7/B3 + Xception/ViT). Soft-voting ensemble output probabilities never rank worst across unseen generators.

---

## 3. Dataset Upgrades & Data Quality Engineering

### A. Immediate Cleaning of the Existing Dataset

1. **Exact-Duplicate Purge:** Run SHA-256 hashing across Train, Val, and Test splits. Delete any Train sample that matches a Val/Test hash (purges 5,334 leakage groups).
2. **Perceptual Near-Duplicate Purge:** Apply `pHash` (dHash <= 8 bits) and CLIP embedding cosine similarity (>0.95) to remove near-duplicate crops across splits.
3. **Cleanlab Label-Noise Filtering:** Run `cleanlab.classification.CleanLearning` on out-of-sample predicted probabilities to automatically identify and correct the ~12 mislabeled high-confidence images and flag the 29% ambiguous band.

### B. Upgrading to Next-Generation Training Datasets

To detect modern 2024–2026 AI fakes (Midjourney v6, Flux, Stable Diffusion 3, Sora/Veo frames), merge or transition to:

| Dataset | Size | Generators Covered | Key Feature | URL |
|---|---|---|---|---|
| **GenImage** | 2.68M images | 8 (SD 1.4/1.5, Midjourney, GLIDE, VQDM, etc.) | Standard benchmark for diffusion fakes | [genimage-dataset.github.io](https://genimage-dataset.github.io/) |
| **OpenFake v2** | 3.96M images | 18+ (Flux, DALL-E 3, Midjourney 6, Kling, Wan) | Modern 2025 fakes + real-world Reddit noise split | [huggingface.co/datasets/ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) |
| **Community Forensics** | 2.7M images | 4,803 generators | Largest generator diversity in literature | [huggingface.co/datasets/OwensLab/CommunityForensics](https://huggingface.co/datasets/OwensLab/CommunityForensics) |
| **ScaleDF** | 14.6M images | 102 deepfake methods | Scaling-law dataset for domain generalization | [huggingface.co/datasets/WenhaoWang/ScaleDF](https://huggingface.co/datasets/WenhaoWang/ScaleDF) |

---

## 4. Advanced Training Strategies & Loss Functions

1. **Two-Phase Differential Fine-Tuning:**
   - *Phase 1:* Freeze backbone, train classification head only for 2 epochs at `LR = 1e-3`.
   - *Phase 2:* Unfreeze top 30-50% of layers, train at `LR = 1e-5` using Cosine Annealing scheduler with 2-epoch warmup. Freeze Batch Normalization statistics (`model.eval()` mode for BN).
2. **Realistic Degradation Augmentation Pipeline (Fixes 49.8% FPR):**
   - Add heavy JPEG Compression augmentation (`quality_range=(10, 95)`), Gaussian Blur (`kernel_size=3..7`), Downscaling/Upscaling, and Gaussian Noise to **BOTH real and fake samples** during training.
   - *Result:* Forces the model to unlearn "JPEG blockiness = Fake" and lowers false positives on compressed social media uploads by up to 10.6% AUC (PMM framework).
3. **Loss Functions:**
   - Combine Focal Loss ($\gamma = 2.0$) with Label Smoothing ($0.1$) and MixUp ($\alpha = 0.4, p = 0.5$).
   - Alternative: Use CosFace / ArcFace margin losses on feature embeddings before the linear head (improves cluster separation by ~9.3%).
4. **Face-Cropping Preprocessing:**
   - Train on MTCNN / BlazeFace cropped faces with a 30% bounding-box margin resized to 256x256 or 300x300, rather than raw uncropped images.

---

## 5. Practical Implementation Blueprint for TruthLens

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TRUTHLENS ML PIPELINE V10                       │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
            [ Spatial Stream ]          [ Frequency Stream ]
         timm tf_efficientnet_b2_ns         DCT / FFT Feature Map
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                       [ Feature Fusion Layer ]
                                   │
                        [ Linear + Dropout 0.4 ]
                                   │
                         [ Temperature Scaling ]
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
        Probability >= Threshold           Confidence Score
```

### Prioritized Action Items

1. **Fix Server Architecture Sync (Immediate - Day 1):**
   - Ensure `server.py` and `server.ts` dynamically detect the model architecture from checkpoint metadata instead of hardcoding `model_def.create_model()` (ResNet18).
2. **Execute Dataset Cleanup Script (Day 1-2):**
   - Run deduplication across Train/Val/Test (`deduplicate_splits.py`).
   - Run `cleanlab` to clean mislabeled samples.
3. **Train EfficientNet-B2 v10 with Degradation Augmentation (Day 2-4):**
   - Implement `train_v10.py` with Noisy Student weights, JPEG degradation augmentation, differential learning rates, and focal loss.
4. **Evaluate on Standardized Benchmark Harness (Day 4-5):**
   - Evaluate model using `DeepfakeBench` metrics (in-domain AUC, cross-dataset AUC on Celeb-DF v2).
5. **Deploy & Calibrate (Day 5):**
   - Save checkpoint with metadata (`architecture`, `image_size`, `threshold`, `temperature`).
   - Update server to load and serve calibrated outputs.

---

## 6. Key Resources & References

- **DeepfakeBench (Unified PyTorch Harness & 36 Models):** [github.com/SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
- **DFDC Winning Solution Code (Selim Seferbekov):** [github.com/selimsef/dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)
- **Effort ICML 2025 SOTA Model:** [github.com/YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection)
- **HuggingFace Pretrained Deepfake Models:** [huggingface.co/abraraltaf92/deepfake-detection-models](https://huggingface.co/abraraltaf92/deepfake-detection-models)
- **F3-Net Frequency Detection:** [github.com/yyk-wew/F3Net](https://github.com/yyk-wew/F3Net)
- **FreqNet AAAI 2024:** [github.com/chuangchuangtan/FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)
- **Cleanlab Label Noise Python SDK:** [docs.cleanlab.ai](https://docs.cleanlab.ai/)
