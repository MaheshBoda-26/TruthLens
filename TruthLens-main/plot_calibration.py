"""
TruthLens Calibration Visualization
====================================
Generates ROC curve, Precision-Recall curve, F1 vs Threshold,
and FPR vs Threshold plots from calibration results.

Usage:
    python plot_calibration.py
    python plot_calibration.py --no-show   # Save only, don't open
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("calibration_results")


def _ensure_matplotlib():
    """Import matplotlib with non-interactive backend fallback."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#111111",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#cccccc",
        "text.color": "#cccccc",
        "xtick.color": "#999999",
        "ytick.color": "#999999",
        "grid.color": "#222222",
        "font.family": "monospace",
        "font.size": 10,
    })
    return plt


def plot_roc_curve(probs, labels, plt, ax=None):
    """Plot ROC curve from raw probabilities and labels."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # Compute ROC points
    thresholds = np.linspace(0, 1, 500)
    tprs, fprs = [], []
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tprs.append(tp / n_pos if n_pos else 0)
        fprs.append(fp / n_neg if n_neg else 0)

    # AUC
    auc = abs(np.trapezoid(tprs, fprs))

    ax.plot(fprs, tprs, color="#00ff80", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], '--', color="#444444", linewidth=1, label="Random Baseline")
    ax.fill_between(fprs, tprs, alpha=0.08, color="#00ff80")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold", color="#00ff80")
    ax.legend(loc="lower right", facecolor="#1a1a1a", edgecolor="#333")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    return ax


def plot_precision_recall(probs, labels, plt, ax=None):
    """Plot Precision-Recall curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    thresholds = np.linspace(0.01, 0.99, 500)
    precisions, recalls = [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        precisions.append(p)
        recalls.append(r)

    ax.plot(recalls, precisions, color="#ff6644", linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.08, color="#ff6644")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold", color="#ff6644")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    return ax


def plot_f1_vs_threshold(sweep, plt, ax=None):
    """Plot F1 Score vs Threshold with optimal marker."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    thresholds = [m["threshold"] for m in sweep]
    f1s = [m["f1"] for m in sweep]

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    ax.plot(thresholds, f1s, color="#00ccff", linewidth=2, label="F1 Score")
    ax.axvline(x=0.52, color="#ff3333", linestyle="--", alpha=0.7, label="Current (0.52)")
    ax.axvline(x=best_t, color="#00ff80", linestyle="--", alpha=0.7, label=f"Best F1 ({best_t:.2f})")
    ax.scatter([best_t], [best_f1], color="#00ff80", s=80, zorder=5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Threshold", fontsize=14, fontweight="bold", color="#00ccff")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333")
    ax.grid(True, alpha=0.3)
    return ax


def plot_fpr_vs_threshold(sweep, plt, ax=None):
    """Plot False Positive Rate vs Threshold."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    thresholds = [m["threshold"] for m in sweep]
    fprs = [m["fpr"] for m in sweep]
    recalls = [m["recall"] for m in sweep]

    ax.plot(thresholds, fprs, color="#ff3333", linewidth=2, label="False Positive Rate")
    ax.plot(thresholds, recalls, color="#00ff80", linewidth=2, alpha=0.6, label="Recall")
    ax.axvline(x=0.52, color="#ffaa00", linestyle="--", alpha=0.7, label="Current (0.52)")
    ax.fill_between(thresholds, fprs, alpha=0.08, color="#ff3333")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("FPR & Recall vs Threshold", fontsize=14, fontweight="bold", color="#ff3333")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333")
    ax.grid(True, alpha=0.3)
    return ax


def plot_probability_histogram(probs, labels, plt, ax=None):
    """Plot distribution of predicted probabilities by class."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]

    ax.hist(real_probs, bins=80, alpha=0.6, color="#00ff80", label=f"Real (n={len(real_probs)})", density=True)
    ax.hist(fake_probs, bins=80, alpha=0.6, color="#ff3333", label=f"Fake (n={len(fake_probs)})", density=True)
    ax.axvline(x=0.52, color="#ffaa00", linestyle="--", linewidth=2, label="Current (0.52)")
    ax.set_xlabel("P(Fake)")
    ax.set_ylabel("Density")
    ax.set_title("Probability Distribution by Class", fontsize=14, fontweight="bold", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333")
    ax.grid(True, alpha=0.3)
    return ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plt = _ensure_matplotlib()

    # Load data
    data = np.load(OUTPUT_DIR / "predictions.npz")
    probs, labels = data["probs"], data["labels"]

    with open(OUTPUT_DIR / "calibration_results.json") as f:
        results = json.load(f)
    sweep = results["threshold_sweep"]

    # Create 3x2 figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle("TruthLens — Threshold Calibration Dashboard",
                 fontsize=18, fontweight="bold", color="#00ff80", y=0.98)

    plot_roc_curve(probs, labels, plt, axes[0, 0])
    plot_precision_recall(probs, labels, plt, axes[0, 1])
    plot_f1_vs_threshold(sweep, plt, axes[1, 0])
    plot_fpr_vs_threshold(sweep, plt, axes[1, 1])
    plot_probability_histogram(probs, labels, plt, axes[2, 0])

    # Summary table in last panel
    ax_table = axes[2, 1]
    ax_table.axis("off")
    rec = results.get("recommendations", {})
    lines = ["RECOMMENDATIONS SUMMARY\n"]
    for method, m in rec.items():
        lines.append(f"  {method}")
        lines.append(f"    Threshold: {m['threshold']:.2f}  F1: {m['f1']:.1%}  FPR: {m['fpr']:.1%}\n")
    lines.append(f"\n  Current threshold: {results['current_threshold']:.2f}")
    lines.append(f"  ROC-AUC: {results['roc_auc']:.4f}")
    ax_table.text(0.05, 0.95, "\n".join(lines), transform=ax_table.transAxes,
                  fontsize=11, verticalalignment="top", fontfamily="monospace",
                  color="#00ff80",
                  bbox=dict(boxstyle="round,pad=0.8", facecolor="#111111", edgecolor="#00ff80", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "calibration_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"📊 Dashboard saved to {out_path}")

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
