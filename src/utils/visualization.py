"""
Visualisation
=============
Plotting utilities for the self-pruning experiment:

1. plot_gate_distribution : Histogram of final gate values for one model.
2. plot_training_curves   : Loss and accuracy over epochs for one run.
3. plot_lambda_comparison  : Bar chart comparing sparsity vs accuracy across λ.
4. save_results_table      : Print and save a Markdown results table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn

from src.layers.prunable_linear import PrunableLinear

logger = logging.getLogger(__name__)

# ── Shared style ──────────────────────────────────────────────────────────────

STYLE = {
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 11,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


# ── 1. Gate distribution histogram ───────────────────────────────────────────

def plot_gate_distribution(
    model: nn.Module,
    lambda_val: float,
    save_path: Path | str,
    bins: int = 80,
    threshold: float = 1e-2,
) -> None:
    """
    Plot a histogram of all gate values in the network.

    A successful pruning run will show:
      - A large spike near 0  (pruned weights)
      - A cluster of values spread between ~0.1 and 1.0 (retained weights)

    Args:
        model      : Trained SelfPruningNet.
        lambda_val : λ used during training (for plot title).
        save_path  : File path for the saved PNG.
        bins       : Number of histogram bins.
        threshold  : Threshold used to mark the pruning boundary.
    """
    _apply_style()

    all_gates: list[np.ndarray] = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().cpu().numpy().flatten())

    if not all_gates:
        logger.warning("No PrunableLinear layers found — nothing to plot.")
        return

    gates = np.concatenate(all_gates)
    pruned_frac = (gates < threshold).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Main histogram
    n, bin_edges, patches = ax.hist(
        gates, bins=bins, range=(0, 1), color="#3b82f6", edgecolor="white",
        linewidth=0.4, alpha=0.85
    )

    # Colour bars below threshold in red to highlight pruned weights
    for patch, left in zip(patches, bin_edges[:-1]):
        if left < threshold:
            patch.set_facecolor("#ef4444")
            patch.set_alpha(0.9)

    # Vertical threshold line
    ax.axvline(threshold, color="#ef4444", linestyle="--", linewidth=1.2,
               label=f"Prune threshold ({threshold})")

    ax.set_xlabel("Gate Value (sigmoid output)", labelpad=8)
    ax.set_ylabel("Weight Count", labelpad=8)
    ax.set_title(
        f"Gate Value Distribution  (λ={lambda_val})\n"
        f"{pruned_frac:.1%} of weights pruned (gate < {threshold})",
        fontsize=12,
    )
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Annotation box
    textstr = f"Total gates: {len(gates):,}\nPruned: {int(pruned_frac * len(gates)):,}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="grey")
    ax.text(0.72, 0.88, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gate distribution plot saved → {save_path}")


# ── 2. Training curves ────────────────────────────────────────────────────────

def plot_training_curves(
    history: list[dict],
    lambda_val: float,
    save_path: Path | str,
) -> None:
    """
    Plot training loss components and accuracy/sparsity over epochs.

    Args:
        history    : List of per-epoch dicts from the training loop.
        lambda_val : λ used during training.
        save_path  : File path for the saved PNG.
    """
    _apply_style()

    epochs      = [r["epoch"]      for r in history]
    train_loss  = [r["train_loss"] for r in history]
    cls_loss    = [r["train_cls"]  for r in history]
    spar_loss   = [r["train_spar"] for r in history]
    train_acc   = [r["train_acc"]  for r in history]
    test_acc    = [r["test_acc"]   for r in history]
    sparsity    = [r["sparsity"]   for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Left: losses ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Total Loss",          color="#1d4ed8", linewidth=1.8)
    ax.plot(epochs, cls_loss,   label="Classification Loss", color="#16a34a", linewidth=1.4,
            linestyle="--")
    ax.plot(epochs, spar_loss,  label=f"Sparsity Loss (×{lambda_val})",
            color="#b45309", linewidth=1.4, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss Curves  (λ={lambda_val})")
    ax.legend(frameon=False, fontsize=9)

    # ── Right: accuracy + sparsity ─────────────────────────────────────────
    ax2 = axes[1]
    colour_acc  = "#1d4ed8"
    colour_spar = "#dc2626"
    colour_tr   = "#93c5fd"

    l1, = ax2.plot(epochs, train_acc, label="Train Acc",  color=colour_tr,  linewidth=1.4)
    l2, = ax2.plot(epochs, test_acc,  label="Test Acc",   color=colour_acc, linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy", color=colour_acc)
    ax2.tick_params(axis="y", labelcolor=colour_acc)
    ax2.set_ylim(0, 1)

    ax3 = ax2.twinx()
    l3, = ax3.plot(epochs, sparsity, label="Sparsity", color=colour_spar,
                   linewidth=1.4, linestyle="-.")
    ax3.set_ylabel("Sparsity", color=colour_spar)
    ax3.tick_params(axis="y", labelcolor=colour_spar)
    ax3.set_ylim(0, 1)

    ax2.legend(handles=[l1, l2, l3], frameon=False, fontsize=9)
    ax2.set_title(f"Accuracy & Sparsity  (λ={lambda_val})")

    fig.suptitle("Self-Pruning Network Training Dynamics", fontsize=13, y=1.02)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved → {save_path}")


# ── 3. Lambda comparison bar chart ────────────────────────────────────────────

def plot_lambda_comparison(
    results: list[dict],
    save_path: Path | str,
) -> None:
    """
    Side-by-side bar chart comparing test accuracy and sparsity for each λ.

    Args:
        results   : List of result dicts, each with keys:
                    lambda, best_test_accuracy, final_sparsity.
        save_path : File path for the saved PNG.
    """
    _apply_style()

    lambdas   = [r["lambda"]             for r in results]
    accs      = [r["best_test_accuracy"] for r in results]
    sparsities= [r["final_sparsity"]     for r in results]

    x = np.arange(len(lambdas))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, accs,       width, label="Test Accuracy", color="#3b82f6",
                    alpha=0.85, edgecolor="white")
    bars2 = ax2.bar(x + width / 2, sparsities, width, label="Sparsity",      color="#ef4444",
                    alpha=0.85, edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"λ={l}" for l in lambdas], fontsize=10)
    ax1.set_ylabel("Test Accuracy",  color="#3b82f6")
    ax1.tick_params(axis="y", labelcolor="#3b82f6")
    ax1.set_ylim(0, 1)

    ax2.set_ylabel("Sparsity (fraction pruned)", color="#ef4444")
    ax2.tick_params(axis="y", labelcolor="#ef4444")
    ax2.set_ylim(0, 1)

    # Value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
                 color="#1d4ed8")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
                 color="#b91c1c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right",
               fontsize=9)

    ax1.set_title("Sparsity vs Accuracy Trade-off Across λ Values", fontsize=12)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Lambda comparison plot saved → {save_path}")


# ── 4. Results table ──────────────────────────────────────────────────────────

def save_results_table(
    results: list[dict],
    save_path: Path | str,
) -> None:
    """
    Print a Markdown results table to stdout and save it to a file.

    Args:
        results   : List of result dicts with keys:
                    lambda, best_test_accuracy, final_sparsity.
        save_path : Path to output Markdown file.
    """
    header = (
        "| Lambda | Test Accuracy | Sparsity Level (%) |\n"
        "|-------:|:-------------:|:------------------:|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r['lambda']:<6} "
            f"| {r['best_test_accuracy']:.4f}        "
            f"| {r['final_sparsity'] * 100:.2f}%              |"
        )

    table = header + "\n".join(rows)
    print("\n── Results Summary ─────────────────────────────────────")
    print(table)
    print()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("# Experiment Results\n\n")
        f.write(table)
        f.write("\n")
    logger.info(f"Results table saved → {save_path}")
