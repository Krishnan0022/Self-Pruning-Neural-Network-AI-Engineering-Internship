"""
Evaluation Metrics
==================
Utility functions for measuring model performance and gate sparsity.

Functions
---------
compute_accuracy       : Classification accuracy on a DataLoader.
compute_network_sparsity : Fraction of weight gates below a threshold.
compute_layer_sparsities : Per-layer sparsity breakdown.
gate_statistics         : Summary statistics (mean, std, min, max) for gates.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.layers.prunable_linear import PrunableLinear


# ── Accuracy ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute top-1 classification accuracy over an entire DataLoader.

    Args:
        model  : Trained neural network in eval mode.
        loader : DataLoader yielding (images, labels) batches.
        device : Device to run inference on.
    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total   = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total if total > 0 else 0.0


# ── Network-level sparsity ────────────────────────────────────────────────────

@torch.no_grad()
def compute_network_sparsity(
    model: nn.Module,
    threshold: float = 1e-2,
) -> float:
    """
    Compute the fraction of weight gates across the entire network that are
    below `threshold`, indicating those weights are effectively pruned.

    Args:
        model     : Neural network containing PrunableLinear layers.
        threshold : Gate value below which a weight is considered pruned.
    Returns:
        Sparsity fraction in [0, 1]. A value of 0.8 means 80% of weights
        have been pruned.
    """
    total_gates  = 0
    pruned_gates = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates()
            total_gates  += gates.numel()
            pruned_gates += (gates < threshold).sum().item()

    return pruned_gates / total_gates if total_gates > 0 else 0.0


# ── Per-layer sparsity breakdown ──────────────────────────────────────────────

@torch.no_grad()
def compute_layer_sparsities(
    model: nn.Module,
    threshold: float = 1e-2,
) -> list[dict]:
    """
    Return a list of per-layer sparsity records.

    Each record is a dict:
        {
            "layer_index" : int,
            "shape"       : tuple (out_features, in_features),
            "sparsity"    : float,
            "total_gates" : int,
            "pruned_gates": int,
        }
    """
    records = []
    idx = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates  = module.get_gates()
            pruned = (gates < threshold).sum().item()
            records.append({
                "layer_index":  idx,
                "shape":        tuple(gates.shape),
                "sparsity":     pruned / gates.numel(),
                "total_gates":  gates.numel(),
                "pruned_gates": pruned,
            })
            idx += 1
    return records


# ── Gate statistics ───────────────────────────────────────────────────────────

@torch.no_grad()
def gate_statistics(model: nn.Module) -> dict[str, float]:
    """
    Compute summary statistics over all gate values in the network.

    Returns:
        Dict with keys: mean, std, min, max, median,
                        fraction_near_zero (< 0.01),
                        fraction_near_one  (> 0.99).
    """
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            all_gates.append(module.get_gates().flatten())

    if not all_gates:
        return {}

    gates_cat = torch.cat(all_gates)
    return {
        "mean":              gates_cat.mean().item(),
        "std":               gates_cat.std().item(),
        "min":               gates_cat.min().item(),
        "max":               gates_cat.max().item(),
        "median":            gates_cat.median().item(),
        "fraction_near_zero": (gates_cat < 0.01).float().mean().item(),
        "fraction_near_one":  (gates_cat > 0.99).float().mean().item(),
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_sparsity_report(
    model: nn.Module,
    threshold: float = 1e-2,
) -> None:
    """Print a formatted sparsity breakdown to stdout."""
    layer_data = compute_layer_sparsities(model, threshold)
    overall    = compute_network_sparsity(model, threshold)
    stats      = gate_statistics(model)

    print("\n" + "=" * 56)
    print(f"  SPARSITY REPORT  (threshold = {threshold})")
    print("=" * 56)
    print(f"{'Layer':>6}  {'Shape':>16}  {'Pruned':>8}  {'Total':>8}  {'Sparsity':>9}")
    print("-" * 56)
    for rec in layer_data:
        shape_str = f"{rec['shape'][0]}×{rec['shape'][1]}"
        print(
            f"  {rec['layer_index']:>4}  {shape_str:>16}  "
            f"{rec['pruned_gates']:>8,}  {rec['total_gates']:>8,}  "
            f"{rec['sparsity']:>8.1%}"
        )
    print("-" * 56)
    print(f"  {'OVERALL':>4}  {'':>16}  {'':>8}  {'':>8}  {overall:>8.1%}")
    print("=" * 56)
    print(f"\nGate statistics:")
    for k, v in stats.items():
        print(f"  {k:<24} {v:.4f}")
    print()
