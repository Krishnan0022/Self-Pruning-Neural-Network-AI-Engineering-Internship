"""
Loss Functions
==============
This module defines the loss components used to train the self-pruning network:

1. Classification loss  — standard cross-entropy over logits.
2. Sparsity loss        — L1 norm of all gate values, summed across every
                          PrunableLinear layer in the network.
3. Combined total loss  — classification_loss + λ * sparsity_loss

Why L1 on sigmoid gates encourages sparsity
--------------------------------------------
The L1 norm (sum of absolute values) has a well-known property: its subgradient
is constant (±1) for any non-zero value, regardless of magnitude. This means
the penalty does *not* diminish as a gate approaches zero, unlike L2 which gives
a penalty proportional to the value itself. As a result, L1 keeps pushing small
gate values all the way to zero rather than letting them settle at some small
positive number. Sigmoid outputs are always in (0, 1), so their absolute value
equals the value itself; thus SparsityLoss = Σ gate_ij over all layers.

The hyperparameter λ balances these two objectives:
  - λ too small  → gates stay open; little pruning; high accuracy.
  - λ too large  → gates collapse to zero; heavy pruning; accuracy degrades.
  - λ optimal    → many gates are pruned while important weights are retained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from src.layers.prunable_linear import PrunableLinear


@dataclass
class LossComponents:
    """Container for the individual loss terms (for logging / debugging)."""
    total: torch.Tensor
    classification: torch.Tensor
    sparsity: torch.Tensor
    lambda_val: float


# ── Individual loss terms ────────────────────────────────────────────────────

def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss between raw logits and integer class labels.

    Args:
        logits  : (B, C) unnormalised model outputs.
        targets : (B,) integer class indices in [0, C).
    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(logits, targets)


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    L1 norm of all gate values across every PrunableLinear layer.

    Since sigmoid outputs are always positive, |gate| = gate, so this
    is simply the sum of all gate values in the network. The gradient of
    this term w.r.t. gate_scores is:
        d/d(gate_scores) [Σ sigmoid(gate_scores)] = Σ sigmoid * (1 - sigmoid)
    which is non-zero almost everywhere, ensuring gates receive gradient signal.

    Args:
        model: The SelfPruningNet (or any nn.Module containing PrunableLinear layers).
    Returns:
        Scalar sparsity loss tensor.
    """
    total_sparsity = torch.tensor(0.0)

    # Move to the same device as the model parameters
    for param in model.parameters():
        total_sparsity = total_sparsity.to(param.device)
        break

    gate_count = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates()         # sigmoid(gate_scores)
            total_sparsity = total_sparsity + gates.sum()
            gate_count += gates.numel()

    # Normalise by the number of gates so λ is scale-independent
    if gate_count > 0:
        total_sparsity = total_sparsity / gate_count

    return total_sparsity


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model: nn.Module,
    lambda_sparsity: float,
) -> LossComponents:
    """
    Compute the combined training loss:
        Total = CrossEntropy(logits, targets) + λ * L1(gates)

    Args:
        logits           : (B, C) raw model outputs.
        targets          : (B,) integer class labels.
        model            : The neural network containing PrunableLinear layers.
        lambda_sparsity  : Weight controlling sparsity regularisation strength.
    Returns:
        LossComponents dataclass with `.total`, `.classification`, `.sparsity`,
        and `.lambda_val` fields.
    """
    cls_loss = classification_loss(logits, targets)
    spar_loss = sparsity_loss(model)
    total = cls_loss + lambda_sparsity * spar_loss

    return LossComponents(
        total=total,
        classification=cls_loss,
        sparsity=spar_loss,
        lambda_val=lambda_sparsity,
    )
