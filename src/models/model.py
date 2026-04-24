"""
Self-Pruning Feedforward Network
=================================
A multi-layer perceptron for CIFAR-10 image classification where every
linear transformation is performed by a PrunableLinear layer. This lets
the network learn which weights to retain and which to prune, guided by
the sparsity regularization loss applied during training.

Architecture
------------
Input  : 3 × 32 × 32 = 3072 raw pixel values (normalised)
Hidden : 1024 → 512 → 256  (each followed by BatchNorm + ReLU + Dropout)
Output : 10 logits (one per CIFAR-10 class)

Design notes
------------
- BatchNorm is placed *after* the prunable linear layer and *before* ReLU
  to stabilise training and reduce sensitivity to weight scale.
- Dropout provides complementary regularisation to the gate-based pruning.
- All linear layers are PrunableLinear so that sparsity regularisation can
  be applied uniformly across the entire network.
"""

import torch
import torch.nn as nn
from src.layers.prunable_linear import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    Feedforward self-pruning network for CIFAR-10.

    Args:
        hidden_dims (list[int]): Width of each hidden layer.
                                 Default: [1024, 512, 256].
        num_classes (int)      : Number of output classes. Default: 10.
        dropout_p   (float)    : Dropout probability. Default: 0.3.
        gate_init_mean (float) : Initial mean gate value (see PrunableLinear).
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        num_classes: int = 10,
        dropout_p: float = 0.3,
        gate_init_mean: float = 0.5,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # CIFAR-10 images: 3 channels × 32 × 32 pixels
        input_dim = 3 * 32 * 32

        # ── Build hidden layers dynamically ─────────────────────────────────
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                PrunableLinear(prev_dim, hidden_dim, gate_init_mean=gate_init_mean),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # ── Output classifier head ───────────────────────────────────────────
        # Also PrunableLinear so gate sparsity spans the whole network.
        self.classifier = PrunableLinear(
            prev_dim, num_classes, gate_init_mean=gate_init_mean
        )

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, 32, 32).
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # Flatten spatial dimensions: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    # ── Gate / sparsity utilities ────────────────────────────────────────────

    def get_all_gates(self) -> list[torch.Tensor]:
        """Return gate tensors for every PrunableLinear layer in the network."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates())
        return gates

    def get_all_gate_scores(self) -> list[torch.Tensor]:
        """Return raw gate score tensors (pre-sigmoid) for every prunable layer."""
        scores = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                scores.append(module.gate_scores)
        return scores

    def network_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Compute the overall fraction of weights considered pruned.

        A weight is pruned when its corresponding gate value is below
        `threshold`. Returns a value in [0, 1].
        """
        total_weights = 0
        pruned_weights = 0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, PrunableLinear):
                    gates = module.get_gates()
                    total_weights += gates.numel()
                    pruned_weights += (gates < threshold).sum().item()
        return pruned_weights / total_weights if total_weights > 0 else 0.0

    def count_parameters(self) -> dict[str, int]:
        """Return a breakdown of total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
