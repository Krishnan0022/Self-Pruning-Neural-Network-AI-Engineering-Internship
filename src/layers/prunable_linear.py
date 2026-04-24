"""
PrunableLinear Layer
====================
A custom PyTorch linear layer augmented with learnable gate parameters.
Each weight has an associated gate score; applying sigmoid to these scores
produces soft gates in [0, 1]. The effective (pruned) weights are computed
as element-wise products of weights and gates.

During training, an L1 penalty on the gate values drives many of them toward
zero, effectively removing the corresponding weights from the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrunableLinear(nn.Module):
    """
    A linear layer whose weights are gated by learnable scalar values.

    For each weight w_ij, there is a corresponding gate score g_ij.
    The gate is computed as:
        gate_ij = sigmoid(g_ij)
    The effective (pruned) weight used in the forward pass is:
        pruned_w_ij = w_ij * gate_ij

    Gradients flow through both `weight` and `gate_scores` because the entire
    computation is expressed in differentiable PyTorch ops.

    Args:
        in_features  (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias         (bool): Whether to include an additive bias. Default: True.
        gate_init_mean (float): Mean for the normal distribution used to
                                initialise gate_scores. A slightly positive
                                value (e.g. 0.5) means gates start open so the
                                network can learn which ones to close.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init_mean: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ── Standard linear parameters ──────────────────────────────────────
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # ── Gate score parameters (same shape as weight) ─────────────────────
        # These are the raw, unconstrained values that get mapped through
        # sigmoid to produce gates in (0, 1).
        self.gate_scores = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self._init_parameters(gate_init_mean)

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_parameters(self, gate_init_mean: float) -> None:
        """Kaiming uniform for weights (PyTorch default); N(mean, 0.1) for gates."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialise gate scores so that sigmoid(gate_scores) ≈ gate_init_mean
        # sigmoid(x) = mean  ⟹  x = log(mean / (1 - mean))
        logit_mean = math.log(gate_init_mean / (1.0 - gate_init_mean))
        nn.init.normal_(self.gate_scores, mean=logit_mean, std=0.1)

    # ── Core computation ────────────────────────────────────────────────────

    def get_gates(self) -> torch.Tensor:
        """Return the gate tensor (sigmoid of gate_scores), shape (out, in)."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            gates         = sigmoid(gate_scores)          # (out, in)
            pruned_weight = weight * gates                # (out, in)  element-wise
            output        = x @ pruned_weight.T + bias   # standard linear
        """
        gates = self.get_gates()
        pruned_weight = self.weight * gates  # gradients flow through both
        return F.linear(x, pruned_weight, self.bias)

    # ── Utility ─────────────────────────────────────────────────────────────

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (considered pruned)."""
        with torch.no_grad():
            gates = self.get_gates()
            pruned = (gates < threshold).float().mean().item()
        return pruned

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
