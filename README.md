# Self-Pruning Neural Network

A PyTorch implementation of a neural network that **learns to prune itself during training** by attaching learnable gate parameters to every weight. Instead of post-training pruning, the network is trained end-to-end with a sparsity-inducing L1 regularisation term, producing a sparse model that retains accuracy while removing unnecessary connections.

---

## Table of Contents

1. [Motivation](#motivation)
2. [How It Works](#how-it-works)
3. [Repository Structure](#repository-structure)
4. [Setup](#setup)
5. [Running Experiments](#running-experiments)
6. [Configuration](#configuration)
7. [Understanding the Results](#understanding-the-results)
8. [Example Results](#example-results)
9. [Design Decisions](#design-decisions)

---

## Motivation

Large neural networks are effective but expensive to deploy. **Pruning** — removing unnecessary weights — is a standard technique to reduce model size and inference cost. Traditional pruning is a *post-training* step: train fully, then remove small weights.

This project implements **online self-pruning**: the network decides during training which weights to discard, using a differentiable gating mechanism and L1 regularisation.

---

## How It Works

### The `PrunableLinear` Layer

Every standard `nn.Linear` layer is replaced with a `PrunableLinear` layer that adds a **gate score tensor** of the same shape as the weight matrix:

```
gates        = sigmoid(gate_scores)           # ∈ (0, 1)  per weight
pruned_weight = weight × gates                # element-wise
output        = input @ pruned_weight.T + bias
```

- `gate_scores` are learned parameters updated by the optimizer, just like `weight` and `bias`.
- When a gate collapses to ≈ 0, the corresponding weight is effectively **removed** from the network.
- Gradients flow through both `weight` and `gate_scores` via standard backpropagation.

### The Sparsity Loss (Why L1?)

The training objective is:

```
Total Loss = CrossEntropy(logits, labels) + λ × SparsityLoss
SparsityLoss = mean(gates)   # ≡ L1 norm of gates (all positive after sigmoid)
```

**Why L1 encourages sparsity:**  
The subgradient of `|x|` is a constant `±1`, regardless of `|x|`. This means the L1 penalty keeps pushing a small gate value towards zero, unlike L2 whose gradient shrinks proportionally to the value and only asymptotically approaches zero. The result: L1 produces exact zeros (truly pruned weights), while L2 produces small-but-nonzero weights.

### The λ Trade-off

| λ value | Effect |
|---------|--------|
| Too small | Gates stay open; little pruning; high accuracy |
| Too large | Most gates collapse; heavy pruning; accuracy degrades |
| Optimal  | Many gates pruned; important connections retained; good accuracy |

---

## Repository Structure

```
self-pruning-nn/
│
├── src/
│   ├── layers/
│   │   └── prunable_linear.py     # Custom gated linear layer
│   ├── models/
│   │   └── model.py               # SelfPruningNet (MLP for CIFAR-10)
│   ├── training/
│   │   ├── train.py               # Data loading, training loop, checkpointing
│   │   └── loss.py                # Classification + sparsity loss functions
│   └── utils/
│       ├── metrics.py             # Accuracy, sparsity, gate statistics
│       └── visualization.py      # Histogram, curves, comparison charts
│
├── configs/
│   └── config.yaml                # All hyperparameters
│
├── scripts/
│   └── run_experiments.py         # Orchestrates the full λ sweep
│
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- pip

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/<your-username>/self-pruning-nn.git
cd self-pruning-nn

# (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

CIFAR-10 data is downloaded automatically on the first run.

---

## Running Experiments

### Full λ sweep (recommended)

```bash
python scripts/run_experiments.py
```

This trains three models (one per λ from `config.yaml`), saves checkpoints, plots, and a Markdown results table under `./results/`.

### Custom λ values via CLI

```bash
python scripts/run_experiments.py --lambda-values 0.00001 0.0001 0.001 0.01 0.1
```

### Specify device

```bash
python scripts/run_experiments.py --device cuda   # GPU
python scripts/run_experiments.py --device cpu    # force CPU
```

### Custom config

```bash
python scripts/run_experiments.py --config path/to/my_config.yaml
```

---

## Configuration

All hyperparameters live in `configs/config.yaml`:

```yaml
training:
  epochs:        30
  batch_size:    128
  learning_rate: 1e-3

model:
  hidden_dims:    [1024, 512, 256]
  dropout_p:      0.3
  gate_init_mean: 0.5     # sigmoid(0) = 0.5 → gates start half-open

experiment:
  lambda_values: [0.0001, 0.001, 0.01]

evaluation:
  sparsity_threshold: 0.01  # gate < 0.01 → weight considered pruned
```

---

## Understanding the Results

After training, the following outputs are generated:

### Per-run (under `results/lambda_<value>/`)

| File | Description |
|------|-------------|
| `gate_distribution.png` | Histogram of final gate values. A successful run shows a large spike near 0 (pruned) and a cluster away from 0 (retained). |
| `training_curves.png` | Loss components and accuracy/sparsity over epochs. |

### Aggregate (under `results/`)

| File | Description |
|------|-------------|
| `lambda_comparison.png` | Side-by-side bar chart of test accuracy and sparsity for each λ. |
| `results_summary.md` | Markdown table: Lambda, Test Accuracy, Sparsity Level. |
| `results_summary.json` | Machine-readable full results including per-epoch history. |

---

## Example Results

*The following table is representative; actual values depend on hardware and random seed.*

| Lambda | Test Accuracy | Sparsity Level (%) |
|-------:|:-------------:|:------------------:|
| 0.0001 | ~0.56         | ~10%               |
| 0.001  | ~0.54         | ~55%               |
| 0.01   | ~0.47         | ~85%               |

**Interpretation:** Higher λ drives more gates to zero (higher sparsity) at the cost of some accuracy. The optimal λ for this architecture on CIFAR-10 is typically in the 0.0001–0.001 range, giving meaningful compression with minimal accuracy degradation.

---

## Design Decisions

### Why sigmoid (not hard threshold)?
Sigmoid keeps the gating differentiable, allowing standard gradient-based optimisation. A hard threshold would require techniques like straight-through estimators.

### Why normalise SparsityLoss by gate count?
This makes λ scale-independent of network size — the same λ produces comparable sparsity across architectures with different numbers of parameters.

### Why BatchNorm between PrunableLinear and ReLU?
BatchNorm stabilises training when gate values vary widely across weights. It also prevents the batch statistics from collapsing when many gates are near zero.

### Why gate_init_mean = 0.5?
Starting gates at ~0.5 means the network begins in a "soft" state — no bias toward full pruning or full retention — letting the gradient signal from the data guide the outcome.

---

## License

MIT
