"""
Training Script
===============
Handles:
  - CIFAR-10 data loading with standard augmentation and normalisation
  - Adam-based training loop with combined (classification + sparsity) loss
  - Per-epoch metrics: train loss, train accuracy, test accuracy, sparsity
  - Early stopping based on validation accuracy (optional)
  - Checkpoint saving for the best model per lambda run
"""

import os
import time
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import yaml

from src.models.model import SelfPruningNet
from src.training.loss import combined_loss, LossComponents
from src.utils.metrics import compute_accuracy, compute_network_sparsity

logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for CIFAR-10.

    Training transforms include random crop and horizontal flip for light
    augmentation. Test transforms are deterministic (resize + normalise only).

    CIFAR-10 channel-wise mean and std (precomputed on the training set):
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ── Single epoch training ─────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lambda_sparsity: float,
    device: torch.device,
    scheduler: Any = None,
) -> dict[str, float]:
    """
    Run one full pass over the training set.

    Returns a dict with keys:
        loss_total, loss_cls, loss_sparsity, accuracy
    """
    model.train()

    total_loss = 0.0
    total_cls  = 0.0
    total_spar = 0.0
    correct    = 0
    seen       = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        components: LossComponents = combined_loss(
            logits, labels, model, lambda_sparsity
        )
        components.total.backward()

        # Gradient clipping stabilises training when gates are near 0 or 1
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += components.total.item() * batch_size
        total_cls  += components.classification.item() * batch_size
        total_spar += components.sparsity.item() * batch_size
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        seen       += batch_size

    if scheduler is not None:
        scheduler.step()

    return {
        "loss_total":    total_loss / seen,
        "loss_cls":      total_cls  / seen,
        "loss_sparsity": total_spar / seen,
        "accuracy":      correct / seen,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sparsity_threshold: float = 1e-2,
) -> dict[str, float]:
    """
    Evaluate the model on a data loader.

    Returns:
        accuracy  : fraction of correctly classified samples
        sparsity  : fraction of pruned weights (gate < threshold)
    """
    model.eval()
    accuracy = compute_accuracy(model, loader, device)
    sparsity = compute_network_sparsity(model, threshold=sparsity_threshold)
    return {"accuracy": accuracy, "sparsity": sparsity}


# ── Full training run ─────────────────────────────────────────────────────────

def train(
    config: dict,
    lambda_sparsity: float,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """
    Train a SelfPruningNet for the given λ and return a results summary.

    Args:
        config           : Parsed YAML config dictionary.
        lambda_sparsity  : Sparsity regularisation strength.
        device           : Torch device to train on.
        checkpoint_dir   : Directory to save best model checkpoints.
        run_id           : Optional string tag for logging (e.g., "lambda_0.01").

    Returns:
        Dict with keys: lambda, best_test_accuracy, final_sparsity, history.
    """
    tag = run_id or f"lambda_{lambda_sparsity}"
    logger.info("=" * 60)
    logger.info(f"[{tag}] Starting training  λ={lambda_sparsity}")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 2),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = SelfPruningNet(
        hidden_dims=config["model"]["hidden_dims"],
        num_classes=config["model"]["num_classes"],
        dropout_p=config["model"]["dropout_p"],
        gate_init_mean=config["model"]["gate_init_mean"],
    ).to(device)

    param_info = model.count_parameters()
    logger.info(
        f"[{tag}] Model params: {param_info['trainable']:,} trainable "
        f"/ {param_info['total']:,} total"
    )

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=config["training"].get("min_lr", 1e-5),
    )

    # ── Training loop ─────────────────────────────────────────────────────
    history: list[dict] = []
    best_test_acc = 0.0
    best_epoch    = 0
    sparsity_threshold = config["evaluation"]["sparsity_threshold"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, lambda_sparsity, device, scheduler
        )
        test_metrics = evaluate(
            model, test_loader, device, sparsity_threshold
        )

        elapsed = time.time() - t0

        epoch_record = {
            "epoch":         epoch,
            "train_loss":    train_metrics["loss_total"],
            "train_cls":     train_metrics["loss_cls"],
            "train_spar":    train_metrics["loss_sparsity"],
            "train_acc":     train_metrics["accuracy"],
            "test_acc":      test_metrics["accuracy"],
            "sparsity":      test_metrics["sparsity"],
            "elapsed":       elapsed,
        }
        history.append(epoch_record)

        # Logging every epoch
        logger.info(
            f"[{tag}] Epoch {epoch:3d}/{config['training']['epochs']} | "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"(cls={train_metrics['loss_cls']:.4f}, "
            f"spar={train_metrics['loss_sparsity']:.4f}) | "
            f"train_acc={train_metrics['accuracy']:.3f} | "
            f"test_acc={test_metrics['accuracy']:.3f} | "
            f"sparsity={test_metrics['sparsity']:.3f} | "
            f"time={elapsed:.1f}s"
        )

        # Save checkpoint for best model
        if test_metrics["accuracy"] > best_test_acc:
            best_test_acc = test_metrics["accuracy"]
            best_epoch    = epoch
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoint_dir / f"{tag}_best.pt"
                torch.save(
                    {
                        "epoch":         epoch,
                        "model_state":   model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "test_accuracy": best_test_acc,
                        "sparsity":      test_metrics["sparsity"],
                        "lambda":        lambda_sparsity,
                    },
                    ckpt_path,
                )
                logger.info(f"[{tag}] ✓ Checkpoint saved → {ckpt_path}")

    # Final sparsity with the last model state
    final_sparsity = compute_network_sparsity(model, threshold=sparsity_threshold)
    logger.info(
        f"[{tag}] Finished. Best test acc={best_test_acc:.4f} "
        f"@ epoch {best_epoch}. Final sparsity={final_sparsity:.4f}"
    )

    return {
        "lambda":            lambda_sparsity,
        "best_test_accuracy": best_test_acc,
        "best_epoch":         best_epoch,
        "final_sparsity":     final_sparsity,
        "history":            history,
        "model":              model,        # caller may use for visualisation
    }


# ── Config loading helper ─────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config from disk and return as a nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logger to emit to stdout (and optionally a file)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_dir / "training.log", mode="a")
        )
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
