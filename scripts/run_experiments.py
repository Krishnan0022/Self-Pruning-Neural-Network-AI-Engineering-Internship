"""
Experiment Runner
=================
Orchestrates the full self-pruning experiment:
  1. Loads configuration from configs/config.yaml.
  2. Trains one model per λ value in config.experiment.lambda_values.
  3. After each run: saves training curves and gate distribution plots.
  4. After all runs: saves the λ comparison chart and Markdown results table.

Usage
-----
    # From the repository root:
    python scripts/run_experiments.py

    # Override config path:
    python scripts/run_experiments.py --config path/to/config.yaml

    # Use GPU if available (auto-detected by default):
    python scripts/run_experiments.py --device cuda
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import train, load_config, setup_logging
from src.utils.metrics import print_sparsity_report
from src.utils.visualization import (
    plot_gate_distribution,
    plot_training_curves,
    plot_lambda_comparison,
    save_results_table,
)

logger = logging.getLogger(__name__)


# ── CLI arguments ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run self-pruning neural network experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'mps', or 'cpu'. Auto-detected if not set.",
    )
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=None,
        help="Override lambda values from config (e.g. --lambda-values 0.0001 0.001 0.01).",
    )
    return parser.parse_args()


# ── Device selection ───────────────────────────────────────────────────────────

def select_device(requested: str | None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        logger.info(f"Using requested device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Auto-selected device: {device}")
    return device


# ── Main experiment loop ───────────────────────────────────────────────────────

def run_all_experiments(
    config: dict,
    lambda_values: list[float],
    device: torch.device,
) -> list[dict]:
    """
    Train one model per λ, generate per-run plots, and return all results.
    """
    results_dir    = Path(config["output"]["results_dir"])
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    results: list[dict] = []

    for lam in lambda_values:
        run_id = f"lambda_{lam}"
        logger.info(f"\n{'#' * 64}")
        logger.info(f"# Experiment: λ = {lam}")
        logger.info(f"{'#' * 64}\n")

        t_start = time.time()
        result = train(
            config=config,
            lambda_sparsity=lam,
            device=device,
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
        )
        elapsed = time.time() - t_start
        logger.info(f"[{run_id}] Total wall time: {elapsed / 60:.1f} min")

        # ── Per-run visualisations ─────────────────────────────────────────
        model = result["model"]

        plot_gate_distribution(
            model=model,
            lambda_val=lam,
            save_path=results_dir / run_id / "gate_distribution.png",
        )
        plot_training_curves(
            history=result["history"],
            lambda_val=lam,
            save_path=results_dir / run_id / "training_curves.png",
        )
        print_sparsity_report(model, threshold=config["evaluation"]["sparsity_threshold"])

        # Strip the model object before storing (not JSON-serialisable)
        result_record = {k: v for k, v in result.items() if k != "model"}
        results.append(result_record)

        logger.info(
            f"[{run_id}] Done.  "
            f"Best acc = {result['best_test_accuracy']:.4f}  |  "
            f"Sparsity = {result['final_sparsity']:.4f}"
        )

    return results


# ── Summary outputs ────────────────────────────────────────────────────────────

def generate_summary(results: list[dict], config: dict) -> None:
    results_dir = Path(config["output"]["results_dir"])

    # Aggregate bar chart
    plot_lambda_comparison(
        results=results,
        save_path=results_dir / "lambda_comparison.png",
    )

    # Markdown table
    save_results_table(
        results=results,
        save_path=results_dir / "results_summary.md",
    )

    # Machine-readable JSON dump
    json_path = results_dir / "results_summary.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Raw results saved → {json_path}")

    # Final console summary
    print("\n" + "=" * 64)
    print("  FINAL EXPERIMENT SUMMARY")
    print("=" * 64)
    print(f"  {'Lambda':<10}  {'Test Accuracy':>14}  {'Sparsity':>10}")
    print("-" * 64)
    for r in results:
        print(
            f"  {r['lambda']:<10}  "
            f"{r['best_test_accuracy']:>14.4f}  "
            f"{r['final_sparsity'] * 100:>9.2f}%"
        )
    print("=" * 64)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Set up logging to both console and file
    setup_logging(log_dir=Path(config["output"]["log_dir"]))

    device = select_device(args.device)

    # Lambda values: CLI override > config
    lambda_values = args.lambda_values or config["experiment"]["lambda_values"]
    logger.info(f"Lambda sweep: {lambda_values}")
    logger.info(f"Epochs per run: {config['training']['epochs']}")

    results = run_all_experiments(config, lambda_values, device)
    generate_summary(results, config)

    logger.info("All experiments complete. Results written to: "
                f"{config['output']['results_dir']}")


if __name__ == "__main__":
    main()
