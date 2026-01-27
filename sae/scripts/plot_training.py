#!/usr/bin/env python3
"""
Plotting script for SAE training metrics.

Reads a history.json file from an SAE training run and generates
plots for loss and other metrics over epochs.

Usage:
    python orbis/sae/plot_training.py /path/to/logs/run_name/history.json
    
    # Or specify multiple runs
    python orbis/sae/plot_training.py /path/to/run1/history.json /path/to/run2/history.json

Plots are saved in a 'plots' subdirectory next to history.json.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_history(history_path: Path) -> List[Dict]:
    """Load training history from JSON file."""
    with open(history_path, "r") as f:
        return json.load(f)


def extract_metrics(history: List[Dict]) -> Dict[str, Dict[str, List]]:
    """
    Extract train and val metrics from history.
    
    Returns:
        Dictionary with 'train' and 'val' keys, each containing
        metric names mapped to lists of (epoch, value) pairs.
    """
    metrics = {"train": {}, "val": {}}
    
    for entry in history:
        epoch = entry["epoch"]
        
        # Extract train metrics
        if entry.get("train"):
            for key, value in entry["train"].items():
                if key not in metrics["train"]:
                    metrics["train"][key] = {"epochs": [], "values": []}
                metrics["train"][key]["epochs"].append(epoch)
                metrics["train"][key]["values"].append(value)
        
        # Extract val metrics (may be None for some epochs)
        if entry.get("val"):
            for key, value in entry["val"].items():
                if key not in metrics["val"]:
                    metrics["val"][key] = {"epochs": [], "values": []}
                metrics["val"][key]["epochs"].append(epoch)
                metrics["val"][key]["values"].append(value)
    
    return metrics


def plot_metric(
    metrics: Dict[str, Dict[str, List]],
    metric_name: str,
    output_path: Path,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Plot a single metric for train and val splits."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Style configuration
    colors = {"train": "#2563eb", "val": "#dc2626"}  # Blue and red
    
    for split in ["train", "val"]:
        if metric_name in metrics[split]:
            data = metrics[split][metric_name]
            label = f"{split.capitalize()}"
            ax.plot(
                data["epochs"],
                data["values"],
                label=label,
                color=colors[split],
                linewidth=2,
                marker="o" if split == "val" else None,
                markersize=6,
                alpha=0.9,
            )
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric_name.replace('_', ' ').title()} over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_metrics(
    metrics: Dict[str, Dict[str, List]],
    output_dir: Path,
    run_name: str,
):
    """Generate a combined plot with all metrics in subplots."""
    # Get all unique metric names
    all_metrics = set(metrics["train"].keys()) | set(metrics["val"].keys())
    n_metrics = len(all_metrics)
    
    if n_metrics == 0:
        print("No metrics to plot!")
        return
    
    # Create subplot grid
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = {"train": "#2563eb", "val": "#dc2626"}
    metric_labels = {
        "loss": "Loss",
        "l0": "L0 (Active Features)",
        "cos_sim": "Cosine Similarity",
        "rel_error": "Relative Error",
    }
    
    for idx, metric_name in enumerate(sorted(all_metrics)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        for split in ["train", "val"]:
            if metric_name in metrics[split]:
                data = metrics[split][metric_name]
                ax.plot(
                    data["epochs"],
                    data["values"],
                    label=split.capitalize(),
                    color=colors[split],
                    linewidth=2,
                    marker="o" if split == "val" else None,
                    markersize=4,
                    alpha=0.9,
                )
        
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(metric_labels.get(metric_name, metric_name.replace("_", " ").title()), fontsize=10)
        ax.set_title(metric_labels.get(metric_name, metric_name.replace("_", " ").title()), fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Hide empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f"Training Metrics: {run_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "all_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_history(history_path: Path) -> Path:
    """
    Generate all plots for a training history file.
    
    Args:
        history_path: Path to history.json
        
    Returns:
        Path to the output plots directory
    """
    history_path = Path(history_path)
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    # Create plots directory next to history.json
    run_dir = history_path.parent
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    run_name = run_dir.name
    print(f"\nPlotting metrics for: {run_name}")
    print(f"Output directory: {plots_dir}")
    
    # Load and extract metrics
    history = load_history(history_path)
    metrics = extract_metrics(history)
    
    # Plot individual metrics
    metric_configs = {
        "loss": {"title": "Training Loss", "ylabel": "Loss (MSE)"},
        "l0": {"title": "L0 Sparsity", "ylabel": "Active Features"},
        "cos_sim": {"title": "Cosine Similarity", "ylabel": "Cosine Similarity"},
        "rel_error": {"title": "Relative Reconstruction Error", "ylabel": "Relative Error"},
    }
    
    for metric_name in metrics["train"].keys():
        config = metric_configs.get(metric_name, {})
        output_path = plots_dir / f"{metric_name}.png"
        plot_metric(
            metrics,
            metric_name,
            output_path,
            title=config.get("title"),
            ylabel=config.get("ylabel"),
        )
    
    # Generate combined plot
    plot_all_metrics(metrics, plots_dir, run_name)
    
    return plots_dir


def main():
    parser = argparse.ArgumentParser(
        description="Plot SAE training metrics from history.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "history_files",
        type=Path,
        nargs="+",
        help="Path(s) to history.json file(s)",
    )
    
    args = parser.parse_args()
    
    for history_path in args.history_files:
        try:
            plot_history(history_path)
        except Exception as e:
            print(f"Error processing {history_path}: {e}")


if __name__ == "__main__":
    main()
