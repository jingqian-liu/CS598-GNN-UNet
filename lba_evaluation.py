"""
LBA (Ligand Binding Affinity) Evaluation Pipeline
CS598 GNN-UNet Project

This file covers:
1. LBA data loading (train/val/test splits)
2. Evaluation metrics (RMSE, Pearson R)
3. Experiment logging framework (CSV-based, wandb-ready)

Usage:
    python lba_evaluation.py --data_dir /mnt/d/598/data/ATOM3D --lba_split 30
"""

import os
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from scipy import stats


# ============================================================
# 1. DATA LOADING
# ============================================================

def get_lba_datamodule(data_dir: str, lba_split: int = 30, batch_size: int = 32):
    """
    Load the LBA datamodule from ProteinWorkshop.

    Split strategy:
      - lba_split=30: split by sequence identity 30% (harder, recommended)
      - lba_split=60: split by sequence identity 60% (easier)

    Data structure per sample:
      - atoms_pocket: DataFrame of pocket atoms
      - atoms_ligand: DataFrame of ligand atoms
      - scores["neglog_aff"]: binding affinity label (-log Kd/Ki)

    After LBATransform:
      - data.graph_y: binding affinity (float, regression target)
      - data.lig_flag: boolean mask, True for ligand atoms
      - data.coords: atom 3D coordinates
      - data.x: placeholder node features (overwritten by featurizer)
    """
    from proteinworkshop.datasets.atom3d_datamodule import ATOM3DDataModule

    datamodule = ATOM3DDataModule(
        task="LBA",
        data_dir=data_dir,
        lba_split=lba_split,
        batch_size=batch_size,
        num_workers=0,      # set >0 on Linux server for speed
        pin_memory=False,
    )

    print(f"[Data] LBA split: sequence-identity-{lba_split}")
    print(f"[Data] Data dir: {data_dir}")
    print(f"[Data] Downloading data if not present...")

    datamodule.prepare_data()
    datamodule.setup()

    train_dl = datamodule.train_dataloader()
    val_dl   = datamodule.val_dataloader()
    test_dl  = datamodule.test_dataloader()

    # Print split sizes
    print(f"[Data] Train batches: {len(train_dl)}")
    print(f"[Data] Val   batches: {len(val_dl)}")
    print(f"[Data] Test  batches: {len(test_dl)}")

    return datamodule, train_dl, val_dl, test_dl


def inspect_lba_batch(data_dir: str, lba_split: int = 30):
    """
    Quick sanity check: load one batch and print its structure.
    Run this first to verify the data pipeline works.
    """
    _, train_dl, _, _ = get_lba_datamodule(data_dir, lba_split, batch_size=4)

    batch = next(iter(train_dl))
    print("\n=== Batch inspection ===")
    print(f"Batch type: {type(batch)}")
    print(f"Batch keys: {batch.keys() if hasattr(batch, 'keys') else 'N/A'}")
    print(f"graph_y (labels): {batch.graph_y}")
    print(f"graph_y shape: {batch.graph_y.shape}")
    print(f"coords shape: {batch.coords.shape}")
    print(f"lig_flag shape: {batch.lig_flag.shape}")
    print(f"lig_flag sum (# ligand atoms): {batch.lig_flag.sum()}")
    print("========================\n")
    return batch


# ============================================================
# 2. EVALUATION METRICS
# ============================================================

class LBAMetrics:
    """
    Metrics for LBA regression task.

    Standard metrics from ATOM3D benchmark:
      - RMSE: Root Mean Square Error (lower is better)
      - Pearson R: Pearson correlation coefficient (higher is better)

    Usage:
        metrics = LBAMetrics()
        metrics.update(preds, targets)   # call each batch
        results = metrics.compute()      # call at end of epoch
        metrics.reset()                  # call before next epoch
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds   = []
        self.all_targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds:   model predictions, shape (N,) or (N,1)
            targets: ground truth labels, shape (N,) or (N,1)
        """
        preds   = preds.detach().cpu().float().view(-1)
        targets = targets.detach().cpu().float().view(-1)
        self.all_preds.append(preds)
        self.all_targets.append(targets)

    def compute(self) -> Dict[str, float]:
        """
        Returns dict with RMSE and Pearson R.
        """
        preds   = torch.cat(self.all_preds).numpy()
        targets = torch.cat(self.all_targets).numpy()

        # RMSE
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

        # Pearson R
        pearson_r, p_value = stats.pearsonr(preds, targets)
        pearson_r = float(pearson_r)

        return {
            "rmse":     rmse,
            "pearson_r": pearson_r,
            "p_value":  float(p_value),
            "n_samples": len(preds),
        }

    def compute_and_reset(self) -> Dict[str, float]:
        results = self.compute()
        self.reset()
        return results


def evaluate_lba_model(model, dataloader, device: str = "cuda") -> Dict[str, float]:
    """
    Run full evaluation on a dataloader.
    Drop-in function for use after training.

    Args:
        model:      trained model with forward() returning graph-level predictions
        dataloader: val or test dataloader
        device:     "cuda" or "cpu"

    Returns:
        dict with rmse, pearson_r
    """
    model.eval()
    metrics = LBAMetrics()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            preds = model(batch)

            # Handle different output shapes
            if isinstance(preds, dict):
                preds = preds["graph_pred"]  # adjust key if needed
            preds = preds.view(-1)

            targets = batch.graph_y.view(-1)
            metrics.update(preds, targets)

    return metrics.compute()


# ============================================================
# 3. EXPERIMENT LOGGING FRAMEWORK
# ============================================================

class ExperimentLogger:
    """
    Lightweight experiment logger that writes results to CSV.
    Designed to be compatible with both local runs and wandb.

    Usage:
        logger = ExperimentLogger(exp_name="schnet_lba_split30")
        logger.log_config({"encoder": "schnet", "lba_split": 30, "lr": 5e-4})
        logger.log_epoch(epoch=1, train_metrics={...}, val_metrics={...})
        logger.log_test(test_metrics={...})
        logger.save()
    """

    def __init__(self, exp_name: str, log_dir: str = "results"):
        self.exp_name  = exp_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir   = Path(log_dir) / f"{exp_name}_{self.timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.config      = {}
        self.epoch_logs  = []
        self.test_result = {}

        print(f"[Logger] Logging to: {self.log_dir}")

    def log_config(self, config: Dict):
        """Log experiment hyperparameters."""
        self.config = config
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[Logger] Config saved: {config_path}")

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ):
        """Log one epoch of train/val metrics."""
        row = {"epoch": epoch}
        if train_metrics:
            row.update({f"train/{k}": v for k, v in train_metrics.items()})
        if val_metrics:
            row.update({f"val/{k}": v for k, v in val_metrics.items()})
        if extra:
            row.update(extra)
        self.epoch_logs.append(row)

        # Print to console
        val_str = ""
        if val_metrics:
            val_str = f"  val_rmse={val_metrics.get('rmse', 'N/A'):.4f}  val_pearson_r={val_metrics.get('pearson_r', 'N/A'):.4f}"
        print(f"[Epoch {epoch:03d}]{val_str}")

    def log_test(self, test_metrics: Dict):
        """Log final test set results."""
        self.test_result = test_metrics
        test_path = self.log_dir / "test_results.json"
        with open(test_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\n[Logger] Test Results:")
        for k, v in test_metrics.items():
            print(f"         {k}: {v:.4f}" if isinstance(v, float) else f"         {k}: {v}")
        print(f"[Logger] Test results saved: {test_path}")

    def save(self):
        """Write all epoch logs to CSV."""
        if not self.epoch_logs:
            return

        csv_path = self.log_dir / "training_log.csv"
        fieldnames = list(self.epoch_logs[0].keys())

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.epoch_logs)

        print(f"[Logger] Training log saved: {csv_path}")

    def to_wandb(self, project: str, entity: str):
        """
        Push results to wandb (optional, use after local logging works).
        Call this after training completes.
        """
        try:
            import wandb
            run = wandb.init(
                project=project,
                entity=entity,
                name=self.exp_name,
                config=self.config,
            )
            for row in self.epoch_logs:
                epoch = row.pop("epoch")
                wandb.log(row, step=epoch)
            if self.test_result:
                wandb.log({"test/" + k: v for k, v in self.test_result.items()})
            run.finish()
            print(f"[Logger] Results pushed to wandb project: {project}")
        except Exception as e:
            print(f"[Logger] wandb upload failed: {e}")


# ============================================================
# 4. RESULTS COMPARISON TABLE
# ============================================================

def print_results_table(results: Dict[str, Dict]):
    """
    Print a formatted comparison table of multiple model results.

    Args:
        results: dict of {model_name: {"rmse": float, "pearson_r": float}}

    Example:
        print_results_table({
            "SchNet":   {"rmse": 1.42, "pearson_r": 0.61},
            "GVP":      {"rmse": 1.38, "pearson_r": 0.63},
            "GNN-UNet": {"rmse": 1.31, "pearson_r": 0.67},
        })
    """
    print("\n" + "="*52)
    print(f"{'Model':<20} {'RMSE':>10} {'Pearson R':>10}")
    print("-"*52)
    for model_name, metrics in results.items():
        rmse      = metrics.get("rmse", float("nan"))
        pearson_r = metrics.get("pearson_r", float("nan"))
        print(f"{model_name:<20} {rmse:>10.4f} {pearson_r:>10.4f}")
    print("="*52 + "\n")


def save_results_table(results: Dict[str, Dict], save_path: str = "results/comparison_table.csv"):
    """Save comparison table to CSV for easy copy-paste into paper."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "RMSE", "Pearson_R"])
        for model_name, metrics in results.items():
            writer.writerow([
                model_name,
                round(metrics.get("rmse", float("nan")), 4),
                round(metrics.get("pearson_r", float("nan")), 4),
            ])
    print(f"[Results] Comparison table saved: {save_path}")


# ============================================================
# 5. MAIN: QUICK SANITY CHECK
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, default="/mnt/d/598/data/ATOM3D")
    parser.add_argument("--lba_split", type=int, default=30, choices=[30, 60])
    parser.add_argument("--inspect",   action="store_true", help="Inspect one batch and exit")
    args = parser.parse_args()

    if args.inspect:
        # Just verify data loading works
        batch = inspect_lba_batch(args.data_dir, args.lba_split)
        print("Data pipeline OK!")
    else:
        # Demo: show what a full logging loop looks like
        print("=== Demo: ExperimentLogger ===")
        logger = ExperimentLogger(exp_name="schnet_lba_demo")
        logger.log_config({
            "encoder":   "schnet",
            "lba_split": args.lba_split,
            "lr":        5e-4,
            "epochs":    200,
            "batch_size": 32,
        })

        # Simulate a few epochs
        for epoch in range(1, 4):
            logger.log_epoch(
                epoch=epoch,
                train_metrics={"rmse": 2.0 - epoch * 0.1, "pearson_r": 0.4 + epoch * 0.05},
                val_metrics=  {"rmse": 2.1 - epoch * 0.1, "pearson_r": 0.38 + epoch * 0.05},
            )

        logger.log_test({"rmse": 1.80, "pearson_r": 0.55, "n_samples": 800})
        logger.save()

        # Demo comparison table
        print_results_table({
            "SchNet (baseline)": {"rmse": 1.80, "pearson_r": 0.55},
            "GVP (baseline)":    {"rmse": 1.75, "pearson_r": 0.58},
            "GNN-UNet (ours)":   {"rmse": 0.00, "pearson_r": 0.00},  # fill in after training
        })

        print("Done! Check the 'results/' directory for output files.")