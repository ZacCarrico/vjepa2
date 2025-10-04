"""Simple experiment tracking to CSV file."""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class ExperimentTracker:
    """Tracks experiments to a shared CSV file."""

    def __init__(self, csv_path: str = "experiments.csv"):
        self.csv_path = Path(csv_path)
        self.fieldnames = [
            "timestamp",
            "approach",
            "num_videos_per_class",
            "num_train_videos",
            "num_val_videos",
            "num_test_videos",
            "frames_per_clip",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "accumulation_steps",
            "num_epochs",
            "trainable_params",
            "total_params",
            "trainable_pct",
            "final_test_acc",
            "best_val_acc",
            "training_time_sec",
            "training_time_min",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
        ]

        # Create CSV with header if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_experiment(self, metrics: Dict[str, Any]):
        """
        Log an experiment to the CSV file.

        Args:
            metrics: Dictionary containing experiment metrics
        """
        # Calculate derived metrics
        if "trainable_params" in metrics and "total_params" in metrics:
            metrics["trainable_pct"] = (
                100 * metrics["trainable_params"] / metrics["total_params"]
            )

        if "total_training_time" in metrics:
            metrics["training_time_sec"] = metrics["total_training_time"]
            metrics["training_time_min"] = metrics["total_training_time"] / 60

        # Fill in None for missing fields
        row = {field: metrics.get(field, None) for field in self.fieldnames}

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        print(f"\n✅ Experiment logged to {self.csv_path}")

    def get_all_experiments(self) -> list:
        """Read all experiments from CSV."""
        experiments = []
        if self.csv_path.exists():
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                experiments = list(reader)
        return experiments

    def print_summary(self):
        """Print a summary of all experiments."""
        experiments = self.get_all_experiments()

        if not experiments:
            print("No experiments logged yet.")
            return

        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY ({len(experiments)} experiments)")
        print(f"{'='*80}")

        for exp in experiments:
            print(f"\n{exp['timestamp']} | {exp['approach']}")
            print(f"  Videos: {exp['num_videos_per_class']}/class")
            print(f"  LR: {exp['learning_rate']}")
            print(
                f"  Params: {exp['trainable_params']} ({exp.get('trainable_pct', 'N/A')}%)"
            )
            print(f"  Test Acc: {exp['final_test_acc']}")
            print(f"  Time: {exp.get('training_time_min', 'N/A')} min")
