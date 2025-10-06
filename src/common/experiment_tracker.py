"""Simple experiment tracking to CSV file."""

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ExperimentTracker:
    """Tracks experiments to a shared CSV file."""

    @staticmethod
    def _get_git_hash() -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def __init__(self, csv_path: str = "experiments.csv"):
        self.csv_path = Path(csv_path)
        self.fieldnames = [
            "timestamp",
            "git_hash",
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
            "final_train_acc",  # NEW
            "best_val_acc",
            "best_epoch",  # NEW
            "training_time_sec",
            "training_time_min",
            "inference_time_ms",  # NEW
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            # Per-class metrics
            "precision_sitting_down",  # NEW
            "recall_sitting_down",  # NEW
            "f1_sitting_down",  # NEW
            "precision_standing_up",  # NEW
            "recall_standing_up",  # NEW
            "f1_standing_up",  # NEW
            "precision_waving",  # NEW
            "recall_waving",  # NEW
            "f1_waving",  # NEW
            "precision_other",  # NEW
            "recall_other",  # NEW
            "f1_other",  # NEW
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
        # Add git hash
        metrics["git_hash"] = self._get_git_hash()

        # Calculate derived metrics
        if "trainable_params" in metrics and "total_params" in metrics:
            metrics["trainable_pct"] = (
                100 * metrics["trainable_params"] / metrics["total_params"]
            )

        if "total_training_time" in metrics:
            metrics["training_time_sec"] = metrics["total_training_time"]
            metrics["training_time_min"] = metrics["total_training_time"] / 60

        # Flatten per-class metrics if present
        if "per_class_metrics" in metrics:
            for class_name, class_metrics in metrics["per_class_metrics"].items():
                # Normalize class name for CSV column
                class_key = class_name.lower().replace(" ", "_")
                metrics[f"precision_{class_key}"] = class_metrics.get("precision")
                metrics[f"recall_{class_key}"] = class_metrics.get("recall")
                metrics[f"f1_{class_key}"] = class_metrics.get("f1")

        # Fill in None for missing fields
        row = {field: metrics.get(field, None) for field in self.fieldnames}

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        print(f"\n✅ Experiment logged to {self.csv_path}")

    def save_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        id2label: Dict[int, str],
        output_prefix: str
    ):
        """
        Save confusion matrix as PNG visualization and JSON data.

        Args:
            confusion_matrix: numpy array with confusion matrix
            id2label: mapping from class IDs to class names
            output_prefix: prefix for output filenames (e.g., "lora_100videos_20251006-123456")
        """
        # Get class names in order
        class_names = [id2label[i] for i in sorted(id2label.keys())]

        # Save as JSON
        json_path = f"confusion_matrix_{output_prefix}.json"
        conf_matrix_data = {
            "matrix": confusion_matrix.tolist(),
            "classes": class_names
        }
        with open(json_path, "w") as f:
            json.dump(conf_matrix_data, f, indent=2)

        # Save as PNG visualization
        png_path = f"confusion_matrix_{output_prefix}.png"
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {output_prefix}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Confusion matrix saved to: {json_path} and {png_path}")

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
