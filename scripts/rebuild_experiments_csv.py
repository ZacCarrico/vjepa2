#!/usr/bin/env python
"""Rebuild experiments.csv from JSON metric files."""

import json
import glob
from src.common.experiment_tracker import ExperimentTracker


def main():
    """Rebuild experiments.csv from all *_metrics_*.json files."""
    tracker = ExperimentTracker()

    # Find all metrics JSON files
    json_files = sorted(glob.glob("*_metrics_*.json"))

    print(f"Found {len(json_files)} metrics files")

    for json_file in json_files:
        print(f"Processing: {json_file}")
        with open(json_file, "r") as f:
            metrics = json.load(f)

        # Log to experiments.csv
        tracker.log_experiment(metrics)

    print(f"\n✅ Rebuilt experiments.csv with {len(json_files)} experiments")


if __name__ == "__main__":
    main()
