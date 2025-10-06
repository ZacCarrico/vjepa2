#!/usr/bin/env python
"""Compare head-only vs LoRA fine-tuning approaches."""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_experiments(csv_path="experiments.csv"):
    """Load experiments from CSV file."""
    experiments = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append(row)
    return experiments


def filter_16_frame_experiments(experiments):
    """Filter experiments that use 16 frames per clip and have git hash."""
    filtered = []
    for exp in experiments:
        frames = exp.get("frames_per_clip", "")
        git_hash = exp.get("git_hash", "")
        # Include if 16 frames and has git hash
        if frames == "16" and git_hash and git_hash != "":
            filtered.append(exp)
    return filtered


def plot_comparison(experiments, output_path="approach_comparison.png"):
    """Create comparison plots for head-only vs LoRA approaches."""

    # Separate by approach
    head_only = [exp for exp in experiments if exp["approach"] == "head_only"]
    lora = [exp for exp in experiments if exp["approach"] == "lora"]

    # Sort by number of videos
    head_only.sort(key=lambda x: int(x["num_videos_per_class"]))
    lora.sort(key=lambda x: int(x["num_videos_per_class"]))

    # Extract data
    head_videos = [int(exp["num_videos_per_class"]) for exp in head_only]
    head_acc = [float(exp["final_test_acc"]) * 100 for exp in head_only]
    head_time = [float(exp["training_time_min"]) for exp in head_only]

    lora_videos = [int(exp["num_videos_per_class"]) for exp in lora]
    lora_acc = [float(exp["final_test_acc"]) * 100 for exp in lora]
    lora_time = [float(exp["training_time_min"]) for exp in lora]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Test Accuracy vs Number of Videos
    ax1.plot(head_videos, head_acc, marker='o', linewidth=2, markersize=8,
             label='Head-Only (4K params)', color='#1f77b4')
    ax1.plot(lora_videos, lora_acc, marker='s', linewidth=2, markersize=8,
             label='LoRA (496K params)', color='#ff7f0e')

    ax1.set_xlabel('Videos per Class', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy vs Training Data Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([50, 100, 200])

    # Add value labels on points
    for x, y in zip(head_videos, head_acc):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    for x, y in zip(lora_videos, lora_acc):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, -15), ha='center', fontsize=9)

    # Plot 2: Training Time vs Number of Videos
    ax2.plot(head_videos, head_time, marker='o', linewidth=2, markersize=8,
             label='Head-Only', color='#1f77b4')
    ax2.plot(lora_videos, lora_time, marker='s', linewidth=2, markersize=8,
             label='LoRA', color='#ff7f0e')

    ax2.set_xlabel('Videos per Class', fontsize=12)
    ax2.set_ylabel('Training Time (minutes)', fontsize=12)
    ax2.set_title('Training Time vs Training Data Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([50, 100, 200])

    # Add value labels on points
    for x, y in zip(head_videos, head_time):
        ax2.annotate(f'{y:.1f}m', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    for x, y in zip(lora_videos, lora_time):
        ax2.annotate(f'{y:.1f}m', (x, y), textcoords="offset points",
                    xytext=(0, -15), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print("\nHead-Only Approach (4,100 trainable params):")
    for exp in head_only:
        print(f"  {exp['num_videos_per_class']:>3} videos: "
              f"{float(exp['final_test_acc'])*100:>5.1f}% accuracy, "
              f"{float(exp['training_time_min']):>5.1f} min")

    print("\nLoRA Approach (495,620 trainable params):")
    for exp in lora:
        print(f"  {exp['num_videos_per_class']:>3} videos: "
              f"{float(exp['final_test_acc'])*100:>5.1f}% accuracy, "
              f"{float(exp['training_time_min']):>5.1f} min")

    # Calculate accuracy improvements
    print("\n" + "="*60)
    print("LORA vs HEAD-ONLY IMPROVEMENTS")
    print("="*60)
    for h, l in zip(head_only, lora):
        if h["num_videos_per_class"] == l["num_videos_per_class"]:
            videos = h["num_videos_per_class"]
            acc_diff = (float(l["final_test_acc"]) - float(h["final_test_acc"])) * 100
            time_ratio = float(l["training_time_min"]) / float(h["training_time_min"])
            print(f"{videos:>3} videos: {acc_diff:+5.1f}% accuracy, "
                  f"{time_ratio:.2f}x training time")


def main():
    """Main function."""
    print("Loading experiments from experiments.csv...")
    experiments = load_experiments()

    print(f"Found {len(experiments)} total experiments")

    # Filter to 16-frame experiments with git hash
    filtered = filter_16_frame_experiments(experiments)
    print(f"Filtered to {len(filtered)} experiments (16 frames, with git hash)")

    # Create comparison plot
    plot_comparison(filtered)


if __name__ == "__main__":
    main()
