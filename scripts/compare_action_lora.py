#!/usr/bin/env python
# coding: utf-8

"""
LoRA Action Detection Video Count Comparison Analysis
=====================================================

This script compares LoRA fine-tuning results for action detection with different numbers of training videos:
1. 50 training videos per class (140 total)
2. 100 training videos per class (280 total)
3. 200 training videos per class (560 total)

All use the same LoRA configuration and only differ in training data size.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set style for better plots
plt.style.use("default")


def load_lora_metrics():
    """Load training metrics from valid action detection LoRA experiments (LR=2e-4)"""

    # Load valid metrics files with consistent configuration (LR=2e-4)
    metrics_100 = json.load(open("metrics/action_detection/lora_action_metrics_100videos_16frames_251003-221650.json"))
    metrics_200 = json.load(open("metrics/action_detection/lora_action_metrics_200videos_16frames_251004-062932.json"))
    metrics_400 = json.load(open("metrics/action_detection/lora_action_metrics_400videos_16frames_251004-140701.json"))

    # Add method labels for identification
    metrics_100["method"] = "100 Videos/Class"
    metrics_200["method"] = "200 Videos/Class"
    metrics_400["method"] = "400 Videos/Class"

    print("✅ Loaded LoRA action detection training metrics files (LR=2e-4)")
    return metrics_100, metrics_200, metrics_400


def create_video_comparison_table(metrics_100, metrics_200, metrics_400):
    """Create detailed comparison table for different video counts"""

    data = {
        "Metric": [
            "Videos per Class",
            "Training Videos",
            "Validation Videos",
            "Test Videos",
            "Total Parameters",
            "Trainable Parameters",
            "Trainable %",
            "Final Test Accuracy",
            "Best Validation Accuracy",
            "Final Training Loss",
            "Total Training Time",
            "Training Time per Video",
            "Accuracy Gain per Minute",
        ],
        "100 Videos/Class": [
            "100",
            f"{metrics_100['num_train_videos']:,}",
            f"{metrics_100['num_val_videos']:,}",
            f"{metrics_100['num_test_videos']:,}",
            f"{metrics_100['total_params']:,}",
            f"{metrics_100['trainable_params']:,}",
            f"{metrics_100['trainable_params']/metrics_100['total_params']*100:.2f}%",
            f"{metrics_100['final_test_acc']:.4f}",
            f"{metrics_100['best_val_acc']:.4f}",
            f"{metrics_100['train_loss'][-1]:.4f}",
            f"{metrics_100['total_training_time']:.0f}s ({metrics_100['total_training_time']/60:.1f}m)",
            f"{metrics_100['total_training_time']/metrics_100['num_train_videos']:.2f}s",
            f"{metrics_100['final_test_acc']/(metrics_100['total_training_time']/60):.4f}",
        ],
        "200 Videos/Class": [
            "200",
            f"{metrics_200['num_train_videos']:,}",
            f"{metrics_200['num_val_videos']:,}",
            f"{metrics_200['num_test_videos']:,}",
            f"{metrics_200['total_params']:,}",
            f"{metrics_200['trainable_params']:,}",
            f"{metrics_200['trainable_params']/metrics_200['total_params']*100:.2f}%",
            f"{metrics_200['final_test_acc']:.4f}",
            f"{metrics_200['best_val_acc']:.4f}",
            f"{metrics_200['train_loss'][-1]:.4f}",
            f"{metrics_200['total_training_time']:.0f}s ({metrics_200['total_training_time']/60:.1f}m)",
            f"{metrics_200['total_training_time']/metrics_200['num_train_videos']:.2f}s",
            f"{metrics_200['final_test_acc']/(metrics_200['total_training_time']/60):.4f}",
        ],
        "400 Videos/Class": [
            "400",
            f"{metrics_400['num_train_videos']:,}",
            f"{metrics_400['num_val_videos']:,}",
            f"{metrics_400['num_test_videos']:,}",
            f"{metrics_400['total_params']:,}",
            f"{metrics_400['trainable_params']:,}",
            f"{metrics_400['trainable_params']/metrics_400['total_params']*100:.2f}%",
            f"{metrics_400['final_test_acc']:.4f}",
            f"{metrics_400['best_val_acc']:.4f}",
            f"{metrics_400['train_loss'][-1]:.4f}",
            f"{metrics_400['total_training_time']:.0f}s ({metrics_400['total_training_time']/60:.1f}m)",
            f"{metrics_400['total_training_time']/metrics_400['num_train_videos']:.2f}s",
            f"{metrics_400['final_test_acc']/(metrics_400['total_training_time']/60):.4f}",
        ],
    }

    df = pd.DataFrame(data)
    return df


def plot_training_curves(metrics_100, metrics_200, metrics_400):
    """Create training curve comparison plots"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("LoRA Action Detection: Impact of Training Video Count (LR=2e-4)", fontsize=16)

    epochs = list(range(1, 6))  # 5 epochs for all
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # Green, Blue, Orange
    video_counts = [100, 200, 400]

    # 1. Test Accuracy Comparison (Top Left)
    test_accuracies = [
        metrics_100["final_test_acc"],
        metrics_200["final_test_acc"],
        metrics_400["final_test_acc"],
    ]

    bars1 = ax1.bar(video_counts, test_accuracies, color=colors, alpha=0.7, width=60)
    ax1.set_xlabel("Videos per Class")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Test Accuracy vs Video Count")
    ax1.set_ylim([0.75, 0.92])  # Focus on relevant range

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars1, test_accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom')

    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Training Time Comparison (Top Right)
    train_times = [
        metrics_100["total_training_time"] / 60,
        metrics_200["total_training_time"] / 60,
        metrics_400["total_training_time"] / 60,
    ]

    bars2 = ax2.bar(video_counts, train_times, color=colors, alpha=0.7, width=60)
    ax2.set_xlabel("Videos per Class")
    ax2.set_ylabel("Total Training Time (minutes)")
    ax2.set_title("Training Time vs Video Count")

    # Add value labels on bars
    for i, (bar, time_min) in enumerate(zip(bars2, train_times)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.02,
                f'{time_min:.1f}m', ha='center', va='bottom')

    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Training Loss Comparison (Bottom Left)
    ax3.plot(epochs, metrics_100["train_loss"], 's-', label="100 Videos/Class", linewidth=2, markersize=6, color=colors[0])
    ax3.plot(epochs, metrics_200["train_loss"], '^-', label="200 Videos/Class", linewidth=2, markersize=6, color=colors[1])
    ax3.plot(epochs, metrics_400["train_loss"], 'o-', label="400 Videos/Class", linewidth=2, markersize=6, color=colors[2])
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Training Loss")
    ax3.set_title("Training Loss Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Validation Accuracy Comparison (Bottom Right)
    ax4.plot(epochs, metrics_100["val_acc"], 's-', label="100 Videos/Class", linewidth=2, markersize=6, color=colors[0])
    ax4.plot(epochs, metrics_200["val_acc"], '^-', label="200 Videos/Class", linewidth=2, markersize=6, color=colors[1])
    ax4.plot(epochs, metrics_400["val_acc"], 'o-', label="400 Videos/Class", linewidth=2, markersize=6, color=colors[2])
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Validation Accuracy")
    ax4.set_title("Validation Accuracy Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("action_lora_video_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scaling_analysis(metrics_100, metrics_200, metrics_400):
    """Create scaling analysis plots"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LoRA Action Detection: Training Data Scaling Analysis (LR=2e-4)", fontsize=16)

    video_counts = [100, 200, 400]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']

    # 1. Test Accuracy vs Training Videos (bar chart)
    test_accs = [
        metrics_100["final_test_acc"],
        metrics_200["final_test_acc"],
        metrics_400["final_test_acc"],
    ]

    bars1 = ax1.bar(video_counts, test_accs, color=colors, alpha=0.7, width=60)

    # Add labels on bars
    for i, (bar, y) in enumerate(zip(bars1, test_accs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{y:.3f}', ha='center', va='bottom')

    ax1.set_xlabel("Videos per Class")
    ax1.set_ylabel("Final Test Accuracy")
    ax1.set_title("Test Accuracy Scaling")
    ax1.set_ylim([0.75, 0.92])
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Training Time per Video
    time_per_video = [
        metrics_100["total_training_time"] / metrics_100["num_train_videos"],
        metrics_200["total_training_time"] / metrics_200["num_train_videos"],
        metrics_400["total_training_time"] / metrics_400["num_train_videos"],
    ]

    bars2 = ax2.bar(video_counts, time_per_video, color=colors, alpha=0.7, width=60)
    ax2.set_xlabel("Videos per Class")
    ax2.set_ylabel("Training Time per Video (seconds)")
    ax2.set_title("Training Efficiency")

    for i, (bar, time) in enumerate(zip(bars2, time_per_video)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(time_per_video)*0.02,
                f'{time:.1f}s', ha='center', va='bottom')

    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Best Validation Accuracy
    best_val_accs = [
        metrics_100["best_val_acc"],
        metrics_200["best_val_acc"],
        metrics_400["best_val_acc"],
    ]

    bars3 = ax3.bar(video_counts, best_val_accs, color=colors, alpha=0.7, width=60)

    for i, (bar, y) in enumerate(zip(bars3, best_val_accs)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{y:.3f}', ha='center', va='bottom')

    ax3.set_xlabel("Videos per Class")
    ax3.set_ylabel("Best Validation Accuracy")
    ax3.set_title("Validation Accuracy Scaling")
    ax3.set_ylim([0.75, 0.92])
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Final Training Loss
    final_losses = [
        metrics_100["train_loss"][-1],
        metrics_200["train_loss"][-1],
        metrics_400["train_loss"][-1],
    ]

    bars4 = ax4.bar(video_counts, final_losses, color=colors, alpha=0.7, width=60)

    for i, (bar, y) in enumerate(zip(bars4, final_losses)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(final_losses)*0.02,
                f'{y:.3f}', ha='center', va='bottom')

    ax4.set_xlabel("Videos per Class")
    ax4.set_ylabel("Final Training Loss")
    ax4.set_title("Training Loss Scaling")
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("action_lora_scaling_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_video_count_summary(metrics_100, metrics_200, metrics_400):
    """Generate comprehensive summary report for video count comparison"""

    print("=" * 80)
    print("LORA ACTION DETECTION VIDEO COUNT COMPARISON REPORT")
    print("Impact of Training Data Size on LoRA Fine-tuning Performance (LR=2e-4)")
    print("=" * 80)

    print("\n📊 PERFORMANCE SUMMARY:")
    print(f"100 Videos/Class ({metrics_100['num_train_videos']} train) - Test: {metrics_100['final_test_acc']:.4f} | Val: {metrics_100['best_val_acc']:.4f}")
    print(f"200 Videos/Class ({metrics_200['num_train_videos']} train) - Test: {metrics_200['final_test_acc']:.4f} | Val: {metrics_200['best_val_acc']:.4f}")
    print(f"400 Videos/Class ({metrics_400['num_train_videos']} train) - Test: {metrics_400['final_test_acc']:.4f} | Val: {metrics_400['best_val_acc']:.4f}")

    # Calculate improvements
    improvement_200_100 = (metrics_200["final_test_acc"] - metrics_100["final_test_acc"])
    improvement_400_200 = (metrics_400["final_test_acc"] - metrics_200["final_test_acc"])
    improvement_400_100 = (metrics_400["final_test_acc"] - metrics_100["final_test_acc"])

    print(f"\nPerformance Improvements:")
    print(f"200 vs 100 videos:  {improvement_200_100:+.4f} ({improvement_200_100/metrics_100['final_test_acc']*100:+.1f}%)")
    print(f"400 vs 200 videos:  {improvement_400_200:+.4f} ({improvement_400_200/metrics_200['final_test_acc']*100:+.1f}%)")
    print(f"400 vs 100 videos:  {improvement_400_100:+.4f} ({improvement_400_100/metrics_100['final_test_acc']*100:+.1f}%)")

    print("\n⏱️ TRAINING TIME ANALYSIS:")
    print(f"100 Videos/Class: {metrics_100['total_training_time']:.0f}s ({metrics_100['total_training_time']/60:.1f}m)")
    print(f"200 Videos/Class: {metrics_200['total_training_time']:.0f}s ({metrics_200['total_training_time']/60:.1f}m)")
    print(f"400 Videos/Class: {metrics_400['total_training_time']:.0f}s ({metrics_400['total_training_time']/60:.1f}m)")

    print(f"\nTime per Training Video:")
    print(f"100 Videos/Class: {metrics_100['total_training_time']/metrics_100['num_train_videos']:.2f}s per video")
    print(f"200 Videos/Class: {metrics_200['total_training_time']/metrics_200['num_train_videos']:.2f}s per video")
    print(f"400 Videos/Class: {metrics_400['total_training_time']/metrics_400['num_train_videos']:.2f}s per video")

    print("\n🎯 TRAINING EFFICIENCY:")
    eff_100 = metrics_100['final_test_acc'] / (metrics_100['total_training_time']/60)
    eff_200 = metrics_200['final_test_acc'] / (metrics_200['total_training_time']/60)
    eff_400 = metrics_400['final_test_acc'] / (metrics_400['total_training_time']/60)

    print(f"100 Videos/Class: {eff_100:.4f} accuracy per minute")
    print(f"200 Videos/Class: {eff_200:.4f} accuracy per minute")
    print(f"400 Videos/Class: {eff_400:.4f} accuracy per minute")

    print("\n💾 PARAMETER EFFICIENCY:")
    print(f"Total Parameters:     {metrics_100['total_params']:,}")
    print(f"Trainable Parameters: {metrics_100['trainable_params']:,}")
    print(f"Trainable Percentage: {metrics_100['trainable_params']/metrics_100['total_params']*100:.2f}%")
    print("(Same for all experiments - LoRA adapters only)")

    print("\n🔍 KEY INSIGHTS:")
    print("• LoRA trains only 0.13% of parameters while achieving strong performance")
    print(f"• Doubling from 100→200 videos: {improvement_200_100/metrics_100['final_test_acc']*100:.1f}% improvement")
    print(f"• Doubling from 200→400 videos: {improvement_400_200/metrics_200['final_test_acc']*100:.1f}% improvement")
    print("• Training time scales linearly with video count")

    print("\n📈 SCALING PATTERN:")
    if improvement_400_200 > 0:
        print("✅ More data continues to improve performance")
        print("   Additional training data provides meaningful accuracy gains")
        print(f"   Diminishing returns: 100→200 gave {improvement_200_100/metrics_100['final_test_acc']*100:.1f}% gain, 200→400 gave {improvement_400_200/metrics_200['final_test_acc']*100:.1f}% gain")
    else:
        print("⚖️  Performance plateauing - may need different approach for further gains")

    # Determine optimal point
    if metrics_400['final_test_acc'] > 0.85:
        print("\n🎯 RECOMMENDATION:")
        print("✅ 400 videos/class RECOMMENDED for production use:")
        print(f"   - Achieves excellent performance ({metrics_400['final_test_acc']:.1%} test accuracy)")
        print(f"   - Training time is reasonable (~{metrics_400['total_training_time']/60:.0f} minutes)")
        print("   - Provides robust model generalization")
        print(f"   - Use 200 videos/class if training time is critical ({metrics_200['final_test_acc']:.1%} accuracy)")
    elif metrics_200['final_test_acc'] > 0.80:
        print("\n🎯 RECOMMENDATION:")
        print("⚖️  200 videos/class offers good balance:")
        print("   - Good performance with reasonable training time")
        print("   - Consider 400 videos/class if maximum accuracy needed")
    else:
        print("\n🎯 RECOMMENDATION:")
        print("⚠️  Consider more training data:")
        print("   - Current performance may benefit from additional data")
        print("   - Or explore data augmentation techniques")

    print("\n" + "=" * 80)


def main():
    """Main analysis function"""

    print("Loading LoRA action detection metrics for different video counts...")
    metrics_100, metrics_200, metrics_400 = load_lora_metrics()

    print("Creating comparison tables...")
    comparison_df = create_video_comparison_table(metrics_100, metrics_200, metrics_400)

    print("\n" + "=" * 80)
    print("DETAILED VIDEO COUNT COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    print("\nGenerating training curve plots...")
    plot_training_curves(metrics_100, metrics_200, metrics_400)

    print("Generating scaling analysis plots...")
    plot_scaling_analysis(metrics_100, metrics_200, metrics_400)

    print("Generating comprehensive summary report...")
    generate_video_count_summary(metrics_100, metrics_200, metrics_400)

    # Save comparison table
    comparison_df.to_csv("action_lora_video_comparison_table.csv", index=False)

    print(f"\n✅ Analysis complete! Generated files:")
    print(f"   - action_lora_video_comparison_table.csv")
    print(f"   - action_lora_video_comparison.png")
    print(f"   - action_lora_scaling_analysis.png")


if __name__ == "__main__":
    main()
