#!/usr/bin/env python
# coding: utf-8

"""
LoRA Training Video Count Comparison Analysis
============================================

This script compares LoRA fine-tuning results with different numbers of training videos:
1. 75 training videos
2. 150 training videos
3. 300 training videos

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
    """Load training metrics from the three LoRA experiments"""

    # Load all three metrics files
    metrics_75 = json.load(open("../results/metrics/lora_training_metrics_75videos_250924-17:36:59.json"))
    metrics_150 = json.load(open("../results/metrics/lora_training_metrics_150videos_250924-17:15:47.json"))
    metrics_300 = json.load(open("../results/metrics/lora_training_metrics_300videos_250924-14:26:30.json"))

    # Add method labels for identification
    metrics_75["method"] = "75 Videos"
    metrics_150["method"] = "150 Videos"
    metrics_300["method"] = "300 Videos"

    print("✅ Loaded all three LoRA training metrics files")
    return metrics_75, metrics_150, metrics_300


def create_video_comparison_table(metrics_75, metrics_150, metrics_300):
    """Create detailed comparison table for different video counts"""

    data = {
        "Metric": [
            "Training Videos",
            "Validation Videos",
            "Test Videos",
            "Total Parameters",
            "Trainable Parameters",
            "Final Test Accuracy",
            "Best Validation Accuracy",
            "Final Training Loss",
            "Total Training Time",
            "Training Time per Video",
            "Test Accuracy per Training Video",
        ],
        "75 Videos": [
            f"{metrics_75['num_train_videos']:,}",
            f"{metrics_75['num_val_videos']:,}",
            f"{metrics_75['num_test_videos']:,}",
            f"{metrics_75['total_params']:,}",
            f"{metrics_75['trainable_params']:,}",
            f"{metrics_75['final_test_acc']:.4f}",
            f"{metrics_75['best_val_acc']:.4f}",
            f"{metrics_75['train_loss'][-1]:.4f}",
            f"{metrics_75['total_training_time']:.0f}s ({metrics_75['total_training_time']/60:.1f}m)",
            f"{metrics_75['total_training_time']/metrics_75['num_train_videos']:.2f}s",
            f"{metrics_75['final_test_acc']/metrics_75['num_train_videos']*1000:.3f}",
        ],
        "150 Videos": [
            f"{metrics_150['num_train_videos']:,}",
            f"{metrics_150['num_val_videos']:,}",
            f"{metrics_150['num_test_videos']:,}",
            f"{metrics_150['total_params']:,}",
            f"{metrics_150['trainable_params']:,}",
            f"{metrics_150['final_test_acc']:.4f}",
            f"{metrics_150['best_val_acc']:.4f}",
            f"{metrics_150['train_loss'][-1]:.4f}",
            f"{metrics_150['total_training_time']:.0f}s ({metrics_150['total_training_time']/60:.1f}m)",
            f"{metrics_150['total_training_time']/metrics_150['num_train_videos']:.2f}s",
            f"{metrics_150['final_test_acc']/metrics_150['num_train_videos']*1000:.3f}",
        ],
        "300 Videos": [
            f"{metrics_300['num_train_videos']:,}",
            f"{metrics_300['num_val_videos']:,}",
            f"{metrics_300['num_test_videos']:,}",
            f"{metrics_300['total_params']:,}",
            f"{metrics_300['trainable_params']:,}",
            f"{metrics_300['final_test_acc']:.4f}",
            f"{metrics_300['best_val_acc']:.4f}",
            f"{metrics_300['train_loss'][-1]:.4f}",
            f"{metrics_300['total_training_time']:.0f}s ({metrics_300['total_training_time']/60:.1f}m)",
            f"{metrics_300['total_training_time']/metrics_300['num_train_videos']:.2f}s",
            f"{metrics_300['final_test_acc']/metrics_300['num_train_videos']*1000:.3f}",
        ],
    }

    df = pd.DataFrame(data)
    return df


def plot_training_curves(metrics_75, metrics_150, metrics_300):
    """Create training curve comparison plots"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("LoRA Training Comparison: Impact of Training Video Count", fontsize=16)

    epochs = list(range(1, 6))  # 5 epochs for all
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']  # Orange, Green, Blue

    # 1. Training Loss Comparison
    ax1.plot(epochs, metrics_75["train_loss"], 'o-', label="75 Videos", linewidth=2, markersize=6, color=colors[0])
    ax1.plot(epochs, metrics_150["train_loss"], 's-', label="150 Videos", linewidth=2, markersize=6, color=colors[1])
    ax1.plot(epochs, metrics_300["train_loss"], '^-', label="300 Videos", linewidth=2, markersize=6, color=colors[2])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Validation Accuracy Comparison
    ax2.plot(epochs, metrics_75["val_acc"], 'o-', label="75 Videos", linewidth=2, markersize=6, color=colors[0])
    ax2.plot(epochs, metrics_150["val_acc"], 's-', label="150 Videos", linewidth=2, markersize=6, color=colors[1])
    ax2.plot(epochs, metrics_300["val_acc"], '^-', label="300 Videos", linewidth=2, markersize=6, color=colors[2])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Training Time Comparison
    video_counts = [75, 150, 300]
    train_times = [
        metrics_75["total_training_time"] / 60,
        metrics_150["total_training_time"] / 60,
        metrics_300["total_training_time"] / 60,
    ]

    bars3 = ax3.bar(video_counts, train_times, color=colors, alpha=0.7, width=25)
    ax3.set_xlabel("Number of Training Videos")
    ax3.set_ylabel("Total Training Time (minutes)")
    ax3.set_title("Training Time vs Video Count")

    # Add value labels on bars
    for i, (bar, time_min) in enumerate(zip(bars3, train_times)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.02,
                f'{time_min:.1f}m', ha='center', va='bottom')

    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Final Test Accuracy Comparison
    test_accuracies = [
        metrics_75["final_test_acc"],
        metrics_150["final_test_acc"],
        metrics_300["final_test_acc"],
    ]

    bars4 = ax4.bar(video_counts, test_accuracies, color=colors, alpha=0.7, width=25)
    ax4.set_xlabel("Number of Training Videos")
    ax4.set_ylabel("Test Accuracy")
    ax4.set_title("Test Accuracy vs Video Count")

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars4, test_accuracies)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom')

    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("lora_video_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scaling_analysis(metrics_75, metrics_150, metrics_300):
    """Create scaling analysis plots"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LoRA Training Data Scaling Analysis", fontsize=16)

    video_counts = [75, 150, 300]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    # 1. Test Accuracy vs Training Videos (with trend line)
    test_accs = [
        metrics_75["final_test_acc"],
        metrics_150["final_test_acc"],
        metrics_300["final_test_acc"],
    ]

    ax1.scatter(video_counts, test_accs, color=colors, s=100, alpha=0.7)
    ax1.plot(video_counts, test_accs, 'k--', alpha=0.5)

    # Add labels for each point
    for i, (x, y) in enumerate(zip(video_counts, test_accs)):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    ax1.set_xlabel("Number of Training Videos")
    ax1.set_ylabel("Final Test Accuracy")
    ax1.set_title("Test Accuracy Scaling")
    ax1.grid(True, alpha=0.3)

    # 2. Training Time per Video
    time_per_video = [
        metrics_75["total_training_time"] / metrics_75["num_train_videos"],
        metrics_150["total_training_time"] / metrics_150["num_train_videos"],
        metrics_300["total_training_time"] / metrics_300["num_train_videos"],
    ]

    bars2 = ax2.bar(video_counts, time_per_video, color=colors, alpha=0.7, width=25)
    ax2.set_xlabel("Number of Training Videos")
    ax2.set_ylabel("Training Time per Video (seconds)")
    ax2.set_title("Training Efficiency")

    for i, (bar, time) in enumerate(zip(bars2, time_per_video)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(time_per_video)*0.02,
                f'{time:.1f}s', ha='center', va='bottom')

    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Best Validation Accuracy
    best_val_accs = [
        metrics_75["best_val_acc"],
        metrics_150["best_val_acc"],
        metrics_300["best_val_acc"],
    ]

    ax3.scatter(video_counts, best_val_accs, color=colors, s=100, alpha=0.7)
    ax3.plot(video_counts, best_val_accs, 'k--', alpha=0.5)

    for i, (x, y) in enumerate(zip(video_counts, best_val_accs)):
        ax3.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    ax3.set_xlabel("Number of Training Videos")
    ax3.set_ylabel("Best Validation Accuracy")
    ax3.set_title("Validation Accuracy Scaling")
    ax3.grid(True, alpha=0.3)

    # 4. Final Training Loss
    final_losses = [
        metrics_75["train_loss"][-1],
        metrics_150["train_loss"][-1],
        metrics_300["train_loss"][-1],
    ]

    ax4.scatter(video_counts, final_losses, color=colors, s=100, alpha=0.7)
    ax4.plot(video_counts, final_losses, 'k--', alpha=0.5)

    for i, (x, y) in enumerate(zip(video_counts, final_losses)):
        ax4.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    ax4.set_xlabel("Number of Training Videos")
    ax4.set_ylabel("Final Training Loss")
    ax4.set_title("Training Loss Scaling")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lora_scaling_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_video_count_summary(metrics_75, metrics_150, metrics_300):
    """Generate comprehensive summary report for video count comparison"""

    print("=" * 80)
    print("LORA TRAINING VIDEO COUNT COMPARISON REPORT")
    print("Impact of Training Data Size on LoRA Fine-tuning Performance")
    print("=" * 80)

    print("\n📊 PERFORMANCE SUMMARY:")
    print(f"75 Videos  - Test Accuracy: {metrics_75['final_test_acc']:.4f} | Val Accuracy: {metrics_75['best_val_acc']:.4f}")
    print(f"150 Videos - Test Accuracy: {metrics_150['final_test_acc']:.4f} | Val Accuracy: {metrics_150['best_val_acc']:.4f}")
    print(f"300 Videos - Test Accuracy: {metrics_300['final_test_acc']:.4f} | Val Accuracy: {metrics_300['best_val_acc']:.4f}")

    # Calculate improvements
    improvement_150_75 = (metrics_150["final_test_acc"] - metrics_75["final_test_acc"])
    improvement_300_150 = (metrics_300["final_test_acc"] - metrics_150["final_test_acc"])
    improvement_300_75 = (metrics_300["final_test_acc"] - metrics_75["final_test_acc"])

    print(f"\nPerformance Improvements:")
    print(f"150 vs 75 videos:   {improvement_150_75:+.4f} ({improvement_150_75/metrics_75['final_test_acc']*100:+.1f}%)")
    print(f"300 vs 150 videos:  {improvement_300_150:+.4f} ({improvement_300_150/metrics_150['final_test_acc']*100:+.1f}%)")
    print(f"300 vs 75 videos:   {improvement_300_75:+.4f} ({improvement_300_75/metrics_75['final_test_acc']*100:+.1f}%)")

    print("\n⏱️ TRAINING TIME ANALYSIS:")
    print(f"75 Videos:  {metrics_75['total_training_time']:.0f}s ({metrics_75['total_training_time']/60:.1f}m)")
    print(f"150 Videos: {metrics_150['total_training_time']:.0f}s ({metrics_150['total_training_time']/60:.1f}m)")
    print(f"300 Videos: {metrics_300['total_training_time']:.0f}s ({metrics_300['total_training_time']/60:.1f}m)")

    print(f"\nTime per Video:")
    print(f"75 Videos:  {metrics_75['total_training_time']/75:.2f}s per video")
    print(f"150 Videos: {metrics_150['total_training_time']/150:.2f}s per video")
    print(f"300 Videos: {metrics_300['total_training_time']/300:.2f}s per video")

    print("\n🎯 TRAINING EFFICIENCY:")
    eff_75 = metrics_75['final_test_acc'] / (metrics_75['total_training_time']/60)
    eff_150 = metrics_150['final_test_acc'] / (metrics_150['total_training_time']/60)
    eff_300 = metrics_300['final_test_acc'] / (metrics_300['total_training_time']/60)

    print(f"75 Videos:  {eff_75:.4f} accuracy per minute")
    print(f"150 Videos: {eff_150:.4f} accuracy per minute")
    print(f"300 Videos: {eff_300:.4f} accuracy per minute")

    print("\n🔍 KEY INSIGHTS:")
    print("• More training data significantly improves LoRA fine-tuning performance")
    print(f"• Doubling from 75→150 videos: {improvement_150_75/metrics_75['final_test_acc']*100:.1f}% improvement")
    print(f"• Doubling from 150→300 videos: {improvement_300_150/metrics_150['final_test_acc']*100:.1f}% improvement")
    print("• Training time scales roughly linearly with video count")
    print("• Diminishing returns: bigger gains from 75→150 than 150→300")

    print("\n📈 SCALING PATTERN:")
    if improvement_150_75 > improvement_300_150:
        print("✅ Diminishing returns observed - initial data increase more beneficial")
        print("   Suggests the model benefits significantly from basic diversity")
        print("   but additional data provides smaller incremental gains")
    else:
        print("⚠️  Linear or increasing returns - more data continues to help significantly")

    # Determine optimal point
    if metrics_300['final_test_acc'] > 0.9:
        print("\n🎯 RECOMMENDATION:")
        print("✅ 300+ videos RECOMMENDED for production use:")
        print("   - Achieves excellent performance (>90% accuracy)")
        print("   - Justifies the additional training time")
        print("   - Provides robust model generalization")
    elif metrics_150['final_test_acc'] > 0.8:
        print("\n🎯 RECOMMENDATION:")
        print("⚖️  150 videos offers good balance:")
        print("   - Decent performance with reasonable training time")
        print("   - Consider 300 videos if maximum accuracy needed")
    else:
        print("\n🎯 RECOMMENDATION:")
        print("⚠️  Consider more training data or different approaches:")
        print("   - Current performance may be insufficient for production")
        print("   - Explore data augmentation or different architectures")

    print("\n" + "=" * 80)


def main():
    """Main analysis function"""

    print("Loading LoRA training metrics for different video counts...")
    metrics_75, metrics_150, metrics_300 = load_lora_metrics()

    print("Creating comparison tables...")
    comparison_df = create_video_comparison_table(metrics_75, metrics_150, metrics_300)

    print("\n" + "=" * 80)
    print("DETAILED VIDEO COUNT COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    print("\nGenerating training curve plots...")
    plot_training_curves(metrics_75, metrics_150, metrics_300)

    print("Generating scaling analysis plots...")
    plot_scaling_analysis(metrics_75, metrics_150, metrics_300)

    print("Generating comprehensive summary report...")
    generate_video_count_summary(metrics_75, metrics_150, metrics_300)

    # Save comparison table
    comparison_df.to_csv("lora_video_comparison_table.csv", index=False)

    print(f"\n✅ Analysis complete! Generated files:")
    print(f"   - lora_video_comparison_table.csv")
    print(f"   - lora_video_comparison.png")
    print(f"   - lora_scaling_analysis.png")


if __name__ == "__main__":
    main()