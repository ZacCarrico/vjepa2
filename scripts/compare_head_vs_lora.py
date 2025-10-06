#!/usr/bin/env python
# coding: utf-8

"""
Head-only vs LoRA Comparison
=============================

Creates a 2-panel comparison figure of Head-only vs LoRA approaches:
- Left: Test accuracy as a function of training data
- Right: Training time as a function of training data

Font: Poppins 14pt (fallback to Arial if unavailable)
"""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Try to use Poppins font, fallback to Arial
try:
    # Check if Poppins is available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Poppins' in available_fonts:
        plt.rcParams['font.family'] = 'Poppins'
        print("✅ Using Poppins font")
    else:
        plt.rcParams['font.family'] = 'Arial'
        print("⚠️  Poppins not available, using Arial")
except:
    plt.rcParams['font.family'] = 'Arial'
    print("⚠️  Poppins not available, using Arial")

# Set font size to 14
plt.rcParams['font.size'] = 14

METRICS_DIR = pathlib.Path("metrics/action_detection")


def load_metrics():
    """Load Head-only and LoRA metrics"""

    metrics = {
        "head_only": {},
        "lora": {},
    }

    # Head-only experiments
    metrics["head_only"][100] = json.load(open(METRICS_DIR / "head_only_metrics_100videos_16frames_251003-223428.json"))
    metrics["head_only"][200] = json.load(open(METRICS_DIR / "head_only_metrics_200videos_16frames_251004-071214.json"))
    metrics["head_only"][400] = json.load(open(METRICS_DIR / "head_only_metrics_400videos_16frames_251005-071045.json"))

    # LoRA experiments
    metrics["lora"][100] = json.load(open(METRICS_DIR / "lora_action_metrics_100videos_16frames_251003-221650.json"))
    metrics["lora"][200] = json.load(open(METRICS_DIR / "lora_action_metrics_200videos_16frames_251004-062932.json"))
    metrics["lora"][400] = json.load(open(METRICS_DIR / "lora_action_metrics_400videos_16frames_251004-140701.json"))

    print("✅ Loaded Head-only and LoRA metrics")
    return metrics


def create_comparison_figure(metrics):
    """Create 2-panel comparison figure"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Colors
    color_head = "#e74c3c"  # Red
    color_lora = "#2ecc71"  # Green

    # Get data
    video_counts = sorted(metrics["head_only"].keys())

    head_test_acc = [metrics["head_only"][v]["final_test_acc"] for v in video_counts]
    lora_test_acc = [metrics["lora"][v]["final_test_acc"] for v in video_counts]

    head_train_time = [metrics["head_only"][v]["total_training_time"]/60 for v in video_counts]
    lora_train_time = [metrics["lora"][v]["total_training_time"]/60 for v in video_counts]

    # Left panel: Test Accuracy
    ax1.plot(video_counts, head_test_acc, 'o-', label='Head-only (4.1K params)',
             color=color_head, linewidth=2.5, markersize=10)
    ax1.plot(video_counts, lora_test_acc, 's-', label='LoRA (496K params)',
             color=color_lora, linewidth=2.5, markersize=10)

    ax1.set_xlabel('Videos per Class', fontsize=14)
    ax1.set_ylabel('Test Accuracy', fontsize=14)
    ax1.set_title('Test Accuracy vs Training Data', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.75, 0.92])

    # Add value labels
    for i, (x, y) in enumerate(zip(video_counts, head_test_acc)):
        ax1.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom',
                fontsize=11, color=color_head)
    for i, (x, y) in enumerate(zip(video_counts, lora_test_acc)):
        ax1.text(x, y - 0.015, f'{y:.3f}', ha='center', va='top',
                fontsize=11, color=color_lora)

    # Right panel: Training Time
    ax2.plot(video_counts, head_train_time, 'o-', label='Head-only',
             color=color_head, linewidth=2.5, markersize=10)
    ax2.plot(video_counts, lora_train_time, 's-', label='LoRA',
             color=color_lora, linewidth=2.5, markersize=10)

    ax2.set_xlabel('Videos per Class', fontsize=14)
    ax2.set_ylabel('Training Time (minutes)', fontsize=14)
    ax2.set_title('Training Time vs Training Data', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for i, (x, y) in enumerate(zip(video_counts, head_train_time)):
        ax2.text(x, y + 2, f'{y:.1f}m', ha='center', va='bottom',
                fontsize=11, color=color_head)
    for i, (x, y) in enumerate(zip(video_counts, lora_train_time)):
        ax2.text(x, y - 3, f'{y:.1f}m', ha='center', va='top',
                fontsize=11, color=color_lora)

    plt.tight_layout()
    plt.savefig("head_vs_lora_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Saved: head_vs_lora_comparison.png")


def print_comparison_summary(metrics):
    """Print summary statistics"""

    print("\n" + "=" * 80)
    print("HEAD-ONLY vs LoRA COMPARISON SUMMARY")
    print("=" * 80)

    video_counts = sorted(metrics["head_only"].keys())

    for videos in video_counts:
        head = metrics["head_only"][videos]
        lora = metrics["lora"][videos]

        print(f"\n📊 {videos} videos/class:")
        print(f"  Head-only: {head['final_test_acc']:.4f} test acc, {head['total_training_time']/60:.1f} min")
        print(f"  LoRA:      {lora['final_test_acc']:.4f} test acc, {lora['total_training_time']/60:.1f} min")

        acc_diff = lora['final_test_acc'] - head['final_test_acc']
        time_diff = lora['total_training_time']/60 - head['total_training_time']/60

        print(f"  Δ Accuracy: {acc_diff:+.4f} ({acc_diff/head['final_test_acc']*100:+.1f}%)")
        print(f"  Δ Time:     {time_diff:+.1f} min ({time_diff/head['total_training_time']*60*100:+.1f}%)")

    print("\n" + "=" * 80)
    print("\n🏆 KEY INSIGHTS:")
    print(f"  • LoRA achieves {metrics['lora'][400]['final_test_acc']:.1%} accuracy vs Head-only's {metrics['head_only'][400]['final_test_acc']:.1%}")
    print(f"  • LoRA uses 121x more parameters (496K vs 4.1K)")
    print(f"  • LoRA scales better: +{(metrics['lora'][400]['final_test_acc']-metrics['lora'][100]['final_test_acc']):.1%} vs Head-only's {(metrics['head_only'][400]['final_test_acc']-metrics['head_only'][100]['final_test_acc']):.1%}")
    print(f"  • Training time similar: {metrics['lora'][400]['total_training_time']/60:.0f}min vs {metrics['head_only'][400]['total_training_time']/60:.0f}min")
    print("=" * 80)


def main():
    """Main function"""

    print("Creating Head-only vs LoRA comparison figure...")
    print(f"Font: {plt.rcParams['font.family'][0] if isinstance(plt.rcParams['font.family'], list) else plt.rcParams['font.family']}, Size: {plt.rcParams['font.size']}")

    metrics = load_metrics()
    create_comparison_figure(metrics)
    print_comparison_summary(metrics)


if __name__ == "__main__":
    main()
