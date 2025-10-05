#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive Action Detection Training Approaches Comparison
=============================================================

This script compares all four training approaches across multiple dataset sizes:
1. Head-only: Train only the classification head (~4K params, 0.001%)
2. LoRA: Train LoRA adapters in pooler attention (~496K params, 0.13%)
3. Pooler+Head: Train full pooler + head (~49M params, 13.15%)
4. Pooler+Head+LoRA: Train pooler + head + LoRA (~50M params, 13.26%)

Dataset sizes: 50, 100, 200, 400 videos per class
"""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style for better plots
plt.style.use("default")

METRICS_DIR = pathlib.Path("metrics/action_detection")


def load_all_metrics():
    """Load all training metrics from action detection experiments"""

    metrics = {
        "head_only": {},
        "lora": {},
        "pooler_head": {},
        "pooler_head_lora": {},
    }

    # Head-only experiments
    metrics["head_only"][100] = json.load(open(METRICS_DIR / "head_only_metrics_100videos_16frames_251003-223428.json"))
    metrics["head_only"][200] = json.load(open(METRICS_DIR / "head_only_metrics_200videos_16frames_251004-071214.json"))
    metrics["head_only"][400] = json.load(open(METRICS_DIR / "head_only_metrics_400videos_16frames_251005-071045.json"))

    # LoRA experiments
    metrics["lora"][100] = json.load(open(METRICS_DIR / "lora_action_metrics_100videos_16frames_251003-221650.json"))
    metrics["lora"][200] = json.load(open(METRICS_DIR / "lora_action_metrics_200videos_16frames_251004-062932.json"))
    metrics["lora"][400] = json.load(open(METRICS_DIR / "lora_action_metrics_400videos_16frames_251004-140701.json"))

    # Pooler+Head experiments
    metrics["pooler_head"][100] = json.load(open(METRICS_DIR / "pooler_head_metrics_100videos_16frames_251003-225029.json"))
    metrics["pooler_head"][200] = json.load(open(METRICS_DIR / "pooler_head_metrics_200videos_16frames_251004-082514.json"))
    metrics["pooler_head"][400] = json.load(open(METRICS_DIR / "pooler_head_metrics_400videos_16frames_251004-210910.json"))

    # Pooler+Head+LoRA experiments
    metrics["pooler_head_lora"][50] = json.load(open(METRICS_DIR / "pooler_head_lora_metrics_50videos_16frames_251004-165639.json"))
    metrics["pooler_head_lora"][100] = json.load(open(METRICS_DIR / "pooler_head_lora_metrics_100videos_16frames_251004-170730.json"))
    metrics["pooler_head_lora"][200] = json.load(open(METRICS_DIR / "pooler_head_lora_metrics_200videos_16frames_251004-091526.json"))
    metrics["pooler_head_lora"][400] = json.load(open(METRICS_DIR / "pooler_head_lora_metrics_400videos_16frames_251004-174217.json"))

    print("✅ Loaded all training metrics")
    return metrics


def create_comparison_table(metrics):
    """Create comprehensive comparison table"""

    rows = []

    for approach in ["head_only", "lora", "pooler_head", "pooler_head_lora"]:
        for num_videos in sorted(metrics[approach].keys()):
            m = metrics[approach][num_videos]
            rows.append({
                "Approach": approach.replace("_", "+").title(),
                "Videos/Class": num_videos,
                "Test Acc": f"{m['final_test_acc']:.4f}",
                "Val Acc": f"{m['best_val_acc']:.4f}",
                "Trainable Params": f"{m['trainable_params']:,}",
                "% Trainable": f"{m['trainable_params']/m['total_params']*100:.2f}%",
                "Training Time (min)": f"{m['total_training_time']/60:.1f}",
            })

    df = pd.DataFrame(rows)
    return df


def plot_accuracy_vs_videos(metrics):
    """Plot test accuracy vs number of videos for all approaches"""

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "head_only": "#e74c3c",      # Red
        "lora": "#2ecc71",            # Green
        "pooler_head": "#3498db",     # Blue
        "pooler_head_lora": "#9b59b6" # Purple
    }

    labels = {
        "head_only": "Head-only (0.001% params)",
        "lora": "LoRA (0.13% params)",
        "pooler_head": "Pooler+Head (13.15% params)",
        "pooler_head_lora": "Pooler+Head+LoRA (13.26% params)"
    }

    for approach in ["head_only", "lora", "pooler_head", "pooler_head_lora"]:
        video_counts = sorted(metrics[approach].keys())
        test_accs = [metrics[approach][v]["final_test_acc"] for v in video_counts]

        ax.plot(video_counts, test_accs, 'o-', label=labels[approach],
                color=colors[approach], linewidth=2.5, markersize=10)

    ax.set_xlabel("Videos per Class", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Test Accuracy Scaling Across Training Approaches", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.75, 0.92])

    plt.tight_layout()
    plt.savefig("all_approaches_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_efficiency_analysis(metrics):
    """Create multi-panel efficiency analysis"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Training Efficiency Analysis: All Approaches", fontsize=16, fontweight='bold')

    colors = {
        "head_only": "#e74c3c",
        "lora": "#2ecc71",
        "pooler_head": "#3498db",
        "pooler_head_lora": "#9b59b6"
    }

    # 1. Training Time vs Dataset Size
    for approach in ["head_only", "lora", "pooler_head", "pooler_head_lora"]:
        video_counts = sorted(metrics[approach].keys())
        train_times = [metrics[approach][v]["total_training_time"]/60 for v in video_counts]
        ax1.plot(video_counts, train_times, 'o-', label=approach.replace("_", "+").title(),
                color=colors[approach], linewidth=2, markersize=8)

    ax1.set_xlabel("Videos per Class")
    ax1.set_ylabel("Training Time (minutes)")
    ax1.set_title("Training Time Scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Parameter Efficiency (400 videos only)
    approaches = ["head_only", "lora", "pooler_head", "pooler_head_lora"]
    approach_labels = [a.replace("_", "+").title() for a in approaches]
    test_accs_400 = [metrics[a][400]["final_test_acc"] if 400 in metrics[a] else 0
                     for a in approaches]
    trainable_params = [metrics[a][400]["trainable_params"] if 400 in metrics[a] else 0
                        for a in approaches]

    # Use log scale for parameters
    colors_list = [colors[a] for a in approaches]
    bars = ax2.bar(range(len(approaches)), test_accs_400, color=colors_list, alpha=0.7)
    ax2.set_xlabel("Approach")
    ax2.set_ylabel("Test Accuracy (400 videos/class)")
    ax2.set_title("Accuracy vs Parameter Efficiency")
    ax2.set_xticks(range(len(approaches)))
    ax2.set_xticklabels(approach_labels, rotation=15, ha='right')
    ax2.set_ylim([0.75, 0.92])

    # Add parameter count labels on bars
    for i, (bar, params) in enumerate(zip(bars, trainable_params)):
        height = bar.get_height()
        if params > 1e6:
            param_label = f"{params/1e6:.1f}M"
        elif params > 1e3:
            param_label = f"{params/1e3:.0f}K"
        else:
            param_label = f"{params}"
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}\n({param_label})', ha='center', va='bottom', fontsize=9)

    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Accuracy per Minute (efficiency metric)
    for approach in ["head_only", "lora", "pooler_head", "pooler_head_lora"]:
        video_counts = sorted(metrics[approach].keys())
        efficiency = [metrics[approach][v]["final_test_acc"] / (metrics[approach][v]["total_training_time"]/60)
                     for v in video_counts]
        ax3.plot(video_counts, efficiency, 'o-', label=approach.replace("_", "+").title(),
                color=colors[approach], linewidth=2, markersize=8)

    ax3.set_xlabel("Videos per Class")
    ax3.set_ylabel("Accuracy per Minute")
    ax3.set_title("Training Efficiency (Accuracy/Time)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Time per Video
    for approach in ["head_only", "lora", "pooler_head", "pooler_head_lora"]:
        video_counts = sorted(metrics[approach].keys())
        time_per_video = [metrics[approach][v]["total_training_time"] / metrics[approach][v]["num_train_videos"]
                          for v in video_counts]
        ax4.plot(video_counts, time_per_video, 'o-', label=approach.replace("_", "+").title(),
                color=colors[approach], linewidth=2, markersize=8)

    ax4.set_xlabel("Videos per Class")
    ax4.set_ylabel("Training Time per Video (seconds)")
    ax4.set_title("Per-Video Training Cost")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("all_approaches_efficiency_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_report(metrics):
    """Generate comprehensive summary report"""

    print("\n" + "=" * 100)
    print("COMPREHENSIVE ACTION DETECTION TRAINING APPROACHES COMPARISON")
    print("=" * 100)

    print("\n📊 BEST RESULTS BY APPROACH (400 videos/class):")
    print("-" * 100)

    for approach in ["head_only", "lora", "pooler_head", "pooler_head_lora"]:
        if 400 in metrics[approach]:
            m = metrics[approach][400]
            print(f"\n{approach.replace('_', '+').upper()}:")
            print(f"  Test Accuracy: {m['final_test_acc']:.4f} ({m['final_test_acc']*100:.2f}%)")
            print(f"  Validation Accuracy: {m['best_val_acc']:.4f}")
            print(f"  Trainable Parameters: {m['trainable_params']:,} ({m['trainable_params']/m['total_params']*100:.2f}%)")
            print(f"  Training Time: {m['total_training_time']/60:.1f} minutes")
            print(f"  Efficiency: {m['final_test_acc']/(m['total_training_time']/60):.4f} accuracy/minute")

    print("\n\n🏆 PERFORMANCE RANKING (400 videos/class - by test accuracy):")
    print("-" * 100)

    results_400 = []
    for approach in ["head_only", "lora", "pooler_head", "pooler_head_lora"]:
        if 400 in metrics[approach]:
            m = metrics[approach][400]
            results_400.append((
                approach.replace('_', '+').upper(),
                m['final_test_acc'],
                m['trainable_params'],
                m['total_training_time']/60
            ))

    results_400.sort(key=lambda x: x[1], reverse=True)

    for rank, (approach, acc, params, time) in enumerate(results_400, 1):
        params_pct = (params / 375803268) * 100
        print(f"{rank}. {approach:25} - {acc:.4f} ({params:>11,} params = {params_pct:>5.2f}% | {time:>5.1f}min)")

    print("\n\n⚖️  EFFICIENCY RANKING (400 videos/class - accuracy per minute):")
    print("-" * 100)

    efficiency_400 = [(name, acc, time, acc/time) for name, acc, params, time in results_400]
    efficiency_400.sort(key=lambda x: x[3], reverse=True)

    for rank, (approach, acc, time, eff) in enumerate(efficiency_400, 1):
        print(f"{rank}. {approach:25} - {eff:.4f} acc/min ({acc:.4f} in {time:.1f}min)")

    print("\n\n📈 SCALING TRENDS:")
    print("-" * 100)

    for approach in ["head_only", "lora", "pooler_head", "pooler_head_lora"]:
        video_counts = sorted(metrics[approach].keys())
        if len(video_counts) >= 2:
            first = video_counts[0]
            last = video_counts[-1]
            acc_first = metrics[approach][first]["final_test_acc"]
            acc_last = metrics[approach][last]["final_test_acc"]
            improvement = acc_last - acc_first
            improvement_pct = (improvement / acc_first) * 100

            print(f"\n{approach.replace('_', '+').upper()}:")
            print(f"  {first} videos: {acc_first:.4f}")
            print(f"  {last} videos: {acc_last:.4f}")
            print(f"  Improvement: +{improvement:.4f} ({improvement_pct:+.1f}%)")
            print(f"  Scales well: {'✅ Yes' if improvement > 0.05 else '⚠️  Moderate' if improvement > 0.02 else '❌ Poor'}")

    print("\n\n🎯 RECOMMENDATIONS:")
    print("-" * 100)

    # Find best approach by accuracy
    best_acc = max(results_400, key=lambda x: x[1])
    # Find most efficient
    best_eff = max(efficiency_400, key=lambda x: x[3])
    # Find best parameter efficiency (LoRA)
    lora_result = [(n, a, p, t) for n, a, p, t in results_400 if "LORA" in n and "HEAD" not in n]

    print(f"\n✅ **Best Overall Performance**: {best_acc[0]}")
    print(f"   - Test Accuracy: {best_acc[1]:.4f} ({best_acc[1]*100:.1f}%)")
    print(f"   - Use when maximum accuracy is required")

    print(f"\n⚡ **Most Time-Efficient**: {best_eff[0]}")
    print(f"   - Efficiency: {best_eff[3]:.4f} accuracy/minute")
    print(f"   - Use when training time is limited")

    if lora_result:
        lora_acc, lora_params = lora_result[0][1], lora_result[0][2]
        print(f"\n💡 **Best Parameter Efficiency**: LORA")
        print(f"   - Test Accuracy: {lora_acc:.4f} ({lora_acc*100:.1f}%)")
        print(f"   - Trainable Parameters: {lora_params:,} (only 0.13% of model)")
        print(f"   - Use when memory/storage is constrained or need to train many models")

    print("\n" + "=" * 100)


def main():
    """Main analysis function"""

    print("Loading all training metrics...")
    metrics = load_all_metrics()

    print("\nCreating comparison table...")
    comparison_df = create_comparison_table(metrics)

    print("\n" + "=" * 100)
    print("DETAILED COMPARISON TABLE")
    print("=" * 100)
    print(comparison_df.to_string(index=False))

    print("\n\nGenerating accuracy comparison plot...")
    plot_accuracy_vs_videos(metrics)

    print("Generating efficiency analysis plots...")
    plot_efficiency_analysis(metrics)

    print("Generating comprehensive summary report...")
    generate_summary_report(metrics)

    # Save comparison table
    comparison_df.to_csv("all_approaches_comparison_table.csv", index=False)

    print(f"\n\n✅ Analysis complete! Generated files:")
    print(f"   - all_approaches_comparison_table.csv")
    print(f"   - all_approaches_accuracy_comparison.png")
    print(f"   - all_approaches_efficiency_analysis.png")


if __name__ == "__main__":
    main()
