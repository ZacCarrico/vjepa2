#!/usr/bin/env python
"""
Plot LoRA vs Head-Only comparison across data scales.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
videos = [50, 100, 200]

# LoRA (Rank=64, Alpha=128)
lora_val = [83.33, 81.67, 77.5]
lora_test = [56.67, 71.67, 76.67]
lora_train = [68.57, 76.43, 77.68]

# Head-Only
head_val = [60.0, 70.0, 68.33]
head_test = [56.67, 68.33, 67.5]
head_train = [64.29, 66.79, 67.14]

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Test Accuracy Comparison
ax1 = axes[0]
ax1.plot(videos, lora_test, 'o-', linewidth=2.5, markersize=10,
         label='LoRA (Rank=64, Alpha=128)', color='#2E86AB')
ax1.plot(videos, head_test, 's-', linewidth=2.5, markersize=10,
         label='Head-Only', color='#A23B72')

ax1.set_xlabel('Videos per Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Test Accuracy vs Training Data Scale', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='lower right')
ax1.set_xticks(videos)
ax1.set_ylim([50, 80])

# Add value labels on points
for i, v in enumerate(videos):
    ax1.text(v, lora_test[i] + 1.5, f'{lora_test[i]:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2E86AB')
    ax1.text(v, head_test[i] - 2.5, f'{head_test[i]:.1f}%',
             ha='center', va='top', fontsize=10, fontweight='bold', color='#A23B72')

# Plot 2: Train/Val/Test for both approaches
ax2 = axes[1]
x = np.arange(len(videos))
width = 0.25

# LoRA bars
ax2.bar(x - width, lora_train, width, label='LoRA Train', color='#2E86AB', alpha=0.6)
ax2.bar(x, lora_val, width, label='LoRA Val', color='#2E86AB', alpha=0.8)
ax2.bar(x + width, lora_test, width, label='LoRA Test', color='#2E86AB', alpha=1.0)

# Head-Only bars (offset)
ax2.bar(x - width + 3*width, head_train, width, label='Head Train', color='#A23B72', alpha=0.6)
ax2.bar(x + 3*width, head_val, width, label='Head Val', color='#A23B72', alpha=0.8)
ax2.bar(x + width + 3*width, head_test, width, label='Head Test', color='#A23B72', alpha=1.0)

ax2.set_xlabel('Videos per Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Train/Val/Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x + 1.5*width)
ax2.set_xticklabels(videos)
ax2.legend(fontsize=9, ncol=2, loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.set_ylim([50, 90])

plt.tight_layout()
plt.savefig('lora_vs_head_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved to: lora_vs_head_comparison.png")

# Create second plot: Data Scaling Efficiency
fig2, ax = plt.subplots(figsize=(10, 7))

# Calculate improvement from baseline (50 videos)
lora_improvement = [(acc - lora_test[0]) for acc in lora_test]
head_improvement = [(acc - head_test[0]) for acc in head_test]

ax.plot(videos, lora_improvement, 'o-', linewidth=3, markersize=12,
        label='LoRA (Rank=64, Alpha=128)', color='#2E86AB')
ax.plot(videos, head_improvement, 's-', linewidth=3, markersize=12,
        label='Head-Only', color='#A23B72')

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Videos per Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy Improvement from 50 Videos (%)', fontsize=13, fontweight='bold')
ax.set_title('Data Scaling Efficiency: Test Accuracy Improvement',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper left')
ax.set_xticks(videos)
ax.set_ylim([-2, 22])

# Add value labels
for i, v in enumerate(videos):
    ax.text(v, lora_improvement[i] + 1.2, f'+{lora_improvement[i]:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2E86AB')
    ax.text(v, head_improvement[i] - 1.2, f'+{head_improvement[i]:.1f}%',
            ha='center', va='top', fontsize=11, fontweight='bold', color='#A23B72')

# Add annotation for winner
ax.annotate('LoRA scales excellently\n+20% improvement!',
            xy=(200, lora_improvement[2]), xytext=(170, 15),
            fontsize=11, fontweight='bold', color='#2E86AB',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2E86AB', alpha=0.9))

ax.annotate('Head-Only plateaus\n+10.8% improvement',
            xy=(200, head_improvement[2]), xytext=(130, 5),
            fontsize=11, fontweight='bold', color='#A23B72',
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#A23B72', alpha=0.9))

plt.tight_layout()
plt.savefig('data_scaling_efficiency.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved to: data_scaling_efficiency.png")

print("\n📊 Plots created successfully!")
print("   1. lora_vs_head_comparison.png - Comprehensive comparison")
print("   2. data_scaling_efficiency.png - Data scaling analysis")
