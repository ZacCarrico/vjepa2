#!/usr/bin/env python
"""
Plot F1-score and training time comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
videos = [50, 100, 200]

# Average F1-scores (calculated from per-class metrics)
# LoRA: averaging sitting_down, standing_up, waving, other F1 scores
lora_f1 = [
    (0.5714 + 0.5455 + 0.75 + 0.1818) / 4,  # 50v: 0.5122
    (0.85 + 0.64 + 0.7143 + 0.5926) / 4,    # 100v: 0.6992
    (0.8451 + 0.8696 + 0.7119 + 0.5366) / 4  # 200v: 0.7408
]

# Head-Only: averaging sitting_down, standing_up, waving, other F1 scores
head_f1 = [
    (0.7692 + 0.5455 + 0.6087 + 0.3077) / 4,  # 50v: 0.5578
    (0.7143 + 0.6154 + 0.72 + 0.6667) / 4,    # 100v: 0.6791
    (0.7838 + 0.6774 + 0.6415 + 0.5490) / 4   # 200v: 0.6629
]

# Training times in minutes
lora_time = [6.44, 11.41, 36.70]
head_time = [8.91, 17.87, 26.89]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Average F1-Score vs Training Data
ax1 = axes[0]
ax1.plot(videos, [f*100 for f in lora_f1], 'o-', linewidth=2.5, markersize=10,
         label='LoRA (Rank=64, Alpha=128)', color='#2E86AB')
ax1.plot(videos, [f*100 for f in head_f1], 's-', linewidth=2.5, markersize=10,
         label='Head-Only', color='#A23B72')

ax1.set_xlabel('Videos per Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average F1-Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Average F1-Score vs Training Data', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='lower right')
ax1.set_xticks(videos)
ax1.set_ylim([48, 78])

# Add value labels on points
for i, v in enumerate(videos):
    ax1.text(v, lora_f1[i]*100 + 1.5, f'{lora_f1[i]*100:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2E86AB')
    ax1.text(v, head_f1[i]*100 - 1.5, f'{head_f1[i]*100:.1f}%',
             ha='center', va='top', fontsize=10, fontweight='bold', color='#A23B72')

# Plot 2: Training Time vs Training Data
ax2 = axes[1]
ax2.plot(videos, lora_time, 'o-', linewidth=2.5, markersize=10,
         label='LoRA (Rank=64, Alpha=128)', color='#2E86AB')
ax2.plot(videos, head_time, 's-', linewidth=2.5, markersize=10,
         label='Head-Only', color='#A23B72')

ax2.set_xlabel('Videos per Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
ax2.set_title('Training Time vs Training Data', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='upper left')
ax2.set_xticks(videos)
ax2.set_ylim([0, 40])

# Add value labels on points
for i, v in enumerate(videos):
    ax2.text(v, lora_time[i] + 1.5, f'{lora_time[i]:.1f} min',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2E86AB')
    ax2.text(v, head_time[i] - 1.5, f'{head_time[i]:.1f} min',
             ha='center', va='top', fontsize=10, fontweight='bold', color='#A23B72')

# Add efficiency annotation on right plot
ax2.annotate('LoRA: Higher cost\nbut better performance',
            xy=(200, lora_time[2]), xytext=(130, 30),
            fontsize=10, fontweight='bold', color='#2E86AB',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2E86AB', alpha=0.9))

plt.tight_layout()
plt.savefig('f1_and_time_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved to: f1_and_time_comparison.png")

# Print summary statistics
print("\n📊 Summary Statistics:")
print("\nAverage F1-Score:")
print(f"  LoRA:      50v={lora_f1[0]*100:.1f}%  100v={lora_f1[1]*100:.1f}%  200v={lora_f1[2]*100:.1f}%")
print(f"  Head-Only: 50v={head_f1[0]*100:.1f}%  100v={head_f1[1]*100:.1f}%  200v={head_f1[2]*100:.1f}%")
print(f"  Winner at 200v: LoRA (+{(lora_f1[2]-head_f1[2])*100:.1f}%)")

print("\nTraining Time:")
print(f"  LoRA:      50v={lora_time[0]:.1f}min  100v={lora_time[1]:.1f}min  200v={lora_time[2]:.1f}min")
print(f"  Head-Only: 50v={head_time[0]:.1f}min  100v={head_time[1]:.1f}min  200v={head_time[2]:.1f}min")
print(f"  LoRA overhead at 200v: +{lora_time[2]-head_time[2]:.1f} min ({((lora_time[2]/head_time[2])-1)*100:.0f}% slower)")

print("\nPerformance per Minute (F1-score / training time):")
lora_efficiency = [(f*100)/t for f, t in zip(lora_f1, lora_time)]
head_efficiency = [(f*100)/t for f, t in zip(head_f1, head_time)]
print(f"  LoRA:      50v={lora_efficiency[0]:.2f}  100v={lora_efficiency[1]:.2f}  200v={lora_efficiency[2]:.2f}")
print(f"  Head-Only: 50v={head_efficiency[0]:.2f}  100v={head_efficiency[1]:.2f}  200v={head_efficiency[2]:.2f}")
