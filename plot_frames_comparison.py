#!/usr/bin/env python
"""
Plot F1-score and training time comparison across different frame counts.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiments (100 videos per class, accumulation_steps=4)
frames = [8, 16, 32]

# F1-scores by class
f1_sitting_down = [0.85, 0.944, 1.0]
f1_standing_up = [0.64, 0.933, 0.968]
f1_waving = [0.714, 0.889, 0.8]
f1_other = [0.593, 0.815, 0.8]

# Average F1-scores
avg_f1 = [
    (0.85 + 0.64 + 0.714 + 0.593) / 4,  # 8 frames: 69.9%
    (0.944 + 0.933 + 0.889 + 0.815) / 4,  # 16 frames: 89.5%
    (1.0 + 0.968 + 0.8 + 0.8) / 4  # 32 frames: 89.2%
]

# Training times in minutes
training_time = [55.22, 84.89, 131.72]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Average F1-Score vs Frames
ax1 = axes[0]
ax1.plot(frames, [f*100 for f in avg_f1], 'o-', linewidth=3, markersize=12,
        color='#2E86AB')

ax1.set_xlabel('Frames per Clip', fontsize=13, fontweight='bold')
ax1.set_ylabel('Average F1-Score (%)', fontsize=13, fontweight='bold')
ax1.set_title('LoRA Performance vs Temporal Resolution',
            fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(frames)
ax1.set_ylim([65, 95])

# Add value labels
for i, f in enumerate(frames):
    ax1.text(f, avg_f1[i]*100 + 1.5, f'{avg_f1[i]*100:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2E86AB')

# Plot 2: Training Time vs Frames
ax2 = axes[1]
ax2.plot(frames, training_time, 'o-', linewidth=3, markersize=12,
        color='#2E86AB')

ax2.set_xlabel('Frames per Clip', fontsize=13, fontweight='bold')
ax2.set_ylabel('Training Time (minutes)', fontsize=13, fontweight='bold')
ax2.set_title('Training Cost vs Temporal Resolution',
            fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(frames)
ax2.set_ylim([40, 145])

# Add value labels
for i, f in enumerate(frames):
    ax2.text(f, training_time[i] + 3, f'{training_time[i]:.1f} min',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2E86AB')

plt.tight_layout()
plt.savefig('frames_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved to: frames_comparison.png")

# Print summary statistics
print("\n📊 Frames per Clip Comparison (100 videos/class):")
print("\nAverage F1-Score:")
for i, f in enumerate(frames):
    print(f"  {f:2d} frames: {avg_f1[i]*100:.1f}% ({training_time[i]:.1f} min)")

print(f"\nPerformance improvement:")
print(f"  8 → 16 frames: +{(avg_f1[1]-avg_f1[0])*100:.1f}% F1-score (+{training_time[1]-training_time[0]:.1f} min)")
print(f"  16 → 32 frames: {(avg_f1[2]-avg_f1[1])*100:.1f}% F1-score (+{training_time[2]-training_time[1]:.1f} min)")

print(f"\nKey insight: 16 frames provides optimal balance between:")
print(f"  - Performance: 89.5% F1-score (+19.6% over 8 frames)")
print(f"  - Efficiency: 85 min training (vs 132 min for 32 frames)")
print(f"  - Diminishing returns beyond 16 frames (-0.3% F1 for +55% training time)")
