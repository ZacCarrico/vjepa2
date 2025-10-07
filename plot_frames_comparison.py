#!/usr/bin/env python
"""
Plot F1-score comparison across different frame counts.
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

# Test accuracies
test_acc = [71.67, 90.00, 90.00]

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot average F1-score
ax.plot(frames, [f*100 for f in avg_f1], 'o-', linewidth=3, markersize=12,
        label='Average F1-Score', color='#2E86AB')

# Plot test accuracy for comparison
ax.plot(frames, test_acc, 's--', linewidth=2.5, markersize=10,
        label='Test Accuracy', color='#A23B72', alpha=0.7)

ax.set_xlabel('Frames per Clip', fontsize=13, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_title('LoRA Fine-tuning: Impact of Temporal Resolution',
            fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='lower right')
ax.set_xticks(frames)
ax.set_ylim([65, 95])

# Add value labels
for i, f in enumerate(frames):
    ax.text(f, avg_f1[i]*100 + 1.5, f'{avg_f1[i]*100:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2E86AB')
    ax.text(f, test_acc[i] - 1.5, f'{test_acc[i]:.1f}%',
            ha='center', va='top', fontsize=10, fontweight='bold', color='#A23B72')

# Add annotation
ax.annotate('Optimal performance\nat 16 frames',
            xy=(16, avg_f1[1]*100), xytext=(20, 80),
            fontsize=11, fontweight='bold', color='#2E86AB',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2E86AB', alpha=0.9))

ax.annotate('Diminishing returns\nafter 16 frames',
            xy=(32, avg_f1[2]*100), xytext=(26, 75),
            fontsize=10, fontweight='bold', color='#666666',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#666666', alpha=0.8))

plt.tight_layout()
plt.savefig('frames_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved to: frames_comparison.png")

# Print summary statistics
print("\n📊 Frames per Clip Comparison (100 videos/class):")
print("\nAverage F1-Score:")
for i, f in enumerate(frames):
    print(f"  {f:2d} frames: {avg_f1[i]*100:.1f}% (Test Acc: {test_acc[i]:.1f}%)")

print(f"\nPerformance improvement:")
print(f"  8 → 16 frames: +{(avg_f1[1]-avg_f1[0])*100:.1f}% F1-score")
print(f"  16 → 32 frames: {(avg_f1[2]-avg_f1[1])*100:.1f}% F1-score (slight decrease)")

print(f"\nKey insight: 16 frames provides optimal balance between:")
print(f"  - Temporal context for action recognition")
print(f"  - Computational efficiency (85 min vs 132 min for 32 frames)")
print(f"  - Model performance (89.5% F1-score)")
