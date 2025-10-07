#!/usr/bin/env python
"""
Plot F1-score comparison across different accumulation steps (effective batch sizes).
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiments (100 videos per class, 8 frames)
accumulation_steps = [1, 2, 4, 8, 16]

# Average F1-scores (calculated from per-class metrics)
avg_f1 = [
    (0.6897 + 0.6667 + 0.6667 + 0.5) / 4,      # acc_steps=1: 63.1%
    (0.7879 + 0.72 + 0.7333 + 0.5) / 4,        # acc_steps=2: 68.5%
    (0.85 + 0.64 + 0.7143 + 0.5926) / 4,       # acc_steps=4: 69.9%
    (0.7429 + 0.56 + 0.64 + 0.5714) / 4,       # acc_steps=8: 62.9%
    (0.7317 + 0.48 + 0.6923 + 0.6429) / 4      # acc_steps=16: 63.7%
]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot F1-Score vs Accumulation Steps
ax.plot(accumulation_steps, [f*100 for f in avg_f1], 'o-', linewidth=3, markersize=12,
        color='#2E86AB')

ax.set_xlabel('Accumulation Steps (Effective Batch Size)', fontsize=13, fontweight='bold')
ax.set_ylabel('Average F1-Score (%)', fontsize=13, fontweight='bold')
ax.set_title('LoRA Performance vs Batch Accumulation Strategy', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(accumulation_steps)
ax.set_ylim([60, 75])

# Add value labels
for i, acc in enumerate(accumulation_steps):
    ax.text(acc, avg_f1[i]*100 + 0.8, f'{avg_f1[i]*100:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2E86AB')

plt.tight_layout()
plt.savefig('accumulation_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved to: accumulation_comparison.png")

# Print summary statistics
print("\n📊 Accumulation Steps Comparison (100 videos/class, 8 frames):")
print("\nAverage F1-Score by Accumulation Steps:")
for i, acc in enumerate(accumulation_steps):
    print(f"  {acc:2d} steps (batch={acc}): {avg_f1[i]*100:.1f}%")

print(f"\nBest performance: accumulation_steps=4 with {avg_f1[2]*100:.1f}% F1-score")
print(f"Performance drop with larger batches:")
print(f"  4 → 8 steps:  {(avg_f1[3]-avg_f1[2])*100:.1f}% decrease")
print(f"  4 → 16 steps: {(avg_f1[4]-avg_f1[2])*100:.1f}% decrease")
print(f"\nKey insight: Moderate batch size (4) balances gradient stability and generalization")
