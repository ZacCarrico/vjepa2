#!/usr/bin/env python
# coding: utf-8

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Results data
results = {
    75: 0.6933,    # From 75-video training
    150: 0.92,     # From 150-video training
    300: 1.0000    # From 300-video training (original LoRA output)
}

# Create bar plot
training_samples = list(results.keys())
test_accuracies = list(results.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(training_samples, test_accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
               alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels on bars
for i, (samples, accuracy) in enumerate(zip(training_samples, test_accuracies)):
    plt.text(samples, accuracy + 0.02, f'{accuracy:.1%}',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

# Customize plot
plt.xlabel('Number of Training Videos', fontsize=14, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
plt.title('LoRA Fine-tuning Performance vs Training Data Size\n(V-JEPA 2 on UCF-101)',
          fontsize=16, fontweight='bold', pad=20)

# Set y-axis limits and format
plt.ylim(0, 1.1)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Customize x-axis
plt.xticks(training_samples, fontsize=12)
plt.gca().set_axisbelow(True)

# Add some styling
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Remove annotation about the experiment

# Save the plot
plt.savefig('lora_training_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('lora_training_comparison.pdf', bbox_inches='tight')
# Remove plt.show() to avoid interactive display

print("Comparison plot created and saved!")
print("\nResults Summary:")
for samples, accuracy in results.items():
    print(f"  {samples:3d} training videos → {accuracy:.1%} test accuracy")