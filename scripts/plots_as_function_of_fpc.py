import matplotlib.pyplot as plt
import numpy as np

# Set font size
plt.rcParams.update({'font.size': 14})

# Data from the training metrics files (including 1 frame)
frames = [1, 8, 16, 32, 64]
test_accuracy = [0.547, 0.573, 0.547, 0.573, 0.573]
training_time = [84, 198, 490, 1585, 8043]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot for performance
bars1 = ax1.bar(frames, test_accuracy, color='skyblue', alpha=0.8)
ax1.set_xlabel('Frames per Clip')
ax1.set_ylabel('Final Test Accuracy')
ax1.set_title('Model Performance vs Frame Count')
ax1.set_ylim(0.5, 0.6)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars1, test_accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{acc:.3f}', ha='center', va='bottom')

# Bar plot for training time
bars2 = ax2.bar(frames, training_time, color='lightcoral', alpha=0.8)
ax2.set_xlabel('Frames per Clip')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time vs Frame Count')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, time in zip(bars2, training_time):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{time:,}s', ha='center', va='bottom')

# Adjust layout and save
plt.tight_layout()
plt.savefig('lora_training_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as lora_training_comparison.png")