#!/usr/bin/env python
# coding: utf-8

# # V-JEPA 2 Adapter-Based Fine-tuning
#
# This notebook demonstrates adapter-based fine-tuning of V-JEPA 2 using LoRA (Low-Rank Adaptation).
# Adapter tuning allows us to fine-tune large models with significantly fewer trainable parameters,
# making it more memory efficient and faster to train while achieving comparable performance.

import argparse
import json
import math
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor

from common.data import (
    CustomVideoDataset,
    create_data_loaders,
    create_label_mappings,
    setup_ucf101_dataset,
)
from common.training import evaluate, setup_tensorboard
from common.utils import count_parameters, get_device, print_parameter_stats, set_seed

print("Torch:", torch.__version__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='V-JEPA 2 LoRA Fine-tuning')
    parser.add_argument(
        '--num_train_videos',
        type=int,
        default=None,
        help='Number of training videos to use (default: use all available)'
    )
    return parser.parse_args()


# Parse command line arguments
args = parse_arguments()
num_train_videos = args.num_train_videos

# Create timestamp for output files
timestamp = datetime.now().strftime("%y%m%d-%H:%M:%S")
print(f"Training session timestamp: {timestamp}")

if num_train_videos:
    print(f"Using {num_train_videos} training videos")
else:
    print("Using all available training videos")

# Set seed for reproducibility
set_seed(1)

# Device Setup
device = get_device()
print(f"Using device: {device}")

# ## LoRA (Low-Rank Adaptation) Implementation


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that can be applied to any linear layer.
    This implementation follows the original LoRA paper: https://arxiv.org/abs/2106.09685
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices: W = W_0 + (B * A) * scaling
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation: x @ (A^T @ B^T) * scaling"""
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class AdaptedLinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    During training, both original weights (frozen) and LoRA weights are used.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Move LoRA to same device as original linear layer
        self.lora = self.lora.to(original_linear.weight.device)

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original linear layer with LoRA adaptation"""
        return self.original_linear(x) + self.lora(x)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> int:
    """
    Apply LoRA adapters to specified modules in the model.

    Args:
        model: The model to adapt
        target_modules: List of module names to adapt (e.g., ['query', 'value'])
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: Dropout rate for LoRA layers

    Returns:
        Number of adapted modules
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
        ]  # Common attention module names

    adapted_count = 0

    def replace_linear_with_lora(module, name):
        nonlocal adapted_count
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and any(
                target in child_name for target in target_modules
            ):
                # Replace with LoRA adapted version
                adapted_linear = AdaptedLinear(
                    child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(module, child_name, adapted_linear)
                adapted_count += 1
                print(f"Applied LoRA to: {name}.{child_name}")
            else:
                replace_linear_with_lora(child_module, f"{name}.{child_name}")

    replace_linear_with_lora(model, "model")
    return adapted_count




# ## Data Loading (Same as original fine-tuning)

# Load UCF-101 dataset
train_video_file_paths, val_video_file_paths, test_video_file_paths, dataset_root_path = setup_ucf101_dataset()
all_video_file_paths = list(dataset_root_path.glob("**/*.avi"))

video_count_train = len(train_video_file_paths)
video_count_val = len(val_video_file_paths)
video_count_test = len(test_video_file_paths)
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

# Create label mappings
label2id, id2label = create_label_mappings(all_video_file_paths)

print(f"Number of classes: {len(label2id)}")

# ## Dataset and DataLoader Setup




# Limit training videos if specified
if num_train_videos and num_train_videos < len(train_video_file_paths):
    train_video_file_paths = train_video_file_paths[:num_train_videos]
    print(f"Limited training videos to {len(train_video_file_paths)}")

# Create datasets
train_ds = CustomVideoDataset(train_video_file_paths, label2id)
val_ds = CustomVideoDataset(val_video_file_paths, label2id)
test_ds = CustomVideoDataset(test_video_file_paths, label2id)

# Update video counts after potential limiting
video_count_train = len(train_video_file_paths)

# ## Model Setup with LoRA Adapters

# Load model and processor
model_name = "facebook/vjepa2-vitl-fpc16-256-ssv2"
processor = VJEPA2VideoProcessor.from_pretrained(model_name)
model = VJEPA2ForVideoClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
).to(device)

print("Original model parameter stats:")
print_parameter_stats(model, "Original V-JEPA 2")

# Apply LoRA adapters
print("\nApplying LoRA adapters...")

# LoRA configuration
lora_config = {
    "rank": 16,
    "alpha": 32.0,
    "dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],  # Attention modules
}

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# Apply LoRA to attention modules
adapted_count = apply_lora_to_model(
    model,
    target_modules=lora_config["target_modules"],
    rank=lora_config["rank"],
    alpha=lora_config["alpha"],
    dropout=lora_config["dropout"],
)

# Unfreeze the classification head
for param in model.classifier.parameters():
    param.requires_grad = True

print(f"\nApplied LoRA to {adapted_count} modules")
print_parameter_stats(model, "LoRA Adapted V-JEPA 2")

# Create DataLoaders using common module
batch_size = 1
num_workers = 0

train_loader, val_loader, test_loader = create_data_loaders(
    train_ds, val_ds, test_ds, processor, batch_size, num_workers, model.config.frames_per_clip
)

# ## Training Setup

# Create output directory name with timestamp and video count
output_suffix = f"{video_count_train}videos_{timestamp}"
tensorboard_dir = f"runs/vjepa2_lora_finetune_{output_suffix}"
writer = setup_tensorboard(tensorboard_dir)




# Optimizer - only train LoRA parameters and classification head
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)

print(f"\nTraining {len(trainable_params)} parameter groups")
print(f"Learning rate: 2e-4 (higher than full fine-tuning due to fewer parameters)")

# ## Training Loop

num_epochs = 5  # Same as original fine-tuning
accumulation_steps = 4

print(f"\nStarting LoRA fine-tuning for {num_epochs} epochs...")
print(f"Gradient accumulation steps: {accumulation_steps}")

# Store metrics for comparison
training_metrics = {
    "timestamp": timestamp,
    "num_train_videos": video_count_train,
    "num_val_videos": video_count_val,
    "num_test_videos": video_count_test,
    "epochs": [],
    "train_loss": [],
    "val_acc": [],
    "trainable_params": count_parameters(model)[1],
    "total_params": count_parameters(model)[0],
}

# Best model tracking
best_val_acc = 0.0
best_model_state = None

# Start timing
total_start_time = time.time()
print(f"Starting LoRA training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    epoch_losses = []

    for step, (vids, labels) in enumerate(train_loader, start=1):
        inputs = processor(vids, return_tensors="pt").to(model.device)
        labels = labels.to(model.device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        running_loss += loss.item()

        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch} Step {step}: Accumulated Loss = {running_loss:.4f}")
            writer.add_scalar(
                "Train Loss", running_loss, epoch * len(train_loader) + step
            )
            epoch_losses.append(running_loss)
            running_loss = 0.0

    # End of epoch evaluation
    val_acc = evaluate(val_loader, model, processor, model.device)
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch} Validation Accuracy: {val_acc:.4f}")
    print(f"Epoch {epoch} Duration: {epoch_duration:.2f} seconds")
    writer.add_scalar("Val Acc", val_acc, epoch * len(train_loader))

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"New best validation accuracy: {best_val_acc:.4f} - saving model state")

    # Store metrics
    training_metrics["epochs"].append(epoch)
    training_metrics["train_loss"].append(
        sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    )
    training_metrics["val_acc"].append(val_acc)

# Load best model for final test evaluation
if best_model_state is not None:
    print(f"\nLoading best model (validation accuracy: {best_val_acc:.4f}) for final test evaluation")
    model.load_state_dict(best_model_state)
else:
    print(f"\nNo best model saved, using final epoch model for test evaluation")

# Final test evaluation
test_acc = evaluate(test_loader, model, processor, model.device)
total_end_time = time.time()
total_duration = total_end_time - total_start_time

print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(
    f"Total LoRA Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
)
print(f"LoRA training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

writer.add_scalar("Final Test Acc", test_acc, num_epochs * len(train_loader))

# Save metrics
training_metrics["final_test_acc"] = test_acc
training_metrics["best_val_acc"] = best_val_acc
training_metrics["total_training_time"] = total_duration

# Save metrics with timestamp and video count in filename
metrics_filename = f"lora_training_metrics_{output_suffix}.json"
with open(metrics_filename, "w") as f:
    json.dump(training_metrics, f, indent=2)

print(f"Training metrics saved to: {metrics_filename}")
print(f"Tensorboard logs saved to: {tensorboard_dir}")

writer.close()

print(f"\nLoRA fine-tuning completed!")
print_parameter_stats(model, "Final LoRA Adapted Model")

# ## Save Model (Optional)

# Uncomment to save the model with timestamp and video count
model_save_dir = f"./vjepa2-lora-ucf101_{output_suffix}"
# model.save_pretrained(model_save_dir)
# processor.save_pretrained(model_save_dir)

print("\nAdapter-based fine-tuning demonstration completed!")
print(f"Training videos used: {video_count_train:,}")
print(f"Training session: {timestamp}")
print(
    f"Memory efficiency: Used only {count_parameters(model)[1]:,} trainable parameters"
)
print(f"vs {55_000_000:,}+ parameters in full fine-tuning (estimated)")
