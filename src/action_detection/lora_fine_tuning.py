#!/usr/bin/env python
# coding: utf-8

"""
V-JEPA 2 LoRA Fine-tuning for Action Detection

This script demonstrates LoRA (Low-Rank Adaptation) fine-tuning for action detection
on the NTU RGB dataset. LoRA allows efficient fine-tuning with fewer trainable parameters.
"""

import argparse
import json
import math
import pathlib
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
from torchvision.transforms import v2
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor

from src.action_detection.config import DEFAULT_CONFIG
from src.action_detection.data import (
    ActionDetectionDataset,
    setup_ntu_action_dataset,
    create_data_loaders,
)
from src.common.experiment_tracker import ExperimentTracker
from src.common.training import evaluate, evaluate_detailed, setup_tensorboard
from src.common.utils import get_device, set_seed, print_parameter_stats, count_parameters


# ============================================================================
# LoRA Implementation
# ============================================================================

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


# All dataset-related code moved to src/action_detection/data.py


# ============================================================================
# Configuration
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="V-JEPA 2 LoRA Fine-tuning for Action Detection"
    )

    # Dataset configuration
    parser.add_argument(
        "--num_videos",
        type=int,
        default=100,
        help="Number of videos per class to use (default: 100)",
    )

    # Model configuration
    parser.add_argument(
        "--frames_per_clip",
        type=int,
        default=None,
        help="Number of frames per video clip (default: use model default)",
    )

    # LoRA configuration
    parser.add_argument(
        "--lora_rank", type=int, default=None, help="LoRA rank (default: use config)"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=None, help="LoRA alpha (default: use config)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=None,
        help="LoRA dropout rate (default: use config)",
    )

    # Training configuration
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Learning rate (default: use config)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay (default: use config)",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )

    return parser.parse_args()


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    print("Starting V-JEPA 2 LoRA Action Detection Fine-tuning")
    print("=" * 50)

    # Parse arguments
    args = parse_arguments()

    # Load shared config (override argparse defaults)
    config = DEFAULT_CONFIG

    # Override config with command-line arguments if provided
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.lora_rank is not None:
        config.lora_rank = args.lora_rank
    if args.lora_alpha is not None:
        config.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        config.lora_dropout = args.lora_dropout
    if args.frames_per_clip is not None:
        config.frames_per_clip = args.frames_per_clip
    if args.accumulation_steps is not None:
        config.accumulation_steps = args.accumulation_steps

    # Create timestamp for output files
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    print(f"Training session timestamp: {timestamp}")

    # Print configuration
    print("\nConfiguration:")
    print(f"  Videos per class: {args.num_videos}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  LoRA dropout: {config.lora_dropout}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Accumulation steps: {config.accumulation_steps}")

    # Set seed for reproducibility
    set_seed(config.seed)

    # Auto-detect best available device
    device = get_device()
    print(f"Using device: {device}")

    # Setup dataset
    train_videos, val_videos, test_videos, label2id, id2label = setup_ntu_action_dataset(
        num_videos_per_class=args.num_videos
    )

    # Create datasets
    train_ds = ActionDetectionDataset(train_videos, label2id)
    val_ds = ActionDetectionDataset(val_videos, label2id)
    test_ds = ActionDetectionDataset(test_videos, label2id)

    # Setup model and processor
    model_name = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)

    # Check if model already exists locally
    local_model_path = models_dir / "vjepa2-vitl-fpc16-256-ssv2"

    if local_model_path.exists():
        print(f"Loading model from local cache: {local_model_path}")
        processor = VJEPA2VideoProcessor.from_pretrained(local_model_path)
        model = VJEPA2ForVideoClassification.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        ).to(device)
    else:
        print(f"Downloading and caching model to: {local_model_path}")
        processor = VJEPA2VideoProcessor.from_pretrained(model_name)
        model = VJEPA2ForVideoClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        ).to(device)

        # Save model and processor locally
        processor.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)

    # Set frames_per_clip from config (allow command line override)
    frames_per_clip = args.frames_per_clip if args.frames_per_clip else config.frames_per_clip
    if frames_per_clip != model.config.frames_per_clip:
        original_frames = model.config.frames_per_clip
        model.config.frames_per_clip = frames_per_clip
        print(
            f"Overriding frames per clip: {original_frames} → {frames_per_clip}"
        )

    print("\nOriginal model parameter stats:")
    print_parameter_stats(model, "Original V-JEPA 2")

    # Apply LoRA adapters
    print("\nApplying LoRA adapters...")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to attention modules
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    adapted_count = apply_lora_to_model(
        model,
        target_modules=lora_target_modules,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )

    # Unfreeze the classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    print(f"\nApplied LoRA to {adapted_count} modules")
    print_parameter_stats(model, "LoRA Adapted V-JEPA 2")

    # Create DataLoaders using config frames_per_clip
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds,
        val_ds,
        test_ds,
        processor,
        config.batch_size,
        num_workers=config.num_workers,
        frames_per_clip=frames_per_clip,
    )

    # Setup training
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Setup tensorboard
    output_suffix = f"{args.num_videos}videos_{frames_per_clip}frames_{timestamp}"
    tensorboard_dir = f"runs/ntu_action_lora_{output_suffix}"
    writer = setup_tensorboard(tensorboard_dir)

    # Store metrics for comparison
    training_metrics = {
        "timestamp": timestamp,
        "approach": "lora",
        "num_videos_per_class": args.num_videos,
        "num_train_videos": len(train_videos),
        "num_val_videos": len(val_videos),
        "num_test_videos": len(test_videos),
        "frames_per_clip": frames_per_clip,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "accumulation_steps": config.accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "model_name": config.model_name,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "epochs": [],
        "train_loss": [],
        "val_acc": [],
        "trainable_params": count_parameters(model)[1],
        "total_params": count_parameters(model)[0],
    }

    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    # Start timing
    total_start_time = time.time()
    print(f"\nStarting LoRA training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"Training for {config.num_epochs} epochs with {len(train_loader)} steps per epoch"
    )

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        epoch_losses = []

        for step, (vids, labels) in enumerate(train_loader, start=1):
            if vids.size(0) == 0:  # Skip empty batches
                continue

            inputs = processor(vids, return_tensors="pt").to(model.device)
            labels = labels.to(model.device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / config.accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if step % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if (
                    step % (config.accumulation_steps * 10) == 0
                ):  # Log every 10 accumulation steps
                    print(
                        f"Epoch {epoch} Step {step}: Accumulated Loss = {running_loss:.4f}"
                    )
                    writer.add_scalar(
                        "Train/Loss",
                        running_loss,
                        (epoch - 1) * len(train_loader) + step,
                    )
                epoch_losses.append(running_loss)
                running_loss = 0.0

        # End of epoch evaluation
        val_acc = evaluate(val_loader, model, processor, model.device)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch} Validation Accuracy: {val_acc:.4f}")
        print(f"Epoch {epoch} Duration: {epoch_duration:.2f} seconds")

        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            # Save model to disk
            model_save_path = (
                models_dir / f"lora_action_detection_{output_suffix}_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "validation_accuracy": best_val_acc,
                    "label2id": label2id,
                    "id2label": id2label,
                    "lora_config": {
                        "rank": config.lora_rank,
                        "alpha": config.lora_alpha,
                        "dropout": config.lora_dropout,
                        "target_modules": lora_target_modules,
                    },
                },
                model_save_path,
            )
            print(
                f"New best validation accuracy: {best_val_acc:.4f} - saved model to {model_save_path}"
            )

        # Store metrics
        training_metrics["epochs"].append(epoch)
        training_metrics["train_loss"].append(
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        )
        training_metrics["val_acc"].append(val_acc)

    # Load best model for final test evaluation
    if best_model_state is not None:
        print(
            f"\nLoading best model (validation accuracy: {best_val_acc:.4f}) for final test evaluation"
        )
        model.load_state_dict(best_model_state)
    else:
        print(f"\nNo best model saved, using final epoch model for test evaluation")

    # Final test evaluation with detailed metrics
    train_results = evaluate_detailed(train_loader, model, processor, model.device, id2label)
    test_results = evaluate_detailed(test_loader, model, processor, model.device, id2label)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Final Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Final Train Accuracy: {train_results['accuracy']:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Inference Time: {test_results['inference_time_ms']:.2f} ms/video")
    print(
        f"Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
    )
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Log final results
    writer.add_scalar("Test/Final_Accuracy", test_results['accuracy'], config.num_epochs)
    writer.close()

    # Save metrics
    training_metrics["final_test_acc"] = test_results['accuracy']
    training_metrics["final_train_acc"] = train_results['accuracy']
    training_metrics["best_val_acc"] = best_val_acc
    training_metrics["best_epoch"] = best_epoch
    training_metrics["inference_time_ms"] = test_results['inference_time_ms']
    training_metrics["per_class_metrics"] = test_results['per_class_metrics']
    training_metrics["total_training_time"] = total_duration

    # Save confusion matrix
    tracker = ExperimentTracker()
    tracker.save_confusion_matrix(
        test_results['confusion_matrix'],
        id2label,
        f"lora_{args.num_videos}videos_{timestamp}"
    )

    # Save metrics with timestamp
    metrics_filename = f"lora_action_metrics_{output_suffix}.json"
    with open(metrics_filename, "w") as f:
        json.dump(training_metrics, f, indent=2)

    print(f"\nTraining metrics saved to: {metrics_filename}")
    print(f"Tensorboard logs saved to: {tensorboard_dir}")

    # Log to shared experiment tracker (tracker already initialized above)
    tracker.log_experiment(training_metrics)

    # Print label mapping for reference
    print(f"\nLabel Mapping:")
    for idx, label in id2label.items():
        print(f"  {idx}: {label}")

    print(f"\nLoRA fine-tuning completed!")
    print_parameter_stats(model, "Final LoRA Adapted Model")
    print(
        f"Memory efficiency: Used only {count_parameters(model)[1]:,} trainable parameters"
    )


if __name__ == "__main__":
    main()
