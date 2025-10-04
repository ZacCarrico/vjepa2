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
from src.common.experiment_tracker import ExperimentTracker
from src.common.training import evaluate, setup_tensorboard
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


# ============================================================================
# Action Detection Dataset
# ============================================================================

class ActionDetectionDataset(Dataset):
    """Dataset for NTU RGB action detection with specific actions."""

    def __init__(self, video_file_paths: List[pathlib.Path], label2id: Dict[str, int]):
        self.video_file_paths = video_file_paths
        self.label2id = label2id

    def __len__(self):
        return len(self.video_file_paths)

    def __getitem__(self, idx):
        video_path = self.video_file_paths[idx]

        # Extract action from filename (e.g., A008, A009, A023)
        filename = video_path.name
        if "A008_rgb.avi" in filename:
            label = "sitting_down"
        elif "A009_rgb.avi" in filename:
            label = "standing_up"
        elif "A023_rgb.avi" in filename:
            label = "waving"
        else:
            label = "other"

        decoder = VideoDecoder(video_path)
        return decoder, self.label2id[label]


def setup_ntu_action_dataset(
    num_videos_per_class: int = 100,
) -> Tuple[List, List, List, Dict, Dict]:
    """
    Setup NTU RGB dataset for action detection.

    Args:
        num_videos_per_class: Number of videos to use per class

    Returns:
        Tuple of (train_paths, val_paths, test_paths, label2id, id2label)
    """
    ntu_rgb_path = pathlib.Path("ntu_rgb")

    # Find target action videos (limit to num_videos_per_class each)
    sitting_videos = list(ntu_rgb_path.glob("*A008_rgb.avi"))[:num_videos_per_class]
    standing_videos = list(ntu_rgb_path.glob("*A009_rgb.avi"))[:num_videos_per_class]
    waving_videos = list(ntu_rgb_path.glob("*A023_rgb.avi"))[:num_videos_per_class]

    # Find negative samples (other actions)
    all_videos = list(ntu_rgb_path.glob("*.avi"))
    target_actions = set(sitting_videos + standing_videos + waving_videos)
    negative_videos = [v for v in all_videos if v not in target_actions]

    print(f"Found {len(sitting_videos)} sitting down videos")
    print(f"Found {len(standing_videos)} standing up videos")
    print(f"Found {len(waving_videos)} waving videos")
    print(f"Found {len(negative_videos)} negative sample videos")

    # Sample negatives to match other classes
    random.seed(42)
    sampled_negatives = random.sample(
        negative_videos, min(num_videos_per_class, len(negative_videos))
    )

    # Combine all videos
    all_action_videos = sitting_videos + standing_videos + waving_videos + sampled_negatives

    # Shuffle and split
    random.shuffle(all_action_videos)

    # 70/15/15 split
    total_videos = len(all_action_videos)
    train_size = int(0.7 * total_videos)
    val_size = int(0.15 * total_videos)

    train_videos = all_action_videos[:train_size]
    val_videos = all_action_videos[train_size : train_size + val_size]
    test_videos = all_action_videos[train_size + val_size :]

    # Create label mappings
    label2id = {"sitting_down": 0, "standing_up": 1, "waving": 2, "other": 3}
    id2label = {v: k for k, v in label2id.items()}

    print(f"\nDataset splits:")
    print(f"Train: {len(train_videos)} videos")
    print(f"Validation: {len(val_videos)} videos")
    print(f"Test: {len(test_videos)} videos")

    return train_videos, val_videos, test_videos, label2id, id2label


def collate_fn(samples, frames_per_clip, transforms):
    """Sample clips and apply transforms to a batch."""
    clips, labels = [], []
    for decoder, lbl in samples:
        try:
            clip = clips_at_random_indices(
                decoder,
                num_clips=1,
                num_frames_per_clip=frames_per_clip,
                num_indices_between_frames=3,
            ).data
            clips.append(clip)
            labels.append(lbl)
        except Exception as e:
            print(f"Error processing video: {e}")
            # Skip this sample
            continue

    if not clips:
        # Return empty tensors if no valid clips
        return torch.empty(0, frames_per_clip, 3, 224, 224), torch.empty(
            0, dtype=torch.long
        )

    videos = torch.cat(clips, dim=0)
    videos = transforms(videos)
    return videos, torch.tensor(labels)


def setup_transforms(processor):
    """Setup train and eval transforms using processor crop size."""
    train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(
                (processor.crop_size["height"], processor.crop_size["width"])
            ),
            v2.RandomHorizontalFlip(),
        ]
    )
    eval_transforms = v2.Compose(
        [v2.CenterCrop((processor.crop_size["height"], processor.crop_size["width"]))]
    )
    return train_transforms, eval_transforms


def create_data_loaders(
    train_ds,
    val_ds,
    test_ds,
    processor,
    batch_size=1,
    num_workers=0,
    frames_per_clip=16,
):
    """Create DataLoaders for train, validation, and test sets."""
    from functools import partial

    train_transforms, eval_transforms = setup_transforms(processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            frames_per_clip=frames_per_clip,
            transforms=train_transforms,
        ),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_fn,
            frames_per_clip=frames_per_clip,
            transforms=eval_transforms,
        ),
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_fn,
            frames_per_clip=frames_per_clip,
            transforms=eval_transforms,
        ),
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


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
        "--lora_rank", type=int, default=16, help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=32.0, help="LoRA alpha (default: 32.0)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate (default: 0.1)",
    )

    # Training configuration
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
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

    # Override frames_per_clip if specified
    if args.frames_per_clip:
        original_frames = model.config.frames_per_clip
        model.config.frames_per_clip = args.frames_per_clip
        print(
            f"Overriding frames per clip: {original_frames} → {args.frames_per_clip}"
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

    # Create DataLoaders
    frames_per_clip = model.config.frames_per_clip
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
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "accumulation_steps": config.accumulation_steps,
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

    # Final test evaluation
    test_acc = evaluate(test_loader, model, processor, model.device)
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(
        f"Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
    )
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Log final results
    writer.add_scalar("Test/Final_Accuracy", test_acc, config.num_epochs)
    writer.close()

    # Save metrics
    training_metrics["final_test_acc"] = test_acc
    training_metrics["best_val_acc"] = best_val_acc
    training_metrics["total_training_time"] = total_duration

    # Save metrics with timestamp
    metrics_filename = f"lora_action_metrics_{output_suffix}.json"
    with open(metrics_filename, "w") as f:
        json.dump(training_metrics, f, indent=2)

    print(f"\nTraining metrics saved to: {metrics_filename}")
    print(f"Tensorboard logs saved to: {tensorboard_dir}")

    # Log to shared experiment tracker
    tracker = ExperimentTracker()
    training_metrics["num_epochs"] = config.num_epochs
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
