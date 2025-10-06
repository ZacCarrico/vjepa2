#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import pathlib
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
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
from src.common.training import evaluate, setup_tensorboard
from src.common.utils import get_device, set_seed, print_parameter_stats, count_parameters


# All dataset-related code moved to src/action_detection/data.py


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='V-JEPA 2 Action Detection Fine-tuning')
    parser.add_argument(
        '--num_videos',
        type=int,
        default=100,
        help='Number of videos per class to use (default: 100)'
    )
    return parser.parse_args()


def main():
    print("Starting V-JEPA 2 Action Detection Pooler+Head Fine-tuning")
    print("=" * 50)

    # Create timestamp for output files
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    print(f"Training session timestamp: {timestamp}")

    # Parse arguments
    args = parse_arguments()
    print(f"Configuration:")
    print(f"  Videos per class: {args.num_videos}")

    # Load shared config
    config = DEFAULT_CONFIG
    print(f"\nUsing shared configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Weight decay: {config.weight_decay}")

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

    # Set frames_per_clip from config
    frames_per_clip = config.frames_per_clip
    if frames_per_clip != model.config.frames_per_clip:
        print(f"Overriding frames per clip: {model.config.frames_per_clip} → {frames_per_clip}")
        model.config.frames_per_clip = frames_per_clip

    # Print model statistics
    print_parameter_stats(model, "V-JEPA 2 Action Detection Model")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, processor, config.batch_size, config.num_workers, frames_per_clip
    )

    # Freeze backbone and only train classification head (pooler + classifier)
    for param in model.vjepa2.parameters():
        param.requires_grad = False

    print_parameter_stats(model, "V-JEPA 2 Action Detection Model (After Freezing)")

    # Setup training
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)

    # Setup tensorboard
    output_suffix = f"{args.num_videos}videos_{frames_per_clip}frames_{timestamp}"
    tensorboard_dir = f"runs/ntu_pooler_head_{output_suffix}"
    writer = setup_tensorboard(tensorboard_dir)

    # Store metrics for comparison
    training_metrics = {
        "timestamp": timestamp,
        "approach": "pooler_and_head",
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
        "trainable_params": count_parameters(model)[1],
        "total_params": count_parameters(model)[0],
    }

    # Best model tracking
    best_val_acc = 0.0
    best_model_state = None

    # Start timing
    total_start_time = time.time()
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training for {config.num_epochs} epochs with {len(train_loader)} steps per epoch")

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

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
                if step % (config.accumulation_steps * 10) == 0:  # Log every 10 accumulation steps
                    print(f"Epoch {epoch} Step {step}: Accumulated Loss = {running_loss:.4f}")
                    writer.add_scalar("Train/Loss", running_loss, (epoch-1) * len(train_loader) + step)
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
            models_dir = pathlib.Path("models")
            models_dir.mkdir(exist_ok=True)
            model_save_path = models_dir / f"best_action_detection_model_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'validation_accuracy': best_val_acc,
                'label2id': label2id,
                'id2label': id2label
            }, model_save_path)
            print(f"New best validation accuracy: {best_val_acc:.4f} - saved model to {model_save_path}")

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

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Log final results
    writer.add_scalar("Test/Final_Accuracy", test_acc, config.num_epochs)
    writer.close()

    # Save metrics
    training_metrics["final_test_acc"] = test_acc
    training_metrics["best_val_acc"] = best_val_acc
    training_metrics["total_training_time"] = total_duration

    # Save metrics with timestamp
    metrics_filename = f"pooler_head_metrics_{output_suffix}.json"
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


if __name__ == "__main__":
    main()