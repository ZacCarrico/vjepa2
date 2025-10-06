#!/usr/bin/env python
# coding: utf-8

"""
Shared data loading utilities for NTU RGB action detection.

This module contains common dataset classes and functions used across
all action detection fine-tuning scripts.
"""

import pathlib
import random
from functools import partial
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
from torchvision.transforms import v2


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
    Setup NTU RGB dataset for action detection with reproducible video selection.

    This function uses random.seed(42) to ensure reproducible video selection
    across runs, even when the total number of available videos changes.

    Args:
        num_videos_per_class: Number of videos to use per class

    Returns:
        Tuple of (train_paths, val_paths, test_paths, label2id, id2label)
    """
    # Set seed at the start for reproducible video selection
    random.seed(42)

    ntu_rgb_path = pathlib.Path("ntu_rgb")

    # Find ALL target action videos, sort for deterministic ordering, then randomly sample
    all_sitting_videos = sorted(ntu_rgb_path.glob("*A008_rgb.avi"))
    all_standing_videos = sorted(ntu_rgb_path.glob("*A009_rgb.avi"))
    all_waving_videos = sorted(ntu_rgb_path.glob("*A023_rgb.avi"))

    sitting_videos = random.sample(
        all_sitting_videos, min(num_videos_per_class, len(all_sitting_videos))
    )
    standing_videos = random.sample(
        all_standing_videos, min(num_videos_per_class, len(all_standing_videos))
    )
    waving_videos = random.sample(
        all_waving_videos, min(num_videos_per_class, len(all_waving_videos))
    )

    # Find negative samples (other actions)
    all_videos = sorted(ntu_rgb_path.glob("*.avi"))
    target_actions = set(sitting_videos + standing_videos + waving_videos)
    negative_videos = [v for v in all_videos if v not in target_actions]

    print(
        f"Found {len(sitting_videos)} sitting down videos (from {len(all_sitting_videos)} available)"
    )
    print(
        f"Found {len(standing_videos)} standing up videos (from {len(all_standing_videos)} available)"
    )
    print(
        f"Found {len(waving_videos)} waving videos (from {len(all_waving_videos)} available)"
    )
    print(f"Found {len(negative_videos)} negative sample videos")

    # Sample negatives to match other classes
    sampled_negatives = random.sample(
        negative_videos, min(num_videos_per_class, len(negative_videos))
    )

    # Combine all videos
    all_action_videos = (
        sitting_videos + standing_videos + waving_videos + sampled_negatives
    )

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
