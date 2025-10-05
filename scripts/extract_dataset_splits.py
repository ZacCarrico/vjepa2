#!/usr/bin/env python
# coding: utf-8

"""
Extract Dataset Splits for All Action Detection Experiments
============================================================

This script recreates the train/val/test splits for all completed experiments
and saves the file lists. Since the splits use random.seed(42), they are
perfectly reproducible.

Usage:
    python scripts/extract_dataset_splits.py
"""

import json
import pathlib
import re
from typing import Dict, List, Tuple
import random


def setup_ntu_action_dataset(num_videos_per_class: int = 100) -> Tuple[List, List, List, Dict, Dict]:
    """
    Setup NTU RGB dataset for action detection.

    This is the EXACT same function used in training scripts to ensure
    we get identical splits.

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

    # Balance dataset by sampling negatives
    random.seed(42)
    sampled_negatives = random.sample(negative_videos, min(num_videos_per_class, len(negative_videos)))

    # Combine all videos
    all_action_videos = sitting_videos + standing_videos + waving_videos + sampled_negatives

    # Shuffle and split
    random.shuffle(all_action_videos)

    # 70/15/15 split
    total_videos = len(all_action_videos)
    train_size = int(0.7 * total_videos)
    val_size = int(0.15 * total_videos)

    train_videos = all_action_videos[:train_size]
    val_videos = all_action_videos[train_size:train_size + val_size]
    test_videos = all_action_videos[train_size + val_size:]

    # Create label mappings
    label2id = {
        "sitting_down": 0,
        "standing_up": 1,
        "waving": 2,
        "other": 3
    }
    id2label = {v: k for k, v in label2id.items()}

    return train_videos, val_videos, test_videos, label2id, id2label


def extract_num_videos_from_filename(filename: str) -> int:
    """Extract number of videos per class from metrics filename"""
    match = re.search(r'(\d+)videos', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract video count from filename: {filename}")


def process_experiment(metrics_file: pathlib.Path) -> Dict:
    """Process a single experiment and extract dataset splits"""

    # Load existing metrics to get metadata
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Extract number of videos from filename
    num_videos_per_class = extract_num_videos_from_filename(metrics_file.name)

    print(f"\nProcessing: {metrics_file.name}")
    print(f"  Videos per class: {num_videos_per_class}")

    # Recreate the exact same splits
    train_videos, val_videos, test_videos, label2id, id2label = setup_ntu_action_dataset(num_videos_per_class)

    # Verify split sizes match original experiment
    expected_train = metrics.get('num_train_videos')
    expected_val = metrics.get('num_val_videos')
    expected_test = metrics.get('num_test_videos')

    actual_train = len(train_videos)
    actual_val = len(val_videos)
    actual_test = len(test_videos)

    print(f"  Train: {actual_train} (expected {expected_train}) {'✅' if actual_train == expected_train else '❌'}")
    print(f"  Val:   {actual_val} (expected {expected_val}) {'✅' if actual_val == expected_val else '❌'}")
    print(f"  Test:  {actual_test} (expected {expected_test}) {'✅' if actual_test == expected_test else '❌'}")

    # Create dataset splits record
    splits_data = {
        "experiment_file": metrics_file.name,
        "timestamp": metrics.get('timestamp'),
        "approach": metrics.get('approach'),
        "num_videos_per_class": num_videos_per_class,
        "num_train_videos": actual_train,
        "num_val_videos": actual_val,
        "num_test_videos": actual_test,
        "train_files": [str(p) for p in train_videos],
        "val_files": [str(p) for p in val_videos],
        "test_files": [str(p) for p in test_videos],
        "label2id": label2id,
        "id2label": id2label,
    }

    return splits_data


def main():
    """Main extraction function"""

    print("=" * 80)
    print("EXTRACTING DATASET SPLITS FOR ALL EXPERIMENTS")
    print("=" * 80)

    metrics_dir = pathlib.Path("metrics/action_detection")
    output_dir = pathlib.Path("dataset_splits/action_detection")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all metrics files
    metrics_files = sorted(metrics_dir.glob("*.json"))

    print(f"\nFound {len(metrics_files)} experiment metrics files")

    # Process each experiment
    all_splits = []
    for metrics_file in metrics_files:
        try:
            splits_data = process_experiment(metrics_file)
            all_splits.append(splits_data)

            # Save individual splits file
            output_filename = metrics_file.stem + "_dataset_splits.json"
            output_path = output_dir / output_filename

            with open(output_path, 'w') as f:
                json.dump(splits_data, f, indent=2)

            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"  ❌ Error processing {metrics_file.name}: {e}")

    # Save combined summary
    summary_path = output_dir / "all_experiments_dataset_splits.json"
    with open(summary_path, 'w') as f:
        json.dump(all_splits, f, indent=2)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\n✅ Processed {len(all_splits)} experiments")
    print(f"✅ Individual split files saved to: {output_dir}/")
    print(f"✅ Combined summary saved to: {summary_path}")

    # Print summary statistics
    print("\n📊 SUMMARY:")
    for splits in all_splits:
        print(f"\n{splits['experiment_file']}:")
        print(f"  Approach: {splits['approach']}")
        print(f"  Videos/class: {splits['num_videos_per_class']}")
        print(f"  Train: {splits['num_train_videos']} files")
        print(f"  Val:   {splits['num_val_videos']} files")
        print(f"  Test:  {splits['num_test_videos']} files")


if __name__ == "__main__":
    main()
