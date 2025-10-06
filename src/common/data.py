import pathlib
import tarfile
from functools import partial
from typing import Dict, List, Tuple

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
from torchvision.transforms import v2


class CustomVideoDataset(Dataset):
    def __init__(self, video_file_paths, label2id):
        self.video_file_paths = video_file_paths
        self.label2id = label2id

    def __len__(self):
        return len(self.video_file_paths)

    def __getitem__(self, idx):
        video_path = self.video_file_paths[idx]
        label = video_path.parts[2]
        decoder = VideoDecoder(video_path)
        return decoder, self.label2id[label]


def collate_fn(samples, frames_per_clip, transforms):
    """Sample clips and apply transforms to a batch."""
    clips, labels = [], []
    for decoder, lbl in samples:
        clip = clips_at_random_indices(
            decoder,
            num_clips=1,
            num_frames_per_clip=frames_per_clip,
            num_indices_between_frames=3,
        ).data
        clips.append(clip)
        labels.append(lbl)

    videos = torch.cat(clips, dim=0)
    videos = transforms(videos)
    return videos, torch.tensor(labels)


def setup_ucf101_dataset() -> Tuple[List, List, List, pathlib.Path]:
    """
    Download and extract UCF-101 dataset, return file paths for train/val/test splits.

    Returns:
        Tuple of (train_paths, val_paths, test_paths, dataset_root_path)
    """
    fpath = hf_hub_download(
        repo_id="sayakpaul/ucf101-subset",
        filename="UCF101_subset.tar.gz",
        repo_type="dataset",
    )

    with tarfile.open(fpath) as t:
        t.extractall(".", filter="data")

    dataset_root_path = pathlib.Path("UCF101_subset")
    # Sort for deterministic ordering across filesystems
    all_video_file_paths = sorted(dataset_root_path.glob("**/*.avi"))

    # Split data
    train_video_file_paths = []
    val_video_file_paths = []
    test_video_file_paths = []

    for video_file_path in all_video_file_paths:
        video_parts = video_file_path.parts
        if "train" in video_parts:
            train_video_file_paths.append(video_file_path)
        elif "val" in video_parts:
            val_video_file_paths.append(video_file_path)
        elif "test" in video_parts:
            test_video_file_paths.append(video_file_path)
        else:
            raise ValueError(f"Unknown video part: {video_parts}")

    return train_video_file_paths, val_video_file_paths, test_video_file_paths, dataset_root_path


def create_label_mappings(all_video_file_paths) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label mappings from video file paths."""
    class_labels = {path.parts[2] for path in all_video_file_paths}
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def setup_transforms(processor):
    """Setup train and eval transforms using processor crop size."""
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(
            (processor.crop_size["height"], processor.crop_size["width"])
        ),
        v2.RandomHorizontalFlip(),
    ])
    eval_transforms = v2.Compose([
        v2.CenterCrop((processor.crop_size["height"], processor.crop_size["width"]))
    ])
    return train_transforms, eval_transforms


def create_data_loaders(
    train_ds, val_ds, test_ds, processor, batch_size=1, num_workers=0, frames_per_clip=16
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