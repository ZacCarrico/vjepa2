#!/usr/bin/env python
# coding: utf-8

# # V-JEPA 2

# V-JEPA 2 is a new open 1.2B video embedding model by Meta, which attempts to capture the physical world modelling through video ⏯️
#
# The model can be used for various tasks for video: fine-tuning for downstream tasks like video classification, or any task involving embeddings (similarity, retrieval and more!).
#
# You can check all V-JEPA 2 checkpoints and the datasets that come with this release [in this collection](collection_link). You can also read about the release [here](blog_link).
#
# In this notebook we will go through:
# 1. Using V-JEPA 2 as feature extractor,
# 2. Using V-JEPA 2 for video classification
# 3. fine-tuning V-JEPA 2, on [UCF-101 action recognition dataset](https://huggingface.co/datasets/sayakpaul/ucf101-subset) using transformers.
#
# Let's go!

# We need to install accelerate, datasets and transformers' main branch.

import pathlib
import tarfile
import time
from functools import partial

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
from torchvision.transforms import v2
from transformers import (
    AutoModel,
    AutoVideoProcessor,
    VJEPA2ForVideoClassification,
    VJEPA2VideoProcessor,
)

print("Torch:", torch.__version__)


# ## Simple Inference
#
# **Device Setup**
#
# This notebook automatically detects and uses the best available device in the following priority order:
# 1. **CUDA** (NVIDIA GPUs) - if available
# 2. **MPS** (Apple Silicon GPUs) - for Apple Silicon Macs
# 3. **CPU** - as fallback
#
# **Inference with Embeddings**
#
# You can initialize the V-JEPA 2 with ViT Giant checkpoint as follows. Feel free to replace the ID with the one you want to use. Here's [the model collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6).


# Auto-detect best available device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

model = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-384").to(device)
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-384")


video_url = "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/holding_phone.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(
    0, 32
)  # choosing some frames. here, you can define more complex sampling strategy
video_frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W

# slow so skipping this =============>
# video = processor(video_frames, return_tensors="pt").to(model.device)
# with torch.no_grad():
#     video_embeddings = model.get_vision_features(**video)

# print(video_embeddings.shape)
# del model
# =================================


# **Inference for Video Classification**
#
# Meta also provides a model trained on SomethingSomething-v2, a dataset of human-object interactions with 174 classes.
# VJEPA2ForVideoClassification.from_pretrained takes the pretrained weights and adds a new head for the classification task
# Use the same device as before
model = VJEPA2ForVideoClassification.from_pretrained(
    "facebook/vjepa2-vitl-fpc16-256-ssv2"
).to(device)
processor = VJEPA2VideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")


# We can pass the same frames to the new processor.
inputs = processor(video_frames, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])


# ## Data Preprocessing for Fine-tuning

# Let's load the dataset first. UCF-101 consists of 101 different actions covering from blowing candles to playing violin. We will use [a smaller subset of UCF-101](https://huggingface.co/datasets/sayakpaul/ucf101-subset).
fpath = hf_hub_download(
    repo_id="sayakpaul/ucf101-subset",
    filename="UCF101_subset.tar.gz",
    repo_type="dataset",
)
with tarfile.open(fpath) as t:
    t.extractall(".")
dataset_root_path = pathlib.Path("UCF101_subset")
all_video_file_paths = list(dataset_root_path.glob("**/*.avi"))

# We gather different splits as lists to later create the `DataLoader` for training.
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

video_count_train = len(train_video_file_paths)
video_count_val = len(val_video_file_paths)
video_count_test = len(test_video_file_paths)
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

# We need to keep a class label to human-readable label mapping and number of classes to later initialize our model.
class_labels = {path.parts[2] for path in all_video_file_paths}
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}


# We will create a `CustomVideoDataset` class and initialize our train/test/validation sets for DataLoader.
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


train_ds = CustomVideoDataset(train_video_file_paths, label2id)
val_ds = CustomVideoDataset(val_video_file_paths, label2id)
test_ds = CustomVideoDataset(test_video_file_paths, label2id)


# V-JEPA 2 is an embedding model. To fine-tune it, we need to load the weights with a randomly initialized task-specific head put on top of them. For this, we can use `VJEPA2ForVideoClassification` class. During the initialization, we should pass in the mapping between the class labels and human readable labels, so the classification head has the same number of classes, and directly outputs human-readable labels with the confidence scores.
# On a separate note, if you want to only use embeddings, you can use `AutoModel` to do so. This can be used for e.g. video-to-video retrieval or calculating similarity between videos.
# We can now define augmentations and create the data collator. This notebook is made for tutorial purposes, so we keep the augmentations minimal. We can finally initialize the DataLoader afterwards.
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
batch_size = 1
num_workers = 0  # Changed from 4 to 0 to fix multiprocessing error

# DataLoaders
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(
        collate_fn,
        frames_per_clip=model.config.frames_per_clip,
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
        frames_per_clip=model.config.frames_per_clip,
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
        frames_per_clip=model.config.frames_per_clip,
        transforms=eval_transforms,
    ),
    num_workers=num_workers,
    pin_memory=True,
)

# ## Model Training

# Before training, we can login to HF Hub (to later push the model) and setup tensorboard.
# from huggingface_hub import login
# login()  # Commented out for non-interactive execution

writer = SummaryWriter("runs/vjepa2_finetune")


# Here's a small evaluation function we use so we evaluate the model training and log the number to tensorboard.
def evaluate(
    loader: DataLoader,
    model: VJEPA2ForVideoClassification,
    processor: VJEPA2VideoProcessor,
    device: torch.device,
) -> float:
    """Compute accuracy over a dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for vids, labels in loader:
            inputs = processor(vids, return_tensors="pt").to(device)
            labels = labels.to(device)
            logits = model(**inputs).logits
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# Finally, write the training loop!
# - On A100 we can fit in single example, so for better performance we will use gradient accumulation to simulate larger batch sizes.
# - We will freeze the backbone completely and only train the head.
# - We will use Adam optimizer.
# - We need to initialize the model head with the new labels.
del model
model_name = "qubvel-hf/vjepa2-vitl-fpc16-256-ssv2"
processor = VJEPA2VideoProcessor.from_pretrained(model_name)
model = VJEPA2ForVideoClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
).to(device)


gradient_accumulation_steps = 4
for param in model.vjepa2.parameters():
    param.requires_grad = False

# Optimizer and loss
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=1e-5)

# Training loop with gradient accumulation and evaluation
num_epochs = 5
accumulation_steps = 4

# Start timing
total_start_time = time.time()
print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
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
            running_loss = 0.0

    # End of epoch evaluation
    val_acc = evaluate(val_loader, model, processor, model.device)
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch} Validation Accuracy: {val_acc:.4f}")
    print(f"Epoch {epoch} Duration: {epoch_duration:.2f} seconds")
    writer.add_scalar("Val Acc", val_acc, epoch * len(train_loader))  # Log loss


# Final test evaluation
test_acc = evaluate(test_loader, model, processor, model.device)
total_end_time = time.time()
total_duration = total_end_time - total_start_time

print(f"Final Test Accuracy: {test_acc:.4f}")
print(
    f"Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
)
print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

writer.add_scalar(
    "Final Test Acc", test_acc, epoch * len(train_loader) + step
)  # Log loss
writer.close()


# Simply infer to get the embeddings.
# Let's push the model and the tensorboard logs to Hub.
# model.push_to_hub("zac-carrico/vjepa2-vitl-fpc16-256-ssv2-ucf101")  # Commented out for non-interactive execution
# processor.push_to_hub("zac-carrico/vjepa2-vitl-fpc16-256-ssv2-ucf101")  # Commented out for non-interactive execution

# from huggingface_hub import upload_folder  # Commented out for non-interactive execution
# upload_folder(
#     repo_id="zac-carrico/vjepa2-vitl-fpc16-256-ssv2-ucf101",
#     folder_path="runs",
#     path_in_repo="runs")  # Commented out for non-interactive execution


# ## Evaluation

# You can view the logs [here](https://huggingface.co/zac-carrico/vjepa2-vitl-fpc16-256-ssv2-ucf101/tensorboard).
#
# Let's put model to test with an in-the-wild video of a concert scene, although we don't exactly have this label, we have "Band Marching" which is the closest one.


# Utility to plot the frames
# (taken from: https://docs.pytorch.org/torchcodec/stable/generated_examples/basic_example.html#sphx-glr-generated-examples-basic-example-py)
def plot(frames, title, fname):
    try:
        from torchvision.utils import make_grid
        from torchvision.transforms.v2.functional import to_pil_image
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return

    plt.rcParams["savefig.bbox"] = "tight"
    _, ax = plt.subplots()
    ax.imshow(to_pil_image(make_grid(frames)))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


video_url = (
    "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/IMG_3830.mp4"
)
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 32)
video_frames = vr.get_frames_at(indices=frame_idx).data

video = clips_at_random_indices(
    vr,
    num_clips=1,
    num_frames_per_clip=model.config.frames_per_clip,
    num_indices_between_frames=4,
).data

plot(video[0], title="Concert Scene", fname="test.png")
Image.open("test.png").show()


inputs = processor(video_frames, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", id2label[predicted_class_idx])


# Concert is not one of the classes, and it appears the closest class was Archery.
