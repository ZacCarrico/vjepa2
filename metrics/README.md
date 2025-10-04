# Metrics Directory

This directory contains experiment metric files organized by dataset.

## Directory Structure

### `action_detection/`
Metric files from action detection experiments on **NTU RGB+D dataset**.

**Dataset characteristics:**
- 4 action classes: sitting_down, standing_up, waving, other
- Experiments with 50, 100, and 200 videos per class
- Training scripts in `src/action_detection/`

**Current valid files:** 7 files with LR=2e-4 configuration

### `vid_classification/`
Metric files from video classification experiments on **UCF-101 dataset**.

**Dataset characteristics:**
- 101 action classes
- Training scripts in `src/vid_classification/`

**Current valid files:** None yet

## File Naming Convention

Format: `{approach}_metrics_{num_videos}videos_{frames}frames_{timestamp}.json`

- **approach**: `head_only`, `lora_action`, `pooler_head`, or `pooler_head_lora`
- **num_videos**: Number of videos per class used for training
- **frames**: Number of frames per clip (always 16 in current experiments)
- **timestamp**: YYMMDD-HHMMSS format

## Archived Files

Older experiments with different configurations are in `../archive/old_lr_experiments/`.

## Usage

See `../METRICS_FILES.md` for detailed documentation on valid metric files and their results.
