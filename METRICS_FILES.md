# Metric Files Documentation

This document describes the valid experiment metric files in this repository.

## Organization

Metric files are organized by dataset in the `metrics/` directory:
- **`metrics/action_detection/`** - NTU RGB+D action detection experiments (4 classes)
- **`metrics/vid_classification/`** - UCF-101 video classification experiments (101 classes)

## Action Detection Experiments (NTU RGB+D)

Location: `metrics/action_detection/`

All files use the consistent configuration from `src/action_detection/config.py`:
- **Dataset**: NTU RGB+D (4 classes: sitting_down, standing_up, waving, other)
- **Learning rate**: 2e-4
- **Weight decay**: 0.01
- **Optimizer**: AdamW
- **Epochs**: 5
- **Batch size**: 1
- **Accumulation steps**: 4

### 100 Videos Per Class

| File | Approach | Test Acc | Val Acc | Trainable Params |
|------|----------|----------|---------|------------------|
| `head_only_metrics_100videos_16frames_251003-223428.json` | Head-only | 83.33% | 81.67% | 4,100 |
| `lora_action_metrics_100videos_16frames_251003-221650.json` | LoRA | 81.67% | 86.67% | 495,620 |
| `pooler_head_metrics_100videos_16frames_251003-225029.json` | Pooler+Head | 76.67% | 88.33% | 49,340,420 |

### 200 Videos Per Class

| File | Approach | Test Acc | Val Acc | Trainable Params |
|------|----------|----------|---------|------------------|
| `head_only_metrics_200videos_16frames_251004-071214.json` | Head-only | 78.33% | 80.00% | 4,100 |
| `lora_action_metrics_200videos_16frames_251004-062932.json` | LoRA | 84.17% | 86.67% | 495,620 |
| `pooler_head_metrics_200videos_16frames_251004-082514.json` | Pooler+Head | 88.33% | 88.33% | 49,340,420 |
| `pooler_head_lora_metrics_200videos_16frames_251004-091526.json` | Pooler+Head+LoRA | 82.50% | 83.33% | 49,831,940 |

## Video Classification Experiments (UCF-101)

Location: `metrics/vid_classification/`

**Status**: No valid experiments yet.

## Archived Files

Older experiments with different configurations are archived in `archive/old_lr_experiments/`.

These used `learning_rate=1e-3` instead of the current `2e-4` and are not directly comparable to the current results, though they used correct optimizer configuration.

## File Naming Convention

Format: `{approach}_metrics_{num_videos}videos_{frames}frames_{timestamp}.json`

- **approach**: `head_only`, `lora_action`, `pooler_head`, or `pooler_head_lora`
- **num_videos**: Number of videos per class used for training
- **frames**: Number of frames per clip (always 16 in current experiments)
- **timestamp**: YYMMDD-HHMMSS format

## Usage

These files are referenced in:
- `experiments.csv` - Master tracking file with all experiment metadata
- `docs/TRAINING_GUIDE.md` - Detailed training documentation
- Analysis scripts (if any)

All valid files should remain committed to version control for reproducibility.
