# Old Learning Rate Experiments

This directory contains metric files from experiments conducted with `learning_rate=1e-3` (0.001) before the final configuration was established at `learning_rate=2e-4` (0.0002).

## Why These Were Archived

- These experiments used `LR=1e-3` instead of the current default `LR=2e-4`
- All experiments used correct optimizer configuration (`AdamW` with `weight_decay=0.01`)
- They are valid experiments but not directly comparable with the current configuration

## Files

All experiments here were run on **October 3, 2025** with the following configuration:
- Learning rate: 1e-3 (OLD)
- Weight decay: 0.01 ✓
- Optimizer: AdamW ✓
- Epochs: 5 ✓
- Batch size: 1 ✓
- Accumulation steps: 4 ✓

### 50 videos per class (LR=1e-3)
- `head_only_metrics_50videos_16frames_251003-194213.json`
- `head_only_metrics_50videos_16frames_251003-205949.json`
- `lora_action_metrics_50videos_16frames_251003-175848.json`
- `lora_action_metrics_50videos_16frames_251003-195936.json`
- `lora_action_metrics_50videos_16frames_251003-204412.json`
- `pooler_head_metrics_50videos_16frames_251003-195014.json`
- `pooler_head_metrics_50videos_16frames_251003-210822.json`

### 100 videos per class (LR=1e-3)
- `head_only_metrics_100videos_16frames_251003-202131.json`

## Current Valid Experiments

For the final results with `LR=2e-4`, see the metric files in the root directory (not archived).
