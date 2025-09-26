import random
import torch
import numpy as np
from typing import Tuple


def set_seed(seed: int = 1):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in the model.

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_parameter_stats(model: torch.nn.Module, model_name: str = "Model"):
    """Print parameter statistics for the model"""
    total, trainable = count_parameters(model)
    print(f"\n{model_name} Parameter Statistics:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Percentage trainable: {100 * trainable / total:.2f}%")
    print(f"  Memory reduction: {100 * (1 - trainable / total):.2f}%")