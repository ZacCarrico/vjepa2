"""Shared configuration for all action detection fine-tuning experiments."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Shared training configuration for fair comparison across approaches."""

    # Training hyperparameters
    num_epochs: int = 5
    batch_size: int = 1
    accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01

    # Data configuration
    num_workers: int = 0
    frames_per_clip: int = 8

    # Model configuration
    model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"

    # LoRA configuration (only used for LoRA approach)
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

    # Random seed
    seed: int = 42


# Create a singleton instance
DEFAULT_CONFIG = TrainingConfig()
