import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor


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


def setup_tensorboard(log_dir: str) -> SummaryWriter:
    """Setup TensorBoard writer for logging."""
    return SummaryWriter(log_dir)