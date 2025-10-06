import time
from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
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


def evaluate_detailed(
    loader: DataLoader,
    model: VJEPA2ForVideoClassification,
    processor: VJEPA2VideoProcessor,
    device: torch.device,
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    """
    Compute detailed metrics including per-class performance and confusion matrix.

    Returns:
        Dictionary with keys:
        - accuracy: overall accuracy
        - per_class_metrics: dict mapping class_name to {precision, recall, f1}
        - confusion_matrix: numpy array
        - avg_confidence: mean prediction confidence
        - inference_time_ms: average inference time per sample in milliseconds
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []

    total_inference_time = 0.0
    num_samples = 0

    with torch.no_grad():
        for vids, labels in loader:
            # Measure inference time
            start_time = time.time()

            inputs = processor(vids, return_tensors="pt").to(device)
            labels_device = labels.to(device)
            logits = model(**inputs).logits

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Get predictions and confidences
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
            confidences = probs.max(dim=-1).values

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())

            num_samples += labels.size(0)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # Calculate overall accuracy
    accuracy = (all_preds == all_labels).mean()

    # Get class names in order
    class_names = [id2label[i] for i in sorted(id2label.keys())]

    # Calculate per-class metrics using sklearn
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Extract per-class metrics
    per_class_metrics = {}
    for class_name in class_names:
        if class_name in report_dict:
            per_class_metrics[class_name] = {
                "precision": report_dict[class_name]["precision"],
                "recall": report_dict[class_name]["recall"],
                "f1": report_dict[class_name]["f1-score"],
            }

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Calculate average inference time
    avg_inference_time_ms = (total_inference_time / num_samples * 1000) if num_samples > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_matrix,
        "avg_confidence": float(all_confidences.mean()),
        "inference_time_ms": avg_inference_time_ms,
    }


def setup_tensorboard(log_dir: str) -> SummaryWriter:
    """Setup TensorBoard writer for logging."""
    return SummaryWriter(log_dir)