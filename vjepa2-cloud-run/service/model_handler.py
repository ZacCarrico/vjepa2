import os
import logging
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoClassifier:
    def __init__(self, model_path="facebook/vjepa2-vitl-fpc16-256-ssv2"):
        """Initialize the V-JEPA2 video classifier.

        Args:
            model_path: HuggingFace model name or path to local model
        """
        # Use CPU for local testing if no GPU available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load processor
        logger.info(f"Loading processor from {model_path}")
        self.processor = VJEPA2VideoProcessor.from_pretrained(model_path)

        # Load model with appropriate dtype
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Loading model with dtype: {dtype}")

        self.model = VJEPA2ForVideoClassification.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded and set to eval mode")

    def predict(self, video_path, frames_per_clip=16):
        """Predict video classification.

        Args:
            video_path: Path to video file
            frames_per_clip: Number of frames to extract from video

        Returns:
            Dictionary with predictions, top class, and metadata
        """
        try:
            logger.info(f"Processing video: {video_path}")

            # Create video decoder
            decoder = VideoDecoder(video_path)
            logger.info(f"Video duration: {decoder.metadata.duration_seconds:.2f}s")

            # Sample clips from video
            clip = clips_at_random_indices(
                decoder,
                num_clips=1,
                num_frames_per_clip=frames_per_clip,
                num_indices_between_frames=3  # Sample every 3rd frame
            ).data

            logger.info(f"Extracted clip shape: {clip.shape}")

            # Run inference
            with torch.no_grad():
                # Process video frames
                inputs = self.processor(clip, return_tensors="pt")

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs)

                # Get predictions
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_k_predictions = torch.topk(predictions, k=5)

            # Prepare results
            top_classes = []
            for i in range(5):
                class_idx = top_k_predictions.indices[0][i].item()
                confidence = top_k_predictions.values[0][i].item()
                class_name = self.model.config.id2label.get(class_idx, f"class_{class_idx}")

                top_classes.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "class_id": class_idx
                })

            result = {
                "predictions": predictions.cpu().numpy().tolist(),
                "top_class": top_classes[0]["class"],
                "top_k_classes": top_classes,
                "frames_processed": frames_per_clip,
                "device_used": self.device,
                "video_duration": decoder.metadata.duration_seconds
            }

            logger.info(f"Prediction complete. Top class: {result['top_class']}")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise e