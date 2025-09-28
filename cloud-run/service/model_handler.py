import os
import logging
import numpy as np
import torch
import cv2
from typing import Dict, Any

try:
    from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from torchcodec.decoders import VideoDecoder
    from torchcodec.samplers import clips_at_random_indices
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoClassifier:
    def __init__(self, model_path="facebook/vjepa2-vitl-fpc16-256-ssv2"):
        """Initialize the V-JEPA2 video classifier.

        Args:
            model_path: HuggingFace model name or path to local model
        """
        # Check if required dependencies are available
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Please install it: pip install transformers")

        # Use CPU for local testing if no GPU available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load processor
        logger.info(f"Loading processor from {model_path}")
        try:
            self.processor = VJEPA2VideoProcessor.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load V-JEPA2 processor: {e}")
            raise ImportError(f"V-JEPA2 model not available: {e}")

        # Load model with appropriate dtype
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Loading model with dtype: {dtype}")

        try:
            self.model = VJEPA2ForVideoClassification.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("V-JEPA2 model loaded and set to eval mode")
        except Exception as e:
            logger.error(f"Failed to load V-JEPA2 model: {e}")
            raise ImportError(f"V-JEPA2 model not available: {e}")

    def _extract_frames_opencv(self, video_path: str, frames_per_clip: int) -> tuple:
        """Extract frames using OpenCV as fallback."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Extract frames
        frames = []
        frame_step = max(1, frame_count // frames_per_clip)

        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            if len(frames) >= frames_per_clip:
                break

        cap.release()

        # Convert to tensor format expected by transformers
        if frames:
            frames_tensor = torch.stack([torch.tensor(frame).permute(2, 0, 1) for frame in frames])
            # Add batch dimension: (1, frames, channels, height, width)
            frames_tensor = frames_tensor.unsqueeze(0)
        else:
            raise ValueError("No frames extracted from video")

        return frames_tensor, duration

    def _extract_frames_torchcodec(self, video_path: str, frames_per_clip: int) -> tuple:
        """Extract frames using torchcodec if available."""
        decoder = VideoDecoder(video_path)
        duration = decoder.metadata.duration_seconds

        # Sample clips from video
        clip = clips_at_random_indices(
            decoder,
            num_clips=1,
            num_frames_per_clip=frames_per_clip,
            num_indices_between_frames=3
        ).data

        return clip, duration

    def predict(self, video_path: str, frames_per_clip: int = 16) -> Dict[str, Any]:
        """Predict video classification.

        Args:
            video_path: Path to video file
            frames_per_clip: Number of frames to extract from video

        Returns:
            Dictionary with predictions, top class, and metadata
        """
        try:
            logger.info(f"Processing video: {video_path}")

            # Extract frames using available method
            if TORCHCODEC_AVAILABLE:
                logger.info("Using torchcodec for video processing")
                clip, duration = self._extract_frames_torchcodec(video_path, frames_per_clip)
            else:
                logger.info("Using OpenCV for video processing")
                clip, duration = self._extract_frames_opencv(video_path, frames_per_clip)

            logger.info(f"Extracted clip shape: {clip.shape}, Duration: {duration:.2f}s")

            # Run inference
            with torch.no_grad():
                # Process video frames - handle different input formats
                try:
                    if hasattr(self.processor, 'process_video'):
                        # If processor has specific video processing method
                        inputs = self.processor.process_video(clip, return_tensors="pt")
                    else:
                        # Try standard processing
                        inputs = self.processor(clip, return_tensors="pt")
                except Exception as e:
                    logger.warning(f"Processor failed with tensor input: {e}")
                    # Convert tensor to numpy for processor
                    clip_np = clip.cpu().numpy() if torch.is_tensor(clip) else clip
                    inputs = self.processor(clip_np, return_tensors="pt")

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
                "frames_processed": len(clip[0]) if torch.is_tensor(clip) else frames_per_clip,
                "device_used": self.device,
                "video_duration": duration
            }

            logger.info(f"V-JEPA2 prediction complete. Top class: {result['top_class']}")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise e