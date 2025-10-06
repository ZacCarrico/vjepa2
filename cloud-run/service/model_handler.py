import os
import tempfile
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
    def __init__(self, model_path=None, use_local_model=True, use_gcs=None):
        """Initialize the V-JEPA2 video classifier.

        Args:
            model_path: Path to local model file or HuggingFace model name
            use_local_model: If True, load from models/best_action_detection_model_epoch_1.pth
            use_gcs: If True, download model from GCS when not available locally
        """
        # Check if required dependencies are available
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Please install it: pip install transformers")

        # Use CPU for local testing if no GPU available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Check GCS usage
        if use_gcs is None:
            use_gcs = os.getenv("USE_GCS", "false").lower() == "true"
        self.use_gcs = use_gcs
        logger.info(f"GCS enabled: {self.use_gcs}")

        # Import GCS utils if needed
        if self.use_gcs:
            try:
                from gcs_utils import download_from_gcs
                self._download_from_gcs = download_from_gcs
                logger.info("GCS utilities loaded successfully")
            except ImportError as e:
                logger.warning(f"Failed to import GCS utilities: {e}")
                self.use_gcs = False

        # Define action detection labels (from fine-tuning script)
        self.label2id = {
            "sitting_down": 0,
            "standing_up": 1,
            "waving": 2,
            "other": 3
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        if use_local_model:
            # Load the fine-tuned model from GCS
            fine_tuned_model_path = self._get_model_path()

            logger.info(f"Loading fine-tuned model from {fine_tuned_model_path}")

            # Load checkpoint first to get the full model
            checkpoint = torch.load(fine_tuned_model_path, map_location=self.device)

            # Verify label mappings match
            saved_label2id = checkpoint.get('label2id', {})
            if saved_label2id != self.label2id:
                logger.warning(f"Label mapping mismatch. Model: {saved_label2id}, Expected: {self.label2id}")

            # Create processor with default config
            logger.info("Initializing processor with default configuration...")
            from transformers import VJEPA2Config
            config = VJEPA2Config(
                num_labels=len(self.label2id),
                label2id=self.label2id,
                id2label=self.id2label
            )
            self.processor = VJEPA2VideoProcessor()

            # Create model architecture with our config
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            logger.info(f"Creating model architecture with dtype: {dtype}")

            self.model = VJEPA2ForVideoClassification(config)
            self.model = self.model.to(dtype)

            # Load fine-tuned weights
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Move model to device
            self.model = self.model.to(self.device)

            logger.info(f"Loaded fine-tuned weights from {fine_tuned_model_path}")
            logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
            logger.info(f"Best validation accuracy: {checkpoint.get('validation_accuracy', 'unknown')}")

        else:
            # Original HuggingFace model loading
            if model_path is None:
                model_path = "facebook/vjepa2-vitl-fpc16-256-ssv2"

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
            except Exception as e:
                logger.error(f"Failed to load V-JEPA2 model: {e}")
                raise ImportError(f"V-JEPA2 model not available: {e}")

        # Set to eval mode
        self.model.eval()
        logger.info("V-JEPA2 model loaded and set to eval mode")

    def _get_model_path(self) -> str:
        """Get path to fine-tuned model, downloading from GCS if needed and available."""
        # Use the best LoRA model (88.75% test accuracy, 400 videos/class)
        model_filename = "best_lora_model_400videos_epoch_4.pth"

        # For Cloud Run, use /tmp which is writable
        # For local development, try to use the models directory
        if self.use_gcs:
            # In Cloud Run, use /tmp which is writable by non-root users
            local_model_path = f"/tmp/{model_filename}"
        else:
            # Check if model exists in current directory (for Docker local testing)
            current_dir_model = os.path.join(os.path.dirname(__file__), model_filename)
            if os.path.exists(current_dir_model):
                local_model_path = current_dir_model
            else:
                # For local development, use the project structure
                # Try the saved model filename first
                project_root = os.path.join(os.path.dirname(__file__), '..', '..')
                saved_model_path = os.path.join(project_root, "models/lora_action_detection_400videos_16frames_251004-140701_epoch_4.pth")
                if os.path.exists(saved_model_path):
                    local_model_path = saved_model_path
                else:
                    local_model_path = os.path.join(project_root, f"models/{model_filename}")

        # If file exists locally, use it
        if os.path.exists(local_model_path):
            logger.info(f"Using cached model: {local_model_path}")
            return local_model_path

        # If GCS is enabled and file doesn't exist locally, try to download
        if self.use_gcs:
            try:
                logger.info("Model not found in cache, downloading from GCS...")
                gcs_model_uri = f"gs://vjepa2/model-artifacts/{model_filename}"

                # Download from GCS directly to the path (no need to create directory for /tmp)
                self._download_from_gcs(gcs_model_uri, local_model_path)
                logger.info(f"Successfully downloaded model from GCS to {local_model_path}")
                return local_model_path

            except Exception as e:
                logger.error(f"Failed to download model from GCS: {e}")
                raise FileNotFoundError(f"Failed to download model from GCS: {e}")

        # If we reach here, the model is not available
        raise FileNotFoundError(f"Fine-tuned model not found: {local_model_path}")


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

        if not frames:
            raise ValueError("No frames extracted from video")

        # Return frames as a list of numpy arrays for the processor
        # The processor will handle the conversion to tensors
        # This matches what torchcodec returns
        return frames, duration

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
                # Log shape for tensor
                logger.info(f"Extracted clip shape: {clip.shape}, Duration: {duration:.2f}s")
            else:
                logger.info("Using OpenCV for video processing")
                clip, duration = self._extract_frames_opencv(video_path, frames_per_clip)
                # Log info for list of frames
                if isinstance(clip, list):
                    logger.info(f"Extracted {len(clip)} frames, shape: {clip[0].shape if clip else 'empty'}, Duration: {duration:.2f}s")
                else:
                    logger.info(f"Extracted clip shape: {clip.shape if hasattr(clip, 'shape') else type(clip)}, Duration: {duration:.2f}s")

            # Run inference
            with torch.no_grad():
                # Process video frames - handle different input formats
                # The processor expects either:
                # - A list of numpy arrays (from OpenCV)
                # - A tensor (from torchcodec)
                # - A list of PIL images
                inputs = self.processor(clip, return_tensors="pt")

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs)

                # Get predictions
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Prepare results - limit to actual number of classes
            num_classes = len(self.id2label)
            top_k = min(5, num_classes)
            top_k_predictions = torch.topk(predictions, k=top_k)

            top_classes = []
            for i in range(top_k):
                class_idx = top_k_predictions.indices[0][i].item()
                confidence = top_k_predictions.values[0][i].item()
                class_name = self.id2label.get(class_idx, f"class_{class_idx}")

                top_classes.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "class_id": class_idx
                })

            result = {
                "predictions": predictions.cpu().numpy().tolist(),
                "top_class": top_classes[0]["class"],
                "top_k_classes": top_classes,
                "frames_processed": len(clip) if isinstance(clip, list) else (len(clip[0]) if torch.is_tensor(clip) else frames_per_clip),
                "device_used": self.device,
                "video_duration": duration
            }

            logger.info(f"V-JEPA2 prediction complete. Top class: {result['top_class']}")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise e