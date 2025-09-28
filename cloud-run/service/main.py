import os
import tempfile
import logging
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the real V-JEPA2 model, fall back to mock if not available
try:
    from model_handler import VideoClassifier
    logger.info("Using real V-JEPA2 model handler")
except ImportError as e:
    logger.warning(f"V-JEPA2 dependencies not available ({e}), falling back to mock")
    from model_handler_mock import VideoClassifier

# Conditional import for GCS (allows local testing without GCP credentials)
USE_GCS = os.getenv("USE_GCS", "false").lower() == "true"
logger.info(f"GCS enabled: {USE_GCS}")

if USE_GCS:
    try:
        from gcs_utils import download_from_gcs, upload_to_gcs
        logger.info("GCS utilities loaded successfully")
    except ImportError as e:
        logger.warning(f"Failed to import GCS utilities: {e}")
        USE_GCS = False

# Initialize FastAPI app
app = FastAPI(
    title="V-JEPA2 Video Classifier",
    description="Video classification API using V-JEPA2 model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier (this will happen at startup)
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global classifier
    try:
        logger.info("Initializing V-JEPA2 classifier...")
        classifier = VideoClassifier()
        logger.info("Classifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        raise e

class VideoRequest(BaseModel):
    video_uri: str  # gs://bucket/path or local path
    frames_per_clip: int = 16

class VideoResponse(BaseModel):
    predictions: list
    top_class: str
    top_k_classes: list
    frames_processed: int
    device_used: str
    video_duration: float
    processing_time: float

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "V-JEPA2 Video Classifier API",
        "status": "running",
        "gcs_enabled": USE_GCS,
        "device": classifier.device if classifier else "not_initialized"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    return {
        "status": "healthy",
        "device": classifier.device,
        "gcs_enabled": USE_GCS,
        "model_ready": True
    }

@app.post("/classify", response_model=VideoResponse)
async def classify_video(request: VideoRequest):
    """Classify video from URI (local path or GCS)"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    start_time = time.time()

    try:
        # Handle GCS URIs
        if request.video_uri.startswith("gs://"):
            if not USE_GCS:
                raise HTTPException(
                    status_code=400,
                    detail="GCS not enabled. Set USE_GCS=true environment variable"
                )

            logger.info(f"Downloading video from GCS: {request.video_uri}")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                download_from_gcs(request.video_uri, tmp.name)
                result = classifier.predict(tmp.name, request.frames_per_clip)
                os.unlink(tmp.name)  # Clean up temp file
        else:
            # Local file path
            if not os.path.exists(request.video_uri):
                raise HTTPException(
                    status_code=404,
                    detail=f"Video file not found: {request.video_uri}"
                )

            logger.info(f"Processing local video: {request.video_uri}")
            result = classifier.predict(request.video_uri, request.frames_per_clip)

        # Add processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        logger.info(f"Classification completed in {processing_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error classifying video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-upload", response_model=VideoResponse)
async def classify_upload(
    file: UploadFile = File(...),
    frames_per_clip: int = 16
):
    """Direct file upload endpoint for easier testing"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    start_time = time.time()

    # Validate file type (allow common video extensions if content_type is not set)
    allowed_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
    if file.content_type and not file.content_type.startswith(('video/', 'application/')):
        if not file.filename or not file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected video file."
            )

    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name
            content = await file.read()
            tmp.write(content)
            tmp.flush()

            logger.info(f"Processing uploaded file: {file.filename} ({len(content)} bytes)")

            # Run prediction
            result = classifier.predict(temp_path, frames_per_clip)

            # Add metadata
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["original_filename"] = file.filename

            logger.info(f"Upload classification completed in {processing_time:.2f}s")
            return result

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/info")
async def get_info():
    """Get model and system information"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    import torch

    # Handle both real and mock models
    try:
        # Check if it's the real V-JEPA2 model
        if hasattr(classifier, 'model') and hasattr(classifier.model, 'config'):
            num_classes = len(classifier.model.config.id2label) if hasattr(classifier.model.config, 'id2label') else "unknown"
            model_name = "facebook/vjepa2-vitl-fpc16-256-ssv2"
        elif hasattr(classifier, 'config'):
            # Mock model
            num_classes = len(classifier.config["id2label"])
            model_name = "Mock V-JEPA2 (for testing)"
        else:
            num_classes = "unknown"
            model_name = "Unknown model"
    except Exception as e:
        logger.warning(f"Error getting model info: {e}")
        num_classes = "unknown"
        model_name = "Unknown model"

    return {
        "model": {
            "name": model_name,
            "device": classifier.device,
            "num_classes": num_classes
        },
        "system": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "config": {
            "gcs_enabled": USE_GCS,
            "max_upload_size": "100MB"  # Can be configured
        }
    }

if __name__ == "__main__":
    import uvicorn

    # For local development
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.getenv("ENVIRONMENT") == "development" else False
    )