#!/usr/bin/env python3
"""
Webcam Monitor Service
Continuously captures 3-second videos with 1-second overlap and sends to vjepa2 service
"""

import cv2
import os
import time
import tempfile
import logging
import yaml
import requests
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any


class WebcamMonitor:
    def __init__(self, config_path: str = "webcam/config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.camera = None
        self.running = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_config.get('file', 'webcam/webcam_clf.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def find_logitech_brio(self) -> Optional[int]:
        """Find Logitech Brio camera index"""
        camera_config = self.config.get('camera', {})

        # If specific index is configured, use it
        if camera_config.get('index') is not None:
            return camera_config['index']

        # Auto-detect Logitech Brio
        self.logger.info("Auto-detecting Logitech Brio camera...")

        for i in range(10):  # Check first 10 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to get camera name/info if available
                    ret, frame = cap.read()
                    if ret:
                        self.logger.info(f"Found camera at index {i}")
                        cap.release()
                        # For now, return the first working camera
                        # In practice, you might want to add device name detection
                        return i
                cap.release()
            except Exception as e:
                continue

        self.logger.warning("No Logitech Brio found, using default camera (index 0)")
        return 0

    def initialize_camera(self) -> bool:
        """Initialize camera with configured settings"""
        camera_index = self.find_logitech_brio()
        if camera_index is None:
            self.logger.error("No camera found")
            return False

        # Try multiple times and backends if needed
        backends_to_try = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION]

        for attempt, backend in enumerate(backends_to_try):
            self.logger.info(f"Attempting to open camera {camera_index} (attempt {attempt + 1}/2)")

            self.camera = cv2.VideoCapture(camera_index, backend)
            if not self.camera.isOpened():
                self.logger.warning(f"Failed to open camera at index {camera_index} with backend {backend}")
                continue

            # Test if we can actually capture frames
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                self.logger.warning(f"Camera {camera_index} opens but cannot capture frames (attempt {attempt + 1})")
                self.camera.release()
                time.sleep(1)  # Wait before retry
                continue

            self.logger.info(f"Successfully captured test frame from camera {camera_index}")
            break
        else:
            self.logger.error(f"Failed to initialize working camera at index {camera_index}")
            return False

        # Set camera properties
        camera_config = self.config.get('camera', {})
        self.camera.set(cv2.CAP_PROP_FPS, camera_config.get('fps', 30))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('width', 640))
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('height', 480))

        # Verify final settings
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(f"Camera initialized at index {camera_index}: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        return True

    def capture_video(self, duration: float) -> str:
        """Capture video for specified duration and return temp file path"""
        if not self.camera or not self.camera.isOpened():
            raise RuntimeError("Camera not initialized")

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f".{self.config['capture']['format']}",
            prefix=self.config['capture']['temp_prefix'],
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()

        # Get camera properties
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        frames_to_capture = int(fps * duration)
        frames_captured = 0

        self.logger.debug(f"Starting video capture: {frames_to_capture} frames at {fps} fps")

        start_time = time.time()
        while frames_captured < frames_to_capture and self.running:
            ret, frame = self.camera.read()
            if ret:
                out.write(frame)
                frames_captured += 1
            else:
                self.logger.warning("Failed to capture frame")
                break

        out.release()
        actual_duration = time.time() - start_time

        self.logger.debug(f"Video captured: {frames_captured} frames in {actual_duration:.2f}s")
        return temp_path

    def send_to_service(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Send video to vjepa2 service for classification"""
        service_config = self.config.get('service', {})
        url = f"{service_config['url']}{service_config['endpoint']}"
        timeout = service_config.get('timeout', 30)

        try:
            with open(video_path, 'rb') as f:
                files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
                data = {'frames_per_clip': 16}

                self.logger.debug(f"Sending video to {url}")
                response = requests.post(url, files=files, data=data, timeout=timeout)
                response.raise_for_status()

                result = response.json()
                self.logger.info(f"Classification result: {result.get('top_class', 'unknown')}")
                return result

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send video to service: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error sending video: {e}")
            return None

    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary video file"""
        try:
            os.unlink(file_path)
            self.logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp file {file_path}: {e}")

    def run(self):
        """Main monitoring loop"""
        self.logger.info("Starting webcam monitor...")

        if not self.initialize_camera():
            self.logger.error("Failed to initialize camera, exiting")
            return

        self.running = True
        capture_config = self.config.get('capture', {})
        duration = capture_config.get('duration', 3)
        overlap = capture_config.get('overlap', 1)

        # Calculate sleep time between captures (duration - overlap)
        sleep_time = duration - overlap

        self.logger.info(f"Monitor started: {duration}s videos, {overlap}s overlap, {sleep_time}s between captures")

        try:
            while self.running:
                capture_start = time.time()

                try:
                    # Capture video
                    video_path = self.capture_video(duration)

                    # Send to service
                    result = self.send_to_service(video_path)

                    if result:
                        self.logger.info(
                            f"Processed video - Top class: {result.get('top_class')}, "
                            f"Confidence: {result.get('top_k_classes', [{}])[0].get('confidence', 0):.3f}, "
                            f"Processing time: {result.get('processing_time', 0):.2f}s"
                        )

                    # Clean up
                    self.cleanup_temp_file(video_path)

                except Exception as e:
                    self.logger.error(f"Error in capture cycle: {e}")

                # Wait for next capture (accounting for processing time)
                capture_duration = time.time() - capture_start
                sleep_duration = max(0, sleep_time - capture_duration)

                if sleep_duration > 0:
                    time.sleep(sleep_duration)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("Shutting down webcam monitor...")
        self.running = False

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()
        self.logger.info("Webcam monitor shut down complete")


def main():
    """Main entry point"""
    try:
        monitor = WebcamMonitor()
        monitor.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
