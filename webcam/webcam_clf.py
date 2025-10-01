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
import subprocess
import numpy as np
import threading
from queue import Queue, Empty
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Any, Union, List


class WebcamMonitor:
    def __init__(self, config_path: str = "webcam/config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.camera = None
        self.running = False
        self.auth_token = None

        # Performance profiling
        self.enable_profiling = self.config.get('profiling', {}).get('enabled', False)
        self.profiling_stats = {
            'capture_times': [],
            'upload_times': [],
            'network_times': [],
            'processing_times': [],
            'cleanup_times': [],
            'total_times': []
        }

        # Performance optimizations
        self.use_memory_buffer = self.config.get('optimizations', {}).get('use_memory_buffer', False)
        self.use_raw_frames = self.config.get('optimizations', {}).get('use_raw_frames', False)
        self.use_parallel_processing = self.config.get('optimizations', {}).get('use_parallel_processing', False)

        # Threading components for parallel processing
        self.capture_queue = Queue(maxsize=2) if self.use_parallel_processing else None
        self.result_queue = Queue(maxsize=2) if self.use_parallel_processing else None
        self.capture_thread = None
        self.inference_thread = None

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

    def _format_classification_table(self, result: Dict[str, Any],
                                     capture_time: float,
                                     network_time: float,
                                     processing_time: float,
                                     total_time: float) -> str:
        """Format classification results as a table with highlighted max confidence"""
        top_k_classes = result.get('top_k_classes', [])

        if not top_k_classes:
            return "No classification results"

        # Sort alphabetically by class name
        sorted_classes = sorted(top_k_classes, key=lambda x: x.get('class', ''))

        # Find max confidence
        max_confidence = max(cls.get('confidence', 0) for cls in sorted_classes)

        # Build table
        lines = []
        lines.append("┌─────────────────────────────────────┬────────────┐")
        lines.append("│ Class                               │ Confidence │")
        lines.append("├─────────────────────────────────────┼────────────┤")

        for cls in sorted_classes:
            class_name = cls.get('class', 'unknown')
            confidence = cls.get('confidence', 0)

            # Highlight max confidence with star
            marker = "★" if confidence == max_confidence else " "

            # Format with padding - ensure consistent width
            class_str = f"{marker} {class_name}"
            # Truncate or pad to exactly 35 characters
            if len(class_str) > 35:
                class_display = class_str[:35]
            else:
                class_display = class_str.ljust(35)

            confidence_display = f"{confidence:.1%}".rjust(9)

            lines.append(f"│ {class_display} │ {confidence_display}  │")

        lines.append("└─────────────────────────────────────┴────────────┘")
        lines.append("")

        # Add timing information
        lines.append(f"Timings: Capture: {capture_time:.3f}s | Network: {network_time:.3f}s | "
                    f"GPU: {processing_time:.3f}s | Total: {total_time:.3f}s")

        return "\n".join(lines)

    def _get_auth_token(self) -> str:
        """Get authentication token for Cloud Run service"""
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-identity-token"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get auth token: {e.stderr}")
            raise

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
        target_fps = camera_config.get('fps', 30)
        target_width = camera_config.get('width', 640)
        target_height = camera_config.get('height', 480)

        # Set properties multiple times for better compatibility
        for _ in range(3):
            self.camera.set(cv2.CAP_PROP_FPS, target_fps)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

        # Verify final settings
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If camera doesn't support target resolution, we'll resize in capture
        self.needs_resize = (actual_width != target_width or actual_height != target_height)
        self.target_width = target_width
        self.target_height = target_height

        self.logger.info(f"Camera initialized at index {camera_index}: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        if self.needs_resize:
            self.logger.info(f"Will resize frames to {target_width}x{target_height} (camera doesn't support target resolution)")

        return True

    def capture_video(self, duration: float) -> tuple:
        """Capture video for specified duration and return temp file path/buffer/frames and capture time"""
        if not self.camera or not self.camera.isOpened():
            raise RuntimeError("Camera not initialized")

        capture_start = time.time()

        # Get camera properties
        fps = self.camera.get(cv2.CAP_PROP_FPS)

        # Use target resolution for output
        output_width = self.target_width
        output_height = self.target_height

        frames_to_capture = int(fps * duration)
        frames_captured = 0

        self.logger.debug(f"Starting video capture: {frames_to_capture} frames at {fps} fps")

        if self.use_raw_frames:
            # Capture frames directly as numpy array (no video encoding)
            frames = []
            while frames_captured < frames_to_capture and self.running:
                ret, frame = self.camera.read()
                if ret:
                    # Resize frame if needed
                    if self.needs_resize:
                        frame = cv2.resize(frame, (output_width, output_height))

                    # Convert BGR to RGB (model expects RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frames_captured += 1
                else:
                    self.logger.warning("Failed to capture frame")
                    break

            # Stack into numpy array
            frames_array = np.array(frames, dtype=np.uint8)

            capture_time = time.time() - capture_start

            if self.enable_profiling:
                self.profiling_stats['capture_times'].append(capture_time)
                self.logger.debug(f"Frames captured to memory: {frames_captured} frames in {capture_time:.3f}s")

            return frames_array, capture_time

        elif self.use_memory_buffer:
            # Capture frames to memory
            frames = []
            while frames_captured < frames_to_capture and self.running:
                ret, frame = self.camera.read()
                if ret:
                    # Resize frame if needed
                    if self.needs_resize:
                        frame = cv2.resize(frame, (output_width, output_height))
                    frames.append(frame)
                    frames_captured += 1
                else:
                    self.logger.warning("Failed to capture frame")
                    break

            # Encode to MP4 in memory
            temp_path = tempfile.mktemp(suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (output_width, output_height))
            for frame in frames:
                out.write(frame)
            out.release()

            # Read encoded video into memory buffer
            with open(temp_path, 'rb') as f:
                buffer = BytesIO(f.read())
            os.unlink(temp_path)  # Clean up temp file immediately

            capture_time = time.time() - capture_start

            if self.enable_profiling:
                self.profiling_stats['capture_times'].append(capture_time)
                self.logger.debug(f"Video captured to memory: {frames_captured} frames in {capture_time:.3f}s")

            return buffer, capture_time

        else:
            # Original file-based approach
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{self.config['capture']['format']}",
                prefix=self.config['capture']['temp_prefix'],
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (output_width, output_height))

            while frames_captured < frames_to_capture and self.running:
                ret, frame = self.camera.read()
                if ret:
                    # Resize frame if needed
                    if self.needs_resize:
                        frame = cv2.resize(frame, (output_width, output_height))

                    out.write(frame)
                    frames_captured += 1
                else:
                    self.logger.warning("Failed to capture frame")
                    break

            out.release()
            capture_time = time.time() - capture_start

            if self.enable_profiling:
                self.profiling_stats['capture_times'].append(capture_time)
                self.logger.debug(f"Video captured: {frames_captured} frames in {capture_time:.3f}s")

            return temp_path, capture_time

    def send_to_service(self, video_source: Union[str, BytesIO, np.ndarray]) -> tuple:
        """Send video/frames to vjepa2 service for classification and return result with timing

        Args:
            video_source: Either a file path (str), BytesIO buffer, or numpy array of frames
        """
        service_config = self.config.get('service', {})
        mode = service_config.get('mode', 'local')

        # Select URL based on mode
        if mode == 'cloud':
            base_url = service_config.get('cloud_url')
        else:
            base_url = service_config.get('local_url', 'http://localhost:8080')

        # Select endpoint based on data type
        if isinstance(video_source, np.ndarray):
            endpoint = '/classify-frames'
        else:
            endpoint = service_config['endpoint']

        url = f"{base_url}{endpoint}"
        timeout = service_config.get('timeout', 30)

        upload_start = time.time()
        file_read_time = 0
        network_time = 0

        try:
            # Only use authentication for Cloud Run
            headers = {}
            if mode == 'cloud':
                if self.auth_token is None:
                    self.auth_token = self._get_auth_token()
                    self.logger.info("Obtained authentication token")
                headers = {"Authorization": f"Bearer {self.auth_token}"}

            # Get file content from source
            read_start = time.time()
            if isinstance(video_source, np.ndarray):
                # Convert numpy array to bytes
                buffer = BytesIO()
                np.save(buffer, video_source)
                buffer.seek(0)
                file_content = buffer.read()
                filename = 'webcam_frames.npy'
                content_type = 'application/octet-stream'
            elif isinstance(video_source, BytesIO):
                video_source.seek(0)  # Reset to beginning
                file_content = video_source.read()
                filename = 'webcam_capture.mp4'
                content_type = 'video/mp4'
            else:
                with open(video_source, 'rb') as f:
                    file_content = f.read()
                filename = os.path.basename(video_source)
                content_type = 'video/mp4'
            file_read_time = time.time() - read_start

            # Get frames_per_clip from config
            frames_per_clip = self.config.get('capture', {}).get('frames_per_clip', 16)

            # Measure network time
            files = {'file': (filename, file_content, content_type)}
            data = {'frames_per_clip': frames_per_clip}

            self.logger.debug(f"Sending to {url} (mode: {mode}, type: {type(video_source).__name__})")
            network_start = time.time()
            response = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
            network_time = time.time() - network_start

            response.raise_for_status()

            result = response.json()
            upload_time = time.time() - upload_start

            if self.enable_profiling:
                self.profiling_stats['upload_times'].append(file_read_time)
                self.profiling_stats['network_times'].append(network_time)
                self.profiling_stats['processing_times'].append(result.get('processing_time', 0))

            return result, upload_time, file_read_time, network_time

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send video to service: {e}")
            return None, 0, file_read_time, network_time
        except Exception as e:
            self.logger.error(f"Unexpected error sending video: {e}")
            return None, 0, file_read_time, network_time

    def cleanup_temp_file(self, file_source: Union[str, BytesIO, np.ndarray]) -> float:
        """Clean up temporary video file/buffer/array and return cleanup time"""
        cleanup_start = time.time()
        try:
            if isinstance(file_source, np.ndarray):
                # Numpy arrays will be garbage collected
                pass
            elif isinstance(file_source, BytesIO):
                file_source.close()
            else:
                os.unlink(file_source)

            cleanup_time = time.time() - cleanup_start

            if self.enable_profiling:
                self.profiling_stats['cleanup_times'].append(cleanup_time)

            self.logger.debug(f"Cleaned up temp file/buffer")
            return cleanup_time
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp file: {e}")
            return time.time() - cleanup_start

    def _capture_worker(self, duration: float):
        """Worker thread for continuous video capture"""
        while self.running:
            try:
                video_source, capture_time = self.capture_video(duration)
                self.capture_queue.put((video_source, capture_time, time.time()))
            except Exception as e:
                self.logger.error(f"Error in capture worker: {e}")
                if self.running:
                    time.sleep(0.1)

    def _inference_worker(self):
        """Worker thread for processing captured videos"""
        while self.running:
            try:
                # Get captured video from queue with timeout
                video_source, capture_time, capture_end_time = self.capture_queue.get(timeout=1.0)

                # Send to service
                result, upload_time, file_read_time, network_time = self.send_to_service(video_source)

                # Clean up
                cleanup_time = self.cleanup_temp_file(video_source)

                # Calculate total time
                total_time = time.time() - (capture_end_time - capture_time)

                # Put result in queue
                self.result_queue.put((result, capture_time, file_read_time, network_time, cleanup_time, total_time))

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in inference worker: {e}")

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

        mode_str = "parallel" if self.use_parallel_processing else "sequential"
        self.logger.info(f"Monitor started ({mode_str} mode): {duration}s videos, {overlap}s overlap, {sleep_time}s between captures")

        try:
            if self.use_parallel_processing:
                # Start worker threads
                self.capture_thread = threading.Thread(target=self._capture_worker, args=(duration,), daemon=True)
                self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)

                self.capture_thread.start()
                self.inference_thread.start()

                # Main loop: just process results from inference thread
                while self.running:
                    try:
                        result, capture_time, file_read_time, network_time, cleanup_time, total_time = self.result_queue.get(timeout=1.0)

                        if self.enable_profiling:
                            self.profiling_stats['total_times'].append(total_time)

                        if result:
                            processing_time = result.get('processing_time', 0)

                            # Clear screen and print table
                            print("\033[2J\033[H")  # Clear screen and move cursor to top
                            table = self._format_classification_table(
                                result, capture_time, network_time, processing_time, total_time
                            )
                            print(table)

                    except Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing result: {e}")

            else:
                # Sequential mode (original)
                while self.running:
                    cycle_start = time.time()

                    try:
                        # Capture video
                        video_path, capture_time = self.capture_video(duration)

                        # Send to service
                        result, upload_time, file_read_time, network_time = self.send_to_service(video_path)

                        # Clean up
                        cleanup_time = self.cleanup_temp_file(video_path)

                        # Calculate total cycle time
                        cycle_time = time.time() - cycle_start

                        if self.enable_profiling:
                            self.profiling_stats['total_times'].append(cycle_time)

                        if result:
                            processing_time = result.get('processing_time', 0)

                            # Clear screen and print table
                            print("\033[2J\033[H")  # Clear screen and move cursor to top
                            table = self._format_classification_table(
                                result, capture_time, network_time, processing_time, cycle_time
                            )
                            print(table)

                    except Exception as e:
                        self.logger.error(f"Error in capture cycle: {e}")

                    # Wait for next capture (accounting for processing time)
                    capture_duration = time.time() - cycle_start
                    sleep_duration = max(0, sleep_time - capture_duration)

                    if sleep_duration > 0:
                        time.sleep(sleep_duration)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.shutdown()

    def _print_profiling_summary(self):
        """Print profiling statistics summary"""
        if not self.enable_profiling or not self.profiling_stats['total_times']:
            return

        from statistics import mean, median

        self.logger.info("=" * 70)
        self.logger.info("PERFORMANCE PROFILING SUMMARY")
        self.logger.info("=" * 70)

        total_cycles = len(self.profiling_stats['total_times'])
        self.logger.info(f"Total cycles: {total_cycles}")
        self.logger.info("")

        # Calculate averages
        avg_capture = mean(self.profiling_stats['capture_times']) if self.profiling_stats['capture_times'] else 0
        avg_upload = mean(self.profiling_stats['upload_times']) if self.profiling_stats['upload_times'] else 0
        avg_network = mean(self.profiling_stats['network_times']) if self.profiling_stats['network_times'] else 0
        avg_processing = mean(self.profiling_stats['processing_times']) if self.profiling_stats['processing_times'] else 0
        avg_cleanup = mean(self.profiling_stats['cleanup_times']) if self.profiling_stats['cleanup_times'] else 0
        avg_total = mean(self.profiling_stats['total_times'])

        self.logger.info("Average Times:")
        self.logger.info(f"  Capture (frames + encode): {avg_capture:.3f}s ({avg_capture/avg_total*100:.1f}%)")
        self.logger.info(f"  File Read:                 {avg_upload:.3f}s ({avg_upload/avg_total*100:.1f}%)")
        self.logger.info(f"  Network (upload + download): {avg_network:.3f}s ({avg_network/avg_total*100:.1f}%)")
        self.logger.info(f"  GPU Processing:            {avg_processing:.3f}s ({avg_processing/avg_total*100:.1f}%)")
        self.logger.info(f"  File Cleanup:              {avg_cleanup:.3f}s ({avg_cleanup/avg_total*100:.1f}%)")
        self.logger.info(f"  Total Cycle Time:          {avg_total:.3f}s")
        self.logger.info("")

        # Median times
        med_total = median(self.profiling_stats['total_times'])
        self.logger.info(f"Median cycle time: {med_total:.3f}s")
        self.logger.info(f"Min cycle time: {min(self.profiling_stats['total_times']):.3f}s")
        self.logger.info(f"Max cycle time: {max(self.profiling_stats['total_times']):.3f}s")

        self.logger.info("=" * 70)

    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("Shutting down webcam monitor...")
        self.running = False

        # Wait for worker threads to finish
        if self.use_parallel_processing:
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=2.0)

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        # Print profiling summary if enabled
        self._print_profiling_summary()

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
