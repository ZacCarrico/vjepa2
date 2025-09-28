#!/usr/bin/env python3
"""
Create a simple test video for testing the video classifier
"""

import cv2
import numpy as np

def create_test_video(output_path="tests/sample_video.mp4", duration_seconds=3, fps=10):
    """Create a simple test video with colorful frames"""

    # Video properties
    width, height = 224, 224
    total_frames = duration_seconds * fps

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration_seconds}s, FPS: {fps}, Frames: {total_frames}")

    for frame_num in range(total_frames):
        # Create a frame with changing colors
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a gradient effect
        progress = frame_num / total_frames

        # Red channel: sine wave
        red = int(128 + 127 * np.sin(progress * 4 * np.pi))
        # Green channel: cosine wave
        green = int(128 + 127 * np.cos(progress * 6 * np.pi))
        # Blue channel: linear progression
        blue = int(255 * progress)

        frame[:, :] = [blue, green, red]  # BGR format for OpenCV

        # Add some geometric shapes for visual interest
        center = (width // 2, height // 2)
        radius = int(50 + 30 * np.sin(progress * 8 * np.pi))
        cv2.circle(frame, center, radius, (255, 255, 255), 2)

        # Add frame number text
        cv2.putText(frame, f'Frame {frame_num}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ Test video created successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_video()