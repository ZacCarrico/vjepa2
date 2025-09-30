#!/usr/bin/env python3
"""Test video processing locally to compare with Cloud Run"""

import sys
import os
sys.path.insert(0, 'cloud-run/service')

from model_handler import VideoClassifier

# Test with the same videos that fail on Cloud Run
test_videos = [
    "cloud-run/test_video.mp4",
    "ntu_rgb/S001C001P001R001A001_rgb.avi"
]

print("Testing video processing locally...")
print("=" * 50)

# Initialize classifier
print("Initializing classifier...")
classifier = VideoClassifier(use_gcs=False)
print(f"✓ Classifier initialized (device: {classifier.device})")
print()

for video_path in test_videos:
    if os.path.exists(video_path):
        print(f"Testing: {video_path}")
        print("-" * 40)
        try:
            result = classifier.predict(video_path, frames_per_clip=16)
            print(f"✓ SUCCESS!")
            print(f"  Top class: {result['top_class']}")
            print(f"  Confidence: {result['top_k_classes'][0]['confidence']:.1%}")
            print(f"  Frames processed: {result['frames_processed']}")
            print()
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            print(f"  Error type: {type(e).__name__}")
            print()
    else:
        print(f"✗ File not found: {video_path}")
        print()

print("=" * 50)
print("Local testing complete")