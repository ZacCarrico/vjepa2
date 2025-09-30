#!/usr/bin/env python3
"""
Test the locally running Docker container
"""
import requests
import json
import time

# Service configuration
SERVICE_URL = "http://localhost:8080"
VIDEO_FILE = "test_video.mp4"  # Small test video

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{SERVICE_URL}/health")
    print("Health check:", response.json())
    return response.status_code == 200

def test_classify_upload():
    """Test the /classify-upload endpoint"""
    print("\n" + "=" * 60)
    print("Testing Video Classification (Docker Container)")
    print("=" * 60)

    try:
        # Test health first
        if not test_health():
            print("❌ Health check failed")
            return

        # Upload video
        print(f"\nUploading video: {VIDEO_FILE}")
        with open(VIDEO_FILE, 'rb') as video_file:
            files = {'file': ('test_video.mp4', video_file, 'video/mp4')}
            data = {'frames_per_clip': '16'}

            start_time = time.time()
            response = requests.post(
                f"{SERVICE_URL}/classify-upload",
                files=files,
                data=data,
                timeout=60
            )
            elapsed_time = time.time() - start_time

            print(f"Response time: {elapsed_time:.2f}s")

            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ SUCCESS! Classification Results:")
                print(f"   Top class: {result['top_class']}")
                print(f"   Confidence: {result['top_k_classes'][0]['confidence']:.1%}")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                print(f"   Frames processed: {result['frames_processed']}")
                print(f"   Device: {result['device_used']}")

                print(f"\n   All predictions:")
                for pred in result['top_k_classes']:
                    bar = "█" * int(pred['confidence'] * 20)
                    print(f"   - {pred['class']:<15} {pred['confidence']:.1%} {bar}")

            else:
                print(f"\n❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_classify_upload()