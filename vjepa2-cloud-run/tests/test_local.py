#!/usr/bin/env python3
"""
Local testing script for V-JEPA2 Video Classifier API
Run this script to test the API endpoints locally
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8080"
TEST_VIDEO_PATH = "tests/sample_video.mp4"

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def test_health_endpoint():
    """Test the health check endpoint"""
    print_section("Health Check")

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"Device: {data.get('device', 'unknown')}")
            print(f"Model Ready: {data.get('model_ready', False)}")
            print(f"GCS Enabled: {data.get('gcs_enabled', False)}")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running?")
        print(f"   Start the server with: ./scripts/run_local.sh")
        return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_info_endpoint():
    """Test the info endpoint"""
    print_section("System Info")

    try:
        response = requests.get(f"{BASE_URL}/info", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print("✅ Info endpoint working!")
            print("Model Info:")
            model_info = data.get('model', {})
            for key, value in model_info.items():
                print(f"  - {key}: {value}")

            print("\nSystem Info:")
            system_info = data.get('system', {})
            for key, value in system_info.items():
                print(f"  - {key}: {value}")

            return True
        else:
            print(f"❌ Info endpoint failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Info endpoint error: {str(e)}")
        return False

def create_sample_video():
    """Create a simple sample video for testing"""
    print("📹 Creating sample video for testing...")

    try:
        import cv2
        import numpy as np

        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(TEST_VIDEO_PATH, fourcc, 10.0, (224, 224))

        # Create 30 frames with different colors
        for i in range(30):
            # Create a frame with changing colors
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 8) % 256  # Red channel
            frame[:, :, 1] = (i * 4) % 256  # Green channel
            frame[:, :, 2] = (i * 2) % 256  # Blue channel

            # Add some text
            cv2.putText(frame, f'Frame {i}', (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        print(f"✅ Sample video created: {TEST_VIDEO_PATH}")
        return True

    except ImportError:
        print("⚠️  OpenCV not available for creating test video")
        print("   You can use any MP4 video file for testing")
        return False
    except Exception as e:
        print(f"❌ Error creating sample video: {str(e)}")
        return False

def test_file_upload():
    """Test the file upload endpoint"""
    print_section("File Upload Classification")

    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"⚠️  Test video not found: {TEST_VIDEO_PATH}")
        if not create_sample_video():
            print("❌ Cannot test file upload without a video file")
            return False

    try:
        print(f"📤 Uploading video file: {TEST_VIDEO_PATH}")

        with open(TEST_VIDEO_PATH, "rb") as video_file:
            files = {"file": video_file}
            params = {"frames_per_clip": 8}  # Use fewer frames for faster testing

            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/classify-upload",
                files=files,
                params=params,
                timeout=120  # Longer timeout for model loading
            )
            end_time = time.time()

        print(f"Status Code: {response.status_code}")
        print(f"Processing Time: {end_time - start_time:.2f} seconds")

        if response.status_code == 200:
            data = response.json()
            print("✅ File upload classification successful!")
            print(f"Top Class: {data.get('top_class', 'unknown')}")
            print(f"Device Used: {data.get('device_used', 'unknown')}")
            print(f"Frames Processed: {data.get('frames_processed', 0)}")
            print(f"Video Duration: {data.get('video_duration', 0):.2f}s")

            # Show top 3 predictions
            top_k = data.get('top_k_classes', [])[:3]
            print("\nTop 3 Predictions:")
            for i, pred in enumerate(top_k, 1):
                print(f"  {i}. {pred.get('class', 'unknown')}: {pred.get('confidence', 0):.4f}")

            return True
        else:
            print(f"❌ File upload failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ File upload error: {str(e)}")
        return False

def test_local_path_classification():
    """Test classification with local file path"""
    print_section("Local Path Classification")

    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"⚠️  Test video not found: {TEST_VIDEO_PATH}")
        return False

    try:
        # Convert to absolute path
        abs_path = os.path.abspath(TEST_VIDEO_PATH)
        print(f"📂 Testing with local path: {abs_path}")

        payload = {
            "video_uri": abs_path,
            "frames_per_clip": 8
        }

        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/classify",
            json=payload,
            timeout=120
        )
        end_time = time.time()

        print(f"Status Code: {response.status_code}")
        print(f"Processing Time: {end_time - start_time:.2f} seconds")

        if response.status_code == 200:
            data = response.json()
            print("✅ Local path classification successful!")
            print(f"Top Class: {data.get('top_class', 'unknown')}")
            print(f"Device Used: {data.get('device_used', 'unknown')}")
            print(f"Processing Time (API): {data.get('processing_time', 0):.2f}s")

            return True
        else:
            print(f"❌ Local path classification failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Local path classification error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🎬 V-JEPA2 Video Classifier - Local Testing Suite")
    print(f"Testing server at: {BASE_URL}")

    # Ensure test directory exists
    os.makedirs("tests", exist_ok=True)

    tests = [
        ("Health Check", test_health_endpoint),
        ("System Info", test_info_endpoint),
        ("File Upload", test_file_upload),
        ("Local Path", test_local_path_classification),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n🛑 Testing interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {str(e)}")
            results.append((test_name, False))

    # Summary
    print_section("Test Results Summary")
    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your local setup is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)