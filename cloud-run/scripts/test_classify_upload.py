#!/usr/bin/env python3
"""
Test the /classify-upload endpoint of the deployed V-JEPA2 service
"""
import requests
import json
import subprocess
import sys
import time

# Service configuration
SERVICE_URL = "https://vjepa2-classifier-7wzotwquka-uc.a.run.app"
VIDEO_FILE = "ntu_rgb/S001C001P001R001A001_rgb.avi"

def get_auth_token():
    """Get authentication token for Cloud Run service"""
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print(f"Error getting auth token: {result.stderr}")
        sys.exit(1)

def test_classify_upload():
    """Test the /classify-upload endpoint with a video file"""

    print("=" * 60)
    print("Testing V-JEPA2 Video Classification Service")
    print("=" * 60)

    # Get authentication token
    print("\n1. Getting authentication token...")
    token = get_auth_token()
    print("   ✓ Token obtained")

    # Prepare the request
    headers = {"Authorization": f"Bearer {token}"}

    # Open and send the video file
    print(f"\n2. Uploading video file: {VIDEO_FILE}")
    print(f"   File size: 1.9 MB")

    try:
        with open(VIDEO_FILE, 'rb') as video_file:
            files = {'file': ('test_video.avi', video_file, 'video/avi')}
            data = {'frames_per_clip': '16'}

            # Make the request
            print("\n3. Sending request to /classify-upload endpoint...")
            print("   This may take 30-60 seconds for CPU processing...")

            start_time = time.time()
            response = requests.post(
                f"{SERVICE_URL}/classify-upload",
                headers=headers,
                files=files,
                data=data,
                timeout=180  # 3 minutes timeout
            )
            elapsed_time = time.time() - start_time

            print(f"   ✓ Response received in {elapsed_time:.2f} seconds")

            # Check response
            if response.status_code == 200:
                print("\n4. Classification Results:")
                print("   " + "=" * 40)

                result = response.json()

                # Display top class
                print(f"\n   Top Prediction: {result['top_class']}")

                # Display top K classes with confidence
                print("\n   Top 4 Classes:")
                for i, pred in enumerate(result['top_k_classes'], 1):
                    confidence_bar = "█" * int(pred['confidence'] * 20)
                    print(f"   {i}. {pred['class']:<15} {pred['confidence']:.1%} {confidence_bar}")

                # Display metadata
                print(f"\n   Processing Details:")
                print(f"   - Frames processed: {result['frames_processed']}")
                print(f"   - Video duration: {result['video_duration']:.2f}s")
                print(f"   - Processing time: {result['processing_time']:.2f}s")
                print(f"   - Device used: {result['device_used']}")

                # Raw predictions for debugging
                print(f"\n   Raw predictions: {result['predictions'][0]}")

                print("\n" + "=" * 60)
                print("✅ Test completed successfully!")

            else:
                print(f"\n❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")

    except requests.exceptions.Timeout:
        print("\n❌ Error: Request timed out (exceeded 3 minutes)")
        print("   The service may be processing a cold start or experiencing high load")

    except FileNotFoundError:
        print(f"\n❌ Error: Video file not found: {VIDEO_FILE}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    test_classify_upload()