#!/usr/bin/env python3
"""
Debug test for the classify-upload endpoint with different parameters
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

def test_with_params(frames_per_clip):
    """Test the /classify-upload endpoint with specific parameters"""

    print(f"\nTesting with frames_per_clip={frames_per_clip}")
    print("-" * 40)

    # Get authentication token
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        with open(VIDEO_FILE, 'rb') as video_file:
            files = {'file': ('test_video.avi', video_file, 'video/avi')}
            data = {'frames_per_clip': str(frames_per_clip)}

            # Make the request
            print(f"Sending request...")
            start_time = time.time()
            response = requests.post(
                f"{SERVICE_URL}/classify-upload",
                headers=headers,
                files=files,
                data=data,
                timeout=180
            )
            elapsed_time = time.time() - start_time

            print(f"Response time: {elapsed_time:.2f}s")

            # Check response
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"   Top class: {result['top_class']}")
                print(f"   Confidence: {result['top_k_classes'][0]['confidence']:.1%}")
                print(f"   Frames processed: {result['frames_processed']}")
                return True
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Message: {response.text}")
                return False

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def check_video_info():
    """Check video information using ffprobe"""
    print("\nChecking video information...")
    print("-" * 40)

    try:
        # Try to get video info with ffprobe
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", VIDEO_FILE],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            info = json.loads(result.stdout)
            stream = info['streams'][0] if info.get('streams') else {}

            print(f"Video codec: {stream.get('codec_name', 'Unknown')}")
            print(f"Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")
            print(f"Frame rate: {stream.get('r_frame_rate', 'Unknown')}")
            print(f"Duration: {info.get('format', {}).get('duration', 'Unknown')}s")
            print(f"Format: {info.get('format', {}).get('format_name', 'Unknown')}")
        else:
            print("ffprobe not available or failed")

    except Exception as e:
        print(f"Could not get video info: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Debug Test for V-JEPA2 Video Classification")
    print("=" * 60)

    # Check video info
    check_video_info()

    # Try with different frame counts
    print("\nTrying different frame counts...")
    print("=" * 40)

    for frames in [8, 4, 2]:
        success = test_with_params(frames)
        if success:
            print(f"\n✅ Found working configuration: frames_per_clip={frames}")
            break

    print("\n" + "=" * 60)