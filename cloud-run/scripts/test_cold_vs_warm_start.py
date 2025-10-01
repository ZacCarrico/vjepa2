#!/usr/bin/env python3
"""
Test script to compare cold start vs warm start performance
Sends multiple requests to the Cloud Run service and measures response times
"""
import requests
import subprocess
import sys
import time
from statistics import mean, median, stdev

# Service configuration
SERVICE_URL = "https://vjepa2-classifier-7wzotwquka-uc.a.run.app"
VIDEO_FILE = "cloud-run/test_video.mp4"
NUM_REQUESTS = 5

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

def send_request(token, request_num):
    """Send a single classification request and measure time"""
    headers = {"Authorization": f"Bearer {token}"}

    try:
        with open(VIDEO_FILE, 'rb') as video_file:
            files = {'file': ('test_video.mp4', video_file, 'video/mp4')}
            data = {'frames_per_clip': '16'}

            start_time = time.time()
            response = requests.post(
                f"{SERVICE_URL}/classify-upload",
                headers=headers,
                files=files,
                data=data,
                timeout=180
            )
            elapsed_time = time.time() - start_time

            success = response.status_code == 200

            if success:
                result = response.json()
                return {
                    'request_num': request_num,
                    'success': True,
                    'elapsed_time': elapsed_time,
                    'processing_time': result.get('processing_time', 0),
                    'top_class': result.get('top_class', 'unknown'),
                    'confidence': result.get('top_k_classes', [{}])[0].get('confidence', 0)
                }
            else:
                return {
                    'request_num': request_num,
                    'success': False,
                    'elapsed_time': elapsed_time,
                    'error': f"HTTP {response.status_code}: {response.text[:100]}"
                }

    except Exception as e:
        return {
            'request_num': request_num,
            'success': False,
            'elapsed_time': 0,
            'error': str(e)
        }

def main():
    print("=" * 70)
    print("Cold Start vs Warm Start Performance Test")
    print("=" * 70)
    print(f"\nService: {SERVICE_URL}")
    print(f"Video: {VIDEO_FILE}")
    print(f"Number of requests: {NUM_REQUESTS}")
    print()

    # Get authentication token
    print("Getting authentication token...")
    token = get_auth_token()
    print("✓ Token obtained\n")

    # Wait a moment to ensure any existing instances have scaled down
    print("Waiting 10 seconds to allow potential cold start...")
    time.sleep(10)
    print()

    # Send requests and collect results
    results = []

    for i in range(1, NUM_REQUESTS + 1):
        print(f"Request {i}/{NUM_REQUESTS}:")
        print("-" * 50)

        result = send_request(token, i)
        results.append(result)

        if result['success']:
            print(f"  ✓ Success")
            print(f"  - Total time: {result['elapsed_time']:.2f}s")
            print(f"  - Processing time: {result['processing_time']:.2f}s")
            print(f"  - Prediction: {result['top_class']} ({result['confidence']:.1%})")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

        print()

        # Small delay between requests
        if i < NUM_REQUESTS:
            time.sleep(2)

    # Analyze results
    print("=" * 70)
    print("Performance Analysis")
    print("=" * 70)
    print()

    successful_results = [r for r in results if r['success']]

    if len(successful_results) == 0:
        print("❌ No successful requests to analyze")
        return

    if len(successful_results) < NUM_REQUESTS:
        print(f"⚠️  Warning: Only {len(successful_results)}/{NUM_REQUESTS} requests succeeded")
        print()

    # Extract timing data
    total_times = [r['elapsed_time'] for r in successful_results]
    processing_times = [r['processing_time'] for r in successful_results]

    # First request (likely cold start)
    first_request = successful_results[0]
    print(f"First Request (Cold Start):")
    print(f"  - Total time: {first_request['elapsed_time']:.2f}s")
    print(f"  - Processing time: {first_request['processing_time']:.2f}s")
    print()

    # Subsequent requests (warm start)
    if len(successful_results) > 1:
        warm_times = total_times[1:]
        warm_processing = processing_times[1:]

        print(f"Subsequent Requests (Warm Start):")
        print(f"  - Average total time: {mean(warm_times):.2f}s")
        print(f"  - Median total time: {median(warm_times):.2f}s")
        if len(warm_times) > 1:
            print(f"  - Std deviation: {stdev(warm_times):.2f}s")
        print(f"  - Min: {min(warm_times):.2f}s")
        print(f"  - Max: {max(warm_times):.2f}s")
        print()

        print(f"  - Average processing time: {mean(warm_processing):.2f}s")
        print(f"  - Median processing time: {median(warm_processing):.2f}s")
        print()

        # Calculate speedup
        speedup = first_request['elapsed_time'] / mean(warm_times)
        improvement = ((first_request['elapsed_time'] - mean(warm_times)) / first_request['elapsed_time']) * 100

        print(f"Performance Improvement:")
        print(f"  - Speedup: {speedup:.2f}x faster")
        print(f"  - Time saved: {improvement:.1f}%")
        print()

        if speedup > 1.2:
            print("✅ Significant warm start advantage detected!")
        elif speedup > 1.05:
            print("✅ Modest warm start advantage detected")
        else:
            print("ℹ️  Similar performance between cold and warm starts")

    print()
    print("=" * 70)
    print("Detailed Results")
    print("=" * 70)
    print()
    print(f"{'Req':<5} {'Status':<10} {'Total (s)':<12} {'Processing (s)':<15} {'Prediction':<15}")
    print("-" * 70)

    for r in results:
        if r['success']:
            print(f"{r['request_num']:<5} {'✓ Success':<10} {r['elapsed_time']:<12.2f} "
                  f"{r['processing_time']:<15.2f} {r['top_class']:<15}")
        else:
            print(f"{r['request_num']:<5} {'✗ Failed':<10} {'-':<12} {'-':<15} {'-':<15}")

    print()

if __name__ == "__main__":
    main()
