#!/usr/bin/env python3
"""
Test script for deployed V-JEPA2 Video Classifier on Cloud Run
"""
import requests
import json
import os

SERVICE_URL = "https://vjepa2-classifier-7wzotwquka-uc.a.run.app"

def get_auth_token():
    """Get authentication token for Cloud Run service"""
    import subprocess
    result = subprocess.run([
        "gcloud", "auth", "print-identity-token"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print("Error getting auth token:", result.stderr)
        return None

def test_service():
    token = get_auth_token()
    if not token:
        return False

    headers = {"Authorization": f"Bearer {token}"}

    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{SERVICE_URL}/health", headers=headers)
    print(f"Health Status: {response.status_code}")
    if response.status_code == 200:
        print("Health Response:", json.dumps(response.json(), indent=2))

    return response.status_code == 200

if __name__ == "__main__":
    success = test_service()
    print("✅ Service is accessible!" if success else "❌ Service test failed")
