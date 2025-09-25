#!/bin/bash

# Deploy V-JEPA2 Video Classifier to Google Cloud Run

set -e

PROJECT_ID="dev-ml-794354"
SERVICE_NAME="vjepa2-classifier"
REGION="us-central1"
IMAGE_NAME="$REGION-docker.pkg.dev/$PROJECT_ID/vjepa2-repo/$SERVICE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Deploying V-JEPA2 Video Classifier to Cloud Run${NC}"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install it first.${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}🔧 Setting project to: $PROJECT_ID${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔌 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Artifact Registry repository
echo -e "${YELLOW}📦 Creating Artifact Registry repository...${NC}"
if ! gcloud artifacts repositories describe vjepa2-repo --location=$REGION &> /dev/null; then
    gcloud artifacts repositories create vjepa2-repo \
        --repository-format=docker \
        --location=$REGION \
        --description="V-JEPA2 Video Classifier repository"
    echo -e "${GREEN}✅ Artifact Registry repository created${NC}"
else
    echo -e "${GREEN}✅ Artifact Registry repository already exists${NC}"
fi

# Check if we're in the right directory
if [ ! -f "service/Dockerfile" ]; then
    echo -e "${RED}❌ Error: Please run this script from the vjepa2-cloud-run directory${NC}"
    echo "Expected files: service/Dockerfile, service/main.py"
    exit 1
fi

# Build and push the container image
echo -e "${YELLOW}🔨 Building container image...${NC}"
echo -e "${BLUE}Image: $IMAGE_NAME${NC}"

cd service

# Build the image using Cloud Build
gcloud builds submit \
    --tag $IMAGE_NAME \
    --timeout=30m \
    .

echo -e "${GREEN}✅ Container image built and pushed${NC}"

cd ..

# Create service account for the Cloud Run service
echo -e "${YELLOW}👤 Setting up service account...${NC}"
SERVICE_ACCOUNT="vjepa2-sa"
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com"

# Create service account if it doesn't exist
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL &> /dev/null; then
    gcloud iam service-accounts create $SERVICE_ACCOUNT \
        --display-name="V-JEPA2 Video Classifier Service Account" \
        --description="Service account for V-JEPA2 video classifier Cloud Run service"
    echo -e "${GREEN}✅ Service account created${NC}"
else
    echo -e "${GREEN}✅ Service account already exists${NC}"
fi

# Grant necessary permissions to the service account
echo -e "${YELLOW}🔑 Granting permissions...${NC}"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/logging.logWriter"

echo -e "${GREEN}✅ Permissions granted${NC}"

# Deploy to Cloud Run
echo -e "${YELLOW}☁️  Deploying to Cloud Run...${NC}"

gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --service-account $SERVICE_ACCOUNT_EMAIL \
    --memory 32Gi \
    --cpu 8 \
    --gpu 1 \
    --gpu-type nvidia-a100 \
    --min-instances 0 \
    --max-instances 4 \
    --timeout 3600 \
    --concurrency 1 \
    --no-allow-unauthenticated \
    --set-env-vars USE_GCS=true \
    --port 8080

echo -e "${GREEN}✅ Service deployed successfully${NC}"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo ""
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${BLUE}Service URL: $SERVICE_URL${NC}"
echo -e "${BLUE}Region: $REGION${NC}"
echo -e "${BLUE}Service Account: $SERVICE_ACCOUNT_EMAIL${NC}"
echo ""

# Display resource configuration
echo -e "${YELLOW}📊 Resource Configuration:${NC}"
echo -e "  - CPU: 8 vCPUs"
echo -e "  - Memory: 32 GiB"
echo -e "  - GPU: 1x NVIDIA A100"
echo -e "  - Min Instances: 0 (scales to zero)"
echo -e "  - Max Instances: 4"
echo -e "  - Timeout: 60 minutes"
echo ""

# Create a test script for the deployed service
echo -e "${YELLOW}📝 Creating test script for deployed service...${NC}"
cat > test_deployed.py << EOF
#!/usr/bin/env python3
"""
Test script for deployed V-JEPA2 Video Classifier on Cloud Run
"""
import requests
import json
import os

SERVICE_URL = "$SERVICE_URL"

def get_auth_token():
    \"\"\"Get authentication token for Cloud Run service\"\"\"
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
EOF

chmod +x test_deployed.py

echo -e "${GREEN}✅ Test script created: test_deployed.py${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Test the deployed service: python3 test_deployed.py"
echo -e "  2. Upload test videos to GCS: gsutil cp video.mp4 gs://vjepa2/raw-videos/"
echo -e "  3. Monitor logs: gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME' --limit=50"
echo ""
echo -e "${BLUE}📚 Useful commands:${NC}"
echo -e "  - View service details: gcloud run services describe $SERVICE_NAME --region=$REGION"
echo -e "  - Update service: ./scripts/deploy.sh (re-run this script)"
echo -e "  - Delete service: gcloud run services delete $SERVICE_NAME --region=$REGION"