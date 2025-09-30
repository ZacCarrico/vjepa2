#!/bin/bash

# Deploy V-JEPA2 Video Classifier to Google Cloud Run
# Usage: ./deploy.sh [step]
# Steps: setup, build, deploy, test, all (default)

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

# Parse command line arguments
STEP=${1:-all}

# Show help if requested
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
    echo -e "${BLUE}🚀 V-JEPA2 Video Classifier Deployment Script${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC} $0 [step]"
    echo ""
    echo -e "${YELLOW}Available steps:${NC}"
    echo -e "  ${GREEN}setup${NC}       - Set up GCP environment (APIs, Artifact Registry)"
    echo -e "  ${GREEN}build${NC}       - Build and push container image (Cloud Build)"
    echo -e "  ${GREEN}build-local${NC} - Build and push container image (local Docker)"
    echo -e "  ${GREEN}deploy${NC}      - Deploy service to Cloud Run (requires image to exist)"
    echo -e "  ${GREEN}test${NC}        - Create test script for deployed service"
    echo -e "  ${GREEN}all${NC}         - Run all steps sequentially (default)"
    echo -e "  ${GREEN}all-local${NC}   - Run all steps with local build (faster)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0                 # Run all steps with Cloud Build"
    echo -e "  $0 all-local       # Run all steps with local Docker build (faster)"
    echo -e "  $0 setup           # Only set up GCP environment"
    echo -e "  $0 build-local     # Only build and push image locally"
    echo -e "  $0 deploy          # Only deploy (after image is built)"
    echo ""
    echo -e "${YELLOW}Step-by-step workflow:${NC}"
    echo -e "  1. $0 setup     # Set up GCP project and dependencies"
    echo -e "  2. $0 build     # Build and push container image"
    echo -e "  3. $0 deploy    # Deploy to Cloud Run"
    echo -e "  4. $0 test      # Create test script"
    echo ""
    exit 0
fi

echo -e "${BLUE}🚀 Deploying V-JEPA2 Video Classifier to Cloud Run${NC}"
echo -e "${BLUE}Step: $STEP${NC}"

# Function definitions
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI not found. Please install it first.${NC}"
        exit 1
    fi

    # Check if we're in the right directory - allow both project root and cloud-run directory
    if [ -f "service/Dockerfile" ] && [ -f "service/main.py" ]; then
        # We're in cloud-run directory
        echo -e "${GREEN}✅ Running from cloud-run directory${NC}"
    elif [ -f "cloud-run/service/Dockerfile" ] && [ -f "cloud-run/service/main.py" ]; then
        # We're in project root, change to cloud-run directory
        echo -e "${GREEN}✅ Running from project root, changing to cloud-run directory${NC}"
        cd cloud-run
    else
        echo -e "${RED}❌ Error: Cannot find required files${NC}"
        echo "Expected files: service/Dockerfile, service/main.py"
        echo "Please run this script from either:"
        echo "  - Project root directory (where cloud-run/ exists)"
        echo "  - cloud-run/ directory (where service/ exists)"
        exit 1
    fi

    echo -e "${GREEN}✅ Prerequisites checked${NC}"
}

setup_gcp() {
    echo -e "${YELLOW}🔧 Setting up GCP environment...${NC}"

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

    echo -e "${GREEN}✅ GCP setup completed${NC}"
}

build_and_push() {
    local build_method=${1:-cloud}

    echo -e "${YELLOW}🔨 Building and pushing container image...${NC}"
    echo -e "${BLUE}Image: $IMAGE_NAME${NC}"
    echo -e "${BLUE}Build method: $build_method${NC}"

    cd service

    if [ "$build_method" = "local" ]; then
        # Build locally with Docker
        echo -e "${YELLOW}🐳 Building image locally...${NC}"
        docker build -t $IMAGE_NAME .

        echo -e "${YELLOW}📤 Pushing image to Artifact Registry...${NC}"
        docker push $IMAGE_NAME

        echo -e "${GREEN}✅ Container image built locally and pushed${NC}"
    else
        # Build the image using Cloud Build
        echo -e "${YELLOW}☁️ Building image with Cloud Build...${NC}"
        gcloud builds submit \
            --tag $IMAGE_NAME \
            --timeout=30m \
            .

        echo -e "${GREEN}✅ Container image built and pushed via Cloud Build${NC}"
    fi

    cd ..
}

setup_service_account() {
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
}

deploy_to_cloudrun() {
    echo -e "${YELLOW}☁️  Deploying to Cloud Run...${NC}"

    SERVICE_ACCOUNT="vjepa2-sa"
    SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com"

    # Track deployment type for later display
    DEPLOYMENT_TYPE=""

    # Try with GPU first (alpha), fallback to CPU if not available
    echo -e "${YELLOW}Attempting GPU deployment with L4...${NC}"

    # Capture the output to check for quota errors
    GPU_OUTPUT=$(gcloud alpha run deploy $SERVICE_NAME \
        --image $IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --service-account $SERVICE_ACCOUNT_EMAIL \
        --memory 32Gi \
        --cpu 8 \
        --gpu 1 \
        --gpu-type nvidia-l4 \
        --min-instances 0 \
        --max-instances 4 \
        --timeout 3600 \
        --concurrency 1 \
        --no-allow-unauthenticated \
        --set-env-vars USE_GCS=true \
        --port 8080 2>&1)

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ GPU deployment successful with L4${NC}"
        DEPLOYMENT_TYPE="GPU"
    else
        # Check if it's a quota error
        if echo "$GPU_OUTPUT" | grep -q "Quota exceeded"; then
            echo -e "${YELLOW}⚠️  GPU quota exceeded. Attempting CPU-only deployment...${NC}"
        else
            echo -e "${YELLOW}⚠️  GPU deployment failed. Attempting CPU-only deployment...${NC}"
            echo -e "${YELLOW}Error: ${GPU_OUTPUT}${NC}"
        fi

        # Deploy with CPU only - use lower memory for CPU-only deployment
        echo -e "${YELLOW}Deploying with CPU configuration (reduced memory)...${NC}"
        if gcloud run deploy $SERVICE_NAME \
            --image $IMAGE_NAME \
            --platform managed \
            --region $REGION \
            --service-account $SERVICE_ACCOUNT_EMAIL \
            --memory 16Gi \
            --cpu 4 \
            --min-instances 0 \
            --max-instances 2 \
            --timeout 3600 \
            --concurrency 1 \
            --no-allow-unauthenticated \
            --set-env-vars USE_GCS=true \
            --port 8080; then
            echo -e "${YELLOW}⚠️  Service deployed with CPU only (4 vCPUs, 16GB RAM)${NC}"
            echo -e "${YELLOW}Note: Performance will be slower without GPU acceleration${NC}"
            DEPLOYMENT_TYPE="CPU"
        else
            echo -e "${RED}❌ Deployment failed for both GPU and CPU configurations${NC}"
            echo -e "${RED}Please check your quotas and container image${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}✅ Service deployed successfully${NC}"

    # Export for use in show_deployment_info
    export DEPLOYMENT_TYPE
}

show_deployment_info() {
    # Get the service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    SERVICE_ACCOUNT_EMAIL="vjepa2-sa@$PROJECT_ID.iam.gserviceaccount.com"

    echo ""
    echo -e "${GREEN}🎉 Deployment Complete!${NC}"
    echo -e "${BLUE}Service URL: $SERVICE_URL${NC}"
    echo -e "${BLUE}Region: $REGION${NC}"
    echo -e "${BLUE}Service Account: $SERVICE_ACCOUNT_EMAIL${NC}"
    echo ""

    # Display resource configuration based on deployment type
    echo -e "${YELLOW}📊 Resource Configuration:${NC}"
    if [ "$DEPLOYMENT_TYPE" = "GPU" ]; then
        echo -e "  - CPU: 8 vCPUs"
        echo -e "  - Memory: 32 GiB"
        echo -e "  - GPU: 1x NVIDIA L4"
        echo -e "  - Min Instances: 0 (scales to zero)"
        echo -e "  - Max Instances: 4"
    else
        echo -e "  - CPU: 4 vCPUs (CPU-only mode)"
        echo -e "  - Memory: 16 GiB"
        echo -e "  - GPU: None (quota exceeded)"
        echo -e "  - Min Instances: 0 (scales to zero)"
        echo -e "  - Max Instances: 2"
        echo ""
        echo -e "${YELLOW}⚠️  Note: Running in CPU-only mode. Performance will be slower.${NC}"
        echo -e "${YELLOW}   To enable GPU, request increased GPU quota in GCP Console.${NC}"
    fi
    echo -e "  - Timeout: 60 minutes"
    echo ""
}

create_test_script() {
    echo -e "${YELLOW}📝 Creating test script for deployed service...${NC}"

    # Get the service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)" 2>/dev/null || echo "")

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
EOF

    chmod +x test_deployed.py

    echo -e "${GREEN}✅ Test script created: test_deployed.py${NC}"
}

show_next_steps() {
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
}

# Main execution logic
case $STEP in
    "setup")
        check_prerequisites
        setup_gcp
        ;;
    "build")
        check_prerequisites
        build_and_push cloud
        ;;
    "build-local")
        check_prerequisites
        build_and_push local
        ;;
    "deploy")
        check_prerequisites
        setup_service_account
        deploy_to_cloudrun
        show_deployment_info
        create_test_script
        show_next_steps
        ;;
    "test")
        create_test_script
        echo -e "${BLUE}Run: python3 test_deployed.py${NC}"
        ;;
    "all")
        check_prerequisites
        setup_gcp
        build_and_push cloud
        setup_service_account
        deploy_to_cloudrun
        show_deployment_info
        create_test_script
        show_next_steps
        ;;
    "all-local")
        check_prerequisites
        setup_gcp
        build_and_push local
        setup_service_account
        deploy_to_cloudrun
        show_deployment_info
        create_test_script
        show_next_steps
        ;;
    *)
        echo -e "${RED}❌ Invalid step: $STEP${NC}"
        echo -e "${YELLOW}Valid steps: setup, build, build-local, deploy, test, all, all-local${NC}"
        echo -e "${BLUE}Run '$0 help' for more information${NC}"
        exit 1
        ;;
esac