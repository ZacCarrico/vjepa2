#!/bin/bash

# Setup GCS buckets for V-JEPA2 video classifier

set -e

PROJECT_ID="dev-ml-794354"
MAIN_BUCKET="dev-ml-794354-demo"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🪣 Setting up GCS buckets for V-JEPA2 Video Classifier${NC}"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install it first.${NC}"
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null 2>&1; then
    echo -e "${RED}❌ Not authenticated with gcloud. Please run: gcloud auth login${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}🔧 Setting project to: $PROJECT_ID${NC}"
gcloud config set project $PROJECT_ID

# Create main bucket
echo -e "${YELLOW}📦 Creating main bucket: $MAIN_BUCKET${NC}"

if gsutil ls -b gs://$MAIN_BUCKET &> /dev/null; then
    echo -e "${GREEN}✅ Bucket $MAIN_BUCKET already exists${NC}"
else
    gsutil mb -p $PROJECT_ID gs://$MAIN_BUCKET
    echo -e "${GREEN}✅ Created bucket: $MAIN_BUCKET${NC}"
fi

# Create subdirectories
echo -e "${YELLOW}📁 Creating subdirectories...${NC}"

directories=(
    "raw-videos"
    "model-artifacts"
    "predictions"
)

for dir in "${directories[@]}"; do
    echo -e "${BLUE}  Creating: gs://$MAIN_BUCKET/$dir/${NC}"

    # Create a placeholder object to establish the directory structure
    echo "This directory contains $dir for the V-JEPA2 video classifier" | \
        gsutil cp - gs://$MAIN_BUCKET/$dir/.gitkeep
done

echo -e "${GREEN}✅ All directories created${NC}"

# Set bucket permissions (optional - adjust based on your needs)
echo -e "${YELLOW}🔐 Setting bucket permissions...${NC}"

# Make bucket private by default (recommended for production)
gsutil iam ch allUsers:objectViewer gs://$MAIN_BUCKET &> /dev/null || echo -e "${YELLOW}⚠️  Public access already removed or not set${NC}"

echo -e "${GREEN}✅ Bucket setup complete!${NC}"

# Display bucket info
echo -e "${BLUE}📊 Bucket Information:${NC}"
gsutil du -sh gs://$MAIN_BUCKET

echo -e "${BLUE}📂 Directory Structure:${NC}"
gsutil ls -r gs://$MAIN_BUCKET/

# Test bucket access
echo -e "${YELLOW}🧪 Testing bucket access...${NC}"
echo "Test file from setup script" | gsutil cp - gs://$MAIN_BUCKET/test-access.txt
gsutil rm gs://$MAIN_BUCKET/test-access.txt
echo -e "${GREEN}✅ Bucket access test successful${NC}"

echo ""
echo -e "${GREEN}🎉 GCS setup complete!${NC}"
echo -e "${BLUE}You can now:${NC}"
echo -e "  1. Upload videos to: gs://$MAIN_BUCKET/raw-videos/"
echo -e "  2. Store model artifacts in: gs://$MAIN_BUCKET/model-artifacts/"
echo -e "  3. Save predictions to: gs://$MAIN_BUCKET/predictions/"
echo ""
echo -e "${YELLOW}Example upload:${NC}"
echo -e "  gsutil cp your_video.mp4 gs://$MAIN_BUCKET/raw-videos/"