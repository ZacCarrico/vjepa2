#!/bin/bash

# Local development server script for V-JEPA2 video classifier

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting V-JEPA2 Video Classifier (Local Mode)${NC}"

# Check if we're in the right directory
if [ ! -f "service/main.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from the cloud-run directory${NC}"
    echo "Current directory: $(pwd)"
    echo "Expected files: service/main.py, service/requirements.txt"
    exit 1
fi

# Change to service directory
cd service

echo -e "${YELLOW}📦 Checking Python dependencies...${NC}"

# Check if requirements are installed
if ! python3 -c "import fastapi, transformers, torch" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
else
    echo -e "${GREEN}✅ Dependencies are already installed${NC}"
fi

# Set environment variables for local testing
export USE_GCS=false
export ENVIRONMENT=development
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo -e "${BLUE}🔧 Configuration:${NC}"
echo "  - GCS Enabled: $USE_GCS"
echo "  - Environment: $ENVIRONMENT"
echo "  - Python Path: $PYTHONPATH"

# Check for GPU availability
if python3 -c "import torch; print('GPU Available:', torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}  - GPU: Available ✅${NC}"
else
    echo -e "${YELLOW}  - GPU: Not available (will use CPU) ⚠️${NC}"
fi

echo -e "${GREEN}🌟 Starting FastAPI development server...${NC}"
echo -e "${BLUE}📡 Server will be available at: http://localhost:8080${NC}"
echo -e "${BLUE}📖 API Documentation: http://localhost:8080/docs${NC}"
echo -e "${BLUE}🔍 Health Check: http://localhost:8080/health${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server with hot reload for development
uvicorn main:app --reload --host localhost --port 8080