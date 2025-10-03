#!/bin/bash
# Build script - validates environment and prepares for serving

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔨 Building Cultivate ML MVP"
echo "============================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python virtual environment
echo "🔍 Checking Python environment..."
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found${NC}"
    echo "   Run: npm run setup"
    exit 1
fi
echo -e "${GREEN}✅ Virtual environment found${NC}"

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
if ! venv/bin/python -c "import fastapi, uvicorn, torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Missing dependencies, installing...${NC}"
    venv/bin/pip install -r requirements.txt
fi
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Check for ML models
echo "🔍 Checking ML models..."
if [ ! -f "models/ensemble_latest.pkl" ] || [ ! -f "models/classic_latest.pkl" ]; then
    echo -e "${YELLOW}⚠️  ML models not found${NC}"
    echo "   API will use heuristic fallback"
    echo "   To train models: venv/bin/python train_7_model_ensemble.py"
else
    echo -e "${GREEN}✅ ML models found${NC}"
fi

# Create logs directory
mkdir -p logs

echo ""
echo -e "${GREEN}✅ Build complete${NC}"
echo ""
echo "Next step: npm run serve"
