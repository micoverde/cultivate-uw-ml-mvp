#!/bin/bash
# Train ML models (ensemble + classic)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🤖 Training ML Models"
echo "===================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check venv
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found${NC}"
    echo "   Run: npm run setup"
    exit 1
fi

# Check training data
echo "🔍 Checking training data..."
if [ ! -f "comprehensive_training_data.csv" ]; then
    echo -e "${YELLOW}⚠️  No comprehensive training data found${NC}"
    echo "   Will use built-in training examples"
fi

# Train full 7-model ensemble
echo ""
echo -e "${BLUE}📊 Training 7-Model Ensemble (this takes ~2-5 minutes)...${NC}"
echo "   Models: Neural Net, Random Forest, SVM, Logistic, Gradient Boost, AdaBoost, Extra Trees"
echo ""

venv/bin/python train_7_model_ensemble.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Training complete!${NC}"
    echo ""
    echo "📁 Models saved to:"
    echo "   - models/ensemble_latest.pkl (7-model voting ensemble)"
    echo "   - models/classic_latest.pkl (single best model)"
    echo ""
    echo "🎯 Test the models:"
    echo "   npm test"
    echo ""
    echo "🚀 Start serving with new models:"
    echo "   npm run serve"
else
    echo ""
    echo -e "${RED}❌ Training failed${NC}"
    exit 1
fi
