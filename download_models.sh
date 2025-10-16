#!/bin/bash
# Download ML models from Azure Blob Storage and start the API
# This ensures models are always available without needing Git LFS in Docker

set -e

echo "🚀 Cultivate ML API Startup"
echo "=================================================="

# Create models directory
mkdir -p /app/models
cd /app/models

# Check if models already exist (for local dev or cached builds)
if [ "$(ls -A /app/models)" ]; then
    echo "✅ Models directory already populated, skipping download"
else
    echo "📥 Downloading ML models from Azure Blob Storage..."

    # Get connection string from environment (set in Container App)
    if [ -z "$AZURE_STORAGE_CONNECTION_STRING" ]; then
        echo "⚠️  AZURE_STORAGE_CONNECTION_STRING not set"
        echo "   Models will not be available in this environment"
        echo "   Ensemble endpoint will use heuristic fallback"
    else
        # Download models using Azure CLI
        # Container name and blob pattern
        CONTAINER="ml-models"

        # Try to download each model type
        echo "   Downloading ensemble models..."
        az storage blob download-batch \
            --source $CONTAINER \
            --destination . \
            --pattern "ensemble_*.pkl" \
            --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
            2>/dev/null || echo "   ℹ️  No ensemble models found in blob storage"

        echo "   Downloading classic models..."
        az storage blob download-batch \
            --source $CONTAINER \
            --destination . \
            --pattern "classic_*.pkl" \
            --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
            2>/dev/null || echo "   ℹ️  No classic models found in blob storage"

        # Check if any models were downloaded
        if [ "$(ls -A /app/models)" ]; then
            echo "✅ Models downloaded successfully"
        else
            echo "⚠️  No models found in blob storage"
        fi
    fi
fi

# List available models
echo ""
echo "📦 Available Models:"
ls -lh /app/models/*.pkl 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}' || echo "   (none)"

echo ""
echo "🎯 Starting ML API Server..."
echo "=================================================="

# Start the API
exec python run_api.py
