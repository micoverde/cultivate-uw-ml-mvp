#!/bin/bash

# Deploy Ensemble ML Models to Azure Container Apps
# Issue #192: Ensemble Containerization
# Issue #196: Model Selection Settings with Feature Flag

set -e

echo "🚀 Starting Ensemble ML Deployment"
echo "=================================="
echo "This will deploy 7-model ensemble to Azure Container Apps"
echo "Models: Neural Network, XGBoost, Random Forest, SVM, Logistic Regression, LightGBM, Gradient Boosting"
echo ""

# Configuration
RESOURCE_GROUP="cultivate-ml-rg"
REGISTRY="cultivatemlregistry"
REGISTRY_URL="${REGISTRY}.azurecr.io"
IMAGE_NAME="cultivate-ml-api"
ENV_NAME="cultivate-env"
APP_NAME="cultivate-ml-ensemble"
LOCATION="eastus"

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Build the ensemble Docker image
echo ""
echo "🔨 Building Docker image with fullml stage..."
docker build \
  --target fullml \
  --tag ${REGISTRY_URL}/${IMAGE_NAME}:ensemble-latest \
  --tag ${REGISTRY_URL}/${IMAGE_NAME}:ensemble-$(date +%Y%m%d-%H%M%S) \
  .

echo ""
echo "✅ Docker build complete"

# Login to Azure
echo ""
echo "🔐 Logging in to Azure..."
az login --only-show-errors 2>/dev/null || echo "Already logged in"

# Login to ACR
echo ""
echo "🔐 Logging in to Azure Container Registry..."
az acr login --name ${REGISTRY}

# Push image to registry
echo ""
echo "📤 Pushing image to registry..."
docker push ${REGISTRY_URL}/${IMAGE_NAME}:ensemble-latest

# Check if Container App Environment exists
echo ""
echo "🔍 Checking Container App Environment..."
ENV_EXISTS=$(az containerapp env show \
  --name ${ENV_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>/dev/null || echo "")

if [ -z "$ENV_EXISTS" ]; then
  echo "📦 Creating Container App Environment..."
  az containerapp env create \
    --name ${ENV_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --location ${LOCATION}
else
  echo "✅ Container App Environment exists"
fi

# Check if Container App exists
echo ""
echo "🔍 Checking Container App..."
APP_EXISTS=$(az containerapp show \
  --name ${APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>/dev/null || echo "")

if [ -z "$APP_EXISTS" ]; then
  echo "📦 Creating new Container App with ensemble configuration..."

  # Get ACR credentials
  ACR_USERNAME=$(az acr credential show --name ${REGISTRY} --query username -o tsv)
  ACR_PASSWORD=$(az acr credential show --name ${REGISTRY} --query passwords[0].value -o tsv)

  # Create Container App with ensemble settings
  az containerapp create \
    --name ${APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --environment ${ENV_NAME} \
    --image ${REGISTRY_URL}/${IMAGE_NAME}:ensemble-latest \
    --target-port 8000 \
    --ingress external \
    --cpu 0.5 \
    --memory 1.0Gi \
    --min-replicas 0 \
    --max-replicas 2 \
    --registry-server ${REGISTRY_URL} \
    --registry-username ${ACR_USERNAME} \
    --registry-password ${ACR_PASSWORD} \
    --env-vars \
      MODEL_TYPE=ensemble \
      USE_AZURE_STORAGE=true \
      ENSEMBLE_VOTING_STRATEGY=soft \
      MODEL_DOWNLOAD_ON_STARTUP=false \
      REDIS_ENABLED=false \
      FEATURE_FLAG_MODEL_SELECTION=true \
      ENVIRONMENT=production
else
  echo "🔄 Updating existing Container App..."
  az containerapp update \
    --name ${APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --image ${REGISTRY_URL}/${IMAGE_NAME}:ensemble-latest \
    --set-env-vars \
      MODEL_TYPE=ensemble \
      FEATURE_FLAG_MODEL_SELECTION=true
fi

# Get application URL
echo ""
echo "🔍 Getting application URL..."
APP_URL=$(az containerapp show \
  --name ${APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "✅ Deployment Complete!"
echo "=================================="
echo "🌐 Application URL: https://${APP_URL}"
echo "🔧 Ensemble API: https://${APP_URL}/api/classify"
echo "📊 Health Check: https://${APP_URL}/api/health"
echo "🎛️ Model Selection: Enabled via FEATURE_FLAG_MODEL_SELECTION"
echo ""
echo "📝 Next Steps:"
echo "1. Test the ensemble API at https://${APP_URL}/api/classify"
echo "2. Model selection settings will be dark deployed (hidden)"
echo "3. Enable in UI with featureFlags.modelSelection = true"
echo ""

# Health check
echo "🏥 Running health check..."
sleep 10
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://${APP_URL}/api/health || echo "000")

if [ "$HTTP_CODE" == "200" ]; then
  echo "✅ Health check passed!"
else
  echo "⚠️ Health check returned: $HTTP_CODE (may still be starting up)"
fi