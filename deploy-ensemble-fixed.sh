#!/bin/bash
#
# Fixed deployment script for ensemble ML to Azure Container Apps
# Issue #192 - Ensemble ML Container Deployment
#

set -e

echo "🚀 Starting FIXED Ensemble ML Deployment to Azure Container Apps"
echo "============================================="
date

# Configuration
RESOURCE_GROUP="cultivate-ml-prod"
ACR_NAME="cultivatemlregistry"
APP_NAME="cultivate-ml-ensemble"
APP_ENV_NAME="cultivate-ml-env"
LOCATION="eastus"

# Image configuration
IMAGE_TAG="ensemble-$(date +%Y%m%d-%H%M%S)"
IMAGE_NAME="${ACR_NAME}.azurecr.io/cultivate-ml:${IMAGE_TAG}"

# Step 1: Ensure logged into Azure
echo "📝 Checking Azure login..."
az account show > /dev/null 2>&1 || az login

# Step 2: Login to ACR first (fix for Docker auth issue)
echo "🔐 Logging into Azure Container Registry..."
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
docker login ${ACR_NAME}.azurecr.io -u $ACR_NAME -p $ACR_PASSWORD

# Step 3: Build Docker image with fullml stage
echo "🏗️ Building Docker image with ensemble ML (fullml stage)..."
docker build \
  --target fullml \
  --build-arg BUILD_ENV=production \
  -t $IMAGE_NAME \
  -f Dockerfile \
  .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed. Checking requirements..."
    ls -la requirements*.txt
    exit 1
fi

# Step 4: Push to ACR
echo "📤 Pushing image to Azure Container Registry..."
docker push $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed. Retrying with ACR build..."
    # Alternative: Use ACR build if local push fails
    az acr build --registry $ACR_NAME --image cultivate-ml:${IMAGE_TAG} --target fullml .
fi

# Step 5: Deploy or update Container App
echo "🌐 Deploying to Azure Container Apps..."

# Check if app exists
APP_EXISTS=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "name" -o tsv 2>/dev/null || echo "")

if [ -z "$APP_EXISTS" ]; then
    echo "📦 Creating new container app..."
    az containerapp create \
        --name ${APP_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --environment ${APP_ENV_NAME} \
        --image ${IMAGE_NAME} \
        --cpu 0.5 \
        --memory 1.0Gi \
        --min-replicas 0 \
        --max-replicas 2 \
        --ingress external \
        --target-port 8000 \
        --registry-server ${ACR_NAME}.azurecr.io \
        --registry-username ${ACR_NAME} \
        --registry-password "${ACR_PASSWORD}" \
        --env-vars \
            MODEL_TYPE=ensemble \
            FEATURE_FLAG_MODEL_SELECTION=true \
            ENVIRONMENT=production \
            AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING:-}" \
            PYTHONUNBUFFERED=1
else
    echo "📦 Updating existing container app..."
    az containerapp update \
        --name ${APP_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --image ${IMAGE_NAME} \
        --cpu 0.5 \
        --memory 1.0Gi \
        --set-env-vars \
            MODEL_TYPE=ensemble \
            FEATURE_FLAG_MODEL_SELECTION=true \
            DEPLOYMENT_TIMESTAMP=$(date +%s)
fi

# Step 6: Get app URL
echo "🔍 Getting application URL..."
APP_URL=$(az containerapp show \
    --name ${APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --query "properties.configuration.ingress.fqdn" -o tsv)

echo "✅ Deployment completed successfully!"
echo "📍 Application URL: https://${APP_URL}"
echo "🔬 API Endpoints:"
echo "   - Classic: https://${APP_URL}/api/classify"
echo "   - Ensemble: https://${APP_URL}/api/v2/classify/ensemble"
echo "   - Health: https://${APP_URL}/health"
echo ""

# Step 7: Test deployment
echo "🧪 Testing deployment..."
sleep 10
curl -s -f "https://${APP_URL}/health" > /dev/null 2>&1 && echo "✅ Health check passed" || echo "⚠️ Health check pending..."

echo "🏁 Deployment script completed!"
date