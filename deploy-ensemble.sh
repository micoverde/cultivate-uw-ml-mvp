#!/bin/bash

# Deploy Ensemble ML Models to Azure Container Apps
# Issue #192: Containerize Ensemble ML Models
#
# This script builds and deploys the fullml stage with 7-model ensemble
# to Azure Container Apps with progressive rollout
#
# Usage: ./deploy-ensemble.sh [staging|production]

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
RESOURCE_GROUP="cultivate-rg"
ACR_NAME="cultivatemldata"
APP_NAME="cultivate-ml-api"
CONTAINER_ENV="cultivate-env"
IMAGE_TAG="ensemble-$(date +%Y%m%d-%H%M%S)"
REDIS_NAME="cultivateml-redis"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} Ensemble ML Deployment Script${NC}"
echo -e "${BLUE} Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE} Image Tag: ${IMAGE_TAG}${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed. Please install it first.${NC}"
        exit 1
    fi
}

# Function to check Azure login
check_azure_login() {
    echo -e "${YELLOW}🔐 Checking Azure login...${NC}"
    if ! az account show &> /dev/null; then
        echo -e "${RED}❌ Not logged into Azure. Running 'az login'...${NC}"
        az login
    else
        ACCOUNT=$(az account show --query name -o tsv)
        echo -e "${GREEN}✓ Logged into Azure account: $ACCOUNT${NC}"
    fi
}

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"
check_command az
check_command docker
check_azure_login

# Step 1: Train models (optional - skip if models already in Azure)
if [ ! -f "models/ensemble_latest.pkl" ]; then
    echo -e "${YELLOW}🎯 Training ensemble models...${NC}"
    if [ -f "train_ensemble_production.py" ]; then
        python train_ensemble_production.py
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Models trained successfully${NC}"
        else
            echo -e "${YELLOW}⚠️ Training failed, will use Azure models if available${NC}"
        fi
    else
        echo -e "${YELLOW}ℹ️ No local models found, will download from Azure${NC}"
    fi
fi

# Step 2: Build Docker image with fullml stage
echo -e "${YELLOW}🐳 Building Docker image (fullml stage)...${NC}"
docker build --target fullml -t ${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Step 3: Push to Azure Container Registry
echo -e "${YELLOW}📤 Pushing to Azure Container Registry...${NC}"

# Login to ACR
az acr login --name ${ACR_NAME}

# Push the image
docker push ${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image pushed to ACR${NC}"
else
    echo -e "${RED}❌ Failed to push image${NC}"
    exit 1
fi

# Step 4: Create or update Redis Cache (if not exists)
echo -e "${YELLOW}🗄️ Checking Redis Cache...${NC}"
REDIS_EXISTS=$(az redis show --name ${REDIS_NAME} --resource-group ${RESOURCE_GROUP} --query name -o tsv 2>/dev/null || echo "")

if [ -z "$REDIS_EXISTS" ]; then
    echo -e "${YELLOW}Creating Redis Cache...${NC}"
    az redis create \
        --name ${REDIS_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --location eastus \
        --sku Standard \
        --vm-size c1 \
        --redis-version 6
    echo -e "${GREEN}✓ Redis Cache created${NC}"
else
    echo -e "${GREEN}✓ Redis Cache already exists${NC}"
fi

# Get Redis connection string
REDIS_KEY=$(az redis list-keys --name ${REDIS_NAME} --resource-group ${RESOURCE_GROUP} --query primaryKey -o tsv)
REDIS_HOST="${REDIS_NAME}.redis.cache.windows.net"

# Step 5: Deploy to Container Apps
echo -e "${YELLOW}🚀 Deploying to Azure Container Apps...${NC}"

# Check if Container App exists
APP_EXISTS=$(az containerapp show --name ${APP_NAME}-${ENVIRONMENT} --resource-group ${RESOURCE_GROUP} --query name -o tsv 2>/dev/null || echo "")

if [ -z "$APP_EXISTS" ]; then
    # Create new Container App
    echo -e "${YELLOW}Creating new Container App...${NC}"
    az containerapp create \
        --name ${APP_NAME}-${ENVIRONMENT} \
        --resource-group ${RESOURCE_GROUP} \
        --environment ${CONTAINER_ENV} \
        --image ${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG} \
        --target-port 8000 \
        --ingress external \
        --cpu 1.0 \
        --memory 2.0Gi \
        --min-replicas 1 \
        --max-replicas 3 \
        --registry-server ${ACR_NAME}.azurecr.io \
        --secrets \
            storage-connection="$(az storage account show-connection-string --name cultivatemldata --query connectionString -o tsv)" \
            redis-key="${REDIS_KEY}" \
        --env-vars \
            MODEL_TYPE=ensemble \
            USE_AZURE_STORAGE=true \
            AZURE_STORAGE_CONNECTION_STRING=secretref:storage-connection \
            REDIS_HOST=${REDIS_HOST} \
            REDIS_PASSWORD=secretref:redis-key \
            REDIS_ENABLED=true \
            ENVIRONMENT=${ENVIRONMENT}
else
    # Update existing Container App with new revision
    echo -e "${YELLOW}Updating existing Container App...${NC}"

    # Create new revision with traffic split for canary deployment
    if [ "$ENVIRONMENT" == "production" ]; then
        # Production: Start with 10% traffic to new revision
        az containerapp update \
            --name ${APP_NAME}-${ENVIRONMENT} \
            --resource-group ${RESOURCE_GROUP} \
            --image ${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG} \
            --revision-suffix ${IMAGE_TAG}

        # Get revision names
        LATEST_REVISION=$(az containerapp revision list \
            --name ${APP_NAME}-${ENVIRONMENT} \
            --resource-group ${RESOURCE_GROUP} \
            --query "[0].name" -o tsv)

        PREVIOUS_REVISION=$(az containerapp revision list \
            --name ${APP_NAME}-${ENVIRONMENT} \
            --resource-group ${RESOURCE_GROUP} \
            --query "[1].name" -o tsv)

        # Split traffic: 10% to new, 90% to previous
        echo -e "${YELLOW}Setting up canary deployment (10% traffic)...${NC}"
        az containerapp ingress traffic set \
            --name ${APP_NAME}-${ENVIRONMENT} \
            --resource-group ${RESOURCE_GROUP} \
            --revision-weight ${LATEST_REVISION}=10 ${PREVIOUS_REVISION}=90

        echo -e "${GREEN}✓ Canary deployment active (10% traffic)${NC}"
        echo -e "${YELLOW}ℹ️ Monitor metrics for 24 hours before increasing traffic${NC}"
    else
        # Staging: 100% traffic to new revision
        az containerapp update \
            --name ${APP_NAME}-${ENVIRONMENT} \
            --resource-group ${RESOURCE_GROUP} \
            --image ${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG}
        echo -e "${GREEN}✓ Staging deployment complete (100% traffic)${NC}"
    fi
fi

# Step 6: Get application URL
echo -e "${YELLOW}📍 Getting application URL...${NC}"
APP_URL=$(az containerapp show \
    --name ${APP_NAME}-${ENVIRONMENT} \
    --resource-group ${RESOURCE_GROUP} \
    --query properties.configuration.ingress.fqdn -o tsv)

echo -e "${GREEN}✓ Application deployed to: https://${APP_URL}${NC}"

# Step 7: Health check
echo -e "${YELLOW}🏥 Running health check...${NC}"
sleep 30  # Wait for container to start

HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://${APP_URL}/api/health || echo "000")

if [ "$HEALTH_STATUS" == "200" ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"

    # Test ensemble endpoint
    echo -e "${YELLOW}🧪 Testing ensemble model...${NC}"
    curl -s https://${APP_URL}/api/v1/models/info | jq '.'
else
    echo -e "${RED}❌ Health check failed (HTTP ${HEALTH_STATUS})${NC}"
    echo -e "${YELLOW}Check logs with: az containerapp logs show -n ${APP_NAME}-${ENVIRONMENT} -g ${RESOURCE_GROUP}${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "📊 Next Steps:"
echo -e "  1. Monitor performance: ${YELLOW}az monitor metrics list --resource ${APP_NAME}-${ENVIRONMENT} --resource-group ${RESOURCE_GROUP}${NC}"
echo -e "  2. View logs: ${YELLOW}az containerapp logs show -n ${APP_NAME}-${ENVIRONMENT} -g ${RESOURCE_GROUP}${NC}"
echo -e "  3. Scale traffic (production): ${YELLOW}az containerapp ingress traffic set -n ${APP_NAME}-production -g ${RESOURCE_GROUP} --revision-weight latest=50${NC}"
echo -e "  4. Test API: ${YELLOW}curl https://${APP_URL}/api/v1/classify${NC}"
echo ""
echo -e "${GREEN}🔗 Application URL: https://${APP_URL}${NC}"
echo -e "${GREEN}📦 Image Tag: ${IMAGE_TAG}${NC}"