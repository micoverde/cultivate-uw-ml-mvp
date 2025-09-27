#!/bin/bash

# Minimal MVP Deployment - Cost Optimized for Demo Period
# Uses minimal resources to keep costs low during low-traffic demo phase
#
# Resource allocation:
# - 0.25 vCPU, 0.5GB RAM (minimum allowed)
# - Scale to 0 when not in use
# - No Redis cache (can add later)
# - Single region deployment

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} MVP Ensemble Deployment (Cost-Optimized)${NC}"
echo -e "${BLUE} Resources: 0.25 vCPU, 0.5GB RAM${NC}"
echo -e "${BLUE} Auto-scale: 0-1 replicas${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuration
RESOURCE_GROUP="cultivate-ml-rg"
LOCATION="eastus"
ENVIRONMENT="cultivate-ml-env"
APP_NAME="cultivate-ml-mvp"
ACR_NAME="cultivatemlacr"
STORAGE_ACCOUNT="cultivatemlvideos"
STORAGE_RG="cultivate-ml-prod"

# Check Azure login
echo -e "${YELLOW}🔐 Checking Azure login...${NC}"
if ! az account show &> /dev/null; then
    echo "Not logged in. Running 'az login'..."
    az login
fi

ACCOUNT=$(az account show --query name -o tsv)
echo -e "${GREEN}✓ Using Azure account: $ACCOUNT${NC}"

# Step 1: Create Container App Environment (if needed)
echo -e "${YELLOW}📦 Setting up Container App Environment...${NC}"
ENV_EXISTS=$(az containerapp env show \
    --name ${ENVIRONMENT} \
    --resource-group ${RESOURCE_GROUP} \
    --query name -o tsv 2>/dev/null || echo "")

if [ -z "$ENV_EXISTS" ]; then
    echo "Creating Container App Environment..."
    az containerapp env create \
        --name ${ENVIRONMENT} \
        --resource-group ${RESOURCE_GROUP} \
        --location ${LOCATION}
    echo -e "${GREEN}✓ Container App Environment created${NC}"
else
    echo -e "${GREEN}✓ Container App Environment already exists${NC}"
fi

# Step 2: Get Storage Connection String
echo -e "${YELLOW}🔑 Getting storage connection string...${NC}"
STORAGE_CONNECTION=$(az storage account show-connection-string \
    --name ${STORAGE_ACCOUNT} \
    --resource-group ${STORAGE_RG} \
    --query connectionString -o tsv 2>/dev/null || echo "")

if [ -z "$STORAGE_CONNECTION" ]; then
    echo -e "${YELLOW}⚠️ Storage account not found. Creating one...${NC}"
    az storage account create \
        --name ${STORAGE_ACCOUNT} \
        --resource-group ${RESOURCE_GROUP} \
        --location ${LOCATION} \
        --sku Standard_LRS \
        --kind StorageV2

    STORAGE_CONNECTION=$(az storage account show-connection-string \
        --name ${STORAGE_ACCOUNT} \
        --resource-group ${RESOURCE_GROUP} \
        --query connectionString -o tsv)
fi

# Step 3: Get ACR credentials
echo -e "${YELLOW}🔐 Getting ACR credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name ${ACR_NAME} --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name ${ACR_NAME} --query passwords[0].value -o tsv)

# Step 4: Build lightweight image locally first
echo -e "${YELLOW}🐳 Building lightweight Docker image...${NC}"
echo "Using production stage for MVP (lighter weight)..."

docker build \
    --target production \
    -t ${ACR_NAME}.azurecr.io/${APP_NAME}:mvp-latest \
    -f Dockerfile \
    .

# Step 5: Push to ACR
echo -e "${YELLOW}📤 Pushing to ACR...${NC}"
docker login ${ACR_NAME}.azurecr.io -u ${ACR_USERNAME} -p ${ACR_PASSWORD}
docker push ${ACR_NAME}.azurecr.io/${APP_NAME}:mvp-latest

# Step 6: Deploy Container App (minimal resources)
echo -e "${YELLOW}🚀 Deploying Container App with minimal resources...${NC}"

APP_EXISTS=$(az containerapp show \
    --name ${APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --query name -o tsv 2>/dev/null || echo "")

if [ -z "$APP_EXISTS" ]; then
    echo "Creating new Container App..."
    az containerapp create \
        --name ${APP_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --environment ${ENVIRONMENT} \
        --image ${ACR_NAME}.azurecr.io/${APP_NAME}:mvp-latest \
        --target-port 8000 \
        --ingress external \
        --cpu 0.25 \
        --memory 0.5Gi \
        --min-replicas 0 \
        --max-replicas 1 \
        --registry-server ${ACR_NAME}.azurecr.io \
        --registry-username ${ACR_USERNAME} \
        --registry-password ${ACR_PASSWORD} \
        --secrets \
            storage-connection="${STORAGE_CONNECTION}" \
        --env-vars \
            MODEL_TYPE=heuristic \
            USE_AZURE_STORAGE=false \
            ENVIRONMENT=mvp \
            LOG_LEVEL=INFO \
            WORKERS=1 \
            MAX_REQUEST_SIZE=10485760
else
    echo "Updating existing Container App..."
    az containerapp update \
        --name ${APP_NAME} \
        --resource-group ${RESOURCE_GROUP} \
        --image ${ACR_NAME}.azurecr.io/${APP_NAME}:mvp-latest \
        --cpu 0.25 \
        --memory 0.5Gi \
        --min-replicas 0 \
        --max-replicas 1
fi

# Step 7: Configure auto-scaling for cost savings
echo -e "${YELLOW}⚖️ Configuring auto-scaling rules...${NC}"
az containerapp revision list \
    --name ${APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --query "[0].name" -o tsv > /dev/null

# Add HTTP scaling rule (scale to 0 when idle)
az containerapp update \
    --name ${APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --scale-rule-name http-rule \
    --scale-rule-type http \
    --scale-rule-http-concurrency 10 \
    --min-replicas 0 \
    --max-replicas 1

# Step 8: Get application URL
APP_URL=$(az containerapp show \
    --name ${APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --query properties.configuration.ingress.fqdn -o tsv)

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ MVP Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}🔗 Application URL: https://${APP_URL}${NC}"
echo -e "${GREEN}📊 Resources: 0.25 vCPU, 0.5GB RAM${NC}"
echo -e "${GREEN}💰 Estimated cost: ~$15-20/month${NC}"
echo -e "${GREEN}⚡ Auto-scales to 0 when idle${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo -e "  1. Test the API: curl https://${APP_URL}/api/health"
echo -e "  2. Monitor: az containerapp logs show -n ${APP_NAME} -g ${RESOURCE_GROUP}"
echo -e "  3. Scale up when needed: az containerapp update -n ${APP_NAME} -g ${RESOURCE_GROUP} --cpu 0.5 --memory 1Gi"
echo ""
echo -e "${YELLOW}💡 Tips for cost optimization:${NC}"
echo -e "  - Container scales to 0 when idle (no charges)"
echo -e "  - First request after idle takes ~30s (cold start)"
echo -e "  - Upgrade resources only when traffic increases"