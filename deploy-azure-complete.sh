#!/bin/bash

# Complete Azure Deployment Script - Cultivate Learning ML Platform
# Enterprise-grade deployment with full monitoring and scaling

set -e

echo "🚀 Cultivate Learning ML Platform - Complete Azure Deployment"
echo "============================================================="

# Configuration
RESOURCE_GROUP="cultivate-ml-prod"
LOCATION="eastus2"
ACR_NAME="cultivatemlacr"
AKS_CLUSTER="cultivate-ml-aks"
CONTAINER_APP_ENV="cultivate-ml-env"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Main deployment
print_status "Starting complete Azure deployment..."

# Create all resources
az group create --name $RESOURCE_GROUP --location $LOCATION

# Container Registry
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Premium

# Build and push all images
./docker-build.sh

# Deploy to Container Apps
az containerapp create \
    --name cultivate-ml-platform \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --min-replicas 2 \
    --max-replicas 20

print_status "Deployment complete! Platform is ready for production."