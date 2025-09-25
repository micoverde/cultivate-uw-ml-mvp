#!/bin/bash

# Complete Docker-based deployment script for Cultivate Learning ML API
# Run this after Docker installation is complete

set -e

echo "🚀 Starting Docker-based Azure deployment..."

# Configuration
REGISTRY="cultivatemlapi.azurecr.io"
IMAGE_NAME="cultivate-ml-api"
TAG="v1.0"
FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"
RESOURCE_GROUP="rg-cultivate-ml-backend-pag"
CONTAINER_NAME="cultivate-ml-api"
DNS_NAME="cultivate-ml-api-pag"

# Step 1: Test Docker installation
echo "🔧 Testing Docker installation..."
docker --version
if [ $? -ne 0 ]; then
    echo "❌ Docker not available. Please ensure Docker is installed and running."
    exit 1
fi

# Step 2: Login to Azure Container Registry
echo "🔐 Logging into Azure Container Registry..."
# Use the refresh token from Azure CLI
ACR_TOKEN="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZYU1M6M1FNWTpZMjZWOkJQR0Y6Q0NORjpRQzNGOkVPUlE6SUJXRDpFUVBVOklUMzQ6RFU1NzpVRTIzIn0.eyJqdGkiOiJlZDg3MWFmNy1iNGEzLTRjMDUtODFjNC1jNmQ4ODg5MzdlOWMiLCJzdWIiOiJsaXZlLmNvbSN3YXJyZW5qb0BwYXJyeWpvaG5zb24uY29tIiwibmJmIjoxNzU4NzUxNjk5LCJleHAiOjE3NTg3NjMzOTksImlhdCI6MTc1ODc1MTY5OSwiaXNzIjoiQXp1cmUgQ29udGFpbmVyIFJlZ2lzdHJ5IiwiYXVkIjoiY3VsdGl2YXRlbWxhcGkuYXp1cmVjci5pbyIsInZlcnNpb24iOiIxLjAiLCJyaWQiOiJhMDAzNmFlMzI5ZjI0YmYxYmFiOTQ5MWQyYjYzNTc2NSIsImdyYW50X3R5cGUiOiJyZWZyZXNoX3Rva2VuIiwiYXBwaWQiOiIwNGIwNzc5NS04ZGRiLTQ2MWEtYmJlZS0wMmY5ZTFiZjdiNDYiLCJ0ZW5hbnQiOiJkYTBhMWM4OC1iNmNkLTQ1ZGUtOWU0YS1hZWJiNDA2NjdiNjQiLCJwZXJtaXNzaW9ucyI6eyJhY3Rpb25zIjpbInJlYWQiLCJ3cml0ZSIsImRlbGV0ZSIsIm1ldGFkYXRhL3JlYWQiLCJtZXRhZGF0YS93cml0ZSIsImRlbGV0ZWQvcmVhZCIsImRlbGV0ZWQvcmVzdG9yZS9hY3Rpb24iXX0sInJvbGVzIjpbXX0.q92B8L_DFcfqtYNE1EUBNaMg8nKe7Ea6y7LW5ZuEucdOUaDH0BwpoIpg9Z_Fy6FJfwyOhcypepbRuYTcvtxV_nIUzOap97quoiy2rHcXYMblMfk0Jx1fytzZ5V-_qmC86aG0SMHmKeSxlIzIhPkFSOlVe3xpQbxYWMNiAvs0w4r1_haxv14wOtRI1wLp_dqZRievczFOtaQ9ESCJfMCKB3DWS2grIb86JkVLX5W5wcGyMyfWjilFYpQdYBefH6m3Rhd5NyNER7YB1Jc1GMmh05QzmVz-D20RhaSxbMllenYKvnk3YkX556gEXDEYvwgQouBNWWy8DDkc74mDXYjZwA"

docker login $REGISTRY -u 00000000-0000-0000-0000-000000000000 -p $ACR_TOKEN

# Step 3: Build the Docker image
echo "🏗️  Building Docker image..."
docker build -f Dockerfile.api -t $FULL_IMAGE .

# Step 4: Push image to Azure Container Registry
echo "⬆️  Pushing image to Azure Container Registry..."
docker push $FULL_IMAGE

# Step 5: Deploy to Azure Container Instance
echo "☁️  Deploying to Azure Container Instance..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $FULL_IMAGE \
    --registry-login-server $REGISTRY \
    --registry-username 00000000-0000-0000-0000-000000000000 \
    --registry-password "$ACR_TOKEN" \
    --cpu 1 \
    --memory 1.5 \
    --restart-policy Always \
    --ports 8000 \
    --dns-name-label $DNS_NAME \
    --os-type Linux \
    --environment-variables PYTHONPATH=/app PORT=8000

# Step 6: Get deployment info
echo "🎉 Deployment complete!"
echo ""
echo "Getting container information..."
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" -o tsv)
IP=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.ip" -o tsv)

echo "=================================================================="
echo "🌐 API Endpoints:"
echo "   Health Check: http://$FQDN:8000/api/health"
echo "   API Documentation: http://$FQDN:8000/docs"
echo "   Transcript Analysis: http://$FQDN:8000/api/v1/analyze/transcript"
echo ""
echo "📍 Container Details:"
echo "   DNS Name: $FQDN"
echo "   IP Address: $IP"
echo "   Container Image: $FULL_IMAGE"
echo ""
echo "🔍 Next Steps:"
echo "   1. Test API health: curl http://$FQDN:8000/api/health"
echo "   2. Update frontend to use: http://$FQDN:8000"
echo "   3. Test full ML analysis flow"
echo "=================================================================="

echo "✅ Deployment successful!"