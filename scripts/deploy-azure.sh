#!/bin/bash

# Azure Container Instance Deployment Script
# Deploys Cultivate Learning ML API to Azure

set -e

echo "🚀 Starting Azure Container Instance deployment..."

# Configuration
RESOURCE_GROUP="rg-cultivate-ml-backend-pag"
CONTAINER_NAME="cultivate-ml-api"
LOCATION="westus2"
DNS_NAME="cultivate-ml-api-pag"

# Check provider registration
echo "📋 Checking provider registration..."
CONTAINER_STATE=$(az provider show -n Microsoft.ContainerInstance --query "registrationState" -o tsv)
echo "Container Instance provider state: $CONTAINER_STATE"

if [ "$CONTAINER_STATE" != "Registered" ]; then
    echo "⏳ Waiting for Container Instance provider to register..."
    while [ "$CONTAINER_STATE" != "Registered" ]; do
        sleep 10
        CONTAINER_STATE=$(az provider show -n Microsoft.ContainerInstance --query "registrationState" -o tsv)
        echo "Current state: $CONTAINER_STATE"
    done
fi

echo "✅ Container Instance provider is registered"

# Create container with startup command approach
echo "🔧 Creating container instance..."

# Create a deployment YAML for complex configuration
cat > container-deploy.yaml << EOF
apiVersion: '2021-09-01'
location: westus2
properties:
  containers:
  - name: $CONTAINER_NAME
    properties:
      image: python:3.12-slim
      resources:
        requests:
          cpu: 1
          memoryInGb: 1.5
      ports:
      - port: 8000
        protocol: TCP
      command:
      - "/bin/bash"
      - "-c"
      - |
        apt-get update && apt-get install -y curl git && \
        pip install fastapi uvicorn pydantic numpy pandas python-dotenv pyyaml python-json-logger && \
        git clone https://github.com/micoverde/cultivate-uw-ml-mvp.git /app && \
        cd /app && \
        python run_api.py
      environmentVariables:
      - name: PYTHONPATH
        value: /app
      - name: PORT
        value: '8000'
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
    dnsNameLabel: $DNS_NAME
EOF

# Deploy using YAML
az container create \
  --resource-group $RESOURCE_GROUP \
  --file container-deploy.yaml \
  --name $CONTAINER_NAME

echo "🎉 Deployment initiated!"

# Get the URL
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" -o tsv)
echo "🌐 API will be available at: http://$FQDN:8000"
echo "🔍 Health check: http://$FQDN:8000/api/health"
echo "📚 API docs: http://$FQDN:8000/docs"

echo "✅ Deployment complete!"