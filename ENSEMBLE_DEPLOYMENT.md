# 🚀 Ensemble Model Deployment Guide

## Container Architecture Decision

### ✅ **Decision: Use Existing Container with `fullml` Stage**

We're using your existing multi-stage Docker architecture. The ensemble models will run in the `fullml` stage, not the default `production` stage.

## 📦 Model Storage Architecture

### **Hybrid Approach: Azure Blob + Container**

```
┌─────────────────────────────────────┐
│         Azure Blob Storage          │
│     Container: ml-models            │
│  ┌────────────────────────────┐     │
│  │ ensemble_latest.pkl (50MB)  │     │
│  │ classic_latest.pkl  (10MB)  │     │
│  │ training_data.csv   (1MB)   │     │
│  └────────────────────────────┘     │
└─────────────────────────────────────┘
              ↓ Download on startup
┌─────────────────────────────────────┐
│      Docker Container (fullml)      │
│  ┌────────────────────────────┐     │
│  │ /app/models/ (local cache)  │     │
│  │ - Model instances in memory │     │
│  │ - FastAPI with hot-swap     │     │
│  └────────────────────────────┘     │
└─────────────────────────────────────┘
```

## 🎯 Deployment Steps

### Step 1: Train Models Locally
```bash
# Train ensemble and upload to Azure
python train_ensemble_production.py

# Verify models are uploaded
az storage blob list --container-name ml-models --account-name cultivatemlstorage
```

### Step 2: Build Docker Image with Ensemble Support
```bash
# Build the fullml stage (not production)
docker build --target fullml -t cultivate-ml:ensemble .

# Test locally
docker run -p 8000:8000 \
  -e AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
  -e MODEL_TYPE=ensemble \
  cultivate-ml:ensemble
```

### Step 3: Push to Azure Container Registry
```bash
# Tag for ACR
docker tag cultivate-ml:ensemble cultivateml.azurecr.io/cultivate-ml:ensemble-v1

# Push to ACR
az acr build --registry cultivateml \
  --image cultivate-ml:ensemble-v1 \
  --file Dockerfile \
  --target fullml .
```

### Step 4: Deploy to Azure App Service
```bash
# Update App Service to use fullml image
az webapp config container set \
  --name cultivate-ml-api \
  --resource-group cultivate-rg \
  --docker-custom-image-name cultivateml.azurecr.io/cultivate-ml:ensemble-v1

# Set environment variables
az webapp config appsettings set \
  --name cultivate-ml-api \
  --resource-group cultivate-rg \
  --settings \
    MODEL_TYPE=ensemble \
    USE_AZURE_STORAGE=true \
    AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING"

# Restart to load new models
az webapp restart --name cultivate-ml-api --resource-group cultivate-rg
```

## 🔄 Model Switching in Production

### Dynamic Model Selection (No Redeploy Needed)
```bash
# Switch to classic model
curl -X POST https://cultivate-ml-api.azurewebsites.net/api/v1/models/select \
  -H "Content-Type: application/json" \
  -d '{"model_type": "classic"}'

# Switch back to ensemble
curl -X POST https://cultivate-ml-api.azurewebsites.net/api/v1/models/select \
  -H "Content-Type: application/json" \
  -d '{"model_type": "ensemble"}'
```

### Retrain Models (No Redeploy Needed)
```bash
# Trigger retraining via API
curl -X POST https://cultivate-ml-api.azurewebsites.net/api/v1/models/retrain \
  -H "Content-Type: application/json" \
  -d '{"model_type": "ensemble", "use_azure": true}'
```

## 📊 Resource Requirements

### Container Sizes
- **production** stage: ~500MB, 512MB RAM
- **fullml** stage: ~2GB, 2GB RAM (with ensemble models loaded)

### Azure App Service Plans
- **Current (production)**: B1 (Basic) - $54/month
- **Recommended (fullml)**: P1V2 (Premium) - $146/month
- **Alternative**: Container Instances - Pay per use

## 🎨 Architecture Benefits

### Why Not a Separate Container?

1. **Single Deployment Pipeline**: One CI/CD process
2. **Shared Code Base**: Models use same API code
3. **Easy Switching**: Change Docker target, not entire infrastructure
4. **Cost Optimization**: Switch between lightweight/full as needed

### Model Loading Strategy

```python
# On container startup (pseudocode)
if os.getenv('MODEL_TYPE') == 'ensemble':
    if azure_blob_exists('ensemble_latest.pkl'):
        model = download_from_azure()
    else:
        model = train_new_ensemble()
        upload_to_azure(model)
else:
    model = load_classic_model()
```

## 🚦 Rollback Strategy

```bash
# If ensemble has issues, rollback to production stage
az webapp config container set \
  --name cultivate-ml-api \
  --resource-group cultivate-rg \
  --docker-custom-image-name cultivateml.azurecr.io/cultivate-ml:production-v1

# Or just switch model type
curl -X POST .../api/v1/models/select -d '{"model_type": "classic"}'
```

## 📈 Monitoring

```bash
# Check which model is active
curl https://cultivate-ml-api.azurewebsites.net/api/v1/models/comparison

# View performance metrics
curl https://cultivate-ml-api.azurewebsites.net/api/v1/models/performance/history

# Health check
curl https://cultivate-ml-api.azurewebsites.net/api/health
```

## 🎯 Next Steps

1. **Run training**: `python train_ensemble_production.py`
2. **Test locally**: `docker build --target fullml -t test . && docker run -p 8000:8000 test`
3. **Deploy to staging**: Use fullml stage
4. **Monitor performance**: Compare ensemble vs classic
5. **Production decision**: Based on real metrics

## 💡 Key Insights

- **No new container needed** - Reuse existing infrastructure
- **Models in Azure Blob** - Not in container image (faster deploys)
- **Hot-swappable models** - Change without redeploy
- **Cost control** - Can switch between stages based on needs

---

**Ready to deploy?** Start with Step 1: Train the models!