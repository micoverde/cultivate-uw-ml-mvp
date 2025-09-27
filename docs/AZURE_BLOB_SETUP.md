# Azure Blob Storage Setup Guide

## Issue #184 - Production ML Training Data Management

This guide walks through setting up Azure Blob Storage for production ML training data management, including ground truth collection and secure model retraining.

## Prerequisites

1. **Azure Account**: Active Azure subscription
2. **Azure CLI**: Installed and authenticated (`az login`)
3. **Python 3.9+**: With pip installed
4. **Redis**: For rate limiting (optional, fallback available)

## Quick Start

### 1. Install Dependencies

```bash
# Install Azure dependencies
pip install -r requirements-azure.txt

# Install Redis (optional, for rate limiting)
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis
```

### 2. Create Azure Resources

```bash
# Set variables
RESOURCE_GROUP="cultivate-ml-rg"
STORAGE_ACCOUNT="cultivatemldata"
LOCATION="eastus"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create storage account
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS \
  --encryption-services blob \
  --https-only true \
  --allow-blob-public-access false

# Get storage key
STORAGE_KEY=$(az storage account keys list \
  --resource-group $RESOURCE_GROUP \
  --account-name $STORAGE_ACCOUNT \
  --query '[0].value' -o tsv)

# Create containers
az storage container create \
  --name training-data \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY

az storage container create \
  --name ground-truth \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY

az storage container create \
  --name audit-logs \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY
```

### 3. Configure Environment

```bash
# Copy template
cp .env.azure.template .env

# Edit .env and add your Azure Storage connection string
# The connection string format is:
# DefaultEndpointsProtocol=https;AccountName=cultivatemldata;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net
```

### 4. Set Up Password for Retraining

The default password is `lev`. To change it:

```python
# Generate new password hash
import bcrypt

password = "your_new_password"
hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))
print(hash.decode('utf-8'))

# Add the hash to your .env file:
# RETRAIN_PASSWORD_HASH=YOUR_HASH_HERE
```

## Architecture Overview

```
Azure Blob Storage Structure:
cultivatemldata/
├── training-data/           # Hot tier - Active training data
│   ├── raw-feedback/        # User feedback submissions
│   ├── validated-labels/    # Validated training batches
│   └── model-checkpoints/   # Trained model states
├── ground-truth/            # Cool tier - Reference data
│   ├── oeq-ceq-labels/     # Question classifications
│   └── video-annotations/   # Video analysis data
└── audit-logs/             # Archive tier - Compliance logs
    └── retraining-events/   # Authentication and training logs
```

## API Endpoints

### 1. Collect Feedback

```javascript
POST /api/v1/feedback/collect
{
  "question": "Why is the sky blue?",
  "ml_prediction": "OEQ",
  "human_label": "CEQ",
  "confidence": 0.85,
  "features": {...},
  "user_id": "teacher123"
}
```

### 2. Retrain Model (Password Protected)

```javascript
POST /api/v1/model/retrain
Headers: {
  "X-Retrain-Password": "lev"
}
Body: {
  "model_type": "oeq_ceq_classifier",
  "training_config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

### 3. Check Training Status

```javascript
GET /api/v1/training/status/{job_id}

Response: {
  "status": "running",
  "progress": 0.75,
  "current_epoch": 7,
  "metrics": {
    "accuracy": 0.92,
    "loss": 0.24
  }
}
```

### 4. Get Statistics

```javascript
GET /api/v1/training/statistics

Response: {
  "total_feedback": 1250,
  "processed_feedback": 1200,
  "pending_feedback": 50,
  "oeq_count": 600,
  "ceq_count": 650
}
```

## Frontend Integration

The frontend automatically shows a password modal when the "Start Retraining" button is clicked:

1. User clicks "Start Retraining" button
2. Password modal appears
3. User enters password (`lev`)
4. System validates and initiates retraining
5. Progress bar shows real-time status
6. Model version updates on completion

## Security Features

1. **Password Protection**: BCrypt hashed password for retraining
2. **Rate Limiting**: Max 3 attempts per hour per IP
3. **Audit Logging**: All operations logged to Azure Blob
4. **Encryption**: Data encrypted at rest and in transit
5. **RBAC**: Role-based access control via Azure AD

## Monitoring

### View Audit Logs

```bash
# List recent authentication attempts
az storage blob list \
  --container-name audit-logs \
  --account-name $STORAGE_ACCOUNT \
  --prefix "retraining-events/$(date +%Y-%m)/" \
  --query "[].name"

# Download specific log
az storage blob download \
  --container-name audit-logs \
  --name "retraining-events/2025-01/auth_20250127_153042_192_168_1_1.json" \
  --account-name $STORAGE_ACCOUNT \
  --file auth_log.json
```

### Check Storage Metrics

```bash
# Get storage account metrics
az monitor metrics list \
  --resource /subscriptions/{sub-id}/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT \
  --metric "BlobCapacity" \
  --aggregation Total

# Get container sizes
az storage blob list \
  --container-name training-data \
  --account-name $STORAGE_ACCOUNT \
  --query "[].{name:name, size:properties.contentLength}" \
  --output table
```

## Cost Optimization

1. **Data Lifecycle**: Automatically move old data to Cool/Archive tiers
2. **Cleanup Policy**: Delete data older than 90 days
3. **Compression**: Use Parquet format for validated labels
4. **Batch Operations**: Process feedback in batches to reduce transactions

### Set Up Lifecycle Policy

```bash
# Create lifecycle policy JSON
cat > lifecycle-policy.json <<EOF
{
  "rules": [
    {
      "name": "MoveToCool",
      "type": "Lifecycle",
      "definition": {
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["raw-feedback/"]
        },
        "actions": {
          "baseBlob": {
            "tierToCool": {
              "daysAfterModificationGreaterThan": 30
            },
            "tierToArchive": {
              "daysAfterModificationGreaterThan": 90
            },
            "delete": {
              "daysAfterModificationGreaterThan": 365
            }
          }
        }
      }
    }
  ]
}
EOF

# Apply lifecycle policy
az storage account management-policy create \
  --account-name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --policy @lifecycle-policy.json
```

## Troubleshooting

### Issue: "Too many failed attempts"
- **Solution**: Wait 1 hour for lockout to expire, or clear Redis key:
  ```bash
  redis-cli DEL retrain_lockout:YOUR_IP
  ```

### Issue: "Insufficient training data"
- **Solution**: Collect at least 10 feedback samples before retraining

### Issue: "Azure Blob Storage not accessible"
- **Solution**: Check connection string and network connectivity:
  ```bash
  az storage account show-connection-string \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP
  ```

### Issue: "Model checkpoint failed to save"
- **Solution**: Check storage quota and permissions:
  ```bash
  az storage account show \
    --name $STORAGE_ACCOUNT \
    --query "sku.name"
  ```

## Production Checklist

- [ ] Azure Storage Account created
- [ ] All containers created with proper access tiers
- [ ] Connection string added to environment variables
- [ ] Redis installed and running (or fallback configured)
- [ ] Password configured (default: `lev`)
- [ ] Lifecycle policies configured
- [ ] Monitoring alerts set up
- [ ] Backup strategy defined
- [ ] Security audit completed

## Support

For issues or questions:
- GitHub Issue: #184
- Documentation: `/docs/AZURE_BLOB_SETUP.md`
- API Docs: `/api/v1/docs`

---

**Author**: Claude (Partner-Level Microsoft SDE)
**Date**: 2025-09-27
**Issue**: #184