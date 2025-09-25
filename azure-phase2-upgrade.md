# Phase 2 Infrastructure Upgrade Plan
## Container Apps + PyTorch for Video Feature Extraction

### 🎯 **Upgrade Scope**
Transition from Container Instance to Container Apps to support FEATURE 4: Automatic Video Feature Extraction

### 📦 **New Azure Resources Needed**

#### **1. Container Apps Environment**
```bash
az containerapp env create \
  --name cultivate-ml-env \
  --resource-group rg-cultivate-ml-backend-pag \
  --location westus2
```

#### **2. Storage Account for Videos**
```bash
az storage account create \
  --name cultivatemlvideostorage \
  --resource-group rg-cultivate-ml-backend-pag \
  --location westus2 \
  --sku Standard_LRS
```

#### **3. Enhanced Container App**
```bash
az containerapp create \
  --name cultivate-ml-api-v2 \
  --resource-group rg-cultivate-ml-backend-pag \
  --environment cultivate-ml-env \
  --image cultivatemlapi.azurecr.io/cultivate-ml-api:pytorch \
  --cpu 2.0 \
  --memory 4Gi \
  --min-replicas 1 \
  --max-replicas 5 \
  --ingress external \
  --target-port 8000
```

### 🐳 **Enhanced Docker Image (Phase 2)**

#### **requirements-phase2.txt**
```
# Phase 1 Dependencies (keep existing)
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0

# Phase 2 Deep Learning Stack
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
librosa>=0.10.0

# Video Processing
ffmpeg-python>=0.2.0
pillow>=10.0.0

# Optional: Whisper for enhanced audio
openai-whisper>=20230918
```

#### **Dockerfile.phase2**
```dockerfile
FROM python:3.12-slim

# Install system dependencies for video processing
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy enhanced requirements
COPY requirements-phase2.txt .
RUN pip install --no-cache-dir -r requirements-phase2.txt

# Copy source code
COPY src/ ./src/
COPY run_api.py .

# Create directories for video processing
RUN mkdir -p /app/uploads /app/cache /app/models

USER apiuser
EXPOSE 8000

CMD ["python", "run_api.py"]
```

### 🔄 **Migration Strategy**

#### **Blue-Green Deployment**
1. **Keep Phase 1 running** (Container Instance)
2. **Deploy Phase 2** (Container Apps)
3. **Test video features** in Phase 2
4. **Switch frontend** to Phase 2 when ready
5. **Decommission Phase 1** after validation

#### **Cost Implications**
- **Phase 1**: ~$30/month (Container Instance)
- **Phase 2**: ~$80-120/month (Container Apps + Storage)
- **Migration Period**: ~$110-150/month (both running)
- **Final State**: ~$80-120/month (Phase 2 only)

### 📊 **Performance Expectations**

#### **Video Processing Capabilities**
- **Upload Size**: Up to 100MB per video
- **Processing Time**: ~2x video length
- **Features Extracted**: Visual, audio, gesture, engagement
- **Auto-scaling**: 1-5 containers based on load

#### **Enhanced ML Analysis**
- **Existing**: Transcript-only analysis (20s)
- **Phase 2**: Video + transcript analysis (60-120s)
- **Features**: Face detection, gesture recognition, speech patterns
- **Quality**: Professional-grade educational analysis

### ⚡ **Implementation Timeline**

#### **Week 1: Infrastructure Setup**
- Create Container Apps environment
- Set up video storage account
- Build Phase 2 Docker image

#### **Week 2: Video Processing Pipeline**
- Implement video upload endpoints
- Add PyTorch feature extraction models
- Create video analysis workflows

#### **Week 3: Integration & Testing**
- Connect video features to transcript analysis
- Test auto-scaling under load
- Performance optimization

#### **Week 4: Production Migration**
- Blue-green deployment to production
- Frontend updates for video upload
- Monitoring and validation

### 🚨 **Risk Mitigation**
- **Backward Compatibility**: Phase 2 supports all Phase 1 endpoints
- **Rollback Plan**: Keep Phase 1 Container Instance as fallback
- **Gradual Migration**: Test thoroughly before full switch
- **Cost Control**: Monitor spending during transition

### ✅ **Go/No-Go Decision Criteria**
- ✅ **Technical**: PyTorch models working in container
- ✅ **Performance**: Video processing < 2x video length
- ✅ **Cost**: Monthly spend under budget approval
- ✅ **Quality**: Feature extraction accuracy > 85%
- ✅ **Stability**: Auto-scaling working without issues

**Recommendation: Proceed with Phase 2 upgrade to enable FEATURE 4 automatic video feature extraction**