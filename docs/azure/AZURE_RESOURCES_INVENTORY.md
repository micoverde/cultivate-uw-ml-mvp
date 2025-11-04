# Azure Resources Inventory - Cultivate Learning ML Platform

## Resource Groups Overview

### Primary Cultivate ML Resource Groups

#### 1. **cultivate-ml-rg** (East US)
**Purpose:** Main development environment for Container Apps and ML API

| Resource Name | Type | Purpose | Status |
|--------------|------|---------|--------|
| cultivate-ml-api | Container App | FastAPI ML service | Active |
| cultivate-ml-env | Managed Environment | Container Apps infrastructure | Active |
| cultivatemlregistry | Container Registry | Docker images storage | Active |
| cultivate-ml-logs | Log Analytics Workspace | Monitoring and logs | Active |
| workspace-cultivatemlrgR94V | Log Analytics Workspace | Additional monitoring | Active |

**Key URLs:**
- API Endpoint: `https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io`
- Container Registry: `cultivatemlregistry.azurecr.io`

---

#### 2. **cultivate-ml-prod** (East US 2 / Central US)
**Purpose:** Production environment for demos and storage

| Resource Name | Type | Purpose | Location | Status |
|--------------|------|---------|----------|--------|
| cultivate-ml-demos | Static Web App | Demo 1 & 2 hosting | Central US | Active |
| cultivatemlacr | Container Registry | Production images | East US 2 | Active |
| cultivatemlvideos | Storage Account | Video/model storage | Central US | Active |

**Key URLs:**
- Static Web App: `https://calm-tree-06f328310.1.azurestaticapps.net`
- Storage: `cultivatemlvideos.blob.core.windows.net`

---

#### 3. **rg-cultivate-ml-backend-pag** (West US 2)
**Purpose:** Backend infrastructure and monitoring

| Resource Name | Type | Purpose | Status |
|--------------|------|---------|--------|
| cultivate-ml-api | Container Instance | Backend API container | Active |
| cultivatemlapi | Container Registry | API images | Active |
| cultivate-ml-insights | Application Insights | Performance monitoring | Active |
| cultivate-insights-prod | Application Insights | Production monitoring | Active |
| cultivate-ml-vnet | Virtual Network | Network infrastructure | Active |
| cultivate-ml-env | Managed Environment | Container environment | Active |
| cultivate-ml-logs | Log Analytics | Centralized logging | Active |

---

### Related Resource Groups

#### 4. **harbormaster-rg** (East US)
**Purpose:** Harbormaster application deployment

| Resource Name | Type | Purpose | Status |
|--------------|------|---------|--------|
| harbormaster | Container App | Harbormaster service | Active |

**URL:** `https://harbormaster.salmonsmoke-b94462a8.eastus.azurecontainerapps.io`

---

#### 5. **scratchjr-resources** (East US)
**Purpose:** ScratchJr related resources
- Contains ScratchJr deployment infrastructure

---

#### 6. **rg-bona-ml-uw** & **rg-bona-ml-dev** (West US 2)
**Purpose:** Bona ML development and production environments
- ML model development
- Testing infrastructure

---

### System-Managed Resource Groups

#### 7. **Application Insights Managed Groups**
- `ai_cultivate-ml-insights_2ac48c68-4449-43ea-8fa4-63d5055f6627_managed` (West US 2)
- `ai_cultivate-insights-prod_0781ca37-9028-4df4-8f0c-94b60de45e97_managed` (West US 2)

**Purpose:** Azure-managed infrastructure for Application Insights

#### 8. **NetworkWatcherRG** (West US 2)
**Purpose:** Azure network monitoring infrastructure

---

## Resource Distribution by Type

### Container Apps & Registries
| Name | Resource Group | Location | Purpose |
|------|---------------|----------|---------|
| cultivate-ml-api | cultivate-ml-rg | East US | Main ML API |
| cultivate-ml-api | rg-cultivate-ml-backend-pag | West US 2 | Backend API |
| harbormaster | harbormaster-rg | East US | Harbormaster service |
| cultivatemlregistry | cultivate-ml-rg | East US | Dev registry |
| cultivatemlacr | cultivate-ml-prod | East US 2 | Prod registry |
| cultivatemlapi | rg-cultivate-ml-backend-pag | West US 2 | Backend registry |

### Storage Accounts
| Name | Resource Group | Location | Purpose |
|------|---------------|----------|---------|
| cultivatemlvideos | cultivate-ml-prod | Central US | Videos, models, data |

### Monitoring & Insights
| Name | Resource Group | Location | Purpose |
|------|---------------|----------|---------|
| cultivate-ml-insights | rg-cultivate-ml-backend-pag | West US 2 | Dev monitoring |
| cultivate-insights-prod | rg-cultivate-ml-backend-pag | West US 2 | Prod monitoring |
| cultivate-ml-logs | Multiple | Multiple | Centralized logging |

### Static Web Apps
| Name | Resource Group | Location | Purpose |
|------|---------------|----------|---------|
| cultivate-ml-demos | cultivate-ml-prod | Central US | Demo 1 & 2 hosting |

---

## Active Deployments Summary

### Production Endpoints
1. **Static Web App (Demos):** `https://calm-tree-06f328310.1.azurestaticapps.net`
   - Demo 1: `/demo1/`
   - Demo 2: `/demo2/`

2. **ML API (Container App):** `https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io`
   - Health: `/health`
   - Predict: `/predict`
   - Batch: `/batch_predict`

3. **Blob Storage:** `https://cultivatemlvideos.blob.core.windows.net`
   - Videos container
   - Models container
   - Data container

### Development Endpoints
- **Harbormaster:** `https://harbormaster.salmonsmoke-b94462a8.eastus.azurecontainerapps.io`

---

## Cost Centers

### Primary Cost Drivers
1. **Container Apps** (cultivate-ml-rg, rg-cultivate-ml-backend-pag)
   - Consumption-based pricing
   - Auto-scales 0-20 instances

2. **Storage** (cultivatemlvideos)
   - Hot tier for active data
   - Videos and model storage

3. **Application Insights** (multiple)
   - Data ingestion and retention
   - Performance monitoring

4. **Container Registries** (3 registries)
   - Image storage and bandwidth

### Optimization Opportunities
1. Consolidate multiple container registries
2. Review duplicate Application Insights instances
3. Consider moving old data to cool/archive storage tiers
4. Consolidate resource groups (currently spread across 3 regions)

---

## Regional Distribution

### East US
- cultivate-ml-rg (Container Apps, Registry)
- harbormaster-rg
- scratchjr-resources

### East US 2
- cultivate-ml-prod (part)

### Central US
- cultivate-ml-prod (Static Web App, Storage)
- Multiple PAG resources

### West US 2
- rg-cultivate-ml-backend-pag
- rg-bona-ml-* resources
- Application Insights managed groups

---

## Recommendations

1. **Resource Consolidation:**
   - Consider consolidating the 3 container registries into 1-2
   - Merge duplicate Application Insights instances

2. **Regional Optimization:**
   - Consider moving all resources to a single region pair (e.g., East US / East US 2) for better latency and cost

3. **Naming Convention:**
   - Standardize naming (some use 'pag' suffix, others don't)
   - Consider renaming for clarity

4. **Cost Management:**
   - Implement tagging strategy for cost allocation
   - Set up budget alerts for each resource group
   - Review unused resources (multiple Log Analytics workspaces)

---

**Last Updated:** November 2025
**Total Resource Groups:** 13 (8 project-related, 5 system-managed)
**Primary Production RG:** cultivate-ml-prod
**Primary Development RG:** cultivate-ml-rg