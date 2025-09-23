# 🚀 Azure Static Web Apps Setup Guide

## ⚠️ INFRASTRUCTURE SETUP REQUIRED

**IMPORTANT**: Azure infrastructure does NOT yet exist for this project. You must create the Azure Static Web App resource first before deployments can work. Follow the steps below to set up the required infrastructure.

## 📋 Overview

This guide covers the complete Azure Static Web Apps (SWA) setup for the Cultivate Learning ML MVP, including GitHub integration, environment configuration, and deployment workflows.

## 🏗️ Architecture

```
GitHub Repository → Azure SWA → Production Deployment
├── Frontend (/demo) → Static Web Content
├── API (/api) → Azure Functions
└── CI/CD Pipeline → Automated Deployment
```

## 🔧 Prerequisites

- **Azure Subscription** with Static Web Apps service available
- **GitHub Repository** with admin access
- **Azure CLI** installed (optional, for manual operations)

## ⚡ Quick Setup Steps

### 1. Create Azure Static Web App Resource

⚠️ **REQUIRED FIRST STEP**: This Azure resource does NOT exist yet and must be created.

#### Via Azure Portal
1. Navigate to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource" → "Static Web App"
3. Configure the resource:
   - **Subscription**: Your Azure subscription
   - **Resource Group**: Create new or use existing (suggested: `cultivate-ml-rg`)
   - **Name**: `cultivate-uw-ml-mvp` (CRITICAL: use this exact name)
   - **Plan Type**: Free (for development) or Standard (for production)
   - **Region**: Choose closest to your users (suggested: East US)
   - **Source**: GitHub
   - **GitHub Account**: Authenticate and select account
   - **Organization**: `micoverde`
   - **Repository**: `cultivate-uw-ml-mvp`
   - **Branch**: `main`
   - **Build Presets**: React
   - **App location**: `/demo`
   - **API location**: `/api`
   - **Output location**: `dist`

⚠️ **IMPORTANT**: After creation, the Azure portal will automatically create a deployment token. This token MUST be added to GitHub secrets as `AZURE_STATIC_WEB_APPS_API_TOKEN`.

#### Via Azure CLI
```bash
# Create resource group
az group create --name cultivate-ml-rg --location "East US"

# Create static web app
az staticwebapp create \
  --name cultivate-uw-ml-mvp \
  --resource-group cultivate-ml-rg \
  --source https://github.com/micoverde/cultivate-uw-ml-mvp \
  --location "East US" \
  --branch main \
  --app-location "/demo" \
  --api-location "/api" \
  --output-location "dist" \
  --login-with-github
```

### 2. Configure GitHub Repository Secrets

The Azure portal will automatically create the GitHub secret, but verify:

1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Ensure secret exists: `AZURE_STATIC_WEB_APPS_API_TOKEN`
3. Value should be the deployment token from Azure

### 3. Configure Environment Variables

#### In Azure Portal
1. Navigate to your Static Web App resource
2. Go to Settings → Configuration
3. Add application settings:

| Name | Value | Description |
|------|-------|-------------|
| `API_HOST` | `your-swa-hostname` | Azure SWA hostname |
| `API_PORT` | `443` | HTTPS port for production |
| `ML_MODEL_PATH` | `./models/` | Path to ML models |
| `NODE_ENV` | `production` | Environment mode |
| `AZURE_STORAGE_CONNECTION_STRING` | `your-storage-connection` | Storage for ML models (if needed) |

#### In GitHub Repository Variables
1. Go to GitHub repository → Settings → Secrets and variables → Actions → Variables
2. Add repository variables:
   - `API_HOST`: Azure SWA hostname
   - `API_PORT`: `443`
   - `ML_MODEL_PATH`: `./models/`

## 📁 Project Structure for Azure SWA

```
cultivate-uw-ml-mvp/
├── demo/                          # Frontend React application
│   ├── public/
│   │   └── staticwebapp.config.json  # SWA configuration
│   ├── src/
│   └── package.json
├── api/                           # Azure Functions API
│   ├── health/
│   │   ├── function.json         # Function binding
│   │   └── index.js             # Health check endpoint
│   ├── host.json                # Functions host configuration
│   └── package.json
├── .github/workflows/
│   └── azure-swa-deploy.yml     # Deployment workflow
└── docs/
    └── AZURE_SWA_SETUP.md       # This guide
```

## 🔄 Deployment Workflow

### Automatic Deployment
- **Trigger**: Push to `main` branch
- **Process**:
  1. GitHub Actions checks out code
  2. Sets up Python and Node.js environments
  3. Installs Python dependencies
  4. Builds React frontend
  5. Installs API dependencies
  6. Deploys to Azure SWA

### Manual Deployment
```bash
# Install Azure Static Web Apps CLI
npm install -g @azure/static-web-apps-cli

# Build the project
npm run build

# Deploy (requires Azure SWA deployment token)
swa deploy ./demo/dist --api-location ./api
```

## 🌐 Accessing Your Deployment

### Production URLs
- **Primary URL**: `https://[generated-name].azurestaticapps.net`
- **Custom Domain**: Configure in Azure portal under Custom domains
- **API Endpoints**: `https://[your-swa-url]/api/[function-name]`

### Health Check
Test your deployment:
```bash
curl https://[your-swa-url]/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-09-23T...",
  "service": "Cultivate Learning ML API",
  "version": "1.0.0",
  "environment": "production"
}
```

## 🔧 Configuration Files

### staticwebapp.config.json
Located in `/demo/public/`, this file configures:
- **Routing rules**: API and frontend routing
- **Authentication**: Role-based access
- **Headers**: Security headers
- **Redirects**: Custom error pages

### host.json
Located in `/api/`, configures Azure Functions:
- **Runtime version**: Functions v4
- **Logging**: Application Insights integration
- **Timeouts**: Function execution limits
- **Extensions**: Required function bindings

## 🛠️ Troubleshooting

### Common Issues

#### Deployment Fails
1. **Check GitHub Actions logs**: Repository → Actions tab
2. **Verify secrets**: Ensure `AZURE_STATIC_WEB_APPS_API_TOKEN` is set
3. **Check build process**: Ensure `npm run build` works locally

#### API Not Working
1. **Check function.json**: Verify binding configuration
2. **Test locally**: Use Azure Functions Core Tools
3. **Check logs**: Azure portal → Functions → Monitor

#### Environment Variables Missing
1. **Azure Portal**: Configuration → Application settings
2. **GitHub Variables**: Repository settings → Variables
3. **Deployment logs**: Check if variables are passed correctly

### Debugging Commands

```bash
# Test local development
cd demo && npm run dev
cd api && func start

# Check build process
npm run build

# Validate staticwebapp.config.json
swa build ./demo

# View deployment logs
az staticwebapp logs show --name cultivate-uw-ml-mvp
```

## 📊 Monitoring & Analytics

### Azure Portal Monitoring
- **Overview**: Traffic, requests, errors
- **Functions**: Individual function performance
- **Logs**: Real-time log streaming
- **Metrics**: Custom metrics and alerts

### Application Insights Integration
Enable in Azure portal for:
- **Performance tracking**: Page load times, API response times
- **Error monitoring**: JavaScript errors, function exceptions
- **Usage analytics**: User flows, popular features

## 🔒 Security Considerations

### HTTPS Only
- Azure SWA enforces HTTPS automatically
- Custom domains require SSL certificate setup

### CORS Configuration
Configured in staticwebapp.config.json:
```json
{
  "globalHeaders": {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization"
  }
}
```

### Environment Secrets
- **Never commit secrets** to repository
- **Use Azure Key Vault** for sensitive configuration
- **Rotate tokens** regularly

## 📚 Additional Resources

- [Azure Static Web Apps Documentation](https://docs.microsoft.com/en-us/azure/static-web-apps/)
- [Azure Functions Documentation](https://docs.microsoft.com/en-us/azure/azure-functions/)
- [GitHub Actions for Azure](https://docs.microsoft.com/en-us/azure/developer/github/)

## ✅ Verification Checklist

After setup, verify:
- [ ] Azure SWA resource created and configured
- [ ] GitHub repository connected to Azure
- [ ] `AZURE_STATIC_WEB_APPS_API_TOKEN` secret configured
- [ ] Environment variables set in Azure portal
- [ ] Push to main branch triggers deployment
- [ ] Production URL loads frontend correctly
- [ ] API health endpoint responds successfully
- [ ] staticwebapp.config.json routing works
- [ ] Build and deployment logs show no errors

---

**Last Updated**: September 23, 2025
**Sprint**: FEATURE 0 (Foundation & Infrastructure)
**Team**: Warren & Claude