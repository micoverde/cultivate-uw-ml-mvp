# 🚀 Demo 2 Azure Static Web App Deployment Plan

## 📊 Current Status
- **Demo 2 Files**: ✅ Merged to main branch
- **Azure SWA**: calm-tree-06f328310.1.azurestaticapps.net (deployed)
- **GitHub Actions**: Configured for automatic deployment on main push

## 🎯 Deployment Strategy for Demo 2

### **Option 1: Add to Existing React App (RECOMMENDED)**
Since Azure SWA already serves the React demo app, we can integrate Demo 2 as additional routes:

```javascript
// In demo/src/App.tsx add:
<Route path="/demo2" element={<Demo2Showcase />} />
<Route path="/demo2-upload" element={<Demo2Upload />} />
```

**Steps:**
1. Copy `demo2_whisper_showcase.html` → `demo/public/demo2.html`
2. Copy `demo2_video_upload.html` → `demo/public/demo2-upload.html`
3. Copy `highland_park_real_data.json` → `demo/public/data/`
4. Update navigation to include Demo 2 links
5. Push to main - automatic deployment via GitHub Actions

### **Option 2: Standalone Static Files**
Deploy Demo 2 HTML files as standalone static assets:

```yaml
# In azure-swa-deploy.yml add:
- name: Copy Demo 2 files
  run: |
    cp demo2_*.html demo/dist/
    cp highland_park_real_data.json demo/dist/
```

**Access URLs:**
- https://calm-tree-06f328310.1.azurestaticapps.net/demo2_whisper_showcase.html
- https://calm-tree-06f328310.1.azurestaticapps.net/demo2_video_upload.html

### **Option 3: Separate Azure SWA (Production-Grade)**
Create dedicated SWA for Demo 2 with full backend support:

```bash
# Azure CLI commands
az staticwebapp create \
  --name "swa-cultivate-demo2" \
  --resource-group "rg-cultivate" \
  --source "https://github.com/micoverde/cultivate-uw-ml-mvp" \
  --branch "main" \
  --app-location "/demo2" \
  --api-location "/api" \
  --output-location "dist"
```

## 📋 **IMMEDIATE ACTION PLAN**

### **Step 1: Prepare Files for SWA**
```bash
# Create demo2 directory in demo/public
mkdir -p demo/public/demo2
cp demo2_whisper_showcase.html demo/public/demo2/index.html
cp demo2_video_upload.html demo/public/demo2/upload.html
cp highland_park_real_data.json demo/public/demo2/data.json
```

### **Step 2: Update Build Process**
```json
// In demo/package.json scripts:
"build": "vite build && npm run copy-demo2",
"copy-demo2": "cp -r public/demo2 dist/"
```

### **Step 3: Add Navigation**
```typescript
// In demo/src/components/Navigation.tsx:
<Link to="/demo2" className="nav-link">
  🎥 Demo 2: Whisper Analysis
</Link>
```

### **Step 4: Configure Routing**
```javascript
// In demo/vite.config.ts:
export default defineConfig({
  // ... existing config
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        demo2: resolve(__dirname, 'public/demo2/index.html'),
        demo2Upload: resolve(__dirname, 'public/demo2/upload.html')
      }
    }
  }
})
```

### **Step 5: Deploy**
```bash
git add .
git commit -m "feat: Add Demo 2 Whisper Analysis to Azure SWA deployment"
git push origin main
# GitHub Actions will automatically deploy to Azure SWA
```

## 🔍 **Backend Integration for Video Processing**

### **Azure Function App for Whisper Processing**
Since SWA doesn't support heavy ML processing, create Azure Functions:

```python
# api/ProcessVideo/__init__.py
import azure.functions as func
import whisper
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    video_url = req.params.get('url')

    # Download video from blob storage
    # Process with Whisper
    # Return transcription results

    return func.HttpResponse(
        json.dumps(results),
        mimetype="application/json"
    )
```

### **Azure Resources Needed:**
1. **Azure Function App** (Python 3.9, Consumption plan)
2. **Azure Storage** (Video upload container)
3. **Azure Container Instances** (GPU for Whisper processing)
4. **Application Insights** (Already exists: appins-cultivate-ml)

## 🎯 **QUICKEST PATH TO PRODUCTION**

### **Today (Static Demo):**
1. Copy HTML files to `demo/public/`
2. Update `demo/src/App.tsx` with iframe:
   ```jsx
   <Route path="/demo2" element={
     <iframe src="/demo2_whisper_showcase.html"
             style={{width: '100%', height: '100vh', border: 'none'}} />
   } />
   ```
3. Push to main → Auto-deploy via GitHub Actions
4. **Live in 5 minutes** at: https://calm-tree-06f328310.1.azurestaticapps.net/demo2

### **This Week (Full Integration):**
1. Create Azure Function for Whisper processing
2. Set up blob storage for video uploads
3. Connect Demo 2 upload to Azure backend
4. Enable real-time processing of educator videos

### **Next Sprint (Production Scale):**
1. GPU-enabled container instances
2. Queue-based processing (Azure Service Bus)
3. CDN for video delivery
4. Auto-scaling based on demand

## 🔥 **IMMEDIATE DEPLOYMENT COMMANDS**

```bash
# 1. Quick static deployment
cp demo2_whisper_showcase.html demo/public/
cp demo2_video_upload.html demo/public/
cp highland_park_real_data.json demo/public/

# 2. Test locally
cd demo
npm run build
npm run preview
# Open http://localhost:4173/demo2_whisper_showcase.html

# 3. Deploy to Azure
git add demo/public/demo2_*
git commit -m "feat: Deploy Demo 2 to Azure SWA"
git push origin main

# 4. Monitor deployment
gh run watch
# or
open https://github.com/micoverde/cultivate-uw-ml-mvp/actions

# 5. Access production
open https://calm-tree-06f328310.1.azurestaticapps.net/demo2_whisper_showcase.html
```

## ✅ **Success Criteria**
- Demo 2 accessible via Azure SWA URL
- Highland Park 004.mp4 analysis loads with real data
- Upload interface functional (static demo mode)
- Performance: <2s page load time
- Mobile responsive design works

## 🎉 **Expected Result**
Within 10 minutes, Demo 2 will be live on Azure Static Web Apps, accessible to stakeholders worldwide with Highland Park 004.mp4 real analysis showcasing genuine AI capabilities.

**Current Azure SWA**: https://calm-tree-06f328310.1.azurestaticapps.net
**Demo 2 URL**: https://calm-tree-06f328310.1.azurestaticapps.net/demo2_whisper_showcase.html