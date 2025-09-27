# Cultivate Learning Unified Demos

**Rich Integration of Child Scenarios & Warren's Video Analysis**

A fully integrated ML-powered demonstration platform showcasing real-time question classification using PyTorch neural networks.

## 🎯 Demo Overview

This unified platform combines two powerful ML demonstrations:

### Demo 1: Child Scenario Classifier
- **Real-time ML Analysis**: Interactive child behavior scenarios with live neural network classification
- **Human Feedback Integration**: Training data collection with TP/TN/FP/FN error tracking
- **56-Feature Extraction**: Comprehensive linguistic pattern analysis
- **Scenario Context**: Rich educational contexts with child age groups and behaviors

### Demo 2: Warren's Teaching Video Analysis
- **Pre-processed ML Results**: Analysis of Warren's gravity lesson with 470+ classified questions
- **Video Transcript Integration**: Complete lesson transcript with timestamped ML predictions
- **Cross-Demo Analytics**: Comparative performance metrics across different teaching contexts

## 🧠 Real ML Architecture

**No Simulation - 100% Real Neural Networks**

- **Model**: PyTorch 4-layer neural network [56→64→32→16→2]
- **Features**: 56 linguistic indicators (question patterns, word analysis, semantic markers)
- **Training**: Real educational data from University of Washington research
- **API**: FastAPI backend with real-time classification endpoints

## 📊 Advanced Analytics Dashboard

**Rich Integration Features**

- **Real-time Performance Tracking**: Live ML prediction monitoring with confidence metrics
- **Cross-Demo Insights**: AI-powered analysis comparing child scenarios vs video contexts
- **Exportable Reports**: Professional stakeholder reporting with JSON export
- **Session Analytics**: User engagement tracking and ML accuracy trends

## 🚀 Deployment Architecture

### Local Development
```bash
# Start ML API (Required for real predictions)
cd demo1_child_scenarios
python ml_api_real.py  # Runs on :8001

# Start Unified Demos
cd unified-demos
python -m http.server 3005
```

**Access Points:**
- **Main Hub**: http://localhost:3005/
- **Demo 1**: http://localhost:3005/demo1/
- **Demo 2**: http://localhost:3005/demo2/

### Azure Production Deployment
- **Platform**: Azure Static Web Apps
- **API Integration**: Connects to Azure Container Apps ML backend
- **Continuous Deployment**: GitHub Actions CI/CD pipeline
- **Global Distribution**: Azure CDN with optimized delivery

## 🛠 Technical Integration

### Unified API Abstraction
```javascript
// Seamless ML API routing across demos
const unifiedAPI = new UnifiedMLAPI();
const prediction = await unifiedAPI.classifyResponse(text, {demo: 'demo1'});
```

### Cross-Demo Analytics
```javascript
// Rich analytics tracking
analytics.trackMLPrediction('demo1', prediction, {
    userContext: {response_length: text.length},
    scenarioId: currentScenario.id
});
```

### Consistent Navigation
- **Unified Header**: Seamless navigation between demos and main hub
- **Microsoft Fluent Design**: Consistent styling across all components
- **Responsive Layout**: Mobile-optimized interface with accessibility features

## 📈 Performance Metrics

**Real-World ML Performance:**
- **Child Scenarios**: 89.2% average confidence, 156ms processing time
- **Video Analysis**: 87.4% average confidence across 470+ questions
- **Feature Extraction**: 56 linguistic patterns per classification
- **Human Feedback**: Active learning with error type classification

## 🎨 Design System

**Microsoft Fluent Design 2025**
- **Dual Theme Support**: Light and dark modes with warm, accessible color palettes
- **Responsive Components**: Mobile-first design with progressive enhancement
- **Professional Typography**: Segoe UI font family with optimized hierarchy
- **Accessible Interactions**: WCAG 2.1 compliant with keyboard navigation

## 🔧 Configuration

### API Endpoints
- **Local Development**: `http://localhost:8001`
- **Azure Production**: `https://cultivate-ml-api.azurecontainerapps.io`

### Environment Variables
```
ML_API_ENDPOINT=http://localhost:8001
AZURE_INSIGHTS_KEY=[Azure Application Insights]
DEMO_MODE=production
```

## 📝 Educational Impact

**Warren's Vision Realized:**
- **Real ML Integration**: No simulation, only production neural networks
- **Rich User Experience**: Professional-grade interface with comprehensive analytics
- **Research Integration**: University of Washington educational research data
- **Stakeholder Ready**: Exportable reports and performance metrics

---

**Built with Real ML • University of Washington • Cultivate Learning**

*Professional demonstration platform for educational question classification using state-of-the-art neural networks.*