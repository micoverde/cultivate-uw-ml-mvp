# Functional Specification: Model Selection Settings UI
## Story 9.1 - Issue #196

### Executive Summary
This specification details the implementation of a user-facing settings interface that allows switching between Classical ML and Ensemble ML models in the Cultivate learning platform. The feature enables real-time model selection without requiring application restart or redeployment.

### Business Requirements
- **Primary Goal:** Allow educators to choose between Classical and Ensemble ML models
- **Success Metrics:**
  - Model switching completes in < 2 seconds
  - Zero downtime during model changes
  - Performance metrics visible for informed decisions

### User Stories

#### Story 1: Educator Discovers Settings
**As an** educator using the Cultivate platform
**I want to** see a settings option in the main hub
**So that** I can access ML model configuration

**Acceptance Criteria:**
- Settings cog icon visible in top navigation
- Icon has tooltip "ML Settings"
- Click opens settings modal/panel

#### Story 2: Model Selection
**As an** educator in settings
**I want to** see current model and available options
**So that** I can make an informed choice

**Acceptance Criteria:**
- Current model clearly indicated
- Radio buttons or toggle for model selection
- Brief description of each model's strengths

#### Story 3: Performance Comparison
**As an** educator evaluating models
**I want to** see performance metrics
**So that** I can choose the best model for my needs

**Acceptance Criteria:**
- Display accuracy percentage for each model
- Show average response time
- Include "last updated" timestamp

### Technical Architecture

#### Frontend Components

```typescript
// Settings Modal Component Structure
interface ModelSettings {
  currentModel: 'classic' | 'ensemble';
  performanceMetrics: {
    classic: PerformanceData;
    ensemble: PerformanceData;
  };
  isAdmin: boolean;
  isSwitching: boolean;
}

interface PerformanceData {
  accuracy: number;
  avgResponseTime: number;
  totalPredictions: number;
  lastUpdated: Date;
}
```

#### API Endpoints

**GET /api/v1/models/current**
```json
{
  "model_type": "ensemble",
  "version": "1.0.0",
  "loaded_at": "2025-09-27T10:00:00Z"
}
```

**POST /api/v1/models/select**
```json
Request:
{
  "model_type": "ensemble"
}

Response:
{
  "success": true,
  "previous_model": "classic",
  "current_model": "ensemble",
  "switch_time_ms": 1250
}
```

**GET /api/v1/models/performance**
```json
{
  "classic": {
    "accuracy": 0.92,
    "f1_score": 0.91,
    "avg_response_ms": 45,
    "total_predictions": 15000
  },
  "ensemble": {
    "accuracy": 0.98,
    "f1_score": 0.97,
    "avg_response_ms": 125,
    "total_predictions": 3000
  }
}
```

### UI/UX Specifications

#### Settings Icon
- **Location:** Top navigation bar, right side
- **Icon:** Font Awesome "fa-cog" or Material "settings"
- **Size:** 24x24px
- **Color:** #666 default, #333 on hover
- **Accessibility:** aria-label="ML Model Settings"

#### Settings Panel
- **Type:** Modal overlay
- **Width:** 600px desktop, 90% mobile
- **Sections:**
  1. Current Model Status
  2. Model Selection
  3. Performance Metrics
  4. Apply/Cancel buttons

#### Visual Mockup
```
┌─────────────────────────────────────┐
│ ML Model Settings                × │
├─────────────────────────────────────┤
│ Current Model: [Classic ML]        │
│                                     │
│ Select Model:                       │
│ ○ Classic ML (Faster, Proven)      │
│ ● Ensemble ML (Higher Accuracy)    │
│                                     │
│ Performance Comparison:             │
│ ┌─────────────────────────────┐    │
│ │ Metric    │ Classic│Ensemble│    │
│ ├───────────┼────────┼────────┤    │
│ │ Accuracy  │  92%   │  98%   │    │
│ │ Speed     │  45ms  │ 125ms  │    │
│ │ Usage     │  15K   │  3K    │    │
│ └─────────────────────────────┘    │
│                                     │
│ [Cancel]            [Apply Changes] │
└─────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Basic Toggle (Week 1)
- Settings icon in navigation
- Simple radio button selection
- LocalStorage persistence
- Basic model switching

#### Phase 2: Performance Metrics (Week 2)
- Add performance comparison table
- Real-time metrics updates
- Historical trend charts
- WebSocket for live updates

#### Phase 3: Advanced Features (Week 3)
- A/B testing framework
- Automatic fallback on poor performance
- Admin-only access control
- Audit logging

### Demo Website Updates

#### File: `/demo/public/demo2_warren_fluent.html`

**Add to Navigation Bar:**
```html
<div class="nav-right">
  <button id="mlSettingsBtn" class="settings-btn" title="ML Model Settings">
    <i class="fas fa-cog"></i>
  </button>
</div>
```

**Settings Modal HTML:**
```html
<div id="mlSettingsModal" class="modal">
  <div class="modal-content">
    <h2>ML Model Settings</h2>
    <div class="model-selection">
      <label>
        <input type="radio" name="model" value="classic" />
        <span>Classic ML - Faster Response</span>
      </label>
      <label>
        <input type="radio" name="model" value="ensemble" />
        <span>Ensemble ML - Higher Accuracy</span>
      </label>
    </div>
    <div class="performance-metrics">
      <!-- Metrics table here -->
    </div>
    <div class="modal-actions">
      <button onclick="closeSettings()">Cancel</button>
      <button onclick="applySettings()">Apply</button>
    </div>
  </div>
</div>
```

**JavaScript Functions:**
```javascript
// Initialize settings on page load
function initializeMLSettings() {
  const savedModel = localStorage.getItem('ml_model') || 'classic';
  setActiveModel(savedModel);
  loadPerformanceMetrics();
}

// Handle model selection
async function applySettings() {
  const selected = document.querySelector('input[name="model"]:checked').value;

  showLoadingSpinner();

  const response = await fetch('/api/v1/models/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_type: selected })
  });

  if (response.ok) {
    localStorage.setItem('ml_model', selected);
    showSuccessMessage('Model switched successfully!');
    updateUIForNewModel(selected);
  } else {
    showErrorMessage('Failed to switch model');
  }

  hideLoadingSpinner();
}

// Update UI elements for new model
function updateUIForNewModel(modelType) {
  // Update status indicator
  document.getElementById('currentModel').textContent =
    modelType === 'ensemble' ? 'Ensemble ML' : 'Classic ML';

  // Update any model-specific UI elements
  document.querySelectorAll('.model-indicator').forEach(el => {
    el.dataset.model = modelType;
  });

  // Refresh performance metrics
  loadPerformanceMetrics();
}
```

**CSS Styling:**
```css
.settings-btn {
  background: none;
  border: none;
  cursor: pointer;
  padding: 8px;
  color: #666;
  transition: color 0.3s;
}

.settings-btn:hover {
  color: #333;
}

.modal {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0,0,0,0.5);
  z-index: 1000;
}

.modal-content {
  background: white;
  margin: 10% auto;
  padding: 20px;
  width: 600px;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.model-selection label {
  display: block;
  padding: 10px;
  margin: 5px 0;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
}

.model-selection label:hover {
  background: #f5f5f5;
}

.model-selection input[type="radio"]:checked + span {
  font-weight: bold;
  color: #007bff;
}

.performance-metrics {
  margin: 20px 0;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 4px;
}

.performance-metrics table {
  width: 100%;
  border-collapse: collapse;
}

.performance-metrics th,
.performance-metrics td {
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 20px;
}

.modal-actions button {
  padding: 8px 16px;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
  background: white;
}

.modal-actions button:last-child {
  background: #007bff;
  color: white;
  border-color: #007bff;
}
```

### Testing Requirements

#### Unit Tests
- Model selection API endpoints
- LocalStorage persistence
- Performance metric calculations
- Model switching logic

#### Integration Tests
- End-to-end model switching
- Performance under load
- Fallback mechanisms
- WebSocket connections

#### User Acceptance Tests
- Settings icon visibility and accessibility
- Modal open/close behavior
- Model selection and persistence
- Performance metrics accuracy
- Response time < 2 seconds

### Performance Requirements
- Model switch completion: < 2 seconds
- Settings panel load: < 500ms
- Performance metrics update: Real-time via WebSocket
- No impact on existing API calls during switch

### Security Considerations
- Admin-only access for production (Phase 1)
- Rate limiting on model switch API (max 10/minute)
- Audit logging of all model changes
- No sensitive data in LocalStorage

### Rollout Plan

#### Week 1: Development
- Implement basic UI components
- Create API endpoints
- Add LocalStorage persistence
- Basic testing

#### Week 2: Testing & Refinement
- Load testing with concurrent users
- UI/UX improvements based on feedback
- Performance optimization
- Documentation

#### Week 3: Gradual Rollout
- Deploy to staging with feature flag
- Enable for 10% of users
- Monitor metrics and feedback
- Full rollout if metrics are positive

### Success Metrics
- 80% of users try the ensemble model
- < 5% switch back to classic after trying ensemble
- No increase in support tickets
- Ensemble model accuracy remains > 95%
- Average response time < 150ms

### Future Enhancements
- User-specific model preferences (Story 10.1)
- Automatic model selection based on use case
- Custom model upload for research
- Model performance analytics dashboard
- A/B testing framework for new models

---

**Document Version:** 1.0
**Author:** Warren & Claude
**Date:** September 27, 2025
**Status:** In Development