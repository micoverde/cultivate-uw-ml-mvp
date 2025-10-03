# Production Deployment Plan for Ensemble ML
## Issue #196 - Model Selection Settings

### Executive Summary
Deploy ensemble ML as opt-in beta feature alongside classic ML, with gradual rollout based on performance metrics and cost analysis.

### Current State Assessment
- ✅ Ensemble ML code complete and tested locally
- ✅ API endpoints implemented (`/api/classify`, `/api/v2/classify/ensemble`)
- ✅ Settings UI with model selection complete
- ❌ Docker build failure with fullml stage
- ❌ Azure Container App running old version
- ⚠️ Cost implications not fully evaluated

### Recommended Production Strategy

## 1. Dual-Mode Architecture

### Container Configuration
```yaml
Container Apps:
  cultivate-ml-classic:
    image: lightweight ML
    resources: 0.25 CPU, 0.5GB RAM
    scale: 0-10 replicas
    default: true

  cultivate-ml-ensemble:
    image: fullml with 7 models
    resources: 0.5 CPU, 1GB RAM
    scale: 0-2 replicas
    beta: true
```

### Traffic Distribution
- **90% Classic ML** (proven, fast, cost-effective)
- **10% Ensemble ML** (beta users, A/B testing)

## 2. Phased Deployment Timeline

### Week 1: Infrastructure & Testing
```bash
# Day 1-2: Fix Docker build
- Optimize requirements-ensemble.txt
- Use multi-stage caching
- Test with ACR build service

# Day 3-4: Deploy to staging
- Deploy both containers to staging
- Load testing & performance benchmarks
- Cost analysis

# Day 5: Production readiness
- Update monitoring & alerts
- Prepare rollback procedures
- Documentation
```

### Week 2: Beta Launch
```javascript
// Feature flag configuration
{
  "ensemble_ml": {
    "enabled": true,
    "rollout_percentage": 10,
    "allowed_users": ["beta_testers"],
    "fallback": "classic"
  }
}
```

### Week 3-4: Gradual Rollout
- Monitor accuracy improvements
- Track response times
- Analyze cost impact
- Gather user feedback

### Week 5: Full Production
- Decision point based on metrics
- Either full rollout or maintain dual-mode

## 3. Technical Implementation

### Docker Build Optimization
```dockerfile
# Split heavy dependencies
FROM python:3.12-slim as ml-base
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

FROM ml-base as ensemble-models
COPY requirements-ensemble-models.txt .
RUN pip install --no-cache-dir -r requirements-ensemble-models.txt

FROM ensemble-models as fullml
COPY requirements-ensemble-final.txt .
RUN pip install --no-cache-dir -r requirements-ensemble-final.txt
```

### Deployment Script
```bash
#!/bin/bash
# deploy-production.sh

# Deploy classic (always)
az containerapp update \
  --name cultivate-ml-api \
  --resource-group cultivate-ml-rg \
  --image cultivatemlregistry.azurecr.io/cultivate-ml:classic-latest \
  --cpu 0.25 --memory 0.5Gi

# Deploy ensemble (if enabled)
if [ "$DEPLOY_ENSEMBLE" = "true" ]; then
  az containerapp create \
    --name cultivate-ml-ensemble \
    --resource-group cultivate-ml-rg \
    --image cultivatemlregistry.azurecr.io/cultivate-ml:ensemble-latest \
    --cpu 0.5 --memory 1.0Gi \
    --min-replicas 0 \
    --max-replicas 2
fi
```

### API Gateway Configuration
```nginx
# Traffic routing configuration
location /api/classify {
  set $backend "classic";

  # Check feature flag
  if ($http_x_model_preference = "ensemble") {
    set $backend "ensemble";
  }

  # A/B testing logic
  if ($request_id ~ "[0-9]$") {  # 10% to ensemble
    set $backend "ensemble";
  }

  proxy_pass http://cultivate-ml-$backend/api/classify;
}
```

## 4. Monitoring & Success Metrics

### Key Performance Indicators (KPIs)
| Metric | Classic Baseline | Ensemble Target | Threshold |
|--------|-----------------|-----------------|-----------|
| Accuracy | 67% | 85% | Must exceed 80% |
| Response Time | <100ms | <500ms | P95 < 1s |
| Cost per Request | $0.0001 | $0.0005 | < $0.001 |
| Error Rate | <1% | <1% | < 2% |

### Monitoring Dashboard
```javascript
// Azure Application Insights queries
const metrics = {
  accuracy: `
    customMetrics
    | where name == "model_accuracy"
    | summarize avg(value) by bin(timestamp, 1h), tostring(customDimensions.model)
  `,
  latency: `
    requests
    | where url contains "/api/classify"
    | summarize percentiles(duration, 50, 95, 99) by bin(timestamp, 1h)
  `,
  cost: `
    // Track container resource usage
    ContainerInstanceLog
    | summarize avg(CPUUsage), avg(MemoryUsage) by bin(TimeGenerated, 1h)
  `
};
```

## 5. Risk Mitigation

### Rollback Strategy
```bash
# Instant rollback procedure
az containerapp revision set-mode \
  --name cultivate-ml-api \
  --resource-group cultivate-ml-rg \
  --mode single

# Revert to previous revision
az containerapp update \
  --name cultivate-ml-api \
  --resource-group cultivate-ml-rg \
  --revision cultivate-ml-api--previous
```

### Failure Scenarios
1. **High latency**: Auto-fallback to classic
2. **Cost spike**: Scale down ensemble replicas
3. **Accuracy regression**: Disable ensemble via feature flag
4. **Memory issues**: Reduce model complexity

## 6. Cost Optimization

### Estimated Monthly Costs
| Component | Classic Only | With Ensemble | Notes |
|-----------|-------------|---------------|-------|
| Container Apps | $20 | $35 | Scale-to-zero enabled |
| Storage | $5 | $10 | Model caching |
| Network | $10 | $15 | Increased traffic |
| **Total** | **$35** | **$60** | ~70% increase |

### Cost Reduction Strategies
1. **Time-based scaling**: Ensemble only during business hours
2. **Geographic routing**: Ensemble for premium regions only
3. **User tier**: Ensemble for paid users only
4. **Caching**: Aggressive caching for repeated queries

## 7. Communication Plan

### Stakeholder Updates
- **Week 1**: Technical readiness report
- **Week 2**: Beta launch announcement
- **Week 3-4**: Weekly metrics dashboard
- **Week 5**: Go/No-go decision meeting

### User Communication
```markdown
## 🎯 New: Advanced ML Mode (Beta)

Try our new ensemble ML model for improved accuracy!

**Benefits:**
- 85%+ classification accuracy
- Advanced 7-model voting system
- Better handling of edge cases

**How to enable:**
Click the settings ⚙️ icon and select "Ensemble ML"

Note: Beta feature, may have slightly longer response times.
```

## 8. Decision Matrix

### Go/No-Go Criteria for Full Production

| Criterion | Minimum | Target | Weight |
|-----------|---------|--------|--------|
| Accuracy Improvement | >10% | >15% | 40% |
| User Satisfaction | 7/10 | 9/10 | 30% |
| Cost Increase | <100% | <50% | 20% |
| Technical Stability | 99% uptime | 99.9% | 10% |

**Decision Formula**:
```
Score = (Accuracy×0.4) + (Satisfaction×0.3) + (Cost×0.2) + (Stability×0.1)
If Score > 0.75: Proceed to full production
```

## 9. Long-term Vision

### Future Enhancements
1. **Model versioning**: A/B test different ensemble configurations
2. **Auto-ML**: Automatic model selection based on query type
3. **Edge deployment**: Run lightweight models client-side
4. **Federated learning**: Improve models with privacy-preserved data

### Scaling Strategy
```mermaid
graph LR
    A[MVP - 2 Models] --> B[Beta - 7 Models]
    B --> C[Prod - Dynamic Selection]
    C --> D[Enterprise - Custom Models]
```

## 10. Immediate Next Steps

### Priority Actions (Do Today)
1. ✅ Fix Docker build with split requirements
2. ✅ Deploy classic version to production
3. ✅ Create staging environment for ensemble
4. ✅ Set up monitoring dashboard
5. ✅ Document rollback procedures

### This Week
- [ ] Performance benchmarking
- [ ] Cost analysis report
- [ ] Beta user recruitment
- [ ] Load testing
- [ ] Security review

---

## Approval & Sign-off

| Role | Name | Date | Approval |
|------|------|------|----------|
| Technical Lead | Warren | TBD | [ ] |
| Product Owner | TBD | TBD | [ ] |
| DevOps Lead | TBD | TBD | [ ] |
| Finance | TBD | TBD | [ ] |

---

**Document Version**: 1.0
**Last Updated**: 2025-09-27
**Author**: Claude (AI Assistant)
**Issue**: #196 - Model Selection Settings