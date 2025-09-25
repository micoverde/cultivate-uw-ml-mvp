// Azure Cost Monitoring and Budget Alerts
// Partner-Level Microsoft SDE Implementation for Production Cost Management

import { monitoring } from './monitoring';

// Cost monitoring configuration
const COST_CONFIG = {
  // Azure subscription and resource group details
  subscriptionId: import.meta.env.VITE_AZURE_SUBSCRIPTION_ID || 'default',
  resourceGroupName: 'cultivate-ml-mvp',

  // Budget thresholds (USD)
  budgetAlerts: {
    daily: 10,    // $10 daily limit for demo
    monthly: 300, // $300 monthly limit
    weekly: 70    // $70 weekly limit
  },

  // Service cost tracking
  services: {
    'azure_swa': { name: 'Azure Static Web Apps', freeLimit: 100, ratePer1000: 0.00 },
    'container_instances': { name: 'Azure Container Instances', ratePer1000: 1.34 },
    'application_insights': { name: 'Application Insights', freeLimit: 5000, ratePer1000: 2.30 },
    'storage_account': { name: 'Azure Storage', freeLimit: 5000, ratePer1000: 0.02 },
    'cognitive_services': { name: 'Cognitive Services API', ratePer1000: 1.00 }
  },

  // Cost estimation factors
  estimatedUsage: {
    apiRequestsPerDemo: 5,      // Average API requests per demo session
    storagePerAnalysis: 0.1,    // MB per analysis stored
    insightsEventsPerSession: 50 // Telemetry events per user session
  }
};

class CostMonitoring {
  constructor() {
    this.sessionCosts = {
      currentSession: 0,
      dailyAccumulated: 0,
      weeklyAccumulated: 0,
      monthlyAccumulated: 0
    };

    this.usage = {
      apiRequests: 0,
      storageUsed: 0,
      telemetryEvents: 0,
      userSessions: 0,
      analysisCount: 0
    };

    // Load persisted cost data
    this.loadCostData();
  }

  // Initialize cost monitoring
  initialize() {
    console.log('[Cost Monitor] Initializing cost tracking and budget alerts');

    // Set up periodic cost calculation
    setInterval(() => {
      this.calculateCurrentCosts();
      this.checkBudgetAlerts();
    }, 60000); // Check every minute

    // Track initial session
    this.trackSessionStart();

    monitoring.trackEvent('CostMonitoringInitialized', {
      budgetAlerts: COST_CONFIG.budgetAlerts,
      services: Object.keys(COST_CONFIG.services).length
    });
  }

  // Track API request costs
  trackAPIRequest(endpoint, responseTime, dataSize, success) {
    const estimatedCost = this.calculateAPICost(endpoint, dataSize);

    this.usage.apiRequests++;
    this.sessionCosts.currentSession += estimatedCost;

    // Track cost event
    monitoring.trackCostEvent('api', 'request_processed', {
      endpoint,
      responseTime,
      dataSize,
      estimatedCost,
      success,
      totalSessionCost: this.sessionCosts.currentSession
    });

    // Check if this request pushes us over budget
    this.checkBudgetAlerts();

    console.log(`[Cost Monitor] API Request Cost: $${estimatedCost.toFixed(4)} | Session Total: $${this.sessionCosts.currentSession.toFixed(4)}`);
  }

  // Track storage costs
  trackStorageUsage(operation, dataSize) {
    const estimatedCost = this.calculateStorageCost(dataSize);

    this.usage.storageUsed += dataSize;
    this.sessionCosts.currentSession += estimatedCost;

    monitoring.trackCostEvent('storage', operation, {
      dataSize,
      estimatedCost,
      totalStorageUsed: this.usage.storageUsed,
      totalSessionCost: this.sessionCosts.currentSession
    });
  }

  // Track telemetry costs (Application Insights)
  trackTelemetryEvent(eventType, properties = {}) {
    const estimatedCost = this.calculateTelemetryCost();

    this.usage.telemetryEvents++;
    this.sessionCosts.currentSession += estimatedCost;

    // Only track cost events for significant telemetry (avoid infinite loop)
    if (eventType !== 'telemetry_cost_tracking') {
      monitoring.trackCostEvent('telemetry', eventType, {
        estimatedCost,
        totalTelemetryEvents: this.usage.telemetryEvents,
        totalSessionCost: this.sessionCosts.currentSession,
        ...properties
      });
    }
  }

  // Track ML analysis costs
  trackMLAnalysisCost(analysisType, duration, dataSize, success) {
    const estimatedCost = this.calculateMLCost(analysisType, duration, dataSize);

    this.usage.analysisCount++;
    this.sessionCosts.currentSession += estimatedCost;

    monitoring.trackCostEvent('ml_analysis', analysisType, {
      duration,
      dataSize,
      estimatedCost,
      success,
      totalAnalyses: this.usage.analysisCount,
      totalSessionCost: this.sessionCosts.currentSession
    });

    console.log(`[Cost Monitor] ML Analysis Cost: $${estimatedCost.toFixed(4)} | Type: ${analysisType}`);
  }

  // Calculate API request costs
  calculateAPICost(endpoint, dataSize) {
    let baseCost = 0;

    if (endpoint.includes('analyze')) {
      // ML analysis endpoints are more expensive
      baseCost = COST_CONFIG.services.cognitive_services.ratePer1000 / 1000;

      // Additional cost for data processing
      const dataCostFactor = Math.max(dataSize / 1000, 0.1); // Minimum 0.1KB
      baseCost *= dataCostFactor;
    } else {
      // Standard API calls
      baseCost = COST_CONFIG.services.container_instances.ratePer1000 / 10000; // Much smaller cost
    }

    return baseCost;
  }

  // Calculate storage costs
  calculateStorageCost(dataSize) {
    const dataSizeGB = dataSize / (1024 * 1024 * 1024); // Convert to GB
    return dataSizeGB * COST_CONFIG.services.storage_account.ratePer1000 / 1000;
  }

  // Calculate telemetry costs
  calculateTelemetryCost() {
    // Application Insights has free tier of 5GB/month
    if (this.usage.telemetryEvents < COST_CONFIG.services.application_insights.freeLimit) {
      return 0;
    }

    return COST_CONFIG.services.application_insights.ratePer1000 / 100000; // Very small per-event cost
  }

  // Calculate ML processing costs
  calculateMLCost(analysisType, duration, dataSize) {
    const baseCost = COST_CONFIG.services.cognitive_services.ratePer1000 / 1000;

    // Factor in processing time (longer analysis = more compute cost)
    const durationFactor = Math.max(duration / 1000 / 60, 0.1); // Minutes, minimum 0.1

    // Factor in data size
    const dataSizeFactor = Math.max(dataSize / 1000, 1); // KB, minimum 1KB

    return baseCost * durationFactor * dataSizeFactor;
  }

  // Calculate current total costs
  calculateCurrentCosts() {
    // This would integrate with Azure Cost Management APIs in production
    const totalSessionCost = this.sessionCosts.currentSession;

    // Update accumulated costs (in production, this would come from Azure APIs)
    this.updateAccumulatedCosts();

    return {
      currentSession: totalSessionCost,
      ...this.sessionCosts,
      projectedDaily: this.projectDailyCost(),
      projectedMonthly: this.projectMonthlyCost()
    };
  }

  // Project daily cost based on current usage
  projectDailyCost() {
    const currentHour = new Date().getHours();
    if (currentHour === 0) return this.sessionCosts.currentSession;

    const hourlyRate = this.sessionCosts.currentSession / ((Date.now() - this.getSessionStartTime()) / (1000 * 60 * 60));
    return hourlyRate * 24;
  }

  // Project monthly cost
  projectMonthlyCost() {
    const currentDay = new Date().getDate();
    if (currentDay === 1) return this.sessionCosts.dailyAccumulated;

    const dailyAverage = this.sessionCosts.weeklyAccumulated / 7;
    return dailyAverage * 30; // Approximate month
  }

  // Check budget alerts
  checkBudgetAlerts() {
    const costs = this.calculateCurrentCosts();

    // Daily budget check
    if (costs.projectedDaily > COST_CONFIG.budgetAlerts.daily) {
      this.triggerBudgetAlert('daily', costs.projectedDaily, COST_CONFIG.budgetAlerts.daily);
    }

    // Weekly budget check
    if (this.sessionCosts.weeklyAccumulated > COST_CONFIG.budgetAlerts.weekly) {
      this.triggerBudgetAlert('weekly', this.sessionCosts.weeklyAccumulated, COST_CONFIG.budgetAlerts.weekly);
    }

    // Monthly budget check
    if (costs.projectedMonthly > COST_CONFIG.budgetAlerts.monthly) {
      this.triggerBudgetAlert('monthly', costs.projectedMonthly, COST_CONFIG.budgetAlerts.monthly);
    }
  }

  // Trigger budget alert
  triggerBudgetAlert(period, currentCost, budgetLimit) {
    const alert = {
      severity: 'critical',
      period,
      currentCost: currentCost.toFixed(2),
      budgetLimit: budgetLimit.toFixed(2),
      overageAmount: (currentCost - budgetLimit).toFixed(2),
      overagePercentage: ((currentCost / budgetLimit - 1) * 100).toFixed(1)
    };

    console.warn(`[Cost Alert] ${period.toUpperCase()} BUDGET EXCEEDED!`, alert);

    // Track budget alert event
    monitoring.trackEvent('BudgetAlertTriggered', alert);

    // In production, this would send alerts via email/SMS/Slack
    if (typeof window !== 'undefined' && window.alert) {
      window.alert(
        `⚠️ BUDGET ALERT: ${period} spending ($${alert.currentCost}) has exceeded the limit ($${alert.budgetLimit}) by $${alert.overageAmount} (${alert.overagePercentage}% over)`
      );
    }
  }

  // Get current cost summary for dashboard
  getCostSummary() {
    const costs = this.calculateCurrentCosts();

    return {
      current: costs,
      usage: this.usage,
      budgets: COST_CONFIG.budgetAlerts,
      services: COST_CONFIG.services,
      alerts: this.getActiveAlerts(),
      recommendations: this.getCostOptimizationRecommendations()
    };
  }

  // Get active budget alerts
  getActiveAlerts() {
    const costs = this.calculateCurrentCosts();
    const alerts = [];

    if (costs.projectedDaily > COST_CONFIG.budgetAlerts.daily * 0.8) {
      alerts.push({
        type: 'warning',
        message: `Daily spending approaching limit (${(costs.projectedDaily / COST_CONFIG.budgetAlerts.daily * 100).toFixed(0)}%)`
      });
    }

    if (this.usage.apiRequests > 100) {
      alerts.push({
        type: 'info',
        message: `High API usage detected (${this.usage.apiRequests} requests this session)`
      });
    }

    return alerts;
  }

  // Get cost optimization recommendations
  getCostOptimizationRecommendations() {
    const recommendations = [];

    if (this.usage.apiRequests > 50) {
      recommendations.push({
        priority: 'medium',
        category: 'API Optimization',
        message: 'Consider implementing caching to reduce API requests',
        potentialSavings: (this.usage.apiRequests * 0.001).toFixed(2)
      });
    }

    if (this.usage.telemetryEvents > 1000) {
      recommendations.push({
        priority: 'low',
        category: 'Telemetry Optimization',
        message: 'Review telemetry collection to reduce unnecessary events',
        potentialSavings: (this.usage.telemetryEvents * 0.00001).toFixed(4)
      });
    }

    return recommendations;
  }

  // Session management
  trackSessionStart() {
    this.usage.userSessions++;
    this.sessionStartTime = Date.now();

    monitoring.trackCostEvent('session', 'session_started', {
      userSessions: this.usage.userSessions,
      sessionStartTime: this.sessionStartTime
    });
  }

  getSessionStartTime() {
    return this.sessionStartTime || Date.now();
  }

  // Persistence helpers
  saveCostData() {
    try {
      const costData = {
        sessionCosts: this.sessionCosts,
        usage: this.usage,
        lastUpdated: Date.now()
      };

      localStorage.setItem('cultivate_cost_data', JSON.stringify(costData));
    } catch (error) {
      console.warn('[Cost Monitor] Failed to save cost data:', error);
    }
  }

  loadCostData() {
    try {
      const saved = localStorage.getItem('cultivate_cost_data');
      if (saved) {
        const costData = JSON.parse(saved);

        // Reset if data is from a different day
        const lastUpdated = new Date(costData.lastUpdated);
        const today = new Date();

        if (lastUpdated.toDateString() !== today.toDateString()) {
          // New day - reset daily costs, update weekly/monthly
          this.sessionCosts.dailyAccumulated = costData.sessionCosts.currentSession || 0;
          this.sessionCosts.currentSession = 0;
        } else {
          this.sessionCosts = { ...this.sessionCosts, ...costData.sessionCosts };
          this.usage = { ...this.usage, ...costData.usage };
        }
      }
    } catch (error) {
      console.warn('[Cost Monitor] Failed to load cost data:', error);
    }
  }

  updateAccumulatedCosts() {
    // In production, this would fetch from Azure Cost Management APIs
    // For now, simulate based on current session
    this.sessionCosts.dailyAccumulated += this.sessionCosts.currentSession * 0.1; // 10% of session cost
    this.sessionCosts.weeklyAccumulated = this.sessionCosts.dailyAccumulated * 7;
    this.sessionCosts.monthlyAccumulated = this.sessionCosts.dailyAccumulated * 30;

    this.saveCostData();
  }

  // Public API for integration
  getCostForDisplay() {
    return {
      currentSession: `$${this.sessionCosts.currentSession.toFixed(4)}`,
      projectedDaily: `$${this.projectDailyCost().toFixed(2)}`,
      projectedMonthly: `$${this.projectMonthlyCost().toFixed(2)}`,
      budgetStatus: {
        daily: (this.projectDailyCost() / COST_CONFIG.budgetAlerts.daily * 100).toFixed(1) + '%',
        monthly: (this.projectMonthlyCost() / COST_CONFIG.budgetAlerts.monthly * 100).toFixed(1) + '%'
      }
    };
  }
}

// Export singleton instance
export const costMonitoring = new CostMonitoring();

// Export configuration for other modules
export { COST_CONFIG };

// Auto-initialize cost monitoring
if (typeof window !== 'undefined') {
  // Initialize after DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => costMonitoring.initialize());
  } else {
    costMonitoring.initialize();
  }
}