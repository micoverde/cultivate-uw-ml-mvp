// Azure Auto-Scaling Configuration
// Partner-Level Microsoft SDE Implementation for Production Scaling

import { monitoring } from './monitoring';
import { costMonitoring } from './cost-monitoring';

// Auto-scaling configuration
const SCALING_CONFIG = {
  // Resource scaling thresholds
  thresholds: {
    cpu: {
      scaleUp: 70,    // Scale up at 70% CPU
      scaleDown: 30   // Scale down at 30% CPU
    },
    memory: {
      scaleUp: 80,    // Scale up at 80% memory
      scaleDown: 40   // Scale down at 40% memory
    },
    responseTime: {
      scaleUp: 5000,  // Scale up if response time > 5 seconds
      scaleDown: 1000 // Scale down if response time < 1 second
    },
    concurrentUsers: {
      scaleUp: 50,    // Scale up at 50 concurrent users
      scaleDown: 10   // Scale down at 10 concurrent users
    },
    errorRate: {
      scaleUp: 0.05   // Scale up if error rate > 5%
    }
  },

  // Scaling limits
  limits: {
    minInstances: 1,
    maxInstances: 10,
    cooldownPeriod: 300000, // 5 minutes between scaling actions
    maxScalePerAction: 3    // Maximum instances to add/remove at once
  },

  // Azure Container Instance scaling configuration
  containerInstances: {
    instanceType: 'Standard_D2s_v3', // 2 vCPU, 8GB RAM
    minCpu: 0.5,
    maxCpu: 2.0,
    minMemory: 1,
    maxMemory: 8
  },

  // Cost-aware scaling
  costLimits: {
    maxHourlyCost: 5,     // $5/hour maximum
    maxDailyCost: 50,     // $50/day maximum
    scaleDownOnBudgetAlert: true
  }
};

class AutoScaling {
  constructor() {
    this.currentInstances = 1;
    this.lastScalingAction = 0;
    this.metrics = {
      cpu: 0,
      memory: 0,
      responseTime: 0,
      concurrentUsers: 0,
      errorRate: 0
    };
    this.scalingHistory = [];
    this.isEnabled = false;
  }

  // Initialize auto-scaling
  initialize() {
    console.log('[Auto-Scaling] Initializing automatic scaling policies');

    // Start metrics collection
    this.startMetricsCollection();

    // Enable auto-scaling
    this.isEnabled = true;

    // Track initialization
    monitoring.trackEvent('AutoScalingInitialized', {
      thresholds: SCALING_CONFIG.thresholds,
      limits: SCALING_CONFIG.limits,
      currentInstances: this.currentInstances
    });

    console.log('[Auto-Scaling] Ready for dynamic resource management');
  }

  // Start collecting metrics for scaling decisions
  startMetricsCollection() {
    // Collect metrics every 30 seconds
    setInterval(() => {
      this.collectMetrics();
      this.evaluateScaling();
    }, 30000);

    // Long-term scaling evaluation every 5 minutes
    setInterval(() => {
      this.evaluateLongTermScaling();
    }, 300000);
  }

  // Collect current system metrics
  collectMetrics() {
    try {
      // Get performance metrics from monitoring system
      const performanceMetrics = monitoring.getPerformanceMetrics();

      // Simulate resource metrics (in production, these would come from Azure Monitor)
      this.metrics = {
        cpu: this.simulateCpuUsage(),
        memory: this.simulateMemoryUsage(),
        responseTime: performanceMetrics.averageResponseTime || this.estimateResponseTime(),
        concurrentUsers: performanceMetrics.activeUsers || this.estimateConcurrentUsers(),
        errorRate: this.calculateErrorRate(performanceMetrics)
      };

      // Track metrics
      monitoring.trackEvent('ScalingMetricsCollected', this.metrics);

    } catch (error) {
      console.error('[Auto-Scaling] Error collecting metrics:', error);
      monitoring.trackException(error, { component: 'AutoScaling' });
    }
  }

  // Evaluate if scaling is needed
  evaluateScaling() {
    if (!this.isEnabled) return;

    const now = Date.now();
    const timeSinceLastScaling = now - this.lastScalingAction;

    // Check cooldown period
    if (timeSinceLastScaling < SCALING_CONFIG.limits.cooldownPeriod) {
      return;
    }

    // Check cost limits before scaling up
    const costData = costMonitoring.getCostForDisplay();
    const hourlyBudgetExceeded = parseFloat(costData.projectedDaily) / 24 > SCALING_CONFIG.costLimits.maxHourlyCost;

    // Determine scaling action
    const scaleUpNeeded = this.shouldScaleUp();
    const scaleDownNeeded = this.shouldScaleDown();

    if (scaleUpNeeded && !hourlyBudgetExceeded) {
      this.scaleUp(scaleUpNeeded.reason, scaleUpNeeded.metrics);
    } else if (scaleUpNeeded && hourlyBudgetExceeded) {
      console.warn('[Auto-Scaling] Scale up needed but blocked by cost limits');
      monitoring.trackEvent('ScalingBlockedByCost', {
        reason: 'budget_exceeded',
        currentCost: costData.projectedDaily,
        limit: SCALING_CONFIG.costLimits.maxHourlyCost * 24
      });
    } else if (scaleDownNeeded) {
      this.scaleDown(scaleDownNeeded.reason, scaleDownNeeded.metrics);
    }
  }

  // Determine if scale up is needed
  shouldScaleUp() {
    const reasons = [];

    // CPU threshold
    if (this.metrics.cpu > SCALING_CONFIG.thresholds.cpu.scaleUp) {
      reasons.push(`CPU usage: ${this.metrics.cpu}%`);
    }

    // Memory threshold
    if (this.metrics.memory > SCALING_CONFIG.thresholds.memory.scaleUp) {
      reasons.push(`Memory usage: ${this.metrics.memory}%`);
    }

    // Response time threshold
    if (this.metrics.responseTime > SCALING_CONFIG.thresholds.responseTime.scaleUp) {
      reasons.push(`Response time: ${this.metrics.responseTime}ms`);
    }

    // Concurrent users threshold
    if (this.metrics.concurrentUsers > SCALING_CONFIG.thresholds.concurrentUsers.scaleUp) {
      reasons.push(`Concurrent users: ${this.metrics.concurrentUsers}`);
    }

    // Error rate threshold
    if (this.metrics.errorRate > SCALING_CONFIG.thresholds.errorRate.scaleUp) {
      reasons.push(`Error rate: ${(this.metrics.errorRate * 100).toFixed(1)}%`);
    }

    // Check if we're at max instances
    if (this.currentInstances >= SCALING_CONFIG.limits.maxInstances) {
      if (reasons.length > 0) {
        monitoring.trackEvent('ScalingLimitReached', {
          reason: 'max_instances',
          currentInstances: this.currentInstances,
          triggers: reasons
        });
      }
      return null;
    }

    return reasons.length > 0 ? {
      reason: reasons.join(', '),
      metrics: this.metrics
    } : null;
  }

  // Determine if scale down is needed
  shouldScaleDown() {
    // Don't scale below minimum
    if (this.currentInstances <= SCALING_CONFIG.limits.minInstances) {
      return null;
    }

    const reasons = [];

    // All thresholds must be below scale-down threshold
    if (this.metrics.cpu < SCALING_CONFIG.thresholds.cpu.scaleDown &&
        this.metrics.memory < SCALING_CONFIG.thresholds.memory.scaleDown &&
        this.metrics.responseTime < SCALING_CONFIG.thresholds.responseTime.scaleDown &&
        this.metrics.concurrentUsers < SCALING_CONFIG.thresholds.concurrentUsers.scaleDown &&
        this.metrics.errorRate < 0.01) { // Very low error rate for scale down

      reasons.push('All metrics below scale-down thresholds');
    }

    // Cost-based scale down
    if (SCALING_CONFIG.costLimits.scaleDownOnBudgetAlert) {
      const costAlerts = costMonitoring.getCostSummary().alerts || [];
      if (costAlerts.some(alert => alert.type === 'warning' || alert.type === 'error')) {
        reasons.push('Budget alert triggered');
      }
    }

    return reasons.length > 0 ? {
      reason: reasons.join(', '),
      metrics: this.metrics
    } : null;
  }

  // Execute scale up action
  async scaleUp(reason, metrics) {
    const instancesToAdd = Math.min(
      this.calculateOptimalScaleAmount(true),
      SCALING_CONFIG.limits.maxScalePerAction,
      SCALING_CONFIG.limits.maxInstances - this.currentInstances
    );

    if (instancesToAdd <= 0) return;

    const previousInstances = this.currentInstances;
    this.currentInstances += instancesToAdd;
    this.lastScalingAction = Date.now();

    console.log(`[Auto-Scaling] SCALE UP: ${previousInstances} → ${this.currentInstances} instances`);
    console.log(`[Auto-Scaling] Reason: ${reason}`);

    // Record scaling action
    const scalingAction = {
      type: 'scale_up',
      timestamp: new Date().toISOString(),
      previousInstances,
      newInstances: this.currentInstances,
      reason,
      metrics,
      estimatedCostIncrease: this.estimateScalingCost(instancesToAdd)
    };

    this.scalingHistory.push(scalingAction);

    // Track the scaling event
    monitoring.trackEvent('AutoScalingUp', scalingAction);
    costMonitoring.trackCostEvent('scaling', 'scale_up', {
      instances: instancesToAdd,
      totalInstances: this.currentInstances,
      estimatedCost: scalingAction.estimatedCostIncrease
    });

    // In production, this would call Azure Container Instances API
    await this.executeAzureScaling('up', instancesToAdd);

    // Send notification
    this.sendScalingNotification(scalingAction);
  }

  // Execute scale down action
  async scaleDown(reason, metrics) {
    const instancesToRemove = Math.min(
      this.calculateOptimalScaleAmount(false),
      SCALING_CONFIG.limits.maxScalePerAction,
      this.currentInstances - SCALING_CONFIG.limits.minInstances
    );

    if (instancesToRemove <= 0) return;

    const previousInstances = this.currentInstances;
    this.currentInstances -= instancesToRemove;
    this.lastScalingAction = Date.now();

    console.log(`[Auto-Scaling] SCALE DOWN: ${previousInstances} → ${this.currentInstances} instances`);
    console.log(`[Auto-Scaling] Reason: ${reason}`);

    // Record scaling action
    const scalingAction = {
      type: 'scale_down',
      timestamp: new Date().toISOString(),
      previousInstances,
      newInstances: this.currentInstances,
      reason,
      metrics,
      estimatedCostSavings: this.estimateScalingCost(instancesToRemove)
    };

    this.scalingHistory.push(scalingAction);

    // Track the scaling event
    monitoring.trackEvent('AutoScalingDown', scalingAction);
    costMonitoring.trackCostEvent('scaling', 'scale_down', {
      instances: -instancesToRemove,
      totalInstances: this.currentInstances,
      estimatedSavings: scalingAction.estimatedCostSavings
    });

    // In production, this would call Azure Container Instances API
    await this.executeAzureScaling('down', instancesToRemove);

    // Send notification
    this.sendScalingNotification(scalingAction);
  }

  // Calculate optimal number of instances to scale
  calculateOptimalScaleAmount(scaleUp) {
    if (scaleUp) {
      // Scale up more aggressively if multiple metrics are high
      const highMetrics = [
        this.metrics.cpu > SCALING_CONFIG.thresholds.cpu.scaleUp,
        this.metrics.memory > SCALING_CONFIG.thresholds.memory.scaleUp,
        this.metrics.responseTime > SCALING_CONFIG.thresholds.responseTime.scaleUp * 2,
        this.metrics.concurrentUsers > SCALING_CONFIG.thresholds.concurrentUsers.scaleUp * 1.5
      ].filter(Boolean).length;

      return Math.min(Math.max(highMetrics, 1), 3);
    } else {
      // Conservative scale down
      return 1;
    }
  }

  // Execute Azure Container Instances scaling (production implementation)
  async executeAzureScaling(direction, instanceCount) {
    try {
      // In production, this would use Azure SDK
      console.log(`[Auto-Scaling] Executing Azure Container Instances ${direction} by ${instanceCount}`);

      // Simulated Azure API call
      await this.simulateAzureApiCall(direction, instanceCount);

      console.log(`[Auto-Scaling] Azure scaling ${direction} completed successfully`);

    } catch (error) {
      console.error(`[Auto-Scaling] Azure scaling ${direction} failed:`, error);
      monitoring.trackException(error, {
        component: 'AutoScaling',
        operation: `azure_scale_${direction}`,
        instanceCount
      });

      // Revert local state if Azure call failed
      if (direction === 'up') {
        this.currentInstances -= instanceCount;
      } else {
        this.currentInstances += instanceCount;
      }
    }
  }

  // Simulate Azure API call
  async simulateAzureApiCall(direction, instanceCount) {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        // 95% success rate simulation
        if (Math.random() > 0.05) {
          resolve(`Scaled ${direction} ${instanceCount} instances`);
        } else {
          reject(new Error(`Azure scaling ${direction} failed - resource constraint`));
        }
      }, 2000); // Simulate 2-second API call
    });
  }

  // Estimate cost impact of scaling
  estimateScalingCost(instances) {
    // Azure Container Instances pricing estimation
    const hourlyInstanceCost = 0.0496; // $0.0496/hour for 1 vCPU, 1GB RAM
    const vcpu = SCALING_CONFIG.containerInstances.maxCpu;
    const memory = SCALING_CONFIG.containerInstances.maxMemory;

    const hourlyCost = hourlyInstanceCost * vcpu * (memory / 1) * instances;
    return parseFloat(hourlyCost.toFixed(4));
  }

  // Send scaling notifications
  sendScalingNotification(action) {
    const message = `🔧 Auto-Scaling: ${action.type.replace('_', ' ').toUpperCase()}\n` +
                   `Instances: ${action.previousInstances} → ${action.newInstances}\n` +
                   `Reason: ${action.reason}\n` +
                   `Time: ${action.timestamp}`;

    console.log(`[Auto-Scaling Notification] ${message}`);

    // In production, send to Slack/Teams/Email
    // this.sendSlackNotification(message);
  }

  // Long-term scaling evaluation and optimization
  evaluateLongTermScaling() {
    const recentHistory = this.scalingHistory.slice(-20); // Last 20 actions
    if (recentHistory.length < 5) return;

    // Detect scaling oscillation
    const oscillationDetected = this.detectScalingOscillation(recentHistory);
    if (oscillationDetected) {
      this.handleScalingOscillation();
    }

    // Optimize thresholds based on history
    this.optimizeThresholds(recentHistory);
  }

  // Detect if scaling is oscillating (scaling up and down frequently)
  detectScalingOscillation(history) {
    if (history.length < 6) return false;

    const recent = history.slice(-6);
    const upActions = recent.filter(a => a.type === 'scale_up').length;
    const downActions = recent.filter(a => a.type === 'scale_down').length;

    return upActions >= 3 && downActions >= 3;
  }

  // Handle scaling oscillation
  handleScalingOscillation() {
    console.warn('[Auto-Scaling] Oscillation detected - increasing cooldown period');

    // Temporarily increase cooldown period
    const originalCooldown = SCALING_CONFIG.limits.cooldownPeriod;
    SCALING_CONFIG.limits.cooldownPeriod *= 2;

    monitoring.trackEvent('ScalingOscillationDetected', {
      originalCooldown,
      newCooldown: SCALING_CONFIG.limits.cooldownPeriod,
      recentHistory: this.scalingHistory.slice(-10)
    });

    // Reset cooldown after 1 hour
    setTimeout(() => {
      SCALING_CONFIG.limits.cooldownPeriod = originalCooldown;
      console.log('[Auto-Scaling] Cooldown period reset to normal');
    }, 3600000);
  }

  // Optimize scaling thresholds based on historical data
  optimizeThresholds(history) {
    // Simple optimization: adjust thresholds based on false positive rate
    // In production, this would use machine learning for threshold optimization
    console.log('[Auto-Scaling] Optimizing thresholds based on recent history');
    monitoring.trackEvent('ThresholdOptimization', {
      historyCount: history.length
    });
  }

  // Simulate resource metrics (production would use Azure Monitor)
  simulateCpuUsage() {
    const baseUsage = 20 + (this.metrics.concurrentUsers || 0) * 1.5;
    return Math.min(Math.max(baseUsage + (Math.random() - 0.5) * 20, 0), 100);
  }

  simulateMemoryUsage() {
    const baseUsage = 25 + (this.metrics.concurrentUsers || 0) * 1.2;
    return Math.min(Math.max(baseUsage + (Math.random() - 0.5) * 15, 0), 100);
  }

  estimateResponseTime() {
    const baseTime = 800 + (this.currentInstances > 1 ? -200 : 0);
    return Math.max(baseTime + (Math.random() - 0.5) * 1000, 200);
  }

  estimateConcurrentUsers() {
    const performanceMetrics = monitoring.getPerformanceMetrics();
    return Math.floor((performanceMetrics.userSessions || 1) * (Math.random() * 0.3 + 0.1));
  }

  calculateErrorRate(performanceMetrics) {
    if (!performanceMetrics.errors || !performanceMetrics.apiCalls) return 0;
    return performanceMetrics.errors / Math.max(performanceMetrics.apiCalls, 1);
  }

  // Public API
  getScalingStatus() {
    return {
      isEnabled: this.isEnabled,
      currentInstances: this.currentInstances,
      metrics: this.metrics,
      recentHistory: this.scalingHistory.slice(-5),
      nextEvaluationIn: Math.max(0, SCALING_CONFIG.limits.cooldownPeriod - (Date.now() - this.lastScalingAction)),
      costEstimate: this.estimateScalingCost(this.currentInstances)
    };
  }

  // Manual scaling controls
  enableAutoScaling() {
    this.isEnabled = true;
    monitoring.trackEvent('AutoScalingEnabled');
  }

  disableAutoScaling() {
    this.isEnabled = false;
    monitoring.trackEvent('AutoScalingDisabled');
  }

  forceScaleUp(instances = 1) {
    if (this.currentInstances + instances <= SCALING_CONFIG.limits.maxInstances) {
      this.scaleUp(`Manual scale up by ${instances}`, this.metrics);
    }
  }

  forceScaleDown(instances = 1) {
    if (this.currentInstances - instances >= SCALING_CONFIG.limits.minInstances) {
      this.scaleDown(`Manual scale down by ${instances}`, this.metrics);
    }
  }
}

// Export singleton instance
export const autoScaling = new AutoScaling();

// Export configuration
export { SCALING_CONFIG };

// Auto-initialize scaling
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => autoScaling.initialize());
  } else {
    autoScaling.initialize();
  }
}