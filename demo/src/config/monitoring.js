// Azure Application Insights Configuration
// Partner-Level Microsoft SDE Implementation for Production Monitoring

import { ApplicationInsights } from '@microsoft/applicationinsights-web';

// Environment-based configuration
const MONITORING_CONFIG = {
  // Azure Application Insights Instrumentation Key (configured in Azure)
  instrumentationKey: import.meta.env.VITE_APPINSIGHTS_INSTRUMENTATIONKEY || '4ae4e7dd-900e-48da-836e-bdadbc323b5a',

  // Production environment detection
  isProduction: import.meta.env.VITE_ENVIRONMENT === 'production' ||
                window.location.hostname.endsWith('.azurestaticapps.net'),

  // Monitoring enablement (default: enabled in production, disabled in dev)
  enableMonitoring: import.meta.env.VITE_ENABLE_MONITORING !== 'false',

  // Performance monitoring thresholds
  performanceThresholds: {
    pageLoadTime: 3000, // 3 seconds max acceptable page load
    apiResponseTime: 5000, // 5 seconds max acceptable API response
    criticalErrorThreshold: 5 // Max errors before alerting
  },

  // Cost monitoring settings
  costTracking: {
    trackDependencies: true,
    trackAjaxRequests: true,
    trackPageViews: true,
    trackUserBehavior: true
  }
};

// Initialize Application Insights instance
class CultivateMonitoring {
  constructor() {
    this.appInsights = null;
    this.isInitialized = false;
    this.performanceMetrics = {
      pageLoads: 0,
      apiCalls: 0,
      errors: 0,
      userSessions: 0
    };
  }

  // Initialize monitoring system
  initialize() {
    try {
      if (!MONITORING_CONFIG.enableMonitoring) {
        console.log('[Monitoring] Disabled in development environment');
        return;
      }

      this.appInsights = new ApplicationInsights({
        config: {
          instrumentationKey: MONITORING_CONFIG.instrumentationKey,

          // Performance monitoring configuration
          enableAutoRouteTracking: true,
          enableRequestHeaderTracking: true,
          enableResponseHeaderTracking: true,
          enableAjaxPerfTracking: true,
          maxAjaxCallsPerView: 500,

          // Privacy and compliance
          enableCorsCorrelation: true,
          disableFetchTracking: false,
          disableAjaxTracking: false,

          // Custom telemetry processors
          telemetryProcessors: [
            // Filter out non-critical telemetry in development
            (envelope) => {
              if (!MONITORING_CONFIG.isProduction) {
                // Only track errors and custom events in dev
                return envelope.baseType === 'ExceptionData' ||
                       envelope.baseType === 'EventData';
              }
              return true;
            }
          ],

          // Session and user tracking
          autoTrackPageVisitTime: true,
          enableSessionStorageBuffer: true,

          // Error tracking
          autoExceptionInstrumented: true,
          enableUnhandledPromiseRejectionTracking: true
        }
      });

      this.appInsights.loadAppInsights();
      this.appInsights.trackPageView(); // Track initial page load

      this.isInitialized = true;
      this.setupEventListeners();
      this.trackSessionStart();

      console.log('[Monitoring] Application Insights initialized successfully');

      // Track initialization success
      this.trackEvent('MonitoringInitialized', {
        environment: MONITORING_CONFIG.isProduction ? 'production' : 'development',
        instrumentationKey: MONITORING_CONFIG.instrumentationKey.substring(0, 8) + '...',
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('[Monitoring] Failed to initialize Application Insights:', error);

      // Fallback: basic error logging
      this.setupFallbackErrorTracking();
    }
  }

  // Track custom events (user interactions, demo usage, etc.)
  trackEvent(name, properties = {}, measurements = {}) {
    try {
      if (this.isInitialized && this.appInsights) {
        this.appInsights.trackEvent({
          name: `Cultivate_${name}`,
          properties: {
            ...properties,
            sessionId: this.getSessionId(),
            userAgent: navigator.userAgent,
            timestamp: new Date().toISOString(),
            environment: MONITORING_CONFIG.isProduction ? 'production' : 'development'
          },
          measurements
        });
      }
    } catch (error) {
      console.error('[Monitoring] Error tracking event:', error);
    }
  }

  // Track API performance and costs
  trackDependency(name, data, duration, resultCode, success) {
    try {
      if (this.isInitialized && this.appInsights) {
        this.appInsights.trackDependencyData({
          target: data,
          name: `API_${name}`,
          data: data,
          duration: duration,
          resultCode: resultCode,
          success: success,
          properties: {
            sessionId: this.getSessionId(),
            timestamp: new Date().toISOString()
          }
        });

        // Track performance against thresholds
        if (duration > MONITORING_CONFIG.performanceThresholds.apiResponseTime) {
          this.trackEvent('SlowAPIResponse', {
            apiName: name,
            duration: duration,
            threshold: MONITORING_CONFIG.performanceThresholds.apiResponseTime
          });
        }

        this.performanceMetrics.apiCalls++;
      }
    } catch (error) {
      console.error('[Monitoring] Error tracking dependency:', error);
    }
  }

  // Track exceptions and errors
  trackException(error, properties = {}) {
    try {
      if (this.isInitialized && this.appInsights) {
        this.appInsights.trackException({
          exception: error,
          properties: {
            ...properties,
            sessionId: this.getSessionId(),
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent
          }
        });

        this.performanceMetrics.errors++;

        // Alert on critical error threshold
        if (this.performanceMetrics.errors >= MONITORING_CONFIG.performanceThresholds.criticalErrorThreshold) {
          this.trackEvent('CriticalErrorThresholdReached', {
            errorCount: this.performanceMetrics.errors,
            threshold: MONITORING_CONFIG.performanceThresholds.criticalErrorThreshold
          });
        }
      }
    } catch (trackingError) {
      console.error('[Monitoring] Error tracking exception:', trackingError);
      console.error('[Original Error]:', error);
    }
  }

  // Track page performance
  trackPageView(name, properties = {}) {
    try {
      if (this.isInitialized && this.appInsights) {
        this.appInsights.trackPageView({
          name: name,
          properties: {
            ...properties,
            sessionId: this.getSessionId(),
            timestamp: new Date().toISOString()
          }
        });

        this.performanceMetrics.pageLoads++;
      }
    } catch (error) {
      console.error('[Monitoring] Error tracking page view:', error);
    }
  }

  // Track demo-specific metrics
  trackDemoInteraction(scenario, action, properties = {}) {
    this.trackEvent('DemoInteraction', {
      scenario,
      action,
      ...properties,
      category: 'demo_engagement'
    });
  }

  // Track ML analysis events
  trackMLAnalysis(analysisType, duration, success, properties = {}) {
    this.trackEvent('MLAnalysis', {
      analysisType,
      duration,
      success,
      ...properties,
      category: 'ml_performance'
    });
  }

  // Track cost-related events (for budget monitoring)
  trackCostEvent(service, operation, properties = {}) {
    this.trackEvent('CostTracking', {
      service,
      operation,
      ...properties,
      category: 'cost_monitoring'
    });
  }

  // Setup event listeners for automatic tracking
  setupEventListeners() {
    // Track unhandled errors
    window.addEventListener('error', (event) => {
      this.trackException(new Error(event.message), {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        type: 'global_error'
      });
    });

    // Track unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.trackException(new Error(event.reason), {
        type: 'unhandled_promise_rejection'
      });
    });

    // Track page visibility changes (for session tracking)
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        this.trackEvent('PageVisible');
      } else {
        this.trackEvent('PageHidden');
      }
    });

    // Performance monitoring
    if ('performance' in window) {
      window.addEventListener('load', () => {
        setTimeout(() => {
          const perfData = performance.timing;
          const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;

          this.trackEvent('PageLoadPerformance', {
            loadTime: pageLoadTime,
            domContentLoaded: perfData.domContentLoadedEventEnd - perfData.navigationStart,
            firstPaint: perfData.responseStart - perfData.navigationStart
          }, {
            pageLoadTime: pageLoadTime
          });

          // Alert on slow page loads
          if (pageLoadTime > MONITORING_CONFIG.performanceThresholds.pageLoadTime) {
            this.trackEvent('SlowPageLoad', {
              loadTime: pageLoadTime,
              threshold: MONITORING_CONFIG.performanceThresholds.pageLoadTime
            });
          }
        }, 0);
      });
    }
  }

  // Session management
  trackSessionStart() {
    const sessionId = this.getSessionId();
    this.trackEvent('SessionStart', {
      sessionId,
      userAgent: navigator.userAgent,
      referrer: document.referrer,
      language: navigator.language
    });
    this.performanceMetrics.userSessions++;
  }

  getSessionId() {
    let sessionId = sessionStorage.getItem('cultivate_session_id');
    if (!sessionId) {
      sessionId = 'cultivate_' + Date.now() + '_' + Math.random().toString(36).substring(2, 15);
      sessionStorage.setItem('cultivate_session_id', sessionId);
    }
    return sessionId;
  }

  // Fallback error tracking when Application Insights fails
  setupFallbackErrorTracking() {
    window.addEventListener('error', (event) => {
      console.error('[Fallback Error Tracking]', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        timestamp: new Date().toISOString()
      });
    });
  }

  // Get current performance metrics
  getPerformanceMetrics() {
    return {
      ...this.performanceMetrics,
      isMonitoringEnabled: this.isInitialized,
      sessionId: this.getSessionId(),
      environment: MONITORING_CONFIG.isProduction ? 'production' : 'development'
    };
  }

  // Manual flush for critical events
  flush() {
    if (this.isInitialized && this.appInsights) {
      this.appInsights.flush();
    }
  }
}

// Export singleton instance
export const monitoring = new CultivateMonitoring();

// Export configuration for other modules
export { MONITORING_CONFIG };

// Auto-initialize monitoring
if (typeof window !== 'undefined') {
  // Initialize after DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => monitoring.initialize());
  } else {
    monitoring.initialize();
  }
}