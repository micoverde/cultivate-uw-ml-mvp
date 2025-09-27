/**
 * Rich Analytics and Insights Engine
 * Advanced ML performance tracking, cross-demo analysis, and stakeholder reporting
 * Warren's Vision: Professional, data-driven, fully functional demo
 */

class AdvancedAnalytics {
    constructor() {
        this.sessionData = {
            startTime: new Date(),
            interactions: [],
            mlPredictions: [],
            userFeedback: [],
            performanceMetrics: {},
            crossDemoInsights: {}
        };

        this.realTimeCharts = {};
        this.initialize();
    }

    initialize() {
        this.createAnalyticsDashboard();
        console.log('📊 Advanced Analytics Engine initialized');
    }

    /**
     * Track ML prediction with rich metadata
     */
    trackMLPrediction(demo, prediction, metadata = {}) {
        const predictionEvent = {
            timestamp: new Date(),
            demo: demo,
            classification: prediction.classification,
            confidence: prediction.confidence,
            processingTime: prediction.processing_time_ms,
            features: prediction.features_extracted,
            sessionId: this.getSessionId(),
            userContext: metadata.userContext || {},
            scenarioId: metadata.scenarioId
        };

        this.sessionData.mlPredictions.push(predictionEvent);
        this.updateRealTimeMetrics();
        this.analyzePerformanceTrends();

        console.log('📈 ML Prediction tracked:', predictionEvent);
    }

    /**
     * Track user interaction with advanced context
     */
    trackUserInteraction(action, demo, context = {}) {
        const interaction = {
            timestamp: new Date(),
            action: action,
            demo: demo,
            context: context,
            sessionId: this.getSessionId()
        };

        this.sessionData.interactions.push(interaction);
        this.analyzeUserBehavior();

        console.log('👤 User interaction tracked:', interaction);
    }

    /**
     * Real-time performance metrics calculation
     */
    updateRealTimeMetrics() {
        const predictions = this.sessionData.mlPredictions;
        const demo1Predictions = predictions.filter(p => p.demo === 'demo1');
        const demo2Predictions = predictions.filter(p => p.demo === 'demo2');

        this.sessionData.performanceMetrics = {
            totalPredictions: predictions.length,
            averageConfidence: this.calculateAverageConfidence(predictions),
            averageProcessingTime: this.calculateAverageProcessingTime(predictions),
            demo1Stats: this.calculateDemoStats(demo1Predictions),
            demo2Stats: this.calculateDemoStats(demo2Predictions),
            crossDemoComparison: this.compareDemoPerformance(demo1Predictions, demo2Predictions),
            confidenceDistribution: this.calculateConfidenceDistribution(predictions),
            performanceTrends: this.calculatePerformanceTrends(predictions)
        };

        this.updateDashboard();
    }

    /**
     * Create rich analytics dashboard
     */
    createAnalyticsDashboard() {
        if (document.getElementById('analytics-dashboard')) return;

        const dashboard = document.createElement('div');
        dashboard.id = 'analytics-dashboard';
        dashboard.className = 'analytics-dashboard hidden';

        dashboard.innerHTML = `
            <div class="dashboard-header">
                <h2>🔬 ML Performance Analytics</h2>
                <div class="dashboard-controls">
                    <button onclick="analytics.exportReport()" class="btn-export">📊 Export Report</button>
                    <button onclick="analytics.toggleDashboard()" class="btn-close">✕</button>
                </div>
            </div>

            <div class="dashboard-content">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="total-predictions">0</div>
                        <div class="metric-label">Total Predictions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="avg-confidence">0%</div>
                        <div class="metric-label">Avg Confidence</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="avg-processing">0ms</div>
                        <div class="metric-label">Avg Processing Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="session-duration">0m</div>
                        <div class="metric-label">Session Duration</div>
                    </div>
                </div>

                <div class="charts-section">
                    <div class="chart-container">
                        <h3>Confidence Distribution</h3>
                        <canvas id="confidence-chart" width="400" height="200"></canvas>
                    </div>
                    <div class="chart-container">
                        <h3>Demo Performance Comparison</h3>
                        <canvas id="performance-chart" width="400" height="200"></canvas>
                    </div>
                </div>

                <div class="insights-section">
                    <h3>🧠 AI Insights</h3>
                    <div id="ai-insights" class="insights-content">
                        <p>Analyzing ML performance patterns...</p>
                    </div>
                </div>

                <div class="real-time-feed">
                    <h3>📡 Real-Time Activity</h3>
                    <div id="activity-feed" class="activity-feed"></div>
                </div>
            </div>
        `;

        // Add dashboard styles
        const style = document.createElement('style');
        style.textContent = `
            .analytics-dashboard {
                position: fixed;
                top: 0;
                right: 0;
                width: 400px;
                height: 100vh;
                background: var(--surface-50, #F4F1EE);
                border-left: 2px solid var(--primary-500, #0078D4);
                box-shadow: -4px 0 12px rgba(0, 0, 0, 0.15);
                z-index: 10000;
                transform: translateX(100%);
                transition: transform 0.3s ease;
                overflow-y: auto;
            }

            .analytics-dashboard.visible {
                transform: translateX(0);
            }

            .dashboard-header {
                background: var(--primary-500, #0078D4);
                color: white;
                padding: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .dashboard-content {
                padding: 20px;
            }

            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-bottom: 24px;
            }

            .metric-card {
                background: var(--surface-200, #E8E3DF);
                padding: 16px;
                border-radius: 8px;
                text-align: center;
            }

            .metric-value {
                font-size: 24px;
                font-weight: 700;
                color: var(--primary-500, #0078D4);
            }

            .metric-label {
                font-size: 12px;
                color: var(--text-secondary, #424242);
                margin-top: 4px;
            }

            .chart-container, .insights-section, .real-time-feed {
                margin-bottom: 24px;
                background: white;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid var(--surface-300, #DFD9D4);
            }

            .chart-container h3, .insights-section h3, .real-time-feed h3 {
                margin-bottom: 12px;
                font-size: 14px;
                color: var(--text-primary, #1F1F1F);
            }

            .activity-feed {
                max-height: 200px;
                overflow-y: auto;
                font-size: 12px;
            }

            .activity-item {
                padding: 8px;
                border-bottom: 1px solid var(--surface-200, #E8E3DF);
                font-family: monospace;
            }

            .btn-export, .btn-close {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            }

            .btn-export:hover, .btn-close:hover {
                background: rgba(255, 255, 255, 0.3);
            }

            .analytics-trigger {
                position: fixed;
                top: 50%;
                right: 20px;
                background: var(--primary-500, #0078D4);
                color: white;
                border: none;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                font-size: 20px;
                z-index: 9999;
                transition: all 0.3s ease;
            }

            .analytics-trigger:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
            }
        `;

        document.head.appendChild(style);
        document.body.appendChild(dashboard);

        // Add analytics trigger button
        const trigger = document.createElement('button');
        trigger.className = 'analytics-trigger';
        trigger.innerHTML = '📊';
        trigger.onclick = () => this.toggleDashboard();
        trigger.title = 'Open Analytics Dashboard';
        document.body.appendChild(trigger);

        this.updateDashboard();
    }

    /**
     * Generate AI-powered insights
     */
    generateAIInsights() {
        const predictions = this.sessionData.mlPredictions;
        const interactions = this.sessionData.interactions;

        if (predictions.length < 3) {
            return ["📊 Collecting more data for meaningful insights...", "🎯 Continue using the demos to see patterns emerge."];
        }

        const insights = [];
        const avgConfidence = this.calculateAverageConfidence(predictions);
        const confidenceVariance = this.calculateVariance(predictions.map(p => p.confidence));

        // Confidence analysis
        if (avgConfidence > 0.85) {
            insights.push("🎯 Excellent ML performance! The model shows high confidence across predictions.");
        } else if (avgConfidence < 0.65) {
            insights.push("⚠️ Model confidence is below optimal levels. Consider additional training data.");
        }

        // Consistency analysis
        if (confidenceVariance < 0.05) {
            insights.push("📈 Highly consistent model performance across different scenarios.");
        } else if (confidenceVariance > 0.15) {
            insights.push("📊 Significant variance in confidence levels - investigate edge cases.");
        }

        // Cross-demo comparison
        const demo1Avg = this.calculateAverageConfidence(predictions.filter(p => p.demo === 'demo1'));
        const demo2Avg = this.calculateAverageConfidence(predictions.filter(p => p.demo === 'demo2'));

        if (Math.abs(demo1Avg - demo2Avg) > 0.1) {
            insights.push(`🔍 Performance difference detected: ${demo1Avg > demo2Avg ? 'Child Scenarios' : 'Video Analysis'} showing ${Math.abs(demo1Avg - demo2Avg).toFixed(2)} higher confidence.`);
        }

        // Usage pattern analysis
        const sessionMinutes = (new Date() - this.sessionData.startTime) / 60000;
        if (sessionMinutes > 5 && interactions.length > 10) {
            insights.push("👥 High engagement detected! User is actively exploring ML capabilities.");
        }

        return insights.length > 0 ? insights : ["🔬 Continue exploring to unlock more insights!"];
    }

    /**
     * Export comprehensive report
     */
    exportReport() {
        const report = {
            metadata: {
                exportTime: new Date().toISOString(),
                sessionDuration: this.getSessionDuration(),
                userName: 'Demo User',
                organization: 'Cultivate Learning'
            },
            summary: this.sessionData.performanceMetrics,
            detailed: {
                predictions: this.sessionData.mlPredictions,
                interactions: this.sessionData.interactions,
                feedback: this.sessionData.userFeedback
            },
            insights: this.generateAIInsights(),
            recommendations: this.generateRecommendations()
        };

        // Create downloadable JSON report
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cultivate-learning-analytics-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);

        console.log('📊 Analytics report exported:', report);
    }

    /**
     * Helper methods
     */
    calculateAverageConfidence(predictions) {
        if (predictions.length === 0) return 0;
        return predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;
    }

    calculateAverageProcessingTime(predictions) {
        if (predictions.length === 0) return 0;
        return predictions.reduce((sum, p) => sum + (p.processingTime || 0), 0) / predictions.length;
    }

    calculateVariance(values) {
        if (values.length === 0) return 0;
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    }

    getSessionId() {
        return this.sessionData.startTime.getTime().toString();
    }

    getSessionDuration() {
        return Math.round((new Date() - this.sessionData.startTime) / 60000);
    }

    toggleDashboard() {
        const dashboard = document.getElementById('analytics-dashboard');
        dashboard.classList.toggle('visible');
    }

    updateDashboard() {
        document.getElementById('total-predictions').textContent = this.sessionData.mlPredictions.length;
        document.getElementById('avg-confidence').textContent = `${(this.calculateAverageConfidence(this.sessionData.mlPredictions) * 100).toFixed(1)}%`;
        document.getElementById('avg-processing').textContent = `${this.calculateAverageProcessingTime(this.sessionData.mlPredictions).toFixed(0)}ms`;
        document.getElementById('session-duration').textContent = `${this.getSessionDuration()}m`;

        // Update AI insights
        const insights = this.generateAIInsights();
        document.getElementById('ai-insights').innerHTML = insights.map(insight => `<p>${insight}</p>`).join('');

        // Add to activity feed
        const feed = document.getElementById('activity-feed');
        if (feed && this.sessionData.mlPredictions.length > 0) {
            const latest = this.sessionData.mlPredictions[this.sessionData.mlPredictions.length - 1];
            const activity = document.createElement('div');
            activity.className = 'activity-item';
            activity.innerHTML = `${latest.timestamp.toLocaleTimeString()}: ${latest.demo} - ${latest.classification} (${(latest.confidence * 100).toFixed(1)}%)`;
            feed.insertBefore(activity, feed.firstChild);

            // Keep only last 10 items
            while (feed.children.length > 10) {
                feed.removeChild(feed.lastChild);
            }
        }
    }
}

// Initialize global analytics
window.analytics = new AdvancedAnalytics();

console.log('%c📊 Advanced Analytics Engine Loaded', 'font-size: 14px; font-weight: bold; color: #0078D4;');
console.log('🎯 Rich integration with real-time insights and professional reporting');