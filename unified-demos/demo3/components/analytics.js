/**
 * Analytics Dashboard Component for Demo 3
 * Comprehensive ML classification metrics and analysis
 *
 * Features:
 * - Accuracy, Precision, Recall, F1 Score calculations
 * - Confusion matrix visualization
 * - Multi-tab interface (Overview, By Video, By Age Group, Model Comparison)
 * - CSV export functionality
 * - Real-time updates on classification changes
 *
 * Warren's Vision: Professional, data-driven, fully functional analytics
 */

class AnalyticsDashboard {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = null;
        this.options = {
            showExport: true,
            autoUpdate: true,
            ...options
        };

        // Data storage
        this.predictions = [];
        this.groundTruth = [];
        this.videoData = [];
        this.classicResults = [];
        this.ensembleResults = [];

        // Current state
        this.activeTab = 'overview';
        this.sortColumn = null;
        this.sortDirection = 'asc';

        // Event handler references for cleanup
        this._boundHandlers = {
            tabHandlers: [],
            exportHandler: null,
            sortHandler: null
        };

        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) {
            console.warn(`Analytics Dashboard: Container #${this.containerId} not found`);
            return;
        }

        this.injectStyles();
        this.render();
        this.bindEvents();

        console.log('Analytics Dashboard initialized');
    }

    // ============================================================
    // HELPER FUNCTIONS: Confusion Matrix & Metrics Calculations
    // ============================================================

    /**
     * Calculate confusion matrix from predictions and ground truth
     * Returns { TP, TN, FP, FN } for OEQ as positive class
     *
     * @param {Array} predictions - Array of predicted labels ('OEQ' or 'CEQ')
     * @param {Array} groundTruth - Array of actual labels ('OEQ' or 'CEQ')
     * @returns {Object} Confusion matrix components
     */
    calculateConfusionMatrix(predictions, groundTruth) {
        if (!predictions || !groundTruth || predictions.length !== groundTruth.length) {
            return { TP: 0, TN: 0, FP: 0, FN: 0, total: 0 };
        }

        let TP = 0; // True Positive: Predicted OEQ, Actual OEQ
        let TN = 0; // True Negative: Predicted CEQ, Actual CEQ
        let FP = 0; // False Positive: Predicted OEQ, Actual CEQ
        let FN = 0; // False Negative: Predicted CEQ, Actual OEQ

        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i];
            const actual = groundTruth[i];

            if (pred === 'OEQ' && actual === 'OEQ') {
                TP++;
            } else if (pred === 'CEQ' && actual === 'CEQ') {
                TN++;
            } else if (pred === 'OEQ' && actual === 'CEQ') {
                FP++;
            } else if (pred === 'CEQ' && actual === 'OEQ') {
                FN++;
            }
        }

        return {
            TP,
            TN,
            FP,
            FN,
            total: predictions.length
        };
    }

    /**
     * Calculate all metrics from confusion matrix
     *
     * @param {Object} confusionMatrix - Object with TP, TN, FP, FN
     * @returns {Object} All calculated metrics
     */
    calculateMetrics(confusionMatrix) {
        const { TP, TN, FP, FN, total } = confusionMatrix;

        // Accuracy: (TP + TN) / Total
        const accuracy = total > 0 ? (TP + TN) / total : 0;

        // Precision (OEQ): TP / (TP + FP)
        const precision = (TP + FP) > 0 ? TP / (TP + FP) : 0;

        // Recall (OEQ): TP / (TP + FN)
        const recall = (TP + FN) > 0 ? TP / (TP + FN) : 0;

        // F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        const f1Score = (precision + recall) > 0
            ? 2 * (precision * recall) / (precision + recall)
            : 0;

        // CEQ metrics (inverse perspective)
        const ceqPrecision = (TN + FN) > 0 ? TN / (TN + FN) : 0;
        const ceqRecall = (TN + FP) > 0 ? TN / (TN + FP) : 0;
        const ceqF1 = (ceqPrecision + ceqRecall) > 0
            ? 2 * (ceqPrecision * ceqRecall) / (ceqPrecision + ceqRecall)
            : 0;

        return {
            accuracy,
            precision,
            recall,
            f1Score,
            ceqPrecision,
            ceqRecall,
            ceqF1,
            total,
            oeqCount: TP + FN,
            ceqCount: TN + FP,
            correctCount: TP + TN,
            incorrectCount: FP + FN
        };
    }

    /**
     * Generate CSV export data
     *
     * @param {Object} data - Data to export (results array or full analysis)
     * @returns {string} CSV formatted string
     */
    generateCSVExport(data) {
        const rows = [];

        // Header row
        rows.push([
            'Video ID',
            'Video Name',
            'Question ID',
            'Timestamp',
            'Ground Truth',
            'Prediction',
            'Confidence',
            'Model',
            'Age Group',
            'Correct',
            'Transcript'
        ].join(','));

        // Data rows
        if (Array.isArray(data)) {
            data.forEach(item => {
                const correct = item.prediction === item.groundTruth ? 'Yes' : 'No';
                const row = [
                    this.escapeCSV(item.videoId || ''),
                    this.escapeCSV(item.videoName || ''),
                    this.escapeCSV(item.questionId || ''),
                    this.escapeCSV(item.timestamp || ''),
                    this.escapeCSV(item.groundTruth || ''),
                    this.escapeCSV(item.prediction || ''),
                    (item.confidence || 0).toFixed(4),
                    this.escapeCSV(item.model || 'classic'),
                    this.escapeCSV(item.ageGroup || ''),
                    correct,
                    this.escapeCSV(item.transcript || '')
                ];
                rows.push(row.join(','));
            });
        }

        return rows.join('\n');
    }

    /**
     * Escape CSV special characters
     * Handles null, undefined, and non-string values safely
     */
    escapeCSV(value) {
        // Handle null/undefined explicitly to avoid "null"/"undefined" strings
        if (value === null || value === undefined) return '';
        if (typeof value !== 'string') return String(value);
        if (value.includes(',') || value.includes('"') || value.includes('\n')) {
            return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
    }

    // ============================================================
    // DATA MANAGEMENT
    // ============================================================

    /**
     * Update dashboard with new classification results
     *
     * @param {Object} result - Classification result
     * @param {string} result.videoId - Video identifier
     * @param {string} result.questionId - Question identifier
     * @param {string} result.prediction - Predicted classification
     * @param {string} result.groundTruth - Actual classification
     * @param {number} result.confidence - Prediction confidence
     * @param {string} result.model - Model used (classic/ensemble)
     */
    addResult(result) {
        const index = this.predictions.findIndex(
            p => p.videoId === result.videoId && p.questionId === result.questionId
        );

        if (index >= 0) {
            // Update existing result
            this.predictions[index] = { ...this.predictions[index], ...result };
        } else {
            // Add new result
            this.predictions.push(result);
        }

        // Track by model type
        if (result.model === 'ensemble') {
            this.ensembleResults = this.predictions.filter(p => p.model === 'ensemble');
        } else {
            this.classicResults = this.predictions.filter(p => p.model === 'classic' || !p.model);
        }

        if (this.options.autoUpdate) {
            this.updateDisplay();
        }
    }

    /**
     * Set video catalog data for context
     */
    setVideoData(videos) {
        this.videoData = videos || [];
        this.updateDisplay();
    }

    /**
     * Clear all results
     */
    clearResults() {
        this.predictions = [];
        this.classicResults = [];
        this.ensembleResults = [];
        this.updateDisplay();
    }

    /**
     * Import results from video catalog
     */
    importFromCatalog(catalog) {
        if (!catalog || !catalog.videos) return;

        this.videoData = catalog.videos;

        catalog.videos.forEach(video => {
            if (!video.questions) return;

            video.questions.forEach(question => {
                if (question.ml_prediction) {
                    this.addResult({
                        videoId: video.id,
                        videoName: video.display_name,
                        questionId: question.id,
                        timestamp: question.timestamp,
                        groundTruth: question.ground_truth_type,
                        prediction: question.ml_prediction.classification,
                        confidence: question.ml_prediction.confidence,
                        model: question.ml_prediction.model || 'classic',
                        ageGroup: video.age_group,
                        transcript: question.transcript_text
                    });
                }
            });
        });
    }

    // ============================================================
    // RENDERING
    // ============================================================

    injectStyles() {
        if (document.getElementById('analytics-dashboard-styles')) return;

        const style = document.createElement('style');
        style.id = 'analytics-dashboard-styles';
        style.textContent = `
            .analytics-dashboard {
                background: var(--surface-50, #ffffff);
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 24px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }

            body[data-theme="dark"] .analytics-dashboard {
                background: #1a1a1a;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }

            .analytics-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
                padding-bottom: 16px;
                border-bottom: 1px solid #e0e0e0;
            }

            body[data-theme="dark"] .analytics-header {
                border-bottom-color: #333;
            }

            .analytics-title {
                font-size: 24px;
                font-weight: 600;
                margin: 0;
                color: #1a1a1a;
            }

            body[data-theme="dark"] .analytics-title {
                color: #e0e0e0;
            }

            .analytics-export-btn {
                background: #0078d4;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: background 0.2s;
            }

            .analytics-export-btn:hover {
                background: #106ebe;
            }

            .analytics-export-btn svg {
                width: 16px;
                height: 16px;
            }

            .analytics-tabs {
                display: flex;
                gap: 4px;
                margin-bottom: 24px;
                border-bottom: 2px solid #e0e0e0;
            }

            body[data-theme="dark"] .analytics-tabs {
                border-bottom-color: #333;
            }

            .analytics-tab {
                padding: 12px 24px;
                background: transparent;
                border: none;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                color: #666;
                position: relative;
                transition: color 0.2s;
            }

            body[data-theme="dark"] .analytics-tab {
                color: #999;
            }

            .analytics-tab:hover {
                color: #0078d4;
            }

            .analytics-tab.active {
                color: #0078d4;
            }

            .analytics-tab.active::after {
                content: '';
                position: absolute;
                bottom: -2px;
                left: 0;
                right: 0;
                height: 2px;
                background: #0078d4;
            }

            .analytics-content {
                min-height: 300px;
            }

            .analytics-panel {
                display: none;
            }

            .analytics-panel.active {
                display: block;
                animation: fadeIn 0.3s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            /* Metrics Grid */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }

            .metric-card {
                background: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }

            body[data-theme="dark"] .metric-card {
                background: #2a2a2a;
            }

            .metric-value {
                font-size: 32px;
                font-weight: 700;
                color: #0078d4;
                margin-bottom: 8px;
            }

            .metric-label {
                font-size: 14px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            body[data-theme="dark"] .metric-label {
                color: #999;
            }

            .metric-value.good { color: #10b981; }
            .metric-value.warning { color: #f59e0b; }
            .metric-value.bad { color: #ef4444; }

            /* Pie Chart (CSS-only) */
            .pie-chart-container {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 32px;
                margin: 24px 0;
            }

            .pie-chart {
                width: 150px;
                height: 150px;
                border-radius: 50%;
                position: relative;
                background: conic-gradient(
                    var(--oeg-color, #0078d4) 0deg var(--oeg-angle, 180deg),
                    var(--ceq-color, #10b981) var(--oeg-angle, 180deg) 360deg
                );
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }

            .pie-chart::after {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 60px;
                height: 60px;
                background: var(--surface-50, #ffffff);
                border-radius: 50%;
            }

            body[data-theme="dark"] .pie-chart::after {
                background: #1a1a1a;
            }

            .pie-legend {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
            }

            .legend-color {
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }

            .legend-color.oeq { background: #0078d4; }
            .legend-color.ceq { background: #10b981; }

            /* Confusion Matrix */
            .confusion-matrix-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin: 24px 0;
            }

            .confusion-matrix-title {
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 16px;
                color: #1a1a1a;
            }

            body[data-theme="dark"] .confusion-matrix-title {
                color: #e0e0e0;
            }

            .confusion-matrix {
                border-collapse: collapse;
                text-align: center;
            }

            .confusion-matrix th,
            .confusion-matrix td {
                padding: 16px 24px;
                border: 1px solid #e0e0e0;
            }

            body[data-theme="dark"] .confusion-matrix th,
            body[data-theme="dark"] .confusion-matrix td {
                border-color: #444;
            }

            .confusion-matrix th {
                background: #f5f5f5;
                font-weight: 600;
                font-size: 13px;
            }

            body[data-theme="dark"] .confusion-matrix th {
                background: #2a2a2a;
                color: #e0e0e0;
            }

            .confusion-matrix .header-row th {
                background: transparent;
                border: none;
            }

            .confusion-matrix td {
                font-size: 20px;
                font-weight: 600;
            }

            .confusion-matrix .tp { background: rgba(16, 185, 129, 0.2); color: #10b981; }
            .confusion-matrix .tn { background: rgba(16, 185, 129, 0.2); color: #10b981; }
            .confusion-matrix .fp { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
            .confusion-matrix .fn { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

            /* Data Table */
            .analytics-table-container {
                overflow-x: auto;
                margin: 16px 0;
            }

            .analytics-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }

            .analytics-table th,
            .analytics-table td {
                padding: 12px 16px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }

            body[data-theme="dark"] .analytics-table th,
            body[data-theme="dark"] .analytics-table td {
                border-bottom-color: #333;
            }

            .analytics-table th {
                background: #f5f5f5;
                font-weight: 600;
                cursor: pointer;
                white-space: nowrap;
                user-select: none;
            }

            body[data-theme="dark"] .analytics-table th {
                background: #2a2a2a;
                color: #e0e0e0;
            }

            .analytics-table th:hover {
                background: #e8e8e8;
            }

            body[data-theme="dark"] .analytics-table th:hover {
                background: #333;
            }

            .analytics-table th .sort-icon {
                margin-left: 4px;
                opacity: 0.5;
            }

            .analytics-table th.sorted .sort-icon {
                opacity: 1;
            }

            .analytics-table tr:hover {
                background: rgba(0, 120, 212, 0.05);
            }

            .analytics-table .correct { color: #10b981; }
            .analytics-table .incorrect { color: #ef4444; }

            /* Age Group Breakdown */
            .age-group-section {
                margin-bottom: 32px;
            }

            .age-group-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 16px;
                color: #1a1a1a;
            }

            body[data-theme="dark"] .age-group-title {
                color: #e0e0e0;
            }

            .age-group-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
                margin-left: 8px;
            }

            .age-group-badge.pk {
                background: rgba(0, 120, 212, 0.1);
                color: #0078d4;
            }

            .age-group-badge.toddler {
                background: rgba(139, 92, 246, 0.1);
                color: #8b5cf6;
            }

            /* Model Comparison */
            .model-comparison-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
            }

            @media (max-width: 768px) {
                .model-comparison-grid {
                    grid-template-columns: 1fr;
                }
            }

            .model-column {
                background: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
            }

            body[data-theme="dark"] .model-column {
                background: #2a2a2a;
            }

            .model-column-title {
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 16px;
                color: #1a1a1a;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            body[data-theme="dark"] .model-column-title {
                color: #e0e0e0;
            }

            .model-badge {
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
            }

            .model-badge.classic {
                background: rgba(107, 114, 128, 0.2);
                color: #6b7280;
            }

            .model-badge.ensemble {
                background: rgba(139, 92, 246, 0.2);
                color: #8b5cf6;
            }

            /* Empty State */
            .analytics-empty {
                text-align: center;
                padding: 48px 24px;
                color: #666;
            }

            body[data-theme="dark"] .analytics-empty {
                color: #999;
            }

            .analytics-empty-icon {
                font-size: 48px;
                margin-bottom: 16px;
                opacity: 0.5;
            }

            .analytics-empty-text {
                font-size: 16px;
            }

            /* Progress Bar */
            .progress-bar {
                width: 100%;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 8px;
            }

            body[data-theme="dark"] .progress-bar {
                background: #333;
            }

            .progress-bar-fill {
                height: 100%;
                background: #0078d4;
                border-radius: 4px;
                transition: width 0.3s ease;
            }

            .progress-bar-fill.good { background: #10b981; }
            .progress-bar-fill.warning { background: #f59e0b; }
            .progress-bar-fill.bad { background: #ef4444; }
        `;
        document.head.appendChild(style);
    }

    render() {
        if (!this.container) return;

        this.container.innerHTML = `
            <div class="analytics-dashboard">
                <div class="analytics-header">
                    <h2 class="analytics-title">ML Classification Analytics</h2>
                    ${this.options.showExport ? `
                    <button class="analytics-export-btn" id="analytics-export">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        Export CSV
                    </button>
                    ` : ''}
                </div>

                <div class="analytics-tabs">
                    <button class="analytics-tab active" data-tab="overview">Overview</button>
                    <button class="analytics-tab" data-tab="by-video">By Video</button>
                    <button class="analytics-tab" data-tab="by-age">By Age Group</button>
                    <button class="analytics-tab" data-tab="model-comparison">Model Comparison</button>
                </div>

                <div class="analytics-content">
                    <div class="analytics-panel active" id="panel-overview">
                        ${this.renderOverviewPanel()}
                    </div>
                    <div class="analytics-panel" id="panel-by-video">
                        ${this.renderByVideoPanel()}
                    </div>
                    <div class="analytics-panel" id="panel-by-age">
                        ${this.renderByAgePanel()}
                    </div>
                    <div class="analytics-panel" id="panel-model-comparison">
                        ${this.renderModelComparisonPanel()}
                    </div>
                </div>
            </div>
        `;
    }

    renderOverviewPanel() {
        const predictions = this.predictions.map(p => p.prediction);
        const groundTruth = this.predictions.map(p => p.groundTruth);
        const confusionMatrix = this.calculateConfusionMatrix(predictions, groundTruth);
        const metrics = this.calculateMetrics(confusionMatrix);

        if (this.predictions.length === 0) {
            return this.renderEmptyState('No classification results yet. Run classifications to see analytics.');
        }

        const oeqPercent = metrics.total > 0 ? (metrics.oeqCount / metrics.total) * 100 : 50;
        const oeqAngle = (oeqPercent / 100) * 360;

        const getAccuracyClass = (value) => {
            if (value >= 0.85) return 'good';
            if (value >= 0.70) return 'warning';
            return 'bad';
        };

        return `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value ${getAccuracyClass(metrics.accuracy)}">${(metrics.accuracy * 100).toFixed(1)}%</div>
                    <div class="metric-label">Overall Accuracy</div>
                    <div class="progress-bar">
                        <div class="progress-bar-fill ${getAccuracyClass(metrics.accuracy)}" style="width: ${metrics.accuracy * 100}%"></div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(metrics.precision * 100).toFixed(1)}%</div>
                    <div class="metric-label">Precision (OEQ)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(metrics.recall * 100).toFixed(1)}%</div>
                    <div class="metric-label">Recall (OEQ)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(metrics.f1Score * 100).toFixed(1)}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>

            <div class="pie-chart-container">
                <div class="pie-chart" style="--oeg-angle: ${oeqAngle}deg"></div>
                <div class="pie-legend">
                    <div class="legend-item">
                        <div class="legend-color oeq"></div>
                        <span>OEQ: ${metrics.oeqCount} (${oeqPercent.toFixed(1)}%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color ceq"></div>
                        <span>CEQ: ${metrics.ceqCount} (${(100 - oeqPercent).toFixed(1)}%)</span>
                    </div>
                </div>
            </div>

            ${this.renderConfusionMatrix(confusionMatrix)}

            <div class="metrics-grid" style="margin-top: 24px;">
                <div class="metric-card">
                    <div class="metric-value">${metrics.total}</div>
                    <div class="metric-label">Total Questions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value good">${metrics.correctCount}</div>
                    <div class="metric-label">Correct Predictions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value bad">${metrics.incorrectCount}</div>
                    <div class="metric-label">Incorrect Predictions</div>
                </div>
            </div>
        `;
    }

    /**
     * Render confusion matrix as HTML table
     */
    renderConfusionMatrix(confusionMatrix) {
        const { TP, TN, FP, FN } = confusionMatrix;

        return `
            <div class="confusion-matrix-container">
                <div class="confusion-matrix-title">Confusion Matrix</div>
                <table class="confusion-matrix">
                    <tr class="header-row">
                        <th></th>
                        <th></th>
                        <th colspan="2" style="text-align: center; border-bottom: 1px solid #e0e0e0;">Predicted</th>
                    </tr>
                    <tr>
                        <th></th>
                        <th></th>
                        <th>OEQ</th>
                        <th>CEQ</th>
                    </tr>
                    <tr>
                        <th rowspan="2" style="vertical-align: middle;">Actual</th>
                        <th>OEQ</th>
                        <td class="tp" title="True Positive">${TP}</td>
                        <td class="fn" title="False Negative">${FN}</td>
                    </tr>
                    <tr>
                        <th>CEQ</th>
                        <td class="fp" title="False Positive">${FP}</td>
                        <td class="tn" title="True Negative">${TN}</td>
                    </tr>
                </table>
            </div>
        `;
    }

    renderByVideoPanel() {
        if (this.predictions.length === 0) {
            return this.renderEmptyState('No classification results yet.');
        }

        // Group by video
        const videoGroups = {};
        this.predictions.forEach(p => {
            if (!videoGroups[p.videoId]) {
                videoGroups[p.videoId] = {
                    videoId: p.videoId,
                    videoName: p.videoName || p.videoId,
                    ageGroup: p.ageGroup || 'Unknown',
                    predictions: [],
                    groundTruth: []
                };
            }
            videoGroups[p.videoId].predictions.push(p.prediction);
            videoGroups[p.videoId].groundTruth.push(p.groundTruth);
        });

        // Calculate metrics per video
        const videoStats = Object.values(videoGroups).map(video => {
            const cm = this.calculateConfusionMatrix(video.predictions, video.groundTruth);
            const metrics = this.calculateMetrics(cm);
            return {
                ...video,
                accuracy: metrics.accuracy,
                total: metrics.total,
                correct: metrics.correctCount
            };
        });

        // Sort if needed
        if (this.sortColumn) {
            videoStats.sort((a, b) => {
                const aVal = a[this.sortColumn];
                const bVal = b[this.sortColumn];
                const multiplier = this.sortDirection === 'asc' ? 1 : -1;
                if (typeof aVal === 'number') {
                    return (aVal - bVal) * multiplier;
                }
                return String(aVal).localeCompare(String(bVal)) * multiplier;
            });
        }

        return `
            <div class="analytics-table-container">
                <table class="analytics-table">
                    <thead>
                        <tr>
                            <th data-sort="videoName" class="${this.sortColumn === 'videoName' ? 'sorted' : ''}">
                                Video Name
                                <span class="sort-icon">${this.sortDirection === 'asc' ? '&#9650;' : '&#9660;'}</span>
                            </th>
                            <th data-sort="ageGroup" class="${this.sortColumn === 'ageGroup' ? 'sorted' : ''}">
                                Age Group
                                <span class="sort-icon">${this.sortDirection === 'asc' ? '&#9650;' : '&#9660;'}</span>
                            </th>
                            <th data-sort="total" class="${this.sortColumn === 'total' ? 'sorted' : ''}">
                                Questions
                                <span class="sort-icon">${this.sortDirection === 'asc' ? '&#9650;' : '&#9660;'}</span>
                            </th>
                            <th data-sort="correct" class="${this.sortColumn === 'correct' ? 'sorted' : ''}">
                                Correct
                                <span class="sort-icon">${this.sortDirection === 'asc' ? '&#9650;' : '&#9660;'}</span>
                            </th>
                            <th data-sort="accuracy" class="${this.sortColumn === 'accuracy' ? 'sorted' : ''}">
                                Accuracy
                                <span class="sort-icon">${this.sortDirection === 'asc' ? '&#9650;' : '&#9660;'}</span>
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        ${videoStats.map(video => `
                            <tr>
                                <td>${video.videoName}</td>
                                <td>
                                    <span class="age-group-badge ${video.ageGroup.toLowerCase()}">${video.ageGroup}</span>
                                </td>
                                <td>${video.total}</td>
                                <td class="${video.correct === video.total ? 'correct' : ''}">${video.correct}/${video.total}</td>
                                <td class="${video.accuracy >= 0.85 ? 'correct' : video.accuracy < 0.70 ? 'incorrect' : ''}">
                                    ${(video.accuracy * 100).toFixed(1)}%
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderByAgePanel() {
        if (this.predictions.length === 0) {
            return this.renderEmptyState('No classification results yet.');
        }

        const ageGroups = ['PK', 'TODDLER'];
        const sections = ageGroups.map(ageGroup => {
            const filtered = this.predictions.filter(p => p.ageGroup === ageGroup);
            if (filtered.length === 0) return null;

            const predictions = filtered.map(p => p.prediction);
            const groundTruth = filtered.map(p => p.groundTruth);
            const cm = this.calculateConfusionMatrix(predictions, groundTruth);
            const metrics = this.calculateMetrics(cm);

            return `
                <div class="age-group-section">
                    <h3 class="age-group-title">
                        ${ageGroup === 'PK' ? 'Pre-K' : 'Toddler'}
                        <span class="age-group-badge ${ageGroup.toLowerCase()}">${filtered.length} questions</span>
                    </h3>

                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.precision * 100).toFixed(1)}%</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.recall * 100).toFixed(1)}%</div>
                            <div class="metric-label">Recall</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.f1Score * 100).toFixed(1)}%</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                    </div>

                    ${this.renderConfusionMatrix(cm)}
                </div>
            `;
        }).filter(Boolean);

        if (sections.length === 0) {
            return this.renderEmptyState('No age group data available.');
        }

        return sections.join('');
    }

    renderModelComparisonPanel() {
        const hasClassic = this.classicResults.length > 0;
        const hasEnsemble = this.ensembleResults.length > 0;

        if (!hasClassic && !hasEnsemble) {
            return this.renderEmptyState('Run classifications with both models to see comparison.');
        }

        const renderModelColumn = (results, modelName, badgeClass) => {
            if (results.length === 0) {
                return `
                    <div class="model-column">
                        <h3 class="model-column-title">
                            ${modelName}
                            <span class="model-badge ${badgeClass}">No data</span>
                        </h3>
                        <p style="color: #666; text-align: center;">
                            Run classifications with this model to see results.
                        </p>
                    </div>
                `;
            }

            const predictions = results.map(r => r.prediction);
            const groundTruth = results.map(r => r.groundTruth);
            const cm = this.calculateConfusionMatrix(predictions, groundTruth);
            const metrics = this.calculateMetrics(cm);

            return `
                <div class="model-column">
                    <h3 class="model-column-title">
                        ${modelName}
                        <span class="model-badge ${badgeClass}">${results.length} predictions</span>
                    </h3>

                    <div class="metrics-grid" style="grid-template-columns: 1fr 1fr;">
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.f1Score * 100).toFixed(1)}%</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.precision * 100).toFixed(1)}%</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.recall * 100).toFixed(1)}%</div>
                            <div class="metric-label">Recall</div>
                        </div>
                    </div>

                    <div style="transform: scale(0.85); transform-origin: center top;">
                        ${this.renderConfusionMatrix(cm)}
                    </div>
                </div>
            `;
        };

        return `
            <div class="model-comparison-grid">
                ${renderModelColumn(this.classicResults, 'Classic ML', 'classic')}
                ${renderModelColumn(this.ensembleResults, 'Ensemble (7 Models)', 'ensemble')}
            </div>
        `;
    }

    renderEmptyState(message) {
        return `
            <div class="analytics-empty">
                <div class="analytics-empty-icon">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M3 3v18h18"/>
                        <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>
                    </svg>
                </div>
                <p class="analytics-empty-text">${message}</p>
            </div>
        `;
    }

    // ============================================================
    // EVENT HANDLING
    // ============================================================

    /**
     * Remove previously bound event handlers to prevent memory leaks
     */
    unbindEvents() {
        if (!this.container) return;

        // Remove tab handlers
        this._boundHandlers.tabHandlers.forEach(({ element, handler }) => {
            element.removeEventListener('click', handler);
        });
        this._boundHandlers.tabHandlers = [];

        // Remove export handler
        if (this._boundHandlers.exportHandler) {
            const exportBtn = this.container.querySelector('#analytics-export');
            if (exportBtn) {
                exportBtn.removeEventListener('click', this._boundHandlers.exportHandler);
            }
            this._boundHandlers.exportHandler = null;
        }

        // Remove sort handler (delegated on container)
        if (this._boundHandlers.sortHandler) {
            this.container.removeEventListener('click', this._boundHandlers.sortHandler);
            this._boundHandlers.sortHandler = null;
        }
    }

    bindEvents() {
        if (!this.container) return;

        // Clean up old handlers first to prevent accumulation
        this.unbindEvents();

        // Tab switching (use currentTarget to handle child element clicks properly)
        this.container.querySelectorAll('.analytics-tab').forEach(tab => {
            const handler = (e) => {
                const tabId = e.currentTarget.dataset.tab;
                if (tabId) {
                    this.switchTab(tabId);
                }
            };
            tab.addEventListener('click', handler);
            this._boundHandlers.tabHandlers.push({ element: tab, handler });
        });

        // Export button
        const exportBtn = this.container.querySelector('#analytics-export');
        if (exportBtn) {
            this._boundHandlers.exportHandler = () => this.exportCSV();
            exportBtn.addEventListener('click', this._boundHandlers.exportHandler);
        }

        // Table sorting (delegated on container)
        this._boundHandlers.sortHandler = (e) => {
            const th = e.target.closest('th[data-sort]');
            if (th) {
                this.handleSort(th.dataset.sort);
            }
        };
        this.container.addEventListener('click', this._boundHandlers.sortHandler);
    }

    switchTab(tabId) {
        this.activeTab = tabId;

        // Update tab buttons
        this.container.querySelectorAll('.analytics-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabId);
        });

        // Update panels
        this.container.querySelectorAll('.analytics-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `panel-${tabId}`);
        });
    }

    handleSort(column) {
        if (this.sortColumn === column) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = column;
            this.sortDirection = 'asc';
        }
        this.updateDisplay();
    }

    updateDisplay() {
        // Re-render the active panel
        const panels = {
            'overview': 'renderOverviewPanel',
            'by-video': 'renderByVideoPanel',
            'by-age': 'renderByAgePanel',
            'model-comparison': 'renderModelComparisonPanel'
        };

        const activePanel = this.container.querySelector('.analytics-panel.active');
        if (activePanel) {
            const panelId = activePanel.id.replace('panel-', '');
            const renderMethod = panels[panelId];
            if (renderMethod && this[renderMethod]) {
                activePanel.innerHTML = this[renderMethod]();
            }
        }

        // Re-bind events for new content
        this.bindEvents();
    }

    exportCSV() {
        const csvContent = this.generateCSVExport(this.predictions);
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `ml-classification-results-${new Date().toISOString().split('T')[0]}.csv`;
        link.click();

        URL.revokeObjectURL(url);
        console.log('CSV exported successfully');
    }

    // ============================================================
    // PUBLIC API
    // ============================================================

    /**
     * Get current metrics summary
     */
    getMetricsSummary() {
        const predictions = this.predictions.map(p => p.prediction);
        const groundTruth = this.predictions.map(p => p.groundTruth);
        const confusionMatrix = this.calculateConfusionMatrix(predictions, groundTruth);
        return this.calculateMetrics(confusionMatrix);
    }

    /**
     * Get metrics for a specific video
     */
    getVideoMetrics(videoId) {
        const videoResults = this.predictions.filter(p => p.videoId === videoId);
        if (videoResults.length === 0) return null;

        const predictions = videoResults.map(p => p.prediction);
        const groundTruth = videoResults.map(p => p.groundTruth);
        const confusionMatrix = this.calculateConfusionMatrix(predictions, groundTruth);
        return this.calculateMetrics(confusionMatrix);
    }

    /**
     * Get raw prediction data
     */
    getPredictions() {
        return [...this.predictions];
    }

    /**
     * Force re-render
     */
    refresh() {
        this.render();
        this.bindEvents();
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AnalyticsDashboard };
}

// Also attach to window for direct script inclusion
if (typeof window !== 'undefined') {
    window.AnalyticsDashboard = AnalyticsDashboard;
}

console.log('Analytics Dashboard component loaded');
