/**
 * Question Classification Panel Component
 * Demo 3 - Video Question Analysis
 *
 * Displays questions for selected video with ML classification results.
 * Features:
 * - Batch classification via ML API
 * - Visual match/mismatch indicators
 * - Clickable timestamps for video navigation
 * - Rating functionality
 * - Request cancellation support
 */

export class QuestionPanel {
    constructor(containerSelector) {
        this.container = document.querySelector(containerSelector);
        this.initialized = false;

        if (!this.container) {
            console.error(`QuestionPanel: Container "${containerSelector}" not found`);
            return;
        }

        this.currentVideo = null;
        this.questions = [];
        this.classifications = new Map();
        this.abortController = null;
        this.isClassifying = false;

        this.init();
        this.initialized = true;
    }

    init() {
        this.injectStyles();
        this.render();
        this.attachEventListeners();
        console.log('QuestionPanel initialized');
    }

    injectStyles() {
        if (document.getElementById('question-panel-styles')) return;

        const style = document.createElement('style');
        style.id = 'question-panel-styles';
        style.textContent = `
            .question-panel {
                background: #ffffff;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }

            body[data-theme="dark"] .question-panel {
                background: #1e1e1e;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }

            .question-panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 16px;
                border-bottom: 1px solid #e5e7eb;
            }

            body[data-theme="dark"] .question-panel-header {
                border-bottom-color: #374151;
            }

            .question-panel-title {
                font-size: 18px;
                font-weight: 600;
                color: #1f2937;
                margin: 0;
            }

            body[data-theme="dark"] .question-panel-title {
                color: #f3f4f6;
            }

            .question-panel-count {
                font-size: 14px;
                color: #6b7280;
                background: #f3f4f6;
                padding: 4px 12px;
                border-radius: 16px;
            }

            body[data-theme="dark"] .question-panel-count {
                background: #374151;
                color: #9ca3af;
            }

            .question-panel-empty {
                text-align: center;
                padding: 48px 24px;
                color: #6b7280;
            }

            body[data-theme="dark"] .question-panel-empty {
                color: #9ca3af;
            }

            .question-panel-empty-icon {
                font-size: 48px;
                margin-bottom: 16px;
                opacity: 0.5;
            }

            .question-list {
                display: flex;
                flex-direction: column;
                gap: 16px;
            }

            .question-item {
                background: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 16px;
                transition: all 0.2s ease;
            }

            body[data-theme="dark"] .question-item {
                background: #262626;
                border-color: #374151;
            }

            .question-item:hover {
                border-color: #0078d4;
                box-shadow: 0 2px 8px rgba(0, 120, 212, 0.1);
            }

            .question-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 12px;
            }

            .question-timestamp {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 10px;
                background: #0078d4;
                color: white;
                border-radius: 4px;
                font-size: 13px;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.2s ease;
                border: none;
            }

            .question-timestamp:hover {
                background: #005a9e;
            }

            .question-timestamp svg {
                width: 14px;
                height: 14px;
            }

            .question-ground-truth {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
            }

            .question-ground-truth.OEQ {
                background: #dbeafe;
                color: #1d4ed8;
            }

            .question-ground-truth.CEQ {
                background: #fef3c7;
                color: #b45309;
            }

            body[data-theme="dark"] .question-ground-truth.OEQ {
                background: #1e3a5f;
                color: #93c5fd;
            }

            body[data-theme="dark"] .question-ground-truth.CEQ {
                background: #3f2d07;
                color: #fcd34d;
            }

            .question-text {
                font-size: 15px;
                line-height: 1.5;
                color: #374151;
                margin-bottom: 12px;
                font-style: italic;
            }

            body[data-theme="dark"] .question-text {
                color: #d1d5db;
            }

            .question-description {
                font-size: 13px;
                color: #6b7280;
                margin-bottom: 12px;
            }

            body[data-theme="dark"] .question-description {
                color: #9ca3af;
            }

            .question-classification {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 12px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                margin-top: 12px;
            }

            body[data-theme="dark"] .question-classification {
                background: #1a1a1a;
                border-color: #374151;
            }

            .classification-result {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .classification-prediction {
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .classification-label {
                font-size: 12px;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            body[data-theme="dark"] .classification-label {
                color: #9ca3af;
            }

            .classification-type {
                font-weight: 600;
                font-size: 14px;
            }

            .classification-confidence {
                font-size: 13px;
                color: #6b7280;
            }

            body[data-theme="dark"] .classification-confidence {
                color: #9ca3af;
            }

            .match-indicator {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
            }

            .match-indicator.match {
                background: #d1fae5;
                color: #059669;
            }

            .match-indicator.mismatch {
                background: #fee2e2;
                color: #dc2626;
            }

            body[data-theme="dark"] .match-indicator.match {
                background: #064e3b;
                color: #6ee7b7;
            }

            body[data-theme="dark"] .match-indicator.mismatch {
                background: #450a0a;
                color: #fca5a5;
            }

            .match-indicator svg {
                width: 14px;
                height: 14px;
            }

            .question-actions {
                display: flex;
                gap: 8px;
            }

            .rate-btn {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 6px 12px;
                background: transparent;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                font-size: 13px;
                color: #4b5563;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .rate-btn:hover {
                background: #f3f4f6;
                border-color: #0078d4;
                color: #0078d4;
            }

            body[data-theme="dark"] .rate-btn {
                border-color: #4b5563;
                color: #9ca3af;
            }

            body[data-theme="dark"] .rate-btn:hover {
                background: #374151;
                border-color: #0078d4;
                color: #60a5fa;
            }

            .rate-btn svg {
                width: 14px;
                height: 14px;
            }

            /* Loading state */
            .question-panel-loading {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 48px 24px;
                gap: 16px;
            }

            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 3px solid #e5e7eb;
                border-top-color: #0078d4;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            body[data-theme="dark"] .loading-spinner {
                border-color: #374151;
                border-top-color: #60a5fa;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .loading-text {
                font-size: 14px;
                color: #6b7280;
            }

            body[data-theme="dark"] .loading-text {
                color: #9ca3af;
            }

            /* Error state */
            .question-panel-error {
                padding: 24px;
                background: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 8px;
                text-align: center;
            }

            body[data-theme="dark"] .question-panel-error {
                background: #450a0a;
                border-color: #7f1d1d;
            }

            .error-icon {
                font-size: 32px;
                margin-bottom: 12px;
            }

            .error-message {
                color: #dc2626;
                font-size: 14px;
                margin-bottom: 12px;
            }

            body[data-theme="dark"] .error-message {
                color: #fca5a5;
            }

            .retry-btn {
                padding: 8px 16px;
                background: #dc2626;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.2s ease;
            }

            .retry-btn:hover {
                background: #b91c1c;
            }

            /* No transcript indicator */
            .no-transcript {
                font-size: 13px;
                color: #9ca3af;
                font-style: italic;
            }
        `;
        document.head.appendChild(style);
    }

    render() {
        this.container.innerHTML = `
            <div class="question-panel">
                <div class="question-panel-header">
                    <h3 class="question-panel-title">Question Classification</h3>
                    <span class="question-panel-count">${this.questions.length} questions</span>
                </div>
                <div class="question-panel-body">
                    ${this.renderContent()}
                </div>
            </div>
        `;
    }

    renderContent() {
        if (!this.currentVideo) {
            return `
                <div class="question-panel-empty">
                    <div class="question-panel-empty-icon">?</div>
                    <p>Select a video to view its questions</p>
                </div>
            `;
        }

        if (this.isClassifying) {
            return `
                <div class="question-panel-loading">
                    <div class="loading-spinner"></div>
                    <p class="loading-text">Classifying ${this.questions.length} questions...</p>
                </div>
            `;
        }

        if (this.questions.length === 0) {
            return `
                <div class="question-panel-empty">
                    <div class="question-panel-empty-icon">?</div>
                    <p>No questions found for this video</p>
                </div>
            `;
        }

        return `
            <div class="question-list">
                ${this.questions.map(q => this.renderQuestion(q)).join('')}
            </div>
        `;
    }

    renderQuestion(question) {
        const classification = this.classifications.get(question.id);
        const hasTranscript = question.has_transcript && question.transcript_text;

        return `
            <div class="question-item" data-question-id="${question.id}">
                <div class="question-header">
                    <button class="question-timestamp" data-timestamp="${question.timestamp_seconds}">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polygon points="5,3 19,12 5,21"/>
                        </svg>
                        ${question.timestamp}
                    </button>
                    <span class="question-ground-truth ${question.ground_truth_type}">
                        Ground Truth: ${question.ground_truth_type}
                    </span>
                </div>

                ${hasTranscript
                    ? `<p class="question-text">"${this.escapeHtml(question.transcript_text)}"</p>`
                    : `<p class="no-transcript">No transcript available</p>`
                }

                ${question.description
                    ? `<p class="question-description">${this.escapeHtml(question.description)}</p>`
                    : ''
                }

                ${this.renderClassification(question, classification)}
            </div>
        `;
    }

    renderClassification(question, classification) {
        if (!classification) {
            if (!question.has_transcript || !question.transcript_text) {
                return `
                    <div class="question-classification">
                        <div class="classification-result">
                            <span class="classification-label">ML Prediction:</span>
                            <span class="classification-type" style="color: #9ca3af;">
                                Cannot classify (no transcript)
                            </span>
                        </div>
                        <div class="question-actions">
                            <button class="rate-btn" data-question-id="${question.id}" disabled>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                                </svg>
                                Rate
                            </button>
                        </div>
                    </div>
                `;
            }
            return `
                <div class="question-classification">
                    <div class="classification-result">
                        <span class="classification-label">ML Prediction:</span>
                        <span class="classification-type" style="color: #9ca3af;">Pending...</span>
                    </div>
                </div>
            `;
        }

        if (classification.error) {
            return `
                <div class="question-classification">
                    <div class="classification-result">
                        <span class="classification-label">ML Prediction:</span>
                        <span class="classification-type" style="color: #dc2626;">
                            Error: ${this.escapeHtml(classification.error)}
                        </span>
                    </div>
                </div>
            `;
        }

        const isMatch = classification.prediction === question.ground_truth_type;
        const confidencePercent = Math.round((classification.confidence || 0) * 100);

        return `
            <div class="question-classification">
                <div class="classification-result">
                    <div class="classification-prediction">
                        <span class="classification-label">ML Prediction:</span>
                        <span class="classification-type">${classification.prediction}</span>
                        <span class="classification-confidence">(${confidencePercent}%)</span>
                    </div>
                    <div class="match-indicator ${isMatch ? 'match' : 'mismatch'}">
                        ${isMatch
                            ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                 <polyline points="20,6 9,17 4,12"/>
                               </svg>
                               Match`
                            : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                 <line x1="18" y1="6" x2="6" y2="18"/>
                                 <line x1="6" y1="6" x2="18" y2="18"/>
                               </svg>
                               Mismatch`
                        }
                    </div>
                </div>
                <div class="question-actions">
                    <button class="rate-btn" data-question-id="${question.id}">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                        Rate
                    </button>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        // Listen for model changes
        window.addEventListener('modelChanged', (e) => {
            console.log('QuestionPanel: Model changed, re-classifying');
            if (this.currentVideo && this.questions.length > 0) {
                this.classifyQuestions();
            }
        });

        // Delegate click events
        this.container.addEventListener('click', (e) => {
            // Handle timestamp clicks
            const timestampBtn = e.target.closest('.question-timestamp');
            if (timestampBtn) {
                const timestamp = parseFloat(timestampBtn.dataset.timestamp);
                this.handleTimestampClick(timestamp);
                return;
            }

            // Handle rate button clicks
            const rateBtn = e.target.closest('.rate-btn');
            if (rateBtn && !rateBtn.disabled) {
                const questionId = rateBtn.dataset.questionId;
                this.handleRateClick(questionId);
                return;
            }

            // Handle retry button clicks
            const retryBtn = e.target.closest('.retry-btn');
            if (retryBtn) {
                this.classifyQuestions();
                return;
            }
        });
    }

    handleTimestampClick(timestampSeconds) {
        console.log(`QuestionPanel: Jump to timestamp ${timestampSeconds}s`);
        // Use 'seconds' key for compatibility with app.js handleJumpToTimestamp
        window.dispatchEvent(new CustomEvent('jumpToTimestamp', {
            detail: { seconds: timestampSeconds, timestamp: timestampSeconds }
        }));
    }

    handleRateClick(questionId) {
        const question = this.questions.find(q => q.id === questionId);
        const classification = this.classifications.get(questionId);

        if (!question) {
            console.error(`QuestionPanel: Question ${questionId} not found`);
            return;
        }

        console.log(`QuestionPanel: Opening rating modal for question ${questionId}`);

        window.dispatchEvent(new CustomEvent('openRatingModal', {
            detail: {
                questionId: questionId,
                question: question,
                classification: classification,
                videoId: this.currentVideo?.id
            }
        }));
    }

    /**
     * Set the current video and its questions
     * @param {Object} video - Video object from catalog
     */
    setVideo(video) {
        // Guard: don't proceed if component wasn't properly initialized
        if (!this.initialized || !this.container) {
            console.warn('QuestionPanel: Cannot setVideo - component not initialized');
            return;
        }

        // Cancel any pending classification request
        this.cancelClassification();

        this.currentVideo = video;
        this.questions = video?.questions || [];
        this.classifications.clear();

        // Update count in header
        this.render();

        // Classify questions if we have any with transcripts
        if (this.questions.some(q => q.has_transcript && q.transcript_text)) {
            this.classifyQuestions();
        }
    }

    /**
     * Cancel any in-progress classification request
     */
    cancelClassification() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
            console.log('QuestionPanel: Classification request cancelled');
        }
        this.isClassifying = false;
    }

    /**
     * Classify all questions using the ML API
     * Uses individual classify endpoints (batch endpoints don't exist)
     */
    async classifyQuestions() {
        // Cancel any existing request
        this.cancelClassification();

        const questionsToClassify = this.questions.filter(
            q => q.has_transcript && q.transcript_text
        );

        if (questionsToClassify.length === 0) {
            console.log('QuestionPanel: No questions with transcripts to classify');
            this.render();
            return;
        }

        this.isClassifying = true;
        this.render();

        // Create new abort controller for this request
        this.abortController = new AbortController();

        try {
            const selectedModel = window.modelSettings?.getSelectedModel() || 'classic';

            // Use apiConfig to get the correct endpoint for current environment/model
            // Falls back to constructing URL manually if apiConfig not available
            let endpoint;
            if (window.apiConfig?.getClassifyEndpoint) {
                endpoint = window.apiConfig.getClassifyEndpoint();
            } else {
                const baseUrl = window.apiConfig?.getBaseUrl() || 'http://localhost:5001';
                endpoint = selectedModel === 'ensemble'
                    ? `${baseUrl}/api/v2/classify/ensemble`
                    : `${baseUrl}/api/classify`;
            }

            console.log(`QuestionPanel: Classifying ${questionsToClassify.length} questions`);
            console.log(`QuestionPanel: Using ${selectedModel} model at ${endpoint}`);

            // Classify each question individually (batch endpoints don't exist)
            const classificationPromises = questionsToClassify.map(async (question) => {
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            text: question.transcript_text,
                            debug_mode: true
                        }),
                        signal: this.abortController.signal
                    });

                    if (!response.ok) {
                        throw new Error(`API error: ${response.status} ${response.statusText}`);
                    }

                    const data = await response.json();
                    return {
                        questionId: question.id,
                        success: true,
                        result: {
                            prediction: data.classification || data.prediction || 'Unknown',
                            confidence: data.confidence || data.score || 0,
                            model: selectedModel
                        }
                    };
                } catch (error) {
                    if (error.name === 'AbortError') {
                        throw error; // Re-throw abort errors to stop all requests
                    }
                    return {
                        questionId: question.id,
                        success: false,
                        error: error.message
                    };
                }
            });

            const results = await Promise.all(classificationPromises);

            // Process results
            results.forEach(result => {
                if (result.success) {
                    this.classifications.set(result.questionId, result.result);
                } else {
                    this.classifications.set(result.questionId, {
                        error: result.error,
                        model: selectedModel
                    });
                }
            });

            console.log(`QuestionPanel: Classification complete. ${results.filter(r => r.success).length}/${results.length} succeeded`);

            this.isClassifying = false;
            this.render();

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('QuestionPanel: Classification request aborted');
                return;
            }

            console.error('QuestionPanel: Classification failed:', error);
            this.isClassifying = false;

            // Store error for each question
            questionsToClassify.forEach(question => {
                this.classifications.set(question.id, {
                    error: error.message,
                    model: window.modelSettings?.getSelectedModel() || 'classic'
                });
            });

            this.renderError(error.message);
        }
    }

    renderError(message) {
        const body = this.container.querySelector('.question-panel-body');
        if (body) {
            body.innerHTML = `
                <div class="question-panel-error">
                    <div class="error-icon">!</div>
                    <p class="error-message">${this.escapeHtml(message)}</p>
                    <button class="retry-btn">Retry Classification</button>
                </div>
                <div class="question-list" style="margin-top: 24px;">
                    ${this.questions.map(q => this.renderQuestion(q)).join('')}
                </div>
            `;
        }
    }

    /**
     * Get classification statistics
     */
    getStats() {
        let matches = 0;
        let mismatches = 0;
        let errors = 0;
        let pending = 0;

        this.questions.forEach(question => {
            const classification = this.classifications.get(question.id);
            if (!classification) {
                pending++;
            } else if (classification.error) {
                errors++;
            } else if (classification.prediction === question.ground_truth_type) {
                matches++;
            } else {
                mismatches++;
            }
        });

        return {
            total: this.questions.length,
            matches,
            mismatches,
            errors,
            pending,
            accuracy: this.questions.length > 0
                ? Math.round((matches / (matches + mismatches)) * 100) || 0
                : 0
        };
    }

    /**
     * Get current questions
     */
    getQuestions() {
        return this.questions;
    }

    /**
     * Get classification for a specific question
     */
    getClassification(questionId) {
        return this.classifications.get(questionId);
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Cleanup resources
     */
    destroy() {
        this.cancelClassification();
        this.container.innerHTML = '';
        console.log('QuestionPanel destroyed');
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuestionPanel };
}

// Also attach to window for script tag usage
window.QuestionPanel = QuestionPanel;
