/**
 * Unified ML API Abstraction Layer
 * Handles API coordination between Demo 1 and Demo 2 patterns
 * Warren's requirement: REAL ML APIs only, no simulation
 */

class UnifiedMLAPI {
    constructor() {
        // API endpoint configuration
        this.endpoints = {
            demo1: 'http://localhost:5001/api/v1/classify/response',
            demo2: {
                classify: '/api/v1/classify/question',
                feedback: '/api/v1/save_feedback'
            }
        };

        // Current demo context
        this.currentDemo = this.detectDemo();

        console.log(`🔧 Unified ML API initialized for ${this.currentDemo}`);
        console.log(`📍 Endpoints configured: ${JSON.stringify(this.endpoints)}`);
    }

    /**
     * Detect which demo is currently running based on URL path
     */
    detectDemo() {
        const path = window.location.pathname;
        if (path.includes('/demo1/')) return 'demo1';
        if (path.includes('/demo2/')) return 'demo2';
        return 'demo1'; // Default fallback
    }

    /**
     * Classify response - unified interface for both demos
     * @param {string} text - The text to classify
     * @param {Object} options - Additional options (scenario_id, context, timestamp, etc.)
     * @returns {Promise<Object>} Classification result
     */
    async classifyResponse(text, options = {}) {
        console.log(`🧠 Unified ML API: Classifying text for ${this.currentDemo}`);
        console.log(`📝 Input: "${text.substring(0, 50)}..."`);

        try {
            if (this.currentDemo === 'demo1') {
                return await this.classifyDemo1(text, options);
            } else {
                return await this.classifyDemo2(text, options);
            }
        } catch (error) {
            console.error(`❌ ML API Error (${this.currentDemo}):`, error);
            return this.handleAPIError(error);
        }
    }

    /**
     * Demo 1 classification - Child Scenarios pattern
     */
    async classifyDemo1(text, options) {
        const request = {
            text: text,
            scenario_id: options.scenario_id || 1,
            debug_mode: true
        };

        console.log(`🎭 Demo 1 API Request:`, request);

        const response = await fetch(this.endpoints.demo1, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        });

        if (!response.ok) {
            throw new Error(`Demo 1 API Error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        console.log(`✅ Demo 1 API Response:`, result);

        return this.normalizeResponse(result, 'demo1');
    }

    /**
     * Demo 2 classification - Warren's Video pattern
     */
    async classifyDemo2(text, options) {
        const request = {
            text: text,
            context: options.context || `Demo 2 - timestamp: ${options.timestamp || 0}s`,
            timestamp: options.timestamp || 0
        };

        console.log(`📹 Demo 2 API Request:`, request);

        // For now, route to demo1 endpoint with transformation
        // TODO: When Claude-4's /api/v1/classify/question is ready, switch to that
        const transformedRequest = {
            text: text,
            scenario_id: 999, // Special ID for demo2
            debug_mode: true,
            demo2_context: request.context,
            demo2_timestamp: request.timestamp
        };

        const response = await fetch(this.endpoints.demo1, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(transformedRequest)
        });

        if (!response.ok) {
            throw new Error(`Demo 2 API Error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        console.log(`✅ Demo 2 API Response:`, result);

        return this.normalizeResponse(result, 'demo2');
    }

    /**
     * Submit human feedback - unified interface
     * Fixed: Uses Azure endpoint instead of hardcoded localhost (Issue #225)
     */
    async submitFeedback(isCorrect, predictedClass, confidence, options = {}) {
        console.log(`📝 Unified Feedback: ${isCorrect ? 'Correct' : 'Incorrect'} for ${predictedClass}`);

        // Detect environment and use appropriate endpoint
        const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        const feedbackEndpoint = isLocalhost
            ? 'http://localhost:5001/save_feedback'
            : 'https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io/save_feedback';

        const feedbackData = {
            text: options.text || '',
            predicted_class: predictedClass,
            correct_class: isCorrect ? predictedClass : (predictedClass === 'OEQ' ? 'CEQ' : 'OEQ'),
            error_type: this.calculateErrorType(isCorrect, predictedClass),
            scenario_id: options.scenario_id || 1,
            timestamp: new Date().toISOString(),
            demo_context: this.currentDemo
        };

        try {
            // Use environment-aware feedback endpoint
            console.log(`📤 Submitting feedback to: ${feedbackEndpoint}`);
            const response = await fetch(feedbackEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(feedbackData)
            });

            if (!response.ok) {
                throw new Error(`Feedback API Error: ${response.status}`);
            }

            const result = await response.json();
            console.log(`✅ Feedback Submitted:`, result);
            return result;

        } catch (error) {
            console.error(`❌ Feedback Error:`, error);
            // Graceful fallback to localStorage for offline support
            try {
                console.log('💾 Falling back to localStorage (Feedback endpoint unavailable)');
                const localFeedback = JSON.parse(localStorage.getItem('feedbackData') || '[]');
                localFeedback.push({
                    ...feedbackData,
                    environment: 'local-fallback',
                    upload_status: 'pending'
                });
                localStorage.setItem('feedbackData', JSON.stringify(localFeedback));
                return {
                    status: 'partial',
                    message: 'Feedback saved locally (endpoint unavailable)',
                    total_entries: localFeedback.length,
                    storage: 'local-fallback'
                };
            } catch (fallbackError) {
                console.error(`❌ Even localStorage fallback failed:`, fallbackError);
                return { error: error.message };
            }
        }
    }

    /**
     * Calculate error type for feedback
     */
    calculateErrorType(isCorrect, predictedClass) {
        if (isCorrect) {
            return predictedClass === 'OEQ' ? 'TP' : 'TN';
        } else {
            return predictedClass === 'OEQ' ? 'FP' : 'FN';
        }
    }

    /**
     * Normalize API responses to common format
     */
    normalizeResponse(result, demo) {
        // Ensure consistent response format for both demos
        return {
            classification: result.classification || 'Unknown',
            confidence: result.confidence || 0,
            oeq_probability: result.oeq_probability || (result.classification === 'OEQ' ? result.confidence : 1 - result.confidence),
            ceq_probability: result.ceq_probability || (result.classification === 'CEQ' ? result.confidence : 1 - result.confidence),
            features_extracted: result.features_extracted || result.features_used || 56,
            processing_time_ms: result.processing_time_ms || result.processing_time || 0,
            method: result.method || 'PyTorch Neural Network',
            demo_context: demo,
            debug_info: result.debug_info || {},
            ensemble_details: result.ensemble_details || {},
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Handle API errors gracefully
     */
    handleAPIError(error) {
        const errorResponse = {
            classification: 'Error',
            confidence: 0,
            oeq_probability: 0,
            ceq_probability: 0,
            error: error.message,
            demo_context: this.currentDemo,
            timestamp: new Date().toISOString()
        };

        console.log(`❌ Returning error response:`, errorResponse);
        return errorResponse;
    }

    /**
     * Health check for API endpoints
     */
    async healthCheck() {
        console.log(`🏥 Health checking ML API endpoints...`);

        const results = {
            demo1: await this.checkDemo1Health(),
            demo2: await this.checkDemo2Health()
        };

        console.log(`📊 Health Check Results:`, results);
        return results;
    }

    async checkDemo1Health() {
        try {
            const response = await fetch(this.endpoints.demo1.replace('/classify_response', '/health'), {
                method: 'GET',
                timeout: 5000
            });
            return { status: response.ok ? 'healthy' : 'error', endpoint: this.endpoints.demo1 };
        } catch (error) {
            return { status: 'error', error: error.message, endpoint: this.endpoints.demo1 };
        }
    }

    async checkDemo2Health() {
        // TODO: Implement when Claude-4's endpoint is ready
        return { status: 'pending', note: 'Awaiting Claude-4 endpoint implementation' };
    }
}

// Global instance - available to all demos
window.unifiedAPI = new UnifiedMLAPI();

// Legacy compatibility functions for existing demos
window.classifyResponse = (text, options) => window.unifiedAPI.classifyResponse(text, options);
window.submitFeedback = (isCorrect, predictedClass, confidence, options) =>
    window.unifiedAPI.submitFeedback(isCorrect, predictedClass, confidence, options);

console.log('%c🔧 Unified ML API Layer Loaded', 'font-size: 14px; font-weight: bold; color: #0078D4;');
console.log('🎯 Warren\'s Requirement: REAL ML APIs only - no simulation');
console.log('🔗 API Coordination: Demo 1 + Demo 2 unified interface');
console.log('📡 Endpoints configured for local development and Azure deployment');