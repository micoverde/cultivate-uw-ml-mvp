/**
 * Unified API Configuration for Cultivate Learning
 * Handles 2 environments × 2 models = 4 scenarios
 * Warren's requirement: Handle all combinations properly
 */

class UnifiedAPI {
    constructor() {
        // Detect environment
        this.isLocalhost = window.location.hostname === 'localhost' ||
                          window.location.hostname === '127.0.0.1';

        // Base URLs
        this.localBase = 'http://localhost:5001';
        this.azureBase = 'https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io';

        // Get current base URL
        this.baseUrl = this.isLocalhost ? this.localBase : this.azureBase;

        console.log(`🌐 UnifiedAPI initialized: ${this.isLocalhost ? 'LOCAL' : 'AZURE'} environment`);
    }

    /**
     * Get the correct classification endpoint based on environment and model
     */
    getClassificationEndpoint() {
        const selectedModel = window.modelSettings?.getSelectedModel() || 'classic';
        const isEnsemble = selectedModel === 'ensemble';

        // Build the endpoint matrix
        if (this.isLocalhost) {
            if (isEnsemble) {
                // Local + Ensemble
                return `${this.localBase}/api/v2/classify/ensemble`;
            } else {
                // Local + Classic (special legacy endpoint)
                return `${this.localBase}/classify_response`;
            }
        } else {
            if (isEnsemble) {
                // Azure + Ensemble
                return `${this.azureBase}/api/v2/classify/ensemble`;
            } else {
                // Azure + Classic
                return `${this.azureBase}/api/classify`;
            }
        }
    }

    /**
     * Get the correct feedback endpoint based on environment
     * Fixed: Now uses Azure blob storage endpoint instead of localhost
     */
    getFeedbackEndpoint() {
        // Always use the Azure API feedback endpoint for Blob Storage
        // Issue #225: Fixed CSP violation - removed localhost:5001 reference
        return `${this.azureBase}/api/feedback`;
    }

    /**
     * Make a classification request with proper endpoint selection
     */
    async classifyText(text, scenarioId = null, debugMode = true) {
        const endpoint = this.getClassificationEndpoint();
        const selectedModel = window.modelSettings?.getSelectedModel() || 'classic';

        console.log(`\n🤖 ML Classification Request`);
        console.log(`📍 Environment: ${this.isLocalhost ? 'LOCAL' : 'AZURE'}`);
        console.log(`🧠 Model: ${selectedModel === 'ensemble' ? 'Ensemble (7 Models)' : 'Classic ML'}`);
        console.log(`🎯 Endpoint: ${endpoint}`);
        console.log(`📝 Text: "${text}"`);

        try {
            let response;

            // Different request formats for different endpoints
            if (endpoint.includes('/classify_response')) {
                // Legacy local classic endpoint
                response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        scenario_id: scenarioId,
                        debug_mode: debugMode
                    })
                });
            } else if (endpoint.includes('/api/classify')) {
                // Standard API endpoints (both classic and ensemble)
                response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        debug: debugMode
                    })
                });
            } else {
                // Ensemble endpoints
                response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        debug: debugMode
                    })
                });
            }

            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            console.log('✅ Classification successful:', data);
            return data;

        } catch (error) {
            console.error('❌ Classification failed:', error);
            throw error;
        }
    }

    /**
     * Submit feedback to Blob Storage via Azure API
     * Issue #225: Fixed CSP violation - now uses Azure API endpoint with Blob Storage
     */
    async submitFeedback(feedbackData) {
        const endpoint = this.getFeedbackEndpoint();

        // Ensure timestamp is included
        if (!feedbackData.timestamp) {
            feedbackData.timestamp = new Date().toISOString();
        }

        console.log(`📤 Submitting feedback to: ${endpoint}`);
        console.log(`📝 Feedback data:`, feedbackData);

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(feedbackData)
            });

            if (!response.ok) {
                throw new Error(`Feedback API error: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            console.log('✅ Feedback submitted successfully to Blob Storage:', result);

            return {
                ...result,
                storage: 'azure-blob',
                environment: this.isLocalhost ? 'local-to-azure' : 'azure'
            };

        } catch (error) {
            console.error('❌ Feedback submission failed:', error);

            // Graceful fallback: save to localStorage for offline support
            try {
                console.log('💾 Falling back to localStorage (Blob Storage unavailable)');
                const localFeedback = JSON.parse(localStorage.getItem('feedbackData') || '[]');
                localFeedback.push({
                    ...feedbackData,
                    timestamp: feedbackData.timestamp,
                    environment: 'local-fallback',
                    upload_status: 'pending'
                });
                localStorage.setItem('feedbackData', JSON.stringify(localFeedback));

                console.log(`📊 Feedback saved to localStorage (${localFeedback.length} total entries)`);
                return {
                    status: 'partial',
                    message: 'Feedback saved locally (Blob Storage unavailable)',
                    total_entries: localFeedback.length,
                    storage: 'local-fallback',
                    environment: 'local'
                };
            } catch (fallbackError) {
                console.error('❌ Even localStorage fallback failed:', fallbackError);
                return {
                    status: 'error',
                    message: 'Failed to save feedback: ' + error.message,
                    original_error: error.message
                };
            }
        }
    }

    /**
     * Get environment info for debugging
     */
    getEnvironmentInfo() {
        const selectedModel = window.modelSettings?.getSelectedModel() || 'classic';

        return {
            environment: this.isLocalhost ? 'LOCAL' : 'AZURE',
            baseUrl: this.baseUrl,
            selectedModel: selectedModel,
            classificationEndpoint: this.getClassificationEndpoint(),
            feedbackEndpoint: this.getFeedbackEndpoint() || 'Not available (local)',
            timestamp: new Date().toISOString()
        };
    }
}

// Initialize globally with a unique name to avoid conflicts
window.cultivateAPI = new UnifiedAPI();

// Log initialization
console.log('🚀 CultivateAPI loaded:', window.cultivateAPI.getEnvironmentInfo());