/**
 * API Configuration for Cultivate Learning Platform
 * Handles 2 environments (Local/Azure) × 2 models (Classic/Ensemble)
 */

class APIConfig {
    constructor() {
        // Detect environment
        this.isLocalhost = window.location.hostname === 'localhost' ||
                          window.location.hostname === '127.0.0.1';

        // Base URLs for each environment
        this.baseUrls = {
            local: 'http://localhost:5001',
            azure: 'https://cultivate-ml-api.wonderfulcliff-bce5d5d4.eastus2.azurecontainerapps.io'
        };

        // Endpoint mappings for each environment and model
        this.endpoints = {
            local: {
                classic: {
                    classify: '/api/v1/classify',  // Classic ML endpoint
                    feedback: '/api/v2/feedback'
                },
                ensemble: {
                    classify: '/api/v2/classify/ensemble',  // Ensemble ML endpoint
                    feedback: '/api/v2/feedback'
                }
            },
            azure: {
                classic: {
                    classify: '/api/classify',
                    feedback: '/api/feedback'
                },
                ensemble: {
                    classify: '/api/v2/classify/ensemble',
                    feedback: '/api/v2/feedback'
                }
            }
        };
    }

    getEnvironment() {
        return this.isLocalhost ? 'local' : 'azure';
    }

    getBaseUrl() {
        return this.isLocalhost ? this.baseUrls.local : this.baseUrls.azure;
    }

    getModel() {
        // Get from model settings if available
        if (window.modelSettings) {
            return window.modelSettings.getSelectedModel();
        }
        // Default to classic
        return localStorage.getItem('selectedModel') || 'classic';
    }

    getClassifyEndpoint() {
        const env = this.getEnvironment();
        const model = this.getModel();
        const baseUrl = this.getBaseUrl();
        const endpoint = this.endpoints[env][model].classify;

        console.log(`📍 API Config: ${env} environment, ${model} model`);
        console.log(`🎯 Classify endpoint: ${baseUrl}${endpoint}`);

        return `${baseUrl}${endpoint}`;
    }

    getFeedbackEndpoint() {
        const env = this.getEnvironment();
        const model = this.getModel();
        const baseUrl = this.getBaseUrl();
        const endpoint = this.endpoints[env][model].feedback;

        console.log(`📍 API Config: ${env} environment, ${model} model`);
        console.log(`💬 Feedback endpoint: ${baseUrl}${endpoint}`);

        return `${baseUrl}${endpoint}`;
    }

    // Check if feedback is available (since local doesn't have it)
    isFeedbackAvailable() {
        // For now, only available in Azure
        // TODO: Enable when local API implements feedback endpoints
        return !this.isLocalhost;
    }

    // Get proper request format for each endpoint
    getClassifyRequest(text, scenarioId = null) {
        const env = this.getEnvironment();
        const model = this.getModel();

        // Different endpoints expect different formats
        if (env === 'local' && model === 'classic') {
            // Old classify_response format
            return {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    scenario_id: scenarioId,
                    debug_mode: true
                })
            };
        } else if (env === 'azure' && model === 'classic') {
            // Azure classic uses query parameter
            return {
                method: 'POST',
                url: `${this.getClassifyEndpoint()}?text=${encodeURIComponent(text)}`
            };
        } else {
            // Ensemble endpoints use JSON body
            return {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    scenario_id: scenarioId,
                    debug_mode: true
                })
            };
        }
    }

    // Debug info
    getDebugInfo() {
        return {
            environment: this.getEnvironment(),
            model: this.getModel(),
            baseUrl: this.getBaseUrl(),
            classifyEndpoint: this.getClassifyEndpoint(),
            feedbackEndpoint: this.getFeedbackEndpoint(),
            feedbackAvailable: this.isFeedbackAvailable()
        };
    }
}

// Initialize global API config
window.apiConfig = new APIConfig();

// Listen for model changes
window.addEventListener('modelChanged', (e) => {
    console.log('🔄 Model changed, updating API config');
    console.log('📊 New config:', window.apiConfig.getDebugInfo());
});

console.log('✅ API Config initialized:', window.apiConfig.getDebugInfo());