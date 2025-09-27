/**
 * Simple, elegant model selection settings
 * Can be added to any page with: <script src="../shared/model-settings.js"></script>
 */

class ModelSettings {
    constructor() {
        // Detect environment - if running on localhost, use local API, otherwise use Azure production
        this.isLocalhost = window.location.hostname === 'localhost' ||
                          window.location.hostname === '127.0.0.1';

        // Set base API URL based on environment
        this.apiBaseUrl = this.isLocalhost
            ? 'http://localhost:8001'
            : 'https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io';

        this.models = [
            {
                id: 'classic',
                name: 'Classic ML',
                description: 'Fast, lightweight model'
            },
            {
                id: 'ensemble',
                name: 'Ensemble (7 Models)',
                description: 'Advanced accuracy with voting'
            }
        ];

        this.selectedModel = localStorage.getItem('selectedModel') || 'classic';
        this.init();
    }

    init() {
        this.injectStyles();
        this.createSettingsButton();
        this.createModal();
    }

    injectStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .model-settings-btn {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 48px;
                height: 48px;
                border-radius: 50%;
                background: white;
                border: 1px solid #e0e0e0;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                z-index: 999;
            }

            .model-settings-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }

            body[data-theme="dark"] .model-settings-btn {
                background: #2d2d2d;
                border-color: #404040;
            }

            .model-settings-modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 10000;
                animation: fadeIn 0.2s ease;
            }

            .model-settings-modal.show {
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .model-settings-content {
                background: white;
                border-radius: 12px;
                padding: 32px;
                max-width: 400px;
                width: 90%;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                animation: slideUp 0.3s ease;
            }

            body[data-theme="dark"] .model-settings-content {
                background: #1a1a1a;
                color: #e0e0e0;
            }

            .model-settings-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
            }

            .model-settings-title {
                font-size: 20px;
                font-weight: 600;
                margin: 0;
            }

            .model-settings-close {
                background: transparent;
                border: none;
                cursor: pointer;
                padding: 4px;
                color: #666;
                transition: color 0.2s;
            }

            .model-settings-close:hover {
                color: #000;
            }

            body[data-theme="dark"] .model-settings-close {
                color: #999;
            }

            body[data-theme="dark"] .model-settings-close:hover {
                color: #fff;
            }

            .model-option {
                padding: 16px;
                margin: 8px 0;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s;
            }

            body[data-theme="dark"] .model-option {
                border-color: #404040;
            }

            .model-option:hover {
                border-color: #0078d4;
                background: rgba(0,120,212,0.05);
            }

            .model-option.selected {
                border-color: #0078d4;
                background: rgba(0,120,212,0.1);
            }

            .model-option-name {
                font-weight: 600;
                margin-bottom: 4px;
            }

            .model-option-desc {
                font-size: 14px;
                color: #666;
            }

            body[data-theme="dark"] .model-option-desc {
                color: #999;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes slideUp {
                from {
                    transform: translateY(20px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
    }

    createSettingsButton() {
        const button = document.createElement('button');
        button.className = 'model-settings-btn';
        button.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v6m0 6v6m4.22-13.22l1.42 1.42M1.54 1.54l1.42 1.42M20.46 20.46l-1.42-1.42M1.54 20.46l-1.42-1.42M23 12h-6m-6 0H1"/>
            </svg>
        `;
        button.onclick = () => this.open();
        document.body.appendChild(button);
    }

    createModal() {
        const modal = document.createElement('div');
        modal.className = 'model-settings-modal';
        modal.innerHTML = `
            <div class="model-settings-content">
                <div class="model-settings-header">
                    <h3 class="model-settings-title">Select ML Model</h3>
                    <button class="model-settings-close">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="model-options">
                    ${this.models.map(model => `
                        <div class="model-option ${model.id === this.selectedModel ? 'selected' : ''}" data-model="${model.id}">
                            <div class="model-option-name">${model.name}</div>
                            <div class="model-option-desc">${model.description}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        this.modal = modal;

        // Event listeners
        modal.querySelector('.model-settings-close').onclick = () => this.close();
        modal.onclick = (e) => {
            if (e.target === modal) this.close();
        };

        modal.querySelectorAll('.model-option').forEach(option => {
            option.onclick = () => this.selectModel(option.dataset.model);
        });
    }

    open() {
        this.modal.classList.add('show');
    }

    close() {
        this.modal.classList.remove('show');
    }

    selectModel(modelId) {
        this.selectedModel = modelId;
        localStorage.setItem('selectedModel', modelId);

        // Update UI
        this.modal.querySelectorAll('.model-option').forEach(option => {
            option.classList.toggle('selected', option.dataset.model === modelId);
        });

        // Trigger event for other scripts to listen to
        window.dispatchEvent(new CustomEvent('modelChanged', { detail: { model: modelId } }));

        console.log(`✅ Model switched to: ${this.models.find(m => m.id === modelId).name}`);

        // Close after selection
        setTimeout(() => this.close(), 300);
    }

    getSelectedModel() {
        return this.selectedModel;
    }

    getModelEndpoint() {
        // Return complete URL with environment-specific base
        const endpoint = this.selectedModel === 'ensemble'
            ? '/api/v2/classify/ensemble'
            : '/api/classify';
        return `${this.apiBaseUrl}${endpoint}`;
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.modelSettings = new ModelSettings();
    });
} else {
    window.modelSettings = new ModelSettings();
}