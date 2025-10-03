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
            ? 'http://localhost:5001'
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
                bottom: 60px; /* Above footer */
                right: 20px;
                width: 48px;
                height: 48px;
                border-radius: 50%;
                background: #6b7280;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
                z-index: 99;
            }

            .model-settings-btn:hover {
                background: #4b5563;
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }

            .model-settings-btn svg {
                width: 24px;
                height: 24px;
                stroke: white;
            }

            body[data-theme="dark"] .model-settings-btn {
                background: #4b5563;
            }

            body[data-theme="dark"] .model-settings-btn:hover {
                background: #374151;
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
                <path d="M12 15.5A3.5 3.5 0 0 1 8.5 12A3.5 3.5 0 0 1 12 8.5a3.5 3.5 0 0 1 3.5 3.5a3.5 3.5 0 0 1-3.5 3.5m7.43-2.53c.04-.32.07-.64.07-.97c0-.33-.03-.65-.07-.97l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65A.488.488 0 0 0 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.97c0 .33.03.65.07.97l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.23.09.49 0 .61-.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65Z"/>
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