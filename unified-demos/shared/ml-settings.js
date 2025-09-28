// ML Settings Component for Cultivate Learning Platform
// Shared across Hub, Demo 1, and Demo 2

(function() {
    'use strict';

    // ML Settings HTML Template
    const mlSettingsHTML = `
        <!-- ML Settings Button (Fixed Position) -->
        <button id="mlSettingsBtn" class="ml-settings-btn" title="ML Model Settings">
            ⚙️
        </button>

        <!-- ML Settings Modal -->
        <div id="mlSettingsModal" class="ml-settings-modal">
            <div class="ml-settings-content">
                <div class="ml-settings-header">
                    <h2>ML Model Settings</h2>
                    <span class="ml-settings-close">&times;</span>
                </div>

                <div class="ml-settings-body">
                    <p class="ml-settings-description">
                        Choose between our Classic ML model or our advanced Ensemble model with 7-model voting system.
                    </p>

                    <div class="ml-model-options">
                        <div class="ml-model-option" data-model="classic">
                            <div class="ml-model-radio">
                                <input type="radio" id="classicModel" name="mlModel" value="classic">
                            </div>
                            <label for="classicModel" class="ml-model-label">
                                <h3>Classic ML Model</h3>
                                <p>Traditional machine learning approach with proven reliability and fast response times.</p>
                                <div class="ml-model-stats">
                                    <span>⚡ Response: 1.2s</span>
                                    <span>📊 Accuracy: 92%</span>
                                </div>
                            </label>
                        </div>

                        <div class="ml-model-option" data-model="ensemble">
                            <div class="ml-model-radio">
                                <input type="radio" id="ensembleModel" name="mlModel" value="ensemble">
                            </div>
                            <label for="ensembleModel" class="ml-model-label">
                                <h3>Ensemble Model (7-Model Voting)</h3>
                                <p>Advanced ensemble using 7 specialized models for higher accuracy through collective intelligence.</p>
                                <div class="ml-model-stats">
                                    <span>⚡ Response: 2.8s</span>
                                    <span>📊 Accuracy: 96%</span>
                                </div>
                            </label>
                        </div>
                    </div>

                    <div class="ml-performance-comparison">
                        <h3>Performance Comparison</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Classic</th>
                                    <th>Ensemble</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Accuracy</td>
                                    <td>92%</td>
                                    <td class="highlight">96%</td>
                                </tr>
                                <tr>
                                    <td>Response Time</td>
                                    <td class="highlight">1.2s</td>
                                    <td>2.8s</td>
                                </tr>
                                <tr>
                                    <td>Models Used</td>
                                    <td>1</td>
                                    <td>7</td>
                                </tr>
                                <tr>
                                    <td>Best For</td>
                                    <td>Real-time analysis</td>
                                    <td>High-stakes decisions</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="ml-settings-footer">
                    <button class="ml-settings-save">Save Settings</button>
                    <button class="ml-settings-cancel">Cancel</button>
                </div>
            </div>
        </div>
    `;

    // ML Settings CSS
    const mlSettingsCSS = `
        <style>
        /* ML Settings Button */
        .ml-settings-btn {
            position: fixed !important;
            top: 20px !important;
            right: 20px !important;
            width: 50px !important;
            height: 50px !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            color: white !important;
            font-size: 24px !important;
            cursor: pointer !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            z-index: 99999 !important;
            transition: all 0.3s ease !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .ml-settings-btn:hover {
            transform: scale(1.1) rotate(90deg);
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
        }

        /* Modal Styles */
        .ml-settings-modal {
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(5px);
            animation: fadeIn 0.3s;
        }

        .ml-settings-modal.show {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .ml-settings-content {
            background: white;
            border-radius: 20px;
            width: 90%;
            max-width: 700px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            animation: slideUp 0.3s;
        }

        .ml-settings-header {
            padding: 25px 30px;
            border-bottom: 1px solid #e5e5e5;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 20px 20px 0 0;
        }

        .ml-settings-header h2 {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }

        .ml-settings-close {
            font-size: 32px;
            cursor: pointer;
            opacity: 0.8;
            transition: opacity 0.2s;
            line-height: 1;
        }

        .ml-settings-close:hover {
            opacity: 1;
        }

        .ml-settings-body {
            padding: 30px;
        }

        .ml-settings-description {
            color: #666;
            margin-bottom: 25px;
            font-size: 16px;
            line-height: 1.6;
        }

        .ml-model-options {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 30px;
        }

        .ml-model-option {
            border: 2px solid #e5e5e5;
            border-radius: 12px;
            padding: 20px;
            display: flex;
            align-items: flex-start;
            gap: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .ml-model-option:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }

        .ml-model-option.selected {
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        }

        .ml-model-radio {
            padding-top: 2px;
        }

        .ml-model-radio input[type="radio"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }

        .ml-model-label {
            flex: 1;
            cursor: pointer;
        }

        .ml-model-label h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 18px;
        }

        .ml-model-label p {
            margin: 0 0 15px 0;
            color: #666;
            line-height: 1.5;
        }

        .ml-model-stats {
            display: flex;
            gap: 20px;
            font-size: 14px;
            color: #667eea;
            font-weight: 500;
        }

        .ml-performance-comparison {
            background: #f9f9f9;
            border-radius: 12px;
            padding: 20px;
        }

        .ml-performance-comparison h3 {
            margin: 0 0 15px 0;
            color: #333;
            font-size: 18px;
        }

        .ml-performance-comparison table {
            width: 100%;
            border-collapse: collapse;
        }

        .ml-performance-comparison th,
        .ml-performance-comparison td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e5e5;
        }

        .ml-performance-comparison th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }

        .ml-performance-comparison td {
            color: #666;
        }

        .ml-performance-comparison td.highlight {
            color: #667eea;
            font-weight: 600;
        }

        .ml-settings-footer {
            padding: 20px 30px;
            border-top: 1px solid #e5e5e5;
            display: flex;
            justify-content: flex-end;
            gap: 15px;
        }

        .ml-settings-footer button {
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
        }

        .ml-settings-save {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .ml-settings-save:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .ml-settings-cancel {
            background: #e5e5e5;
            color: #666;
        }

        .ml-settings-cancel:hover {
            background: #d5d5d5;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from {
                transform: translateY(50px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        </style>
    `;

    // Initialize ML Settings
    function initMLSettings() {
        // Add CSS to head
        if (!document.getElementById('mlSettingsStyles')) {
            const styleElement = document.createElement('style');
            styleElement.id = 'mlSettingsStyles';
            styleElement.innerHTML = mlSettingsCSS.replace('<style>', '').replace('</style>', '');
            document.head.appendChild(styleElement);
        }

        // Add HTML to body
        if (!document.getElementById('mlSettingsBtn')) {
            const container = document.createElement('div');
            container.innerHTML = mlSettingsHTML;
            while (container.firstChild) {
                document.body.appendChild(container.firstChild);
            }

            // Set up event listeners
            setupEventListeners();

            // Load saved settings
            loadSavedSettings();
        }
    }

    // Setup event listeners
    function setupEventListeners() {
        const btn = document.getElementById('mlSettingsBtn');
        const modal = document.getElementById('mlSettingsModal');
        const closeBtn = document.querySelector('.ml-settings-close');
        const saveBtn = document.querySelector('.ml-settings-save');
        const cancelBtn = document.querySelector('.ml-settings-cancel');
        const modelOptions = document.querySelectorAll('.ml-model-option');

        // Open modal
        btn.addEventListener('click', () => {
            modal.classList.add('show');
        });

        // Close modal
        closeBtn.addEventListener('click', () => {
            modal.classList.remove('show');
        });

        cancelBtn.addEventListener('click', () => {
            modal.classList.remove('show');
            loadSavedSettings(); // Reset to saved state
        });

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('show');
                loadSavedSettings(); // Reset to saved state
            }
        });

        // Model selection
        modelOptions.forEach(option => {
            option.addEventListener('click', () => {
                const radio = option.querySelector('input[type="radio"]');
                radio.checked = true;

                modelOptions.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
            });
        });

        // Save settings
        saveBtn.addEventListener('click', () => {
            const selectedModel = document.querySelector('input[name="mlModel"]:checked').value;
            localStorage.setItem('ml_model', selectedModel);

            // Call API to update model selection
            updateModelSelection(selectedModel);

            modal.classList.remove('show');
            showNotification(`ML Model switched to ${selectedModel === 'ensemble' ? 'Ensemble (7-Model)' : 'Classic'}`);
        });
    }

    // Load saved settings
    function loadSavedSettings() {
        const savedModel = localStorage.getItem('ml_model') || 'classic';
        const radio = document.querySelector(`input[value="${savedModel}"]`);
        if (radio) {
            radio.checked = true;
            const option = radio.closest('.ml-model-option');
            document.querySelectorAll('.ml-model-option').forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
        }
    }

    // Update model selection via API
    function updateModelSelection(model) {
        fetch('/api/v1/models/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model: model })
        }).then(response => {
            if (response.ok) {
                console.log('Model selection updated:', model);
            }
        }).catch(error => {
            console.error('Error updating model selection:', error);
        });
    }

    // Show notification
    function showNotification(message) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            border-radius: 8px;
            z-index: 10000;
            animation: slideUp 0.3s;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        `;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'fadeOut 0.3s';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initMLSettings);
    } else {
        initMLSettings();
    }

    // Export for manual initialization if needed
    window.MLSettings = {
        init: initMLSettings
    };
})();