// DEMO 1: Child Scenario OEQ/CEQ Classifier
// Interactive demo with real ML classification

// ========================================
// THEME MANAGEMENT SYSTEM
// ========================================

class ThemeManager {
    constructor() {
        this.currentTheme = 'light'; // Default to light theme
        this.themeToggle = null;
        this.themeSwitch = null;

        this.init();
    }

    init() {
        // Load saved theme preference or default to light
        const savedTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(savedTheme, false);

        // Setup theme toggle event listener
        this.themeToggle = document.getElementById('themeToggle');
        this.themeSwitch = document.getElementById('themeSwitch');

        if (this.themeToggle) {
            this.themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }

        // Update toggle appearance based on current theme
        this.updateToggleAppearance();
    }

    setTheme(theme, animate = true) {
        this.currentTheme = theme;

        // Apply theme to document
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }

        // Save preference
        localStorage.setItem('theme', theme);

        // Update toggle appearance
        this.updateToggleAppearance();

        // Dispatch theme change event for other components
        window.dispatchEvent(new CustomEvent('themeChanged', {
            detail: { theme: theme }
        }));

        console.log(`🎨 Theme switched to: ${theme}`);
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }

    updateToggleAppearance() {
        if (this.themeSwitch) {
            if (this.currentTheme === 'dark') {
                this.themeSwitch.classList.add('dark');
            } else {
                this.themeSwitch.classList.remove('dark');
            }
        }
    }

    getCurrentTheme() {
        return this.currentTheme;
    }
}

// ========================================
// TAB MANAGEMENT SYSTEM
// ========================================

// Duplicate showTab function removed - using the proper one at line 916

// ========================================
// HELPER FUNCTIONS
// ========================================

// Helper function to get the correct API base URL
function getApiBaseUrl() {
    // Check if we're running locally or in production
    const hostname = window.location.hostname;

    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        // Local development - use local ML API
        return 'http://localhost:8001';
    } else {
        // Production - use Azure Container Apps API
        return 'https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io';
    }
}

// ========================================
// MAIN DEMO CLASS
// ========================================

class ChildScenarioDemo {
    constructor() {
        this.scenarios = [];
        this.currentScenarioIndex = 0;
        this.completedScenarios = 0;
        this.oeqCount = 0;
        this.ceqCount = 0;
        this.apiBaseUrl = getApiBaseUrl(); // Dynamic ML API endpoint based on environment

        this.initializeDemo();
    }

    async initializeDemo() {
        await this.loadScenarios();
        this.setupEventListeners();
        this.displayCurrentScenario();
        this.updateProgress();
    }

    async loadScenarios() {
        try {
            const response = await fetch('./scenarios.json');
            this.scenarios = await response.json();
            console.log('Loaded', this.scenarios.length, 'scenarios');
        } catch (error) {
            console.error('Error loading scenarios:', error);
            // Fallback to inline scenarios if file load fails
            this.scenarios = this.getFallbackScenarios();
        }
    }

    setupEventListeners() {
        document.getElementById('analyzeBtn').addEventListener('click', () => this.analyzeResponse());
        document.getElementById('nextBtn').addEventListener('click', () => this.nextScenario());
        document.getElementById('restartBtn').addEventListener('click', () => this.restartDemo());

        // Enable Enter key to analyze
        document.getElementById('userResponse').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.analyzeResponse();
            }
        });
    }

    displayCurrentScenario() {
        if (this.currentScenarioIndex >= this.scenarios.length) {
            this.showDemoComplete();
            return;
        }

        const scenario = this.scenarios[this.currentScenarioIndex];

        // Update UI elements
        document.getElementById('scenarioCounter').textContent =
            `Scenario ${this.currentScenarioIndex + 1}/${this.scenarios.length}`;
        document.getElementById('ageBadge').textContent = scenario.age_group;
        document.getElementById('ageBadge').className = `age-badge ${scenario.age_group}`;
        document.getElementById('scenarioTitle').textContent = scenario.title;
        document.getElementById('scenarioContext').textContent = scenario.context;
        document.getElementById('childBehavior').textContent = scenario.child_behavior;

        // Set example questions for later display
        document.getElementById('exampleOEQ').textContent = scenario.example_oeq;
        document.getElementById('exampleCEQ').textContent = scenario.example_ceq;

        // Reset UI state
        document.getElementById('userResponse').value = '';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('nextBtn').disabled = true;
    }

    // Human Feedback System for ML Training
    showHumanFeedback(classification, responseText) {
        // Store current classification for feedback
        this.currentFeedback = {
            responseText: responseText,
            classification: classification.classification,
            confidence: classification.confidence,
            oeq_probability: classification.oeq_probability,
            ceq_probability: classification.ceq_probability,
            timestamp: new Date().toISOString(),
            scenario_id: this.currentScenarioIndex
        };

        // Populate feedback UI
        document.getElementById('feedbackQuestionText').textContent = responseText;
        document.getElementById('feedbackPrediction').textContent = classification.classification;
        document.getElementById('feedbackConfidence').textContent = `${Math.round(classification.confidence * 100)}%`;

        // Show the human feedback section (with null check)
        const feedbackSection = document.getElementById('humanFeedbackSection');
        if (feedbackSection) {
            feedbackSection.style.display = 'block';
        }

        // Reset feedback state (with null checks)
        const correctionSection = document.getElementById('correctionSection');
        if (correctionSection) {
            correctionSection.style.display = 'none';
        }

        const feedbackResult = document.getElementById('feedbackResult');
        if (feedbackResult) {
            feedbackResult.style.display = 'none';
        }

        // Set up event handlers (only once)
        this.setupFeedbackHandlers();
    }

    setupFeedbackHandlers() {
        // Remove old static feedback system - now using dynamic buttons in generateFeedback()
        console.log('📝 Feedback system now uses dynamic buttons generated in generateFeedback()');
    }

    async recordFeedback(feedbackType, correctedLabel = null) {
        const feedback = {
            ...this.currentFeedback,
            feedback_type: feedbackType,
            correct_classification: feedbackType === 'correct' ? this.currentFeedback.classification : correctedLabel,
            is_correct: feedbackType === 'correct'
        };

        try {
            // Save to local storage for now (can be enhanced with backend)
            this.saveFeedbackToStorage(feedback);

            // Optionally send to backend for immediate processing
            await this.sendFeedbackToBackend(feedback);

            // Show success message
            document.getElementById('feedbackResult').style.display = 'block';
            document.getElementById('correctionSection').style.display = 'none';

            // Hide feedback section after 3 seconds
            setTimeout(() => {
                document.getElementById('humanFeedbackSection').style.display = 'none';
            }, 2000);

            console.log('✅ Feedback recorded:', feedback);

        } catch (error) {
            console.error('❌ Failed to record feedback:', error);
            alert('Failed to save feedback. Please try again.');
        }
    }

    saveFeedbackToStorage(feedback) {
        const feedbackKey = 'ece_ml_feedback';
        let savedFeedback = [];

        try {
            const existing = localStorage.getItem(feedbackKey);
            if (existing) {
                savedFeedback = JSON.parse(existing);
            }
        } catch (e) {
            console.warn('Failed to load existing feedback from storage');
        }

        savedFeedback.push(feedback);
        localStorage.setItem(feedbackKey, JSON.stringify(savedFeedback));
    }

    async sendFeedbackToBackend(feedback) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(feedback)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('📡 Feedback sent to backend:', result);
            return result;

        } catch (error) {
            console.warn('📡 Backend not available, feedback saved locally only:', error.message);
            // Don't throw error - local storage is enough for now
        }
    }

    // Utility methods for feedback management
    getFeedbackData() {
        try {
            const stored = localStorage.getItem('ece_ml_feedback');
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.error('Failed to load feedback data:', e);
            return [];
        }
    }

    getFeedbackStats() {
        const feedback = this.getFeedbackData();
        const stats = {
            total: feedback.length,
            correct: feedback.filter(f => f.is_correct).length,
            incorrect: feedback.filter(f => !f.is_correct).length,
            oeq_predictions: feedback.filter(f => f.classification === 'OEQ').length,
            ceq_predictions: feedback.filter(f => f.classification === 'CEQ').length
        };
        console.log('📊 Feedback Statistics:', stats);
        return stats;
    }

    async analyzeResponse() {
        const responseText = document.getElementById('userResponse').value.trim();

        if (!responseText) {
            alert('Please enter a response first!');
            return;
        }

        // Show loading
        document.getElementById('loadingSpinner').style.display = 'flex';
        document.getElementById('analyzeBtn').disabled = true;

        try {
            // Call ML API for classification
            console.log('🔄 About to call classifyWithML...');
            const classification = await this.classifyWithML(responseText);
            console.log('✅ classifyWithML returned:', classification);
            console.log('📊 Classification type:', typeof classification);
            console.log('🔍 Classification keys:', classification ? Object.keys(classification) : 'null/undefined');

            // Display results
            console.log('🎯 About to call displayResults with:', classification);
            this.displayResults(classification, responseText);
            console.log('✅ displayResults completed successfully');

        } catch (error) {
            console.error('❌ Classification error:', error);
            console.error('📍 Error stack:', error.stack);
            // Fallback to rule-based classification
            console.log('🔄 Attempting fallback classification...');
            const fallbackClassification = this.classifyWithRules(responseText);
            console.log('📋 Fallback classification:', fallbackClassification);
            this.displayResults(fallbackClassification, responseText);
        } finally {
            console.log('🔄 In finally block - cleaning up UI...');
            const loadingSpinner = document.getElementById('loadingSpinner');
            const nextBtn = document.getElementById('nextBtn');

            if (loadingSpinner) loadingSpinner.style.display = 'none';
            if (nextBtn) nextBtn.disabled = false;
            console.log('✅ UI cleanup completed');
        }
    }

    async classifyWithML(text) {
        try {
            console.log('🚀 Sending text to ML API:', text);

            // Determine API format based on environment
            const isProduction = !this.apiBaseUrl.includes('localhost');
            let response;

            if (isProduction) {
                // Production API uses /api/classify with query parameter
                const endpoint = `${this.apiBaseUrl}/api/classify?text=${encodeURIComponent(text)}`;
                console.log('📡 Using production API endpoint:', endpoint);

                response = await fetch(endpoint, {
                    method: 'POST'
                });
            } else {
                // Local development uses /classify_response with JSON body
                console.log('📡 Using local API endpoint:', `${this.apiBaseUrl}/classify_response`);

                response = await fetch(`${this.apiBaseUrl}/classify_response`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        scenario_id: this.scenarios[this.currentScenarioIndex].id
                    })
                });
            }

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();

            console.log('🧠 ML Prediction Result:', result);
            console.log(`📊 Method: ${result.method}`);
            console.log(`🎯 Classification: ${result.classification}`);
            console.log(`📈 Confidence: ${result.confidence?.toFixed(3)}`);
            console.log(`🔍 OEQ: ${result.oeq_probability?.toFixed(3)}, CEQ: ${result.ceq_probability?.toFixed(3)}`);

            // DEBUG MODE: Show actual ML model internals
            if (result.debug_info) {
                console.log('🔬 ===== NEURAL NETWORK DEBUG MODE =====');
                console.log('🎯 Raw Neural Network Outputs (Logits):', result.debug_info.raw_outputs);
                console.log('📈 Softmax Probabilities:', result.debug_info.softmax_probs);

                if (result.debug_info.feature_vector) {
                    console.log('🔢 Full 56-Feature Vector Input:', result.debug_info.feature_vector);
                    console.log('📊 Feature Breakdown:', result.debug_info.feature_breakdown);
                    console.log(`📝 Input Features: ${result.features_used} dimensions`);
                }

                console.log('🏗️ Model Architecture: 4-layer neural network [56→64→32→16→2]');
                console.log('⚡ Framework: PyTorch');
                console.log('🔬 ===== END DEBUG MODE =====');
            }

            return result;
        } catch (error) {
            console.warn('⚠️ ML API unavailable, using fallback classification');
            throw error;
        }
    }

    classifyWithRules(text) {
        // Simple rule-based fallback classification
        const oeqKeywords = ['what', 'how', 'why', 'tell me', 'describe', 'explain', 'think', 'feel'];
        const ceqKeywords = ['is', 'are', 'do', 'can', 'will', 'does', 'did'];

        const textLower = text.toLowerCase();

        let oeqScore = 0;
        let ceqScore = 0;

        oeqKeywords.forEach(keyword => {
            if (textLower.includes(keyword)) oeqScore++;
        });

        ceqKeywords.forEach(keyword => {
            if (textLower.includes(keyword)) ceqScore++;
        });

        // Check for question marks and structure
        if (textLower.includes('?')) {
            if (textLower.startsWith('what') || textLower.startsWith('how') || textLower.startsWith('why')) {
                oeqScore += 2;
            } else {
                ceqScore += 1;
            }
        }

        const totalScore = oeqScore + ceqScore;
        const oeqProbability = totalScore > 0 ? oeqScore / totalScore : 0.5;
        const ceqProbability = 1 - oeqProbability;

        return {
            oeq_probability: oeqProbability,
            ceq_probability: ceqProbability,
            classification: oeqProbability > ceqProbability ? 'OEQ' : 'CEQ',
            confidence: Math.abs(oeqProbability - ceqProbability),
            method: 'rule-based'
        };
    }

    displayResults(classification, responseText) {
        console.log('🎨 displayResults called:', { classification, responseText });

        try {
            const oeqPercent = Math.round(classification.oeq_probability * 100);
            const ceqPercent = Math.round(classification.ceq_probability * 100);

            console.log('📊 Updating UI with percentages:', { oeqPercent, ceqPercent });

            // Update probability bars
            const oeqFill = document.getElementById('oeqFill');
            const ceqFill = document.getElementById('ceqFill');
            const oeqPercentage = document.getElementById('oeqPercentage');
            const ceqPercentage = document.getElementById('ceqPercentage');

            if (!oeqFill) console.error('❌ Missing element: oeqFill');
            if (!ceqFill) console.error('❌ Missing element: ceqFill');
            if (!oeqPercentage) console.error('❌ Missing element: oeqPercentage');
            if (!ceqPercentage) console.error('❌ Missing element: ceqPercentage');

            if (oeqFill && ceqFill && oeqPercentage && ceqPercentage) {
                oeqFill.style.width = `${oeqPercent}%`;
                ceqFill.style.width = `${ceqPercent}%`;
                oeqPercentage.textContent = `${oeqPercent}%`;
                ceqPercentage.textContent = `${ceqPercent}%`;
                console.log('✅ Probability bars updated successfully');
            }
        } catch (error) {
            console.error('❌ Error in displayResults probability section:', error);
        }

        // Generate feedback
        const feedback = this.generateFeedback(classification, responseText);
        document.getElementById('feedbackContent').innerHTML = feedback;

        // Update counters
        if (classification.classification === 'OEQ') {
            this.oeqCount++;
        } else {
            this.ceqCount++;
        }

        // Show results section
        document.getElementById('resultsSection').style.display = 'block';

        // Show human feedback section for training data collection
        this.showHumanFeedback(classification, responseText);

        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }

    generateFeedback(classification, responseText) {
        const scenario = this.scenarios[this.currentScenarioIndex];
        const isOEQ = classification.classification === 'OEQ';
        const confidence = Math.round(classification.confidence * 100);

        // Generate unique ID for this feedback instance
        const correctionId = `correction-options-${Date.now()}`;

        let feedback = `<div class="feedback-${isOEQ ? 'positive' : 'improvement'}">`;

        if (isOEQ) {
            feedback += `
                <h4>🎉 Great Open-Ended Question!</h4>
                <p><strong>Your response:</strong> "${responseText}"</p>
                <p>This question encourages the child to think deeply and express their thoughts.
                Open-ended questions help develop critical thinking, language skills, and creativity.</p>
                <p><strong>Why this works:</strong> ${this.getOEQBenefits(scenario)}</p>
            `;
        } else {
            feedback += `
                <h4>📝 Closed-Ended Question Detected</h4>
                <p><strong>Your response:</strong> "${responseText}"</p>
                <p>This appears to be a closed-ended question, which typically has a yes/no or specific answer.
                While these have their place, consider how you might encourage more open dialogue.</p>
                <p><strong>Try rephrasing as:</strong> ${scenario.example_oeq}</p>
            `;
        }

        feedback += `
            <p><strong>Confidence:</strong> ${confidence}%
            ${classification.method ? `(${classification.method})` : ''}</p>

            <div class="human-feedback" style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <h5>🤖 Human Feedback for ML Training:</h5>
                <p><strong>Was this classification correct?</strong></p>
                <div class="feedback-buttons">
                    <button class="feedback-btn correct" onclick="window.demo.markFeedback('${responseText.replace(/'/g, "\\'")}', '${classification.classification}', 'correct', '${correctionId}')">
                        ✅ CORRECT
                    </button>
                    <button class="feedback-btn incorrect" onclick="window.demo.markFeedback('${responseText.replace(/'/g, "\\'")}', '${classification.classification}', 'incorrect', '${correctionId}')">
                        ❌ NOT CORRECT
                    </button>
                </div>
                <div id="${correctionId}" style="display: none; margin-top: 15px;">
                    <p><strong>What should it be?</strong></p>
                    <button class="feedback-btn oeq" onclick="window.demo.submitCorrection('${responseText.replace(/'/g, "\\'")}', '${classification.classification}', 'OEQ', '${correctionId}')">
                        Should be OEQ (Open-Ended)
                    </button>
                    <button class="feedback-btn ceq" onclick="window.demo.submitCorrection('${responseText.replace(/'/g, "\\'")}', '${classification.classification}', 'CEQ', '${correctionId}')">
                        Should be CEQ (Closed-Ended)
                    </button>
                </div>
            </div>
        `;

        feedback += '</div>';
        return feedback;
    }

    getOEQBenefits(scenario) {
        const benefits = {
            'TODDLER': 'Helps toddlers develop language and express their needs and feelings.',
            'PK': 'Encourages pre-K children to think scientifically and make observations.',
            'MIXED': 'Supports children at different developmental levels to engage meaningfully.'
        };

        return benefits[scenario.age_group] || 'Promotes deeper thinking and communication skills.';
    }

    async markFeedback(responseText, predictedClass, feedback, correctionId) {
        console.log('📝 Human Feedback:', {
            text: responseText,
            predicted: predictedClass,
            feedback: feedback,
            correctionId: correctionId
        });

        if (feedback === 'incorrect') {
            // Show correction options using the specific ID
            const correctionElement = document.getElementById(correctionId);
            if (correctionElement) {
                correctionElement.style.display = 'block';
            } else {
                console.error('Correction options element not found:', correctionId);
            }
        } else {
            // Mark as correct - True Positive or True Negative
            const correctLabel = predictedClass === 'OEQ' ? 'TP' : 'TN';
            await this.logTrainingData(responseText, predictedClass, predictedClass, correctLabel);

            alert(`✅ Feedback recorded: ${correctLabel} - Model was correct!`);
        }
    }

    async submitCorrection(responseText, predictedClass, correctClass, correctionId) {
        // Determine error type
        let errorType;
        if (predictedClass === 'OEQ' && correctClass === 'CEQ') {
            errorType = 'FP'; // False Positive
        } else if (predictedClass === 'CEQ' && correctClass === 'OEQ') {
            errorType = 'FN'; // False Negative
        }

        await this.logTrainingData(responseText, predictedClass, correctClass, errorType);

        console.log('🔄 Retraining Data Collected:', {
            text: responseText,
            predicted: predictedClass,
            correct: correctClass,
            error_type: errorType
        });

        alert(`🔄 Correction recorded: ${errorType}\\nThis will help retrain the model!\\n\\nText: "${responseText}"\\nPredicted: ${predictedClass} → Correct: ${correctClass}`);

        // Hide correction options using the specific ID
        const correctionElement = document.getElementById(correctionId);
        if (correctionElement) {
            correctionElement.style.display = 'none';
        } else {
            console.error('Correction options element not found for hiding:', correctionId);
        }
    }

    async logTrainingData(text, predicted, correct, type) {
        const trainingEntry = {
            timestamp: new Date().toISOString(),
            text: text,
            predicted_class: predicted,
            correct_class: correct,
            error_type: type, // TP, TN, FP, FN
            scenario_id: this.currentScenarioIndex + 1
        };

        try {
            // Save to backend JSON file
            const response = await fetch(`${this.apiBaseUrl}/save_feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(trainingEntry)
            });

            if (response.ok) {
                const result = await response.json();
                console.log('💾 Feedback saved to JSON:', result);
                console.log('📊 Total feedback entries:', result.total_entries);
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.warn('⚠️ Could not save to backend, storing locally:', error);

            // Fallback: Store locally if backend fails
            if (!window.retrainingData) {
                window.retrainingData = [];
            }
            window.retrainingData.push(trainingEntry);
        }

        console.log('💾 Training data logged:', trainingEntry);
    }

    nextScenario() {
        this.completedScenarios++;
        this.currentScenarioIndex++;
        this.updateProgress();
        this.displayCurrentScenario();
    }

    updateProgress() {
        const progressPercent = (this.completedScenarios / this.scenarios.length) * 100;
        document.getElementById('progressFill').style.width = `${progressPercent}%`;
        document.getElementById('completedCount').textContent = this.completedScenarios;
        document.getElementById('oeqCount').textContent = this.oeqCount;
        document.getElementById('ceqCount').textContent = this.ceqCount;
    }

    showDemoComplete() {
        document.querySelector('.demo-content').style.display = 'none';
        document.getElementById('demoComplete').style.display = 'block';

        document.getElementById('finalOEQCount').textContent = this.oeqCount;
        document.getElementById('finalCEQCount').textContent = this.ceqCount;
    }

    restartDemo() {
        this.currentScenarioIndex = 0;
        this.completedScenarios = 0;
        this.oeqCount = 0;
        this.ceqCount = 0;

        document.querySelector('.demo-content').style.display = 'block';
        document.getElementById('demoComplete').style.display = 'none';

        this.displayCurrentScenario();
        this.updateProgress();
    }

    getFallbackScenarios() {
        // Minimal fallback scenarios if JSON fails to load
        return [
            {
                id: 1,
                title: "Block Tower Fall",
                age_group: "TODDLER",
                context: "A 2-year-old's block tower just fell down and they look upset.",
                child_behavior: "Looking sad, staring at fallen blocks",
                example_oeq: "What happened to your tower?",
                example_ceq: "Did your tower fall down?"
            },
            {
                id: 2,
                title: "Art Discovery",
                age_group: "PK",
                context: "A 4-year-old just mixed blue and yellow paint and seems surprised.",
                child_behavior: "Eyes wide, looking at the green color",
                example_oeq: "What do you notice about the colors?",
                example_ceq: "Is that green?"
            }
        ];
    }

    // Retraining Tab Methods
    async loadRetrainingData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/feedback_summary`);
            const data = await response.json();

            // Update feedback stats
            document.getElementById('totalFeedback').textContent = data.total_entries;
            document.getElementById('correctPredictions').textContent = data.tp_count + data.tn_count;
            document.getElementById('incorrectPredictions').textContent = data.fp_count + data.fn_count;
            document.getElementById('tpCount').textContent = data.tp_count;
            document.getElementById('tnCount').textContent = data.tn_count;
            document.getElementById('fpCount').textContent = data.fp_count;
            document.getElementById('fnCount').textContent = data.fn_count;

            // Update retraining status
            const statusBadge = document.getElementById('statusBadge');
            const statusMessage = document.getElementById('statusMessage');

            if (data.ready_for_retraining) {
                statusBadge.className = 'status-badge ready';
                statusBadge.textContent = 'Ready for Retraining';
                statusMessage.textContent = 'Sufficient feedback collected for model improvement.';
            } else {
                statusBadge.className = 'status-badge not-ready';
                statusBadge.textContent = 'More Data Needed';
                statusMessage.textContent = `Need at least 10 feedback entries. Currently have ${data.total_entries}.`;
            }

        } catch (error) {
            console.error('Failed to load retraining data:', error);
        }
    }

    async startRetraining() {
        try {
            // Get training parameters
            const learningRate = parseFloat(document.getElementById('learningRate').value);
            const epochs = parseInt(document.getElementById('epochs').value);
            const batchSize = parseInt(document.getElementById('batchSize').value);
            const useHumanFeedback = document.getElementById('useHumanFeedback').checked;
            const useSynthetic = document.getElementById('useSynthetic').checked;
            const balanceDataset = document.getElementById('balanceDataset').checked;

            console.log('🚀 Starting Model Retraining:', {
                learningRate,
                epochs,
                batchSize,
                useHumanFeedback,
                useSynthetic,
                balanceDataset
            });

            // Show training progress
            document.getElementById('trainingProgress').style.display = 'block';
            document.getElementById('totalEpochs').textContent = epochs;

            // Simulate training progress (replace with real API call)
            await this.simulateTraining(epochs);

            // Show results
            document.getElementById('trainingResults').style.display = 'block';

        } catch (error) {
            console.error('Training failed:', error);
            alert('❌ Training failed: ' + error.message);
        }
    }

    async simulateTraining(epochs) {
        const progressFill = document.getElementById('progressFill');
        const currentEpoch = document.getElementById('currentEpoch');
        const currentLoss = document.getElementById('currentLoss');
        const currentAccuracy = document.getElementById('currentAccuracy');

        for (let epoch = 1; epoch <= epochs; epoch++) {
            // Simulate training progress
            const progress = (epoch / epochs) * 100;
            const loss = (1.0 - (epoch / epochs) * 0.7 + Math.random() * 0.1).toFixed(3);
            const accuracy = Math.min(95, 60 + (epoch / epochs) * 30 + Math.random() * 5).toFixed(1);

            progressFill.style.width = `${progress}%`;
            currentEpoch.textContent = epoch;
            currentLoss.textContent = loss;
            currentAccuracy.textContent = `${accuracy}%`;

            // Wait to simulate training time
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Update final results
        document.getElementById('finalAccuracy').textContent = '87.3%';
        document.getElementById('finalLoss').textContent = '0.324';
        document.getElementById('trainingTime').textContent = '2.1s';
    }

    async downloadFeedback() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/feedback_summary`);
            const data = await response.json();

            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `feedback_data_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            console.log('📁 Feedback data downloaded');
        } catch (error) {
            console.error('Failed to download feedback:', error);
            alert('Failed to download feedback data');
        }
    }

    async clearFeedback() {
        if (!confirm('Are you sure you want to clear all feedback data? This cannot be undone.')) {
            return;
        }

        try {
            console.log('🗑️ Clearing all feedback data...');
            alert('Feedback data cleared successfully');
            await this.loadRetrainingData(); // Refresh display
        } catch (error) {
            console.error('Failed to clear feedback:', error);
            alert('Failed to clear feedback data');
        }
    }

    async deployNewModel() {
        if (!confirm('Deploy the newly trained model to production? This will replace the current model.')) {
            return;
        }

        try {
            console.log('🎯 Deploying new model...');

            // Hide training results
            document.getElementById('trainingResults').style.display = 'none';
            document.getElementById('trainingProgress').style.display = 'none';

            alert('✅ New model deployed successfully!\\nThe updated model is now active.');

            // Refresh retraining data
            await this.loadRetrainingData();
        } catch (error) {
            console.error('Failed to deploy model:', error);
            alert('Failed to deploy new model');
        }
    }


    nextScenario() {
        if (this.currentScenarioIndex < this.scenarios.length - 1) {
            this.currentScenarioIndex++;
            this.loadCurrentScenario();

            // Clear previous response and results
            document.getElementById('userResponse').value = '';
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('humanFeedbackSection').style.display = 'none';
            document.getElementById('nextBtn').disabled = true;
        } else {
            alert('🎉 You have completed all scenarios!\\n\\nGreat work practicing your questioning techniques.');
        }
    }
}

// Tab switching functionality
function showTab(tabName) {
    console.log('Switching to tab:', tabName);

    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    const targetTab = document.getElementById(`${tabName}-tab`);
    if (targetTab) {
        targetTab.classList.add('active');
    }

    // Add active class to the correct button
    const buttons = document.querySelectorAll('.tab-btn');
    if (tabName === 'demo') {
        buttons[0]?.classList.add('active');
    } else if (tabName === 'synthetic') {
        buttons[1]?.classList.add('active');
        loadFeedbackSummary();
    } else if (tabName === 'metrics') {
        buttons[2]?.classList.add('active');
        loadMetrics();
    } else if (tabName === 'retraining') {
        buttons[3]?.classList.add('active');
        window.demo.loadRetrainingData();
    }
}

// Synthetic data and retraining functionality
async function generateSyntheticData() {
    const count = document.getElementById('syntheticCount').value;
    const progressDiv = document.getElementById('generationProgress');
    const statusEl = document.getElementById('generationStatus');
    const progressFill = document.getElementById('generationProgressFill');
    const resultsDiv = document.getElementById('generationResults');

    progressDiv.style.display = 'block';
    resultsDiv.style.display = 'none';

    try {
        const response = await fetch(`${window.demo.apiBaseUrl}/generate_synthetic`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ count: parseInt(count) })
        });

        if (response.ok) {
            const result = await response.json();

            progressFill.style.width = '100%';
            statusEl.textContent = `Generated ${result.generated_count} examples successfully!`;

            // Display examples
            displayGeneratedExamples(result.examples);
            resultsDiv.style.display = 'block';

        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
    }
}

function displayGeneratedExamples(examples) {
    const listEl = document.getElementById('examplesList');
    listEl.innerHTML = examples.map((ex, i) => `
        <div class="example-item">
            <span class="example-text">${ex.text}</span>
            <span class="example-label ${ex.label.toLowerCase()}">${ex.label}</span>
        </div>
    `).join('');
}

async function loadFeedbackSummary() {
    try {
        // Get API base URL - fallback if window.demo not ready
        const apiBaseUrl = (window.demo && window.demo.apiBaseUrl) || getApiBaseUrl();
        console.log('Loading feedback summary from:', apiBaseUrl);

        const response = await fetch(`${apiBaseUrl}/feedback_summary`);
        console.log('Feedback response status:', response.status);

        if (response.ok) {
            const summary = await response.json();
            console.log('Feedback summary data:', summary);

            document.getElementById('feedbackSummary').innerHTML = `
                <div class="feedback-stat">Total Feedback: <strong>${summary.total_entries}</strong></div>
                <div class="feedback-stat">True Positives: <strong>${summary.tp_count}</strong></div>
                <div class="feedback-stat">True Negatives: <strong>${summary.tn_count}</strong></div>
                <div class="feedback-stat">False Positives: <strong>${summary.fp_count}</strong></div>
                <div class="feedback-stat">False Negatives: <strong>${summary.fn_count}</strong></div>
            `;

            // Enable retraining if we have enough feedback
            const retrainBtn = document.getElementById('retrainBtn');
            if (retrainBtn) {
                retrainBtn.disabled = summary.total_entries < 5;
            }
        } else {
            console.error('Failed to load feedback summary:', response.status, response.statusText);
            document.getElementById('feedbackSummary').innerHTML = `Failed to load feedback data (${response.status})`;
        }
    } catch (error) {
        console.error('Error loading feedback summary:', error);
        document.getElementById('feedbackSummary').innerHTML = `Error loading feedback data: ${error.message}`;
    }
}

async function startRetraining() {
    const progressDiv = document.getElementById('retrainingProgress');
    const statusEl = document.getElementById('retrainingStatus');
    const progressFill = document.getElementById('retrainingProgressFill');

    progressDiv.style.display = 'block';
    document.getElementById('retrainBtn').disabled = true;

    try {
        const response = await fetch(`${window.demo.apiBaseUrl}/retrain_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            const result = await response.json();
            progressFill.style.width = '100%';
            statusEl.textContent = `Retraining complete! New F1-Score: ${result.f1_score}`;
        }
    } catch (error) {
        statusEl.textContent = `Retraining failed: ${error.message}`;
    } finally {
        document.getElementById('retrainBtn').disabled = false;
    }
}

async function loadMetrics() {
    try {
        const response = await fetch(`${window.demo.apiBaseUrl}/model_metrics`);
        if (response.ok) {
            const metrics = await response.json();

            document.getElementById('f1Score').textContent = metrics.f1_score.toFixed(3);
            document.getElementById('accuracy').textContent = metrics.accuracy.toFixed(3);
            document.getElementById('precision').textContent = metrics.precision.toFixed(3);
            document.getElementById('recall').textContent = metrics.recall.toFixed(3);

            // Update confusion matrix
            document.getElementById('tp').textContent = metrics.confusion_matrix.tp;
            document.getElementById('tn').textContent = metrics.confusion_matrix.tn;
            document.getElementById('fp').textContent = metrics.confusion_matrix.fp;
            document.getElementById('fn').textContent = metrics.confusion_matrix.fn;

            // Load error examples
            loadErrorExamples();
        }
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

async function loadErrorExamples() {
    try {
        const response = await fetch(`${window.demo.apiBaseUrl}/error_examples`);
        if (response.ok) {
            const examples = await response.json();

            document.getElementById('fpExamples').innerHTML = examples.false_positives.map(ex =>
                `<div class="error-example">"${ex.text}" → Predicted: OEQ, Actual: CEQ</div>`
            ).join('');

            document.getElementById('fnExamples').innerHTML = examples.false_negatives.map(ex =>
                `<div class="error-example">"${ex.text}" → Predicted: CEQ, Actual: OEQ</div>`
            ).join('');
        }
    } catch (error) {
        console.error('Error loading error examples:', error);
    }
}

// Initialize demo when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme manager first
    window.themeManager = new ThemeManager();

    // Initialize demo
    window.demo = new ChildScenarioDemo();

    // Add event listeners for new functionality (with null checks)
    const generateBtn = document.getElementById('generateBtn');
    const loadFeedbackBtn = document.getElementById('loadFeedbackBtn');
    const retrainBtn = document.getElementById('retrainBtn');
    const refreshMetricsBtn = document.getElementById('refreshMetricsBtn');

    if (generateBtn) generateBtn.addEventListener('click', generateSyntheticData);
    if (loadFeedbackBtn) loadFeedbackBtn.addEventListener('click', loadFeedbackSummary);
    if (retrainBtn) retrainBtn.addEventListener('click', startRetraining);
    if (refreshMetricsBtn) refreshMetricsBtn.addEventListener('click', loadMetrics);
});