/**
 * Cultivate Learning - Interactive Demo
 * OEQ/CEQ Question Classification Demo
 */

// API Configuration
const API_CONFIG = {
    // Try multiple endpoints
    endpoints: [
        'http://localhost:8000',
        'http://localhost:8001',
        'https://cultivate-ml-api.azurecontainerapps.io'
    ],
    currentEndpoint: null,
    isOnline: false
};

// DOM Elements
const elements = {
    questionInput: document.getElementById('question-input'),
    classifyBtn: document.getElementById('classify-btn'),
    clearBtn: document.getElementById('clear-btn'),
    modelSelect: document.getElementById('model-select'),
    resultsPlaceholder: document.getElementById('results-placeholder'),
    resultsContent: document.getElementById('results-content'),
    classificationBadge: document.getElementById('classification-badge'),
    classificationLabel: document.getElementById('classification-label'),
    confidenceValue: document.getElementById('confidence-value'),
    confidenceFill: document.getElementById('confidence-fill'),
    oeqProb: document.getElementById('oeq-prob'),
    ceqProb: document.getElementById('ceq-prob'),
    votingSection: document.getElementById('voting-section'),
    oeqVotes: document.getElementById('oeq-votes'),
    ceqVotes: document.getElementById('ceq-votes'),
    modelVotes: document.getElementById('model-votes'),
    featuresList: document.getElementById('features-list'),
    teachingTip: document.getElementById('teaching-tip'),
    tipText: document.getElementById('tip-text'),
    apiStatus: document.getElementById('api-status'),
    apiEndpoint: document.getElementById('api-endpoint'),
    apiModel: document.getElementById('api-model')
};

// Teaching tips based on classification
const TEACHING_TIPS = {
    OEQ: [
        "Excellent open-ended question! This invites children to think deeply and share their ideas.",
        "Great use of an open-ended prompt! Remember to wait 3-5 seconds for children to formulate their response.",
        "This question encourages exploration and creative thinking. Follow up with 'Tell me more about that!'",
        "Open-ended questions like this help develop children's language and reasoning skills."
    ],
    CEQ: [
        "This is a closed-ended question that limits responses. Try rephrasing to 'What do you notice about...' instead.",
        "Consider transforming this into an open-ended question by asking 'How' or 'Why' instead.",
        "Closed questions are useful for quick checks, but balance them with open-ended follow-ups.",
        "Try extending this with 'Tell me more' or 'Why do you think that?' after the child responds."
    ]
};

// Feature extraction for display
function extractFeatures(text) {
    const features = [];
    const textLower = text.toLowerCase();

    // OEQ indicators
    const oeqKeywords = ['what', 'how', 'why', 'tell me', 'describe', 'explain', 'think', 'feel', 'imagine'];
    oeqKeywords.forEach(keyword => {
        if (textLower.includes(keyword)) {
            features.push({ name: keyword, type: 'positive' });
        }
    });

    // CEQ indicators
    const ceqKeywords = ['is', 'are', 'do', 'does', 'did', 'can', 'could', 'would'];
    ceqKeywords.forEach(keyword => {
        const pattern = new RegExp(`\\b${keyword}\\b`, 'i');
        if (pattern.test(textLower) && textLower.startsWith(keyword)) {
            features.push({ name: `starts with "${keyword}"`, type: 'negative' });
        }
    });

    // Structural features
    if (text.includes('?')) {
        features.push({ name: 'has question mark', type: 'neutral' });
    }
    if (text.split(' ').length > 7) {
        features.push({ name: 'longer question', type: 'positive' });
    }

    return features;
}

// Fallback local classification when API is unavailable
function classifyLocally(text) {
    const textLower = text.toLowerCase();
    let oeqScore = 0.5;

    // OEQ indicators
    const oeqKeywords = {
        'what do you think': 0.25,
        'how do you': 0.2,
        'why do you': 0.2,
        'tell me': 0.15,
        'describe': 0.15,
        'explain': 0.15,
        'what': 0.1,
        'how': 0.1,
        'why': 0.1,
        'think': 0.1,
        'feel': 0.1,
        'imagine': 0.1
    };

    for (const [keyword, weight] of Object.entries(oeqKeywords)) {
        if (textLower.includes(keyword)) {
            oeqScore += weight;
        }
    }

    // CEQ indicators
    const ceqStarts = ['is ', 'are ', 'do ', 'does ', 'did ', 'can ', 'could ', 'would ', 'should '];
    ceqStarts.forEach(start => {
        if (textLower.startsWith(start)) {
            oeqScore -= 0.25;
        }
    });

    // Clamp score
    oeqScore = Math.max(0.05, Math.min(0.95, oeqScore));

    const classification = oeqScore > 0.5 ? 'OEQ' : 'CEQ';
    const confidence = classification === 'OEQ' ? oeqScore : (1 - oeqScore);

    return {
        classification,
        confidence,
        probabilities: {
            OEQ: oeqScore,
            CEQ: 1 - oeqScore
        },
        model: 'local-heuristic',
        voting_details: null
    };
}

// API Classification
async function classifyQuestion(text, model = 'ensemble') {
    const endpoint = model === 'ensemble'
        ? '/api/v2/classify/ensemble'
        : '/api/v1/classify';

    // Try API first
    if (API_CONFIG.currentEndpoint) {
        try {
            const response = await fetch(`${API_CONFIG.currentEndpoint}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    scenario_id: 1,
                    debug_mode: true
                })
            });

            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.warn('API request failed, using local classification:', error);
        }
    }

    // Fallback to local classification
    return classifyLocally(text);
}

// Display results
function displayResults(result) {
    const { classification, confidence, probabilities, voting_details, model } = result;

    // Show results content
    elements.resultsPlaceholder.classList.add('hidden');
    elements.resultsContent.classList.remove('hidden');

    // Update classification badge
    elements.classificationBadge.className = `classification-badge ${classification.toLowerCase()}`;
    elements.classificationBadge.querySelector('.badge-type').textContent = classification;
    elements.classificationLabel.textContent = classification === 'OEQ'
        ? 'Open-Ended Question'
        : 'Closed-Ended Question';

    // Update confidence
    const confidencePercent = Math.round(confidence * 100);
    elements.confidenceValue.textContent = `${confidencePercent}%`;
    elements.confidenceFill.style.width = `${confidencePercent}%`;
    elements.confidenceFill.className = `meter-fill ${classification.toLowerCase()}`;

    // Update probabilities
    elements.oeqProb.textContent = (probabilities?.OEQ || (classification === 'OEQ' ? confidence : 1 - confidence)).toFixed(2);
    elements.ceqProb.textContent = (probabilities?.CEQ || (classification === 'CEQ' ? confidence : 1 - confidence)).toFixed(2);

    // Update ensemble voting (if available)
    if (voting_details && voting_details.individual_predictions) {
        elements.votingSection.style.display = 'block';

        const oeqCount = voting_details.vote_tally?.OEQ || 0;
        const ceqCount = voting_details.vote_tally?.CEQ || 0;

        elements.oeqVotes.textContent = `${oeqCount} model${oeqCount !== 1 ? 's' : ''} OEQ`;
        elements.ceqVotes.textContent = `${ceqCount} model${ceqCount !== 1 ? 's' : ''} CEQ`;

        // Display individual model votes
        elements.modelVotes.innerHTML = '';
        for (const [modelName, prediction] of Object.entries(voting_details.individual_predictions)) {
            const voteEl = document.createElement('div');
            voteEl.className = `model-vote-item ${prediction.prediction.toLowerCase()}`;
            voteEl.innerHTML = `
                <span class="model-name">${modelName}</span>
                <span class="model-pred">${prediction.prediction}</span>
                <span class="model-confidence">(${Math.round(prediction.confidence * 100)}%)</span>
            `;
            elements.modelVotes.appendChild(voteEl);
        }
    } else {
        elements.votingSection.style.display = 'none';
    }

    // Update features
    const features = extractFeatures(elements.questionInput.value);
    elements.featuresList.innerHTML = features.map(f =>
        `<span class="feature-tag ${f.type}">${f.name}</span>`
    ).join('');

    // Update teaching tip
    const tips = TEACHING_TIPS[classification];
    const randomTip = tips[Math.floor(Math.random() * tips.length)];
    elements.tipText.textContent = randomTip;
    elements.teachingTip.className = `teaching-tip ${classification.toLowerCase()}`;

    // Update API info
    elements.apiModel.textContent = model || 'unknown';
}

// Check API availability
async function checkApiStatus() {
    const statusDot = elements.apiStatus.querySelector('.status-dot');
    const statusText = elements.apiStatus.querySelector('.status-text');

    // Skip localhost endpoints when running on production (Azure)
    const isProduction = window.location.hostname.includes('azurestaticapps.net');
    const endpointsToCheck = isProduction
        ? API_CONFIG.endpoints.filter(e => !e.includes('localhost'))
        : API_CONFIG.endpoints;

    for (const endpoint of endpointsToCheck) {
        try {
            const response = await fetch(`${endpoint}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(3000)
            });

            if (response.ok) {
                API_CONFIG.currentEndpoint = endpoint;
                API_CONFIG.isOnline = true;
                statusDot.className = 'status-dot online';
                statusText.textContent = 'API Online';
                elements.apiEndpoint.textContent = endpoint;
                return;
            }
        } catch (error) {
            console.log(`Endpoint ${endpoint} not available`);
        }
    }

    // No endpoint available - use local mode
    API_CONFIG.isOnline = false;
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Local Mode (Demo)';
    elements.apiEndpoint.textContent = 'Using local classification';
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Check API status
    checkApiStatus();

    // Classify button
    elements.classifyBtn.addEventListener('click', async () => {
        const text = elements.questionInput.value.trim();
        if (!text) return;

        const btnText = elements.classifyBtn.querySelector('.btn-text');
        const btnLoading = elements.classifyBtn.querySelector('.btn-loading');

        btnText.style.display = 'none';
        btnLoading.style.display = 'inline-flex';
        elements.classifyBtn.disabled = true;

        try {
            const model = elements.modelSelect.value;
            const result = await classifyQuestion(text, model);
            displayResults(result);
        } catch (error) {
            console.error('Classification error:', error);
            alert('Classification failed. Please try again.');
        } finally {
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
            elements.classifyBtn.disabled = false;
        }
    });

    // Clear button
    elements.clearBtn.addEventListener('click', () => {
        elements.questionInput.value = '';
        elements.resultsContent.classList.add('hidden');
        elements.resultsPlaceholder.classList.remove('hidden');
    });

    // Example tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.example-content').forEach(content => {
                content.classList.add('hidden');
            });
            document.getElementById(`tab-${tabId}`).classList.remove('hidden');
        });
    });

    // Example items - click to classify
    document.querySelectorAll('.example-item').forEach(item => {
        item.addEventListener('click', () => {
            const question = item.dataset.question;
            elements.questionInput.value = question;
            elements.classifyBtn.click();
        });
    });

    // Scenario questions
    document.querySelectorAll('.scenario-q').forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.dataset.question;
            elements.questionInput.value = question;
            elements.classifyBtn.click();

            // Scroll to results
            elements.resultsContent.scrollIntoView({ behavior: 'smooth' });
        });
    });

    // Enter key to classify
    elements.questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            elements.classifyBtn.click();
        }
    });

    // Model change updates voting visibility
    elements.modelSelect.addEventListener('change', () => {
        if (elements.modelSelect.value === 'classic') {
            elements.votingSection.style.display = 'none';
        }
    });
});
