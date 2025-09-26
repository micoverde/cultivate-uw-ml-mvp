// DEMO 1: Child Scenario OEQ/CEQ Classifier
// Interactive demo with real ML classification

class ChildScenarioDemo {
    constructor() {
        this.scenarios = [];
        this.currentScenarioIndex = 0;
        this.completedScenarios = 0;
        this.oeqCount = 0;
        this.ceqCount = 0;
        this.apiBaseUrl = 'http://localhost:8001'; // ML API endpoint

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
            const classification = await this.classifyWithML(responseText);

            // Display results
            this.displayResults(classification, responseText);

        } catch (error) {
            console.error('Classification error:', error);
            // Fallback to rule-based classification
            const fallbackClassification = this.classifyWithRules(responseText);
            this.displayResults(fallbackClassification, responseText);
        } finally {
            document.getElementById('loadingSpinner').style.display = 'none';
            document.getElementById('nextBtn').disabled = false;
        }
    }

    async classifyWithML(text) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/classify_response`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    scenario_id: this.scenarios[this.currentScenarioIndex].id
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.warn('ML API unavailable, using fallback classification');
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
        const oeqPercent = Math.round(classification.oeq_probability * 100);
        const ceqPercent = Math.round(classification.ceq_probability * 100);

        // Update probability bars
        document.getElementById('oeqFill').style.width = `${oeqPercent}%`;
        document.getElementById('ceqFill').style.width = `${ceqPercent}%`;
        document.getElementById('oeqPercentage').textContent = `${oeqPercent}%`;
        document.getElementById('ceqPercentage').textContent = `${ceqPercent}%`;

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
}

// Initialize demo when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChildScenarioDemo();
});