/**
 * learning-spiral.js - Creative Learning Spiral Implementation
 *
 * Based on Mitch Resnick's Creative Learning Spiral framework from MIT Media Lab's
 * Lifelong Kindergarten group. This module implements the pedagogical principles
 * that guide effective learning through iterative cycles of imagination, creation,
 * play, sharing, and reflection.
 *
 * References:
 * - Resnick, M. (2017). Lifelong Kindergarten: Cultivating Creativity
 * - Brennan, K. & Resnick, M. (2012). New frameworks for computational thinking
 *
 * @author Cultivate Learning
 * @version 2.0.0
 */

(function(global) {
    'use strict';

    // ==========================================================================
    // Configuration Constants
    // ==========================================================================

    const STORAGE_KEYS = {
        LEARNER_ID: 'cultivate_learner_id',
        SPIRAL_PROGRESS: 'cultivate_spiral_progress',
        UMC_PROGRESSION: 'cultivate_umc_progression',
        REFLECTIONS: 'cultivate_reflections',
        TINKER_HISTORY: 'cultivate_tinker_history',
        FOUR_PS_DATA: 'cultivate_four_ps',
        SESSION_DATA: 'cultivate_session_data'
    };

    const SPIRAL_STAGES = ['imagine', 'create', 'play', 'share', 'reflect'];
    const UMC_LEVELS = ['use', 'modify', 'create'];

    /**
     * Reflection prompts based on Karen Brennan's framework.
     */
    const REFLECTION_PROMPTS = {
        imagine: [
            "What patterns did you notice in the data?",
            "How does this connect to your understanding of education?",
            "What possibilities are you excited to explore?"
        ],
        create: [
            "What was the most challenging part of building this?",
            "How did you decide which approach to take?",
            "What would you do differently if you started over?"
        ],
        play: [
            "What surprised you when you experimented?",
            "What 'happy accidents' led to interesting discoveries?",
            "How did playful exploration change your understanding?"
        ],
        share: [
            "How did sharing your work change your perspective?",
            "What feedback was most valuable to you?",
            "How might others use or build upon your ideas?"
        ],
        reflect: [
            "What would you want to explore next?",
            "How has your understanding evolved through this experience?",
            "How might you apply these insights in practice?"
        ]
    };

    // ==========================================================================
    // Utility Functions
    // ==========================================================================

    function generateLearnerId() {
        const timestamp = Date.now().toString(36);
        const randomPart = Math.random().toString(36).substring(2, 8);
        return `learner_${timestamp}_${randomPart}`;
    }

    function getStorageItem(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            return defaultValue;
        }
    }

    function setStorageItem(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (e) {
            return false;
        }
    }

    function formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        if (hours > 0) return `${hours}h ${minutes % 60}m`;
        if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
        return `${seconds}s`;
    }

    // ==========================================================================
    // CreativeLearningSpiral Class
    // ==========================================================================

    /**
     * Core navigation component for the 5-stage learning spiral.
     * Pedagogical Reasoning: The spiral represents iterative learning cycles.
     */
    class CreativeLearningSpiral {
        constructor(options = {}) {
            this.container = options.container || null;
            this.onStageChange = options.onStageChange || (() => {});
            this.stages = SPIRAL_STAGES;
            this.currentStage = 0;
            this.cycleCount = 0;
            this._loadProgress();
            this._initializeSessionTracking();
        }

        _loadProgress() {
            const saved = getStorageItem(STORAGE_KEYS.SPIRAL_PROGRESS, {
                currentStage: 0, cycleCount: 0, stageVisits: {}, stageTime: {}
            });
            this.currentStage = saved.currentStage;
            this.cycleCount = saved.cycleCount;
            this.stageVisits = saved.stageVisits || {};
            this.stageTime = saved.stageTime || {};
            this.stages.forEach(stage => {
                if (!this.stageVisits[stage]) this.stageVisits[stage] = 0;
                if (!this.stageTime[stage]) this.stageTime[stage] = 0;
            });
        }

        _saveProgress() {
            setStorageItem(STORAGE_KEYS.SPIRAL_PROGRESS, {
                currentStage: this.currentStage,
                cycleCount: this.cycleCount,
                stageVisits: this.stageVisits,
                stageTime: this.stageTime
            });
        }

        _initializeSessionTracking() {
            this._stageStartTime = Date.now();
            window.addEventListener('beforeunload', () => {
                this._recordStageTime();
                this._saveProgress();
            });
        }

        _recordStageTime() {
            if (this._stageStartTime) {
                const elapsed = Date.now() - this._stageStartTime;
                const currentStageName = this.stages[this.currentStage];
                this.stageTime[currentStageName] += elapsed;
            }
        }

        getCurrentStageName() {
            return this.stages[this.currentStage];
        }

        getStageMetadata(stageName) {
            const metadata = {
                imagine: { icon: 'lightbulb', color: '#f59e0b', title: 'Imagine', description: 'Wonder about possibilities.' },
                create: { icon: 'hammer', color: '#10b981', title: 'Create', description: 'Build something based on your ideas.' },
                play: { icon: 'gamepad', color: '#6366f1', title: 'Play', description: 'Experiment freely.' },
                share: { icon: 'users', color: '#ec4899', title: 'Share', description: 'Show others what you\'ve made.' },
                reflect: { icon: 'brain', color: '#8b5cf6', title: 'Reflect', description: 'Think about what you learned.' }
            };
            return metadata[stageName] || metadata.imagine;
        }

        goToStage(stageIdentifier) {
            let targetIndex = typeof stageIdentifier === 'string'
                ? this.stages.indexOf(stageIdentifier)
                : stageIdentifier;
            if (targetIndex < 0 || targetIndex >= this.stages.length) return false;

            this._recordStageTime();
            const previousStage = this.currentStage;
            this.currentStage = targetIndex;
            const stageName = this.stages[targetIndex];
            this.stageVisits[stageName]++;
            this._stageStartTime = Date.now();

            if (targetIndex === 0 && previousStage === this.stages.length - 1) {
                this.cycleCount++;
            }

            this._saveProgress();
            this.onStageChange({
                previousStage: this.stages[previousStage],
                currentStage: stageName,
                cycleCount: this.cycleCount,
                metadata: this.getStageMetadata(stageName)
            });
            return true;
        }

        nextStage() {
            return this.goToStage((this.currentStage + 1) % this.stages.length);
        }

        previousStage() {
            return this.goToStage(this.currentStage === 0 ? this.stages.length - 1 : this.currentStage - 1);
        }

        getProgress() {
            const totalTime = Object.values(this.stageTime).reduce((a, b) => a + b, 0);
            return {
                currentStage: this.getCurrentStageName(),
                cycleCount: this.cycleCount,
                stageVisits: { ...this.stageVisits },
                stageTime: { ...this.stageTime },
                totalTime,
                formattedTotalTime: formatDuration(totalTime)
            };
        }

        reset() {
            this.currentStage = 0;
            this.cycleCount = 0;
            this.stageVisits = {};
            this.stageTime = {};
            this.stages.forEach(stage => {
                this.stageVisits[stage] = 0;
                this.stageTime[stage] = 0;
            });
            this._stageStartTime = Date.now();
            this._saveProgress();
        }
    }

    // ==========================================================================
    // UMCProgression Class
    // ==========================================================================

    /**
     * Use-Modify-Create Progression Tracker.
     * Pedagogical Reasoning: Scaffolded entry reduces cognitive load.
     */
    class UMCProgression {
        constructor(options = {}) {
            this.onProgressChange = options.onProgressChange || (() => {});
            this.onBadgeEarned = options.onBadgeEarned || (() => {});
            this.levels = UMC_LEVELS;
            this._loadProgress();
        }

        _loadProgress() {
            const saved = getStorageItem(STORAGE_KEYS.UMC_PROGRESSION, {
                activities: {}, badges: [], currentLevel: 'use', unlockedContent: []
            });
            this.activities = saved.activities || {};
            this.badges = saved.badges || [];
            this.currentLevel = saved.currentLevel || 'use';
            this.unlockedContent = saved.unlockedContent || [];
        }

        _saveProgress() {
            setStorageItem(STORAGE_KEYS.UMC_PROGRESSION, {
                activities: this.activities,
                badges: this.badges,
                currentLevel: this.currentLevel,
                unlockedContent: this.unlockedContent
            });
        }

        getLevelMetadata(level) {
            const metadata = {
                use: { title: 'Use', description: 'Explore existing examples', color: '#3b82f6', requiredActivities: 3, badgeName: 'Explorer' },
                modify: { title: 'Modify', description: 'Adapt and customize', color: '#f59e0b', requiredActivities: 3, badgeName: 'Tinkerer' },
                create: { title: 'Create', description: 'Build original work', color: '#10b981', requiredActivities: 2, badgeName: 'Creator' }
            };
            return metadata[level] || metadata.use;
        }

        completeActivity(activityId, level, metadata = {}) {
            if (!this.levels.includes(level)) return { success: false };
            if (this.activities[activityId]) return { success: true, alreadyCompleted: true };

            this.activities[activityId] = { level, completedAt: Date.now(), ...metadata };
            const result = { success: true, activityId, level, badgeEarned: null, levelAdvanced: false };

            const levelActivities = Object.values(this.activities).filter(a => a.level === level);
            const levelMeta = this.getLevelMetadata(level);

            if (levelActivities.length >= levelMeta.requiredActivities) {
                const badgeId = `${level}_master`;
                if (!this.badges.includes(badgeId)) {
                    this.badges.push(badgeId);
                    result.badgeEarned = { id: badgeId, name: levelMeta.badgeName, level, earnedAt: Date.now() };
                    this.onBadgeEarned(result.badgeEarned);
                }

                const currentLevelIndex = this.levels.indexOf(this.currentLevel);
                const activityLevelIndex = this.levels.indexOf(level);
                if (activityLevelIndex >= currentLevelIndex && currentLevelIndex < this.levels.length - 1) {
                    this.currentLevel = this.levels[currentLevelIndex + 1];
                    result.levelAdvanced = true;
                }
            }

            this._saveProgress();
            this.onProgressChange(this.getProgress());
            return result;
        }

        isContentUnlocked(contentId) {
            if (this.currentLevel === 'use') {
                return ['basic-exploration', 'guided-tour', 'intro-examples'].includes(contentId);
            }
            return this.unlockedContent.includes(contentId);
        }

        getProgress() {
            const levelProgress = {};
            this.levels.forEach(level => {
                const levelActivities = Object.values(this.activities).filter(a => a.level === level);
                const meta = this.getLevelMetadata(level);
                levelProgress[level] = {
                    completed: levelActivities.length,
                    required: meta.requiredActivities,
                    percentage: Math.min(100, Math.round((levelActivities.length / meta.requiredActivities) * 100)),
                    isComplete: levelActivities.length >= meta.requiredActivities
                };
            });
            return {
                currentLevel: this.currentLevel,
                levelProgress,
                totalActivities: Object.keys(this.activities).length,
                badges: [...this.badges],
                unlockedContent: [...this.unlockedContent]
            };
        }

        reset() {
            this.activities = {};
            this.badges = [];
            this.currentLevel = 'use';
            this.unlockedContent = [];
            this._saveProgress();
        }
    }

    // ==========================================================================
    // ReflectionJournal Class
    // ==========================================================================

    /**
     * Structured reflection component based on Karen Brennan's framework.
     */
    class ReflectionJournal {
        constructor() {
            this.prompts = REFLECTION_PROMPTS;
            this._loadReflections();
        }

        _loadReflections() {
            const saved = getStorageItem(STORAGE_KEYS.REFLECTIONS, { entries: [], promptResponses: {} });
            this.entries = saved.entries || [];
            this.promptResponses = saved.promptResponses || {};
        }

        _saveReflections() {
            setStorageItem(STORAGE_KEYS.REFLECTIONS, { entries: this.entries, promptResponses: this.promptResponses });
        }

        getPromptsForStage(stage) {
            return this.prompts[stage] || this.prompts.reflect;
        }

        getRandomPrompt(stage) {
            const stagePrompts = this.getPromptsForStage(stage);
            return stagePrompts[Math.floor(Math.random() * stagePrompts.length)];
        }

        addEntry(entry) {
            const newEntry = {
                id: `reflection_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                prompt: entry.prompt,
                response: entry.response,
                stage: entry.stage || 'reflect',
                module: entry.module || null,
                createdAt: Date.now(),
                wordCount: entry.response.split(/\s+/).filter(w => w.length > 0).length
            };
            this.entries.push(newEntry);
            if (!this.promptResponses[entry.prompt]) this.promptResponses[entry.prompt] = [];
            this.promptResponses[entry.prompt].push(newEntry.id);
            this._saveReflections();
            return newEntry;
        }

        updateEntry(entryId, newResponse) {
            const entry = this.entries.find(e => e.id === entryId);
            if (!entry) return false;
            entry.response = newResponse;
            entry.updatedAt = Date.now();
            entry.wordCount = newResponse.split(/\s+/).filter(w => w.length > 0).length;
            this._saveReflections();
            return true;
        }

        deleteEntry(entryId) {
            const index = this.entries.findIndex(e => e.id === entryId);
            if (index === -1) return false;
            this.entries.splice(index, 1);
            this._saveReflections();
            return true;
        }

        getEntries(filters = {}) {
            let results = [...this.entries];
            if (filters.stage) results = results.filter(e => e.stage === filters.stage);
            if (filters.module) results = results.filter(e => e.module === filters.module);
            return results.sort((a, b) => b.createdAt - a.createdAt);
        }

        getStats() {
            const totalWords = this.entries.reduce((sum, e) => sum + e.wordCount, 0);
            return {
                totalEntries: this.entries.length,
                totalWords,
                averageWords: this.entries.length > 0 ? Math.round(totalWords / this.entries.length) : 0,
                uniquePrompts: Object.keys(this.promptResponses).length
            };
        }

        exportSummary(format = 'text') {
            const stats = this.getStats();
            const entries = this.getEntries();
            const learnerId = getStorageItem(STORAGE_KEYS.LEARNER_ID) || 'Unknown';

            if (format === 'html') {
                return `<!DOCTYPE html><html><head><title>Learning Journey</title>
                <style>body{font-family:sans-serif;max-width:800px;margin:0 auto;padding:2rem;}
                .entry{border-left:3px solid #4f46e5;padding-left:1rem;margin-bottom:1.5rem;}</style></head>
                <body><h1>Learning Journey Summary</h1><p>Learner: ${learnerId}</p>
                <p>Entries: ${stats.totalEntries} | Words: ${stats.totalWords}</p>
                ${entries.map(e => `<div class="entry"><h4>${e.prompt}</h4><p>${e.response}</p>
                <small>${new Date(e.createdAt).toLocaleDateString()}</small></div>`).join('')}
                </body></html>`;
            }

            return `LEARNING JOURNEY SUMMARY\nLearner: ${learnerId}\nEntries: ${stats.totalEntries}\nWords: ${stats.totalWords}\n\n` +
                entries.map(e => `Prompt: ${e.prompt}\n${e.response}\n${new Date(e.createdAt).toLocaleDateString()}\n---`).join('\n');
        }

        reset() {
            this.entries = [];
            this.promptResponses = {};
            this._saveReflections();
        }
    }

    // ==========================================================================
    // TinkerZone Class
    // ==========================================================================

    /**
     * Interactive parameter adjustment environment.
     * Pedagogical Reasoning: Safe experimentation encourages bold exploration.
     */
    class TinkerZone {
        constructor(options = {}) {
            this.parameters = options.parameters || {};
            this.onParameterChange = options.onParameterChange || (() => {});
            this.feedbackGenerator = options.feedbackGenerator || (() => 'Experiment with different values!');
            this._loadHistory();
            this._initializeState();
        }

        _loadHistory() {
            const saved = getStorageItem(STORAGE_KEYS.TINKER_HISTORY, {
                history: [], savedConfigs: {}, explorationCount: 0
            });
            this.history = saved.history || [];
            this.savedConfigs = saved.savedConfigs || {};
            this.explorationCount = saved.explorationCount || 0;
        }

        _saveHistory() {
            setStorageItem(STORAGE_KEYS.TINKER_HISTORY, {
                history: this.history.slice(-100),
                savedConfigs: this.savedConfigs,
                explorationCount: this.explorationCount
            });
        }

        _initializeState() {
            this.currentState = {};
            this.defaultState = {};
            Object.entries(this.parameters).forEach(([key, param]) => {
                this.currentState[key] = param.default;
                this.defaultState[key] = param.default;
            });
            this._pushState('initial');
        }

        defineParameter(name, config) {
            this.parameters[name] = {
                type: config.type || 'number',
                default: config.default,
                min: config.min,
                max: config.max,
                step: config.step || 1,
                options: config.options,
                label: config.label || name,
                description: config.description || '',
                explorationPrompt: config.explorationPrompt || `What happens if you change ${name}?`
            };
            this.currentState[name] = config.default;
            this.defaultState[name] = config.default;
        }

        updateParameter(name, value) {
            if (!this.parameters[name]) return { success: false };

            const param = this.parameters[name];
            const oldValue = this.currentState[name];
            let validatedValue = value;

            if (param.type === 'number') {
                validatedValue = Math.max(param.min, Math.min(param.max, Number(value)));
            } else if (param.type === 'select' && !param.options.includes(value)) {
                validatedValue = param.default;
            } else if (param.type === 'boolean') {
                validatedValue = Boolean(value);
            }

            this.currentState[name] = validatedValue;
            this._pushState(`changed_${name}`);
            this.explorationCount++;

            const feedback = this.feedbackGenerator(name, validatedValue, oldValue, this.currentState);
            this._saveHistory();

            this.onParameterChange({
                parameter: name, oldValue, newValue: validatedValue,
                currentState: { ...this.currentState }, feedback
            });

            return { success: true, parameter: name, oldValue, newValue: validatedValue, feedback };
        }

        _pushState(action) {
            this.history.push({ state: { ...this.currentState }, action, timestamp: Date.now() });
        }

        undo() {
            if (this.history.length <= 1) return false;
            this.history.pop();
            const previousState = this.history[this.history.length - 1];
            this.currentState = { ...previousState.state };
            this._saveHistory();
            this.onParameterChange({ action: 'undo', currentState: { ...this.currentState }, feedback: 'Reverted to previous state' });
            return true;
        }

        reset() {
            const oldState = { ...this.currentState };
            this.currentState = { ...this.defaultState };
            this._pushState('reset');
            this._saveHistory();
            this.onParameterChange({ action: 'reset', oldState, currentState: { ...this.currentState }, feedback: 'All parameters reset' });
            return { success: true, oldState, newState: { ...this.currentState } };
        }

        saveConfiguration(name) {
            this.savedConfigs[name] = { state: { ...this.currentState }, savedAt: Date.now() };
            this._saveHistory();
            return true;
        }

        loadConfiguration(name) {
            if (!this.savedConfigs[name]) return false;
            this.currentState = { ...this.savedConfigs[name].state };
            this._pushState(`loaded_${name}`);
            this._saveHistory();
            this.onParameterChange({ action: 'load', configName: name, currentState: { ...this.currentState } });
            return true;
        }

        getExplorationPrompts() {
            return Object.values(this.parameters).map(p => p.explorationPrompt);
        }

        getRandomExplorationPrompt() {
            const prompts = this.getExplorationPrompts();
            return prompts[Math.floor(Math.random() * prompts.length)] || 'What happens if...?';
        }

        getStats() {
            return {
                explorationCount: this.explorationCount,
                historyLength: this.history.length,
                savedConfigCount: Object.keys(this.savedConfigs).length,
                parameterCount: Object.keys(this.parameters).length,
                currentState: { ...this.currentState }
            };
        }
    }

    // ==========================================================================
    // FourPsIntegration Class
    // ==========================================================================

    /**
     * Projects, Passion, Peers, Play tracking.
     * Pedagogical Reasoning: Learning emerges from the intersection of all 4 P's.
     */
    class FourPsIntegration {
        constructor() {
            this._loadData();
        }

        _loadData() {
            const saved = getStorageItem(STORAGE_KEYS.FOUR_PS_DATA, {
                projects: [], passionAreas: [], peerInsights: [],
                playMetrics: { explorationBadges: [], curiosityPoints: 0, experimentsRun: 0 }
            });
            this.projects = saved.projects || [];
            this.passionAreas = saved.passionAreas || [];
            this.peerInsights = saved.peerInsights || [];
            this.playMetrics = saved.playMetrics || { explorationBadges: [], curiosityPoints: 0, experimentsRun: 0 };
        }

        _saveData() {
            setStorageItem(STORAGE_KEYS.FOUR_PS_DATA, {
                projects: this.projects, passionAreas: this.passionAreas,
                peerInsights: this.peerInsights, playMetrics: this.playMetrics
            });
        }

        // Projects
        completeProject(projectId, metadata = {}) {
            const existing = this.projects.find(p => p.id === projectId);
            if (existing) {
                existing.completedAt = Date.now();
                existing.completionCount = (existing.completionCount || 0) + 1;
            } else {
                this.projects.push({
                    id: projectId, name: metadata.name || projectId,
                    module: metadata.module || null, completedAt: Date.now(), completionCount: 1
                });
            }
            this._saveData();
            return this.projects.find(p => p.id === projectId);
        }

        getCompletedProjects() {
            return this.projects.filter(p => p.completedAt);
        }

        // Passion
        setPassionAreas(areas) {
            const validAreas = ['technical', 'impact', 'pedagogy'];
            this.passionAreas = areas.filter(a => validAreas.includes(a));
            this._saveData();
        }

        getPassionMetadata(area) {
            const metadata = {
                technical: { title: 'Technical Deep Dive', color: '#3b82f6', recommendedModules: ['module3-ml-models', 'module2-feature-extraction'] },
                impact: { title: 'Educational Impact', color: '#ec4899', recommendedModules: ['module4-live-demo', 'module1-data-collection'] },
                pedagogy: { title: 'Pedagogical Approach', color: '#10b981', recommendedModules: ['module1-data-collection', 'module4-live-demo'] }
            };
            return metadata[area] || metadata.technical;
        }

        getRecommendations() {
            const recommendations = [];
            this.passionAreas.forEach(area => {
                const meta = this.getPassionMetadata(area);
                meta.recommendedModules.forEach(module => {
                    if (!recommendations.find(r => r.module === module)) {
                        recommendations.push({ module, reason: `Matches your interest in ${meta.title}`, passionArea: area });
                    }
                });
            });
            return recommendations;
        }

        // Peers
        addPeerInsight(insight) {
            this.peerInsights.push({
                id: `insight_${Date.now()}`, content: insight.content,
                type: insight.type || 'observation', module: insight.module || null, createdAt: Date.now()
            });
            if (this.peerInsights.length > 50) this.peerInsights = this.peerInsights.slice(-50);
            this._saveData();
        }

        getPeerInsights(filters = {}) {
            let results = [...this.peerInsights];
            if (filters.module) results = results.filter(i => i.module === filters.module);
            if (filters.type) results = results.filter(i => i.type === filters.type);
            return results.sort((a, b) => b.createdAt - a.createdAt);
        }

        seedSamplePeerInsights() {
            const samples = [
                { content: '73% of learners found feature extraction most valuable', type: 'statistic', module: 'module2' },
                { content: 'Combining multiple features improves model accuracy significantly', type: 'discovery', module: 'module3' },
                { content: 'Interactive parameter tuning clarified hyperparameter effects', type: 'feedback', module: 'module3' }
            ];
            samples.forEach(s => this.addPeerInsight(s));
        }

        // Play
        awardExplorationBadge(badgeId, metadata = {}) {
            if (this.playMetrics.explorationBadges.find(b => b.id === badgeId)) return false;
            this.playMetrics.explorationBadges.push({
                id: badgeId, name: metadata.name || badgeId,
                description: metadata.description || '', earnedAt: Date.now()
            });
            this._saveData();
            return true;
        }

        addCuriosityPoints(points, reason = '') {
            this.playMetrics.curiosityPoints += points;
            this.playMetrics.experimentsRun++;
            this._saveData();
        }

        getAvailableBadges() {
            return [
                { id: 'first_tinker', name: 'First Tinker', description: 'Made first parameter adjustment', points: 10 },
                { id: 'explorer', name: 'Explorer', description: 'Visited all spiral stages', points: 25 },
                { id: 'reflector', name: 'Deep Thinker', description: 'Written 5+ reflections', points: 30 },
                { id: 'experimenter', name: 'Mad Scientist', description: 'Ran 20+ experiments', points: 50 },
                { id: 'cycle_master', name: 'Cycle Master', description: 'Completed 3 full cycles', points: 100 }
            ];
        }

        getSummary() {
            return {
                projects: { completed: this.getCompletedProjects().length, total: this.projects.length },
                passion: { areas: this.passionAreas, recommendations: this.getRecommendations().slice(0, 3) },
                peers: { insightCount: this.peerInsights.length, recent: this.getPeerInsights().slice(0, 3) },
                play: {
                    curiosityPoints: this.playMetrics.curiosityPoints,
                    experimentsRun: this.playMetrics.experimentsRun,
                    badges: this.playMetrics.explorationBadges,
                    badgeCount: this.playMetrics.explorationBadges.length
                }
            };
        }

        reset() {
            this.projects = [];
            this.passionAreas = [];
            this.peerInsights = [];
            this.playMetrics = { explorationBadges: [], curiosityPoints: 0, experimentsRun: 0 };
            this._saveData();
        }
    }

    // ==========================================================================
    // LearningProgressManager - Main Integration Class
    // ==========================================================================

    class LearningProgressManager {
        constructor() {
            this._initializeLearnerId();
            this._initializeSession();

            this.spiral = new CreativeLearningSpiral({
                onStageChange: (data) => this._handleStageChange(data)
            });
            this.umc = new UMCProgression({
                onProgressChange: (data) => this._handleUMCChange(data),
                onBadgeEarned: (badge) => this._handleBadgeEarned(badge)
            });
            this.journal = new ReflectionJournal();
            this.tinker = new TinkerZone({
                onParameterChange: (data) => this._handleTinkerChange(data)
            });
            this.fourPs = new FourPsIntegration();
        }

        _initializeLearnerId() {
            let learnerId = getStorageItem(STORAGE_KEYS.LEARNER_ID);
            if (!learnerId) {
                learnerId = generateLearnerId();
                setStorageItem(STORAGE_KEYS.LEARNER_ID, learnerId);
            }
            this.learnerId = learnerId;
        }

        _initializeSession() {
            const session = getStorageItem(STORAGE_KEYS.SESSION_DATA, {
                sessions: [], currentSessionStart: null, totalSessions: 0
            });
            session.currentSessionStart = Date.now();
            session.totalSessions++;
            session.sessions.push({ start: Date.now(), end: null });
            if (session.sessions.length > 100) session.sessions = session.sessions.slice(-100);
            this.sessionData = session;
            setStorageItem(STORAGE_KEYS.SESSION_DATA, session);

            window.addEventListener('beforeunload', () => {
                const currentSession = this.sessionData.sessions[this.sessionData.sessions.length - 1];
                if (currentSession) {
                    currentSession.end = Date.now();
                    setStorageItem(STORAGE_KEYS.SESSION_DATA, this.sessionData);
                }
            });
        }

        _handleStageChange(data) {
            this.fourPs.addCuriosityPoints(5, `Explored ${data.currentStage} stage`);
            const progress = this.spiral.getProgress();
            const allStagesVisited = Object.values(progress.stageVisits).every(v => v > 0);
            if (allStagesVisited) {
                this.fourPs.awardExplorationBadge('explorer', { name: 'Explorer', description: 'Visited all spiral stages' });
            }
            if (progress.cycleCount >= 3) {
                this.fourPs.awardExplorationBadge('cycle_master', { name: 'Cycle Master', description: 'Completed 3 full learning cycles' });
            }
        }

        _handleUMCChange(data) {
            this.fourPs.addCuriosityPoints(10, 'Completed UMC activity');
        }

        _handleBadgeEarned(badge) {
            console.log('Badge earned:', badge);
        }

        _handleTinkerChange(data) {
            const stats = this.tinker.getStats();
            if (stats.explorationCount === 1) {
                this.fourPs.awardExplorationBadge('first_tinker', { name: 'First Tinker', description: 'Made first parameter adjustment' });
            }
            if (stats.explorationCount >= 20) {
                this.fourPs.awardExplorationBadge('experimenter', { name: 'Mad Scientist', description: 'Ran 20+ experiments' });
            }
            this.fourPs.addCuriosityPoints(2, 'Tinker experiment');
        }

        getLearnerId() { return this.learnerId; }

        getCompleteSummary() {
            return {
                learnerId: this.learnerId,
                sessionInfo: { currentSession: this.sessionData.currentSessionStart, totalSessions: this.sessionData.totalSessions },
                spiral: this.spiral.getProgress(),
                umc: this.umc.getProgress(),
                journal: this.journal.getStats(),
                tinker: this.tinker.getStats(),
                fourPs: this.fourPs.getSummary()
            };
        }

        exportAllData() {
            return {
                learnerId: this.learnerId, exportedAt: Date.now(),
                spiral: this.spiral.getProgress(), umc: this.umc.getProgress(),
                reflections: this.journal.getEntries(), journalStats: this.journal.getStats(),
                tinkerStats: this.tinker.getStats(), fourPs: this.fourPs.getSummary(),
                sessions: this.sessionData
            };
        }

        resetAllProgress() {
            this.spiral.reset();
            this.umc.reset();
            this.journal.reset();
            this.tinker.reset();
            this.fourPs.reset();
            this.learnerId = generateLearnerId();
            setStorageItem(STORAGE_KEYS.LEARNER_ID, this.learnerId);
        }
    }

    // ==========================================================================
    // Public API
    // ==========================================================================

    function initializeLearningSpiral() {
        return new LearningProgressManager();
    }

    global.LearningSpiral = {
        initialize: initializeLearningSpiral,
        CreativeLearningSpiral,
        UMCProgression,
        ReflectionJournal,
        TinkerZone,
        FourPsIntegration,
        LearningProgressManager,
        utils: { generateLearnerId, formatDuration, getStorageItem, setStorageItem },
        SPIRAL_STAGES,
        UMC_LEVELS,
        REFLECTION_PROMPTS,
        STORAGE_KEYS
    };

    // Also export for ES modules compatibility
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = global.LearningSpiral;
    }

})(typeof window !== 'undefined' ? window : this);
