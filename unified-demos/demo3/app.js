/**
 * Demo 3: Video Question Analysis Application
 * Production-ready orchestration layer for all Demo 3 components
 *
 * Principal SDE 5 implementation - Warren's Cultivate Learning ML MVP
 *
 * Features:
 * - Video catalog browsing with ground truth annotations
 * - Real-time question classification (Classic ML + Ensemble models)
 * - Analytics dashboard with classification metrics
 * - Human feedback collection for model improvement
 *
 * Architecture:
 * - ES6 modules for clean component separation
 * - Event-driven communication between components
 * - LocalStorage for user preferences and ratings persistence
 * - Azure Blob Storage fallback for video files
 *
 * Usage in HTML:
 * <script type="module" src="./app.js"></script>
 * <!-- Or load components separately and app.js will detect them from window -->
 */

// ============================================================================
// COMPONENT LOADING STRATEGY
// ============================================================================
// Components can be loaded either via:
// 1. ES6 dynamic imports (preferred for module bundlers)
// 2. Script tags that attach to window (for direct browser usage)
//
// The app will detect which method is available and use it appropriately.

let VideoSelector, QuestionPanel, AnalyticsDashboard;

/**
 * Dynamically load components with fallback to window attachments
 * @returns {Promise<void>}
 */
async function loadComponents() {
    try {
        // Try ES6 dynamic imports first
        const [vsMod, qpMod, adMod] = await Promise.allSettled([
            import('./components/video-selector.js'),
            import('./components/question-panel.js'),
            import('./components/analytics.js')
        ]);

        VideoSelector = vsMod.status === 'fulfilled'
            ? (vsMod.value.VideoSelector || vsMod.value.default)
            : window.VideoSelector;

        QuestionPanel = qpMod.status === 'fulfilled'
            ? (qpMod.value.QuestionPanel || qpMod.value.default)
            : window.QuestionPanel;

        AnalyticsDashboard = adMod.status === 'fulfilled'
            ? (adMod.value.AnalyticsDashboard || adMod.value.default)
            : window.AnalyticsDashboard;

        console.log('[Demo3] Components loaded via dynamic imports');

    } catch (error) {
        // Fallback to window-attached components
        console.log('[Demo3] Falling back to window-attached components');
        VideoSelector = window.VideoSelector;
        QuestionPanel = window.QuestionPanel;
        AnalyticsDashboard = window.AnalyticsDashboard;
    }
}

// ============================================================================
// GLOBAL STATE MANAGEMENT
// ============================================================================

/**
 * Centralized application state
 * Manages all shared state across components
 */
const AppState = {
    // Currently selected video object from catalog
    currentVideo: null,

    // Selected classification model: 'classic' | 'ensemble'
    currentModel: 'classic',

    // Classification results cache: Map<video_id, Array<QuestionClassification>>
    classifications: new Map(),

    // Full video catalog loaded from JSON
    catalog: null,

    // Loading states for async operations
    loading: {
        catalog: false,
        video: false,
        classification: false,
        analytics: false
    },

    // User ratings stored in memory (persisted to localStorage)
    ratings: new Map(),

    // Error state for user-friendly messages
    lastError: null
};

// ============================================================================
// CONFIGURATION
// ============================================================================

const Config = {
    // API endpoints based on environment
    get apiBaseUrl() {
        const hostname = window.location.hostname;
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:5001';
        }
        return 'https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io';
    },

    // Azure Blob Storage for video fallback
    azureBlobBase: 'https://cultivatemlstorage.blob.core.windows.net/videos',

    // LocalStorage keys
    storageKeys: {
        selectedModel: 'demo3_selectedModel',
        lastVideo: 'demo3_lastVideo',
        ratings: 'demo3_ratings',
        preferences: 'demo3_preferences'
    },

    // Catalog data path
    catalogPath: './data/video_catalog.json'
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Parse timestamp string to seconds
 * @param {string} timestamp - Format "M:SS" or "MM:SS"
 * @returns {number} Total seconds
 */
function parseTimestamp(timestamp) {
    if (typeof timestamp === 'number') return timestamp;
    if (!timestamp || typeof timestamp !== 'string') return 0;

    const parts = timestamp.split(':').map(Number);
    if (parts.length === 2) {
        return parts[0] * 60 + parts[1];
    } else if (parts.length === 3) {
        return parts[0] * 3600 + parts[1] * 60 + parts[2];
    }
    return 0;
}

/**
 * Format seconds as timestamp string
 * @param {number} seconds - Total seconds
 * @returns {string} Formatted as "M:SS"
 */
function formatTimestamp(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Show loading indicator
 * @param {string} type - Loading type from AppState.loading
 * @param {boolean} show - Whether to show or hide
 */
function setLoading(type, show) {
    AppState.loading[type] = show;

    const indicator = document.getElementById('loadingIndicator');
    const message = document.getElementById('loadingMessage');

    if (indicator) {
        const anyLoading = Object.values(AppState.loading).some(v => v);
        indicator.style.display = anyLoading ? 'flex' : 'none';

        if (message && show) {
            const messages = {
                catalog: 'Loading video catalog...',
                video: 'Loading video...',
                classification: 'Classifying questions...',
                analytics: 'Calculating analytics...'
            };
            message.textContent = messages[type] || 'Loading...';
        }
    }
}

/**
 * Display user-friendly error message
 * @param {string} message - Error message to display
 * @param {Error} error - Optional error object for logging
 */
function showError(message, error = null) {
    AppState.lastError = { message, error, timestamp: new Date().toISOString() };

    console.error(`[Demo3 Error] ${message}`, error);

    const errorContainer = document.getElementById('errorContainer');
    if (errorContainer) {
        errorContainer.innerHTML = `
            <div class="error-message">
                <span class="error-icon">!</span>
                <span class="error-text">${message}</span>
                <button class="error-dismiss" onclick="this.parentElement.style.display='none'">Dismiss</button>
            </div>
        `;
        errorContainer.style.display = 'block';

        // Auto-hide after 10 seconds
        setTimeout(() => {
            errorContainer.style.display = 'none';
        }, 10000);
    }
}

/**
 * Clear error display
 */
function clearError() {
    AppState.lastError = null;
    const errorContainer = document.getElementById('errorContainer');
    if (errorContainer) {
        errorContainer.style.display = 'none';
    }
}

// ============================================================================
// LOCAL STORAGE MANAGEMENT
// ============================================================================

/**
 * Load user preferences from localStorage
 */
function loadUserPreferences() {
    try {
        // Load selected model
        const savedModel = localStorage.getItem(Config.storageKeys.selectedModel);
        if (savedModel && (savedModel === 'classic' || savedModel === 'ensemble')) {
            AppState.currentModel = savedModel;
        }

        // Load ratings
        const savedRatings = localStorage.getItem(Config.storageKeys.ratings);
        if (savedRatings) {
            const ratingsObj = JSON.parse(savedRatings);
            AppState.ratings = new Map(Object.entries(ratingsObj));
        }

        console.log('[Demo3] User preferences loaded:', {
            model: AppState.currentModel,
            ratingsCount: AppState.ratings.size
        });

    } catch (error) {
        console.warn('[Demo3] Failed to load user preferences:', error);
    }
}

/**
 * Save user preferences to localStorage
 */
function saveUserPreferences() {
    try {
        localStorage.setItem(Config.storageKeys.selectedModel, AppState.currentModel);

        // Convert Map to object for JSON serialization
        const ratingsObj = Object.fromEntries(AppState.ratings);
        localStorage.setItem(Config.storageKeys.ratings, JSON.stringify(ratingsObj));

        // Save last video ID
        if (AppState.currentVideo) {
            localStorage.setItem(Config.storageKeys.lastVideo, AppState.currentVideo.id);
        }

    } catch (error) {
        console.warn('[Demo3] Failed to save user preferences:', error);
    }
}

/**
 * Get last selected video ID from localStorage
 * @returns {string|null} Video ID or null
 */
function getLastVideoId() {
    return localStorage.getItem(Config.storageKeys.lastVideo);
}

// ============================================================================
// VIDEO CATALOG MANAGEMENT
// ============================================================================

/**
 * Load video catalog from JSON file
 * @returns {Promise<Object>} Catalog data
 */
async function loadCatalog() {
    setLoading('catalog', true);
    clearError();

    try {
        const response = await fetch(Config.catalogPath);

        if (!response.ok) {
            throw new Error(`Failed to load catalog: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        AppState.catalog = data;

        console.log('[Demo3] Catalog loaded:', {
            totalVideos: data.videos?.length || 0,
            statistics: data.statistics
        });

        return data;

    } catch (error) {
        showError('Failed to load video catalog. Please refresh the page.', error);
        throw error;

    } finally {
        setLoading('catalog', false);
    }
}

/**
 * Get video source URL with Azure Blob fallback
 * @param {Object} video - Video object from catalog
 * @returns {string} Video URL
 */
function getVideoSourceUrl(video) {
    // First try local path if it exists
    if (video.file_info?.exists && video.file_info?.local_path) {
        // For web playback, we need to handle local paths differently
        // In production, videos should be served from Azure Blob Storage
        console.log('[Demo3] Local file exists but using Azure Blob for web playback');
    }

    // Construct Azure Blob Storage URL
    const filename = encodeURIComponent(video.filename);
    const extension = video.file_info?.local_path?.split('.').pop() || 'mp4';

    return `${Config.azureBlobBase}/${filename}.${extension}`;
}

// ============================================================================
// CLASSIFICATION API
// ============================================================================

/**
 * Classify a single question text using the ML API
 * @param {string} text - Question text to classify
 * @param {Object} options - Additional options (videoId, questionId, etc.)
 * @returns {Promise<Object>} Classification result
 */
async function classifyQuestion(text, options = {}) {
    const endpoint = AppState.currentModel === 'ensemble'
        ? `${Config.apiBaseUrl}/api/v2/classify/ensemble`
        : `${Config.apiBaseUrl}/api/classify`;

    console.log(`[Demo3] Classifying: "${text.substring(0, 50)}..." with ${AppState.currentModel} model`);

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                debug: true,
                ...options
            })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();

        // Normalize response format
        return {
            classification: result.classification || 'Unknown',
            confidence: result.confidence || 0,
            oeq_probability: result.oeq_probability || (result.classification === 'OEQ' ? result.confidence : 1 - result.confidence),
            ceq_probability: result.ceq_probability || (result.classification === 'CEQ' ? result.confidence : 1 - result.confidence),
            model: AppState.currentModel,
            method: result.method || (AppState.currentModel === 'ensemble' ? 'Ensemble (7 Models)' : 'Classic ML'),
            processing_time_ms: result.processing_time_ms || 0,
            ensemble_details: result.ensemble_details || null,
            timestamp: new Date().toISOString()
        };

    } catch (error) {
        console.error('[Demo3] Classification failed:', error);
        return {
            classification: 'Error',
            confidence: 0,
            error: error.message,
            model: AppState.currentModel,
            timestamp: new Date().toISOString()
        };
    }
}

/**
 * Classify all questions for a video
 * @param {Object} video - Video object with questions array
 * @returns {Promise<Array>} Array of classification results
 */
async function classifyVideoQuestions(video) {
    if (!video?.questions?.length) {
        return [];
    }

    setLoading('classification', true);

    try {
        const results = [];

        for (const question of video.questions) {
            if (question.transcript_text) {
                const result = await classifyQuestion(question.transcript_text, {
                    videoId: video.id,
                    questionId: question.id,
                    timestamp: question.timestamp_seconds
                });

                results.push({
                    ...question,
                    ml_prediction: result
                });
            } else {
                results.push({
                    ...question,
                    ml_prediction: null
                });
            }
        }

        // Cache results
        AppState.classifications.set(video.id, results);

        console.log(`[Demo3] Classified ${results.length} questions for video ${video.id}`);

        return results;

    } finally {
        setLoading('classification', false);
    }
}

// ============================================================================
// RATING/FEEDBACK MANAGEMENT
// ============================================================================

/**
 * Submit a rating for a classification
 * @param {Object} ratingData - Rating data including videoId, questionId, isCorrect, etc.
 * @returns {Promise<Object>} Result of rating submission
 */
async function submitRating(ratingData) {
    const {
        videoId,
        questionId,
        predictedClass,
        correctClass,
        isCorrect,
        text
    } = ratingData;

    // Generate rating key
    const ratingKey = `${videoId}_${questionId}_${AppState.currentModel}`;

    // Store locally
    AppState.ratings.set(ratingKey, {
        ...ratingData,
        model: AppState.currentModel,
        timestamp: new Date().toISOString()
    });

    // Persist to localStorage
    saveUserPreferences();

    // Determine error type for ML feedback
    let errorType = isCorrect ? (predictedClass === 'OEQ' ? 'TP' : 'TN') :
                    (predictedClass === 'OEQ' ? 'FP' : 'FN');

    // Try to submit to backend
    try {
        const feedbackEndpoint = `${Config.apiBaseUrl}/save_feedback`;

        const response = await fetch(feedbackEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                predicted_class: predictedClass,
                correct_class: correctClass,
                error_type: errorType,
                video_id: videoId,
                question_id: questionId,
                model: AppState.currentModel,
                timestamp: new Date().toISOString()
            })
        });

        if (response.ok) {
            const result = await response.json();
            console.log('[Demo3] Rating submitted to backend:', result);
            return { success: true, storage: 'backend', ...result };
        }

    } catch (error) {
        console.warn('[Demo3] Backend rating submission failed, stored locally:', error);
    }

    return { success: true, storage: 'local', ratingKey };
}

// ============================================================================
// TAB MANAGEMENT
// ============================================================================

/**
 * Switch to a different tab
 * @param {string} tabName - Tab identifier: 'video-library' | 'question-analysis' | 'ground-truth' | 'analytics'
 */
function switchTab(tabName) {
    console.log('[Demo3] Switching to tab:', tabName);

    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update tab content panels
    document.querySelectorAll('.tab-content').forEach(panel => {
        panel.classList.toggle('active', panel.id === `${tabName}-tab`);
    });

    // Tab-specific initialization
    switch (tabName) {
        case 'analytics':
            // Refresh analytics when switching to that tab
            window.dispatchEvent(new CustomEvent('refreshAnalytics'));
            break;

        case 'ground-truth':
            // Load ground truth data if needed
            if (AppState.currentVideo) {
                displayGroundTruth(AppState.currentVideo);
            }
            break;
    }
}

/**
 * Display ground truth annotations for a video
 * @param {Object} video - Video object with questions
 */
function displayGroundTruth(video) {
    const container = document.getElementById('groundTruthContainer');
    if (!container) return;

    if (!video?.questions?.length) {
        container.innerHTML = '<p class="empty-state">No ground truth annotations available for this video.</p>';
        return;
    }

    const html = video.questions.map((q, index) => `
        <div class="ground-truth-item">
            <div class="gt-header">
                <span class="gt-timestamp" data-seconds="${q.timestamp_seconds}">
                    ${q.timestamp}
                </span>
                <span class="gt-type gt-type-${q.ground_truth_type?.toLowerCase() || 'unknown'}">
                    ${q.ground_truth_type || 'Unknown'}
                </span>
            </div>
            <div class="gt-description">${q.description || 'No description'}</div>
            ${q.transcript_text ? `<div class="gt-transcript">"${q.transcript_text}"</div>` : ''}
            ${q.behavioral_notes ? `
                <div class="gt-notes">
                    ${Object.entries(q.behavioral_notes)
                        .filter(([_, value]) => value)
                        .map(([key, _]) => `<span class="gt-note">${key.replace(/_/g, ' ')}</span>`)
                        .join('')}
                </div>
            ` : ''}
        </div>
    `).join('');

    container.innerHTML = html;

    // Add click handlers for timestamps
    container.querySelectorAll('.gt-timestamp').forEach(el => {
        el.addEventListener('click', () => {
            const seconds = parseFloat(el.dataset.seconds);
            window.dispatchEvent(new CustomEvent('jumpToTimestamp', { detail: { seconds } }));
        });
    });
}

// ============================================================================
// VIDEO PLAYER INTEGRATION
// ============================================================================

/**
 * Update video player with selected video
 * @param {Object} video - Video object from catalog
 */
function updateVideoPlayer(video) {
    const videoElement = document.getElementById('videoPlayer');
    const videoTitle = document.getElementById('videoTitle');
    const videoDescription = document.getElementById('videoDescription');

    if (!videoElement) {
        console.warn('[Demo3] Video player element not found');
        return;
    }

    setLoading('video', true);

    try {
        // Update source
        const sourceUrl = getVideoSourceUrl(video);
        videoElement.src = sourceUrl;

        // Update metadata display
        if (videoTitle) {
            videoTitle.textContent = video.display_name || video.filename;
        }

        if (videoDescription) {
            videoDescription.textContent = video.description || '';
        }

        // Set up event handlers
        videoElement.onloadeddata = () => {
            setLoading('video', false);
            console.log('[Demo3] Video loaded:', video.display_name);
        };

        videoElement.onerror = (e) => {
            setLoading('video', false);
            showError(`Failed to load video: ${video.display_name}. The file may not be available.`);
        };

        // Load the video
        videoElement.load();

    } catch (error) {
        setLoading('video', false);
        showError('Failed to set up video player.', error);
    }
}

/**
 * Seek video to specific timestamp
 * @param {number} seconds - Target time in seconds
 */
function seekToTimestamp(seconds) {
    const videoElement = document.getElementById('videoPlayer');

    if (videoElement && !isNaN(seconds)) {
        videoElement.currentTime = seconds;

        // If video is paused, start playing
        if (videoElement.paused) {
            videoElement.play().catch(e => {
                console.warn('[Demo3] Auto-play blocked:', e);
            });
        }

        console.log(`[Demo3] Seeking to ${formatTimestamp(seconds)}`);
    }
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

/**
 * Handle video selection event
 * @param {CustomEvent} event - Event with video object in detail
 */
async function handleVideoSelected(event) {
    const video = event.detail?.video || event.detail;

    if (!video) {
        console.warn('[Demo3] Video selected event received without video data');
        return;
    }

    console.log('[Demo3] Video selected:', video.display_name);

    // Update state
    AppState.currentVideo = video;
    saveUserPreferences();

    // Update video player
    updateVideoPlayer(video);

    // Load questions and classify
    const cachedClassifications = AppState.classifications.get(video.id);

    if (cachedClassifications) {
        // Use cached results
        window.dispatchEvent(new CustomEvent('questionsLoaded', {
            detail: { video, questions: cachedClassifications }
        }));
    } else {
        // Classify questions
        const classifiedQuestions = await classifyVideoQuestions(video);

        window.dispatchEvent(new CustomEvent('questionsLoaded', {
            detail: { video, questions: classifiedQuestions }
        }));
    }

    // Update ground truth display
    displayGroundTruth(video);

    // Refresh analytics
    window.dispatchEvent(new CustomEvent('refreshAnalytics'));
}

/**
 * Handle model change event
 * @param {CustomEvent} event - Event with new model in detail
 */
async function handleModelChanged(event) {
    const newModel = event.detail?.model || event.detail;

    if (!newModel || (newModel !== 'classic' && newModel !== 'ensemble')) {
        console.warn('[Demo3] Invalid model specified:', newModel);
        return;
    }

    console.log('[Demo3] Model changed to:', newModel);

    // Update state
    AppState.currentModel = newModel;
    saveUserPreferences();

    // Clear cached classifications (they're model-specific)
    AppState.classifications.clear();

    // Re-classify current video if one is selected
    if (AppState.currentVideo) {
        const classifiedQuestions = await classifyVideoQuestions(AppState.currentVideo);

        window.dispatchEvent(new CustomEvent('questionsLoaded', {
            detail: {
                video: AppState.currentVideo,
                questions: classifiedQuestions
            }
        }));
    }

    // Refresh analytics
    window.dispatchEvent(new CustomEvent('refreshAnalytics'));
}

/**
 * Handle jump to timestamp event
 * @param {CustomEvent} event - Event with seconds in detail
 */
function handleJumpToTimestamp(event) {
    const seconds = event.detail?.seconds ?? event.detail;

    if (typeof seconds === 'number') {
        seekToTimestamp(seconds);
    }
}

/**
 * Handle rating submitted event
 * @param {CustomEvent} event - Event with rating data in detail
 */
async function handleRatingSubmitted(event) {
    const ratingData = event.detail;

    if (!ratingData) {
        console.warn('[Demo3] Rating submitted event received without data');
        return;
    }

    console.log('[Demo3] Rating submitted:', ratingData);

    const result = await submitRating(ratingData);

    // Notify analytics to refresh
    window.dispatchEvent(new CustomEvent('refreshAnalytics'));

    // Dispatch result event
    window.dispatchEvent(new CustomEvent('ratingProcessed', {
        detail: result
    }));
}

// ============================================================================
// COMPONENT INITIALIZATION
// ============================================================================

/**
 * Initialize all Demo 3 components
 * Components have varying constructor signatures, so we adapt to each
 */
async function initializeComponents() {
    console.log('[Demo3] Initializing components...');

    try {
        // Initialize VideoSelector component
        // Constructor signature: (containerId: string, options?: object)
        if (VideoSelector) {
            window.videoSelector = new VideoSelector('videoSelectorContainer', {
                catalogPath: Config.catalogPath,
                onVideoSelect: (video) => {
                    window.dispatchEvent(new CustomEvent('videoSelected', { detail: { video } }));
                },
                autoLoad: true
            });
            console.log('[Demo3] VideoSelector initialized');
        } else {
            console.warn('[Demo3] VideoSelector not available');
        }

        // Initialize QuestionPanel component
        // Constructor signature: (containerSelector: string)
        if (QuestionPanel) {
            window.questionPanel = new QuestionPanel('#questionPanelContainer');
            console.log('[Demo3] QuestionPanel initialized');
        } else {
            console.warn('[Demo3] QuestionPanel not available');
        }

        // Initialize AnalyticsDashboard component
        // Constructor signature: (containerId: string, options?: object)
        if (AnalyticsDashboard) {
            window.analyticsDashboard = new AnalyticsDashboard('analyticsDashboardContainer', {
                showExport: true,
                autoUpdate: true
            });
            console.log('[Demo3] AnalyticsDashboard initialized');
        } else {
            console.warn('[Demo3] AnalyticsDashboard not available');
        }

        // Make state accessible to components
        window.Demo3State = AppState;
        window.Demo3Config = Config;

    } catch (error) {
        console.error('[Demo3] Failed to initialize components:', error);
        showError('Failed to initialize application components. Some features may not work correctly.');
    }
}

// ============================================================================
// TAB BUTTON SETUP
// ============================================================================

/**
 * Set up tab switching buttons
 */
function setupTabButtons() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            if (tabName) {
                switchTab(tabName);
            }
        });
    });
}

// ============================================================================
// MAIN INITIALIZATION
// ============================================================================

/**
 * Main application entry point
 * Called when DOM is fully loaded
 */
async function initializeApp() {
    console.log('========================================');
    console.log('[Demo3] Video Question Analysis - Initializing');
    console.log(`[Demo3] Environment: ${Config.apiBaseUrl.includes('localhost') ? 'LOCAL' : 'AZURE'}`);
    console.log('========================================');

    try {
        // Load user preferences first
        loadUserPreferences();

        // Load component modules/classes
        await loadComponents();

        // Load video catalog
        await loadCatalog();

        // Initialize components
        await initializeComponents();

        // Set up tab buttons
        setupTabButtons();

        // Register event listeners for cross-component communication
        window.addEventListener('videoSelected', handleVideoSelected);
        window.addEventListener('modelChanged', handleModelChanged);
        window.addEventListener('jumpToTimestamp', handleJumpToTimestamp);
        window.addEventListener('ratingSubmitted', handleRatingSubmitted);

        // Additional event for timestamp from QuestionPanel (uses different event name)
        window.addEventListener('jumpToTimestamp', (e) => {
            const timestamp = e.detail?.timestamp ?? e.detail?.seconds;
            if (typeof timestamp === 'number') {
                seekToTimestamp(timestamp);
            }
        });

        // Listen for questions loaded to update QuestionPanel
        window.addEventListener('questionsLoaded', (e) => {
            if (window.questionPanel && e.detail?.video) {
                window.questionPanel.setVideo(e.detail.video);
            }
        });

        // Try to restore last selected video
        const lastVideoId = getLastVideoId();
        if (lastVideoId && AppState.catalog?.videos) {
            const lastVideo = AppState.catalog.videos.find(v => v.id === lastVideoId);
            if (lastVideo) {
                console.log('[Demo3] Restoring last video:', lastVideo.display_name);
                // Delay to ensure components are fully initialized
                setTimeout(() => {
                    window.dispatchEvent(new CustomEvent('videoSelected', { detail: { video: lastVideo } }));
                }, 100);
            }
        }

        console.log('[Demo3] Application initialized successfully');
        console.log(`[Demo3] Catalog: ${AppState.catalog?.videos?.length || 0} videos`);
        console.log(`[Demo3] Model: ${AppState.currentModel}`);

    } catch (error) {
        console.error('[Demo3] Application initialization failed:', error);
        showError('Failed to initialize the application. Please refresh the page and try again.');
    }
}

// ============================================================================
// DOM READY HANDLER
// ============================================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    // DOM already loaded
    initializeApp();
}

// ============================================================================
// EXPORTS - Make key functions available globally for debugging
// ============================================================================

window.Demo3 = {
    // State access
    getState: () => AppState,
    getConfig: () => Config,

    // Actions
    classifyQuestion,
    classifyVideoQuestions,
    submitRating,
    switchTab,
    seekToTimestamp,

    // Utilities
    parseTimestamp,
    formatTimestamp,
    getVideoSourceUrl,

    // Reload functions
    reloadCatalog: loadCatalog,
    reinitialize: initializeApp
};

console.log('[Demo3] Global Demo3 object available for debugging');
