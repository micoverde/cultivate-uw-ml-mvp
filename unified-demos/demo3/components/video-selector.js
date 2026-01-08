/**
 * VideoSelector Component
 *
 * A production-ready video selection component for the Cultivate UW ML Demo.
 * Displays video cards in a grid with filtering, searching, and sorting capabilities.
 *
 * @module VideoSelector
 * @version 1.0.0
 */

/**
 * VideoSelector class for rendering and managing video selection UI
 */
export class VideoSelector {
    /**
     * Creates a new VideoSelector instance
     * @param {string} containerId - The ID of the container element to render into
     * @param {Object} options - Configuration options
     * @param {string} [options.catalogPath='../data/video_catalog.json'] - Path to video catalog JSON
     * @param {Function} [options.onVideoSelect] - Callback when video is selected
     * @param {boolean} [options.autoLoad=true] - Whether to auto-load catalog on init
     */
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = null;
        this.options = {
            catalogPath: options.catalogPath || '../data/video_catalog.json',
            onVideoSelect: options.onVideoSelect || null,
            autoLoad: options.autoLoad !== false
        };

        // State
        this.videos = [];
        this.filteredVideos = [];
        this.selectedVideoId = null;
        this.currentFilter = 'ALL';
        this.currentSearch = '';
        this.currentSort = 'name';
        this.isLoading = false;
        this.error = null;

        // Bind methods
        this._handleFilterChange = this._handleFilterChange.bind(this);
        this._handleSearchInput = this._handleSearchInput.bind(this);
        this._handleSortChange = this._handleSortChange.bind(this);
        this._handleVideoClick = this._handleVideoClick.bind(this);

        // Initialize
        this._init();
    }

    /**
     * Initialize the component
     * @private
     */
    async _init() {
        this.container = document.getElementById(this.containerId);

        if (!this.container) {
            console.error(`[VideoSelector] Container element with ID "${this.containerId}" not found`);
            return;
        }

        // Render initial structure
        this._renderSkeleton();

        // Auto-load if enabled
        if (this.options.autoLoad) {
            await this.loadCatalog();
        }
    }

    /**
     * Load the video catalog from JSON
     * @param {string} [path] - Optional override path to catalog
     * @returns {Promise<void>}
     */
    async loadCatalog(path = null) {
        const catalogPath = path || this.options.catalogPath;

        this.isLoading = true;
        this.error = null;
        this._updateLoadingState();

        try {
            const response = await fetch(catalogPath);

            if (!response.ok) {
                throw new Error(`Failed to load video catalog: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();

            if (!data.videos || !Array.isArray(data.videos)) {
                throw new Error('Invalid video catalog format: expected "videos" array');
            }

            this.videos = data.videos;
            this.filteredVideos = [...this.videos];
            this._applyFiltersAndSort();

            console.log(`[VideoSelector] Loaded ${this.videos.length} videos from catalog`);

        } catch (err) {
            this.error = err.message;
            console.error('[VideoSelector] Error loading catalog:', err);
        } finally {
            this.isLoading = false;
            this._render();
        }
    }

    /**
     * Render the skeleton/loading structure
     * @private
     */
    _renderSkeleton() {
        this.container.innerHTML = `
            <div class="video-selector">
                <div class="video-selector__header">
                    <h2 class="video-selector__title">Video Library</h2>
                    <div class="video-selector__controls">
                        <div class="video-selector__search">
                            <input
                                type="text"
                                class="video-selector__search-input"
                                placeholder="Search videos..."
                                aria-label="Search videos by name"
                            />
                        </div>
                        <div class="video-selector__filters">
                            <button class="video-selector__filter-btn active" data-filter="ALL">All</button>
                            <button class="video-selector__filter-btn" data-filter="PK">Pre-K</button>
                            <button class="video-selector__filter-btn" data-filter="TODDLER">Toddler</button>
                        </div>
                        <div class="video-selector__sort">
                            <label for="video-sort">Sort by:</label>
                            <select id="video-sort" class="video-selector__sort-select">
                                <option value="name">Name</option>
                                <option value="questions">Question Count</option>
                                <option value="age_group">Age Group</option>
                            </select>
                        </div>
                    </div>
                </div>
                <div class="video-selector__content">
                    <div class="video-selector__grid">
                        <div class="video-selector__loading">Loading videos...</div>
                    </div>
                </div>
                <div class="video-selector__stats"></div>
            </div>
        `;

        this._attachEventListeners();
        this._injectStyles();
    }

    /**
     * Attach event listeners to controls
     * @private
     */
    _attachEventListeners() {
        // Search input
        const searchInput = this.container.querySelector('.video-selector__search-input');
        if (searchInput) {
            searchInput.addEventListener('input', this._handleSearchInput);
        }

        // Filter buttons
        const filterBtns = this.container.querySelectorAll('.video-selector__filter-btn');
        filterBtns.forEach(btn => {
            btn.addEventListener('click', this._handleFilterChange);
        });

        // Sort select
        const sortSelect = this.container.querySelector('.video-selector__sort-select');
        if (sortSelect) {
            sortSelect.addEventListener('change', this._handleSortChange);
        }
    }

    /**
     * Handle filter button clicks
     * @private
     * @param {Event} e - Click event
     */
    _handleFilterChange(e) {
        const filter = e.target.dataset.filter;
        if (!filter) return;

        this.currentFilter = filter;

        // Update active state
        const filterBtns = this.container.querySelectorAll('.video-selector__filter-btn');
        filterBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });

        this._applyFiltersAndSort();
        this._renderGrid();
    }

    /**
     * Handle search input
     * @private
     * @param {Event} e - Input event
     */
    _handleSearchInput(e) {
        this.currentSearch = e.target.value.toLowerCase().trim();
        this._applyFiltersAndSort();
        this._renderGrid();
    }

    /**
     * Handle sort change
     * @private
     * @param {Event} e - Change event
     */
    _handleSortChange(e) {
        this.currentSort = e.target.value;
        this._applyFiltersAndSort();
        this._renderGrid();
    }

    /**
     * Handle video card click
     * @private
     * @param {Event} e - Click event
     */
    _handleVideoClick(e) {
        const card = e.target.closest('.video-card');
        if (!card) return;

        const videoId = card.dataset.videoId;
        const video = this.videos.find(v => v.id === videoId);

        if (!video) return;

        // Update selection
        this.selectedVideoId = videoId;

        // Update visual state
        const allCards = this.container.querySelectorAll('.video-card');
        allCards.forEach(c => {
            c.classList.toggle('selected', c.dataset.videoId === videoId);
        });

        // Dispatch custom event
        const event = new CustomEvent('videoSelected', {
            bubbles: true,
            detail: { video }
        });
        this.container.dispatchEvent(event);

        // Call callback if provided
        if (this.options.onVideoSelect) {
            this.options.onVideoSelect(video);
        }

        console.log(`[VideoSelector] Video selected: ${video.display_name} (${video.id})`);
    }

    /**
     * Apply current filters and sorting to videos
     * @private
     */
    _applyFiltersAndSort() {
        // Start with all videos
        let result = [...this.videos];

        // Apply age group filter
        if (this.currentFilter !== 'ALL') {
            result = result.filter(v => v.age_group === this.currentFilter);
        }

        // Apply search filter
        if (this.currentSearch) {
            result = result.filter(v =>
                v.display_name.toLowerCase().includes(this.currentSearch) ||
                (v.description && v.description.toLowerCase().includes(this.currentSearch))
            );
        }

        // Apply sorting
        result.sort((a, b) => {
            switch (this.currentSort) {
                case 'name':
                    return a.display_name.localeCompare(b.display_name);
                case 'questions':
                    return (b.questions?.length || 0) - (a.questions?.length || 0);
                case 'age_group':
                    return a.age_group.localeCompare(b.age_group);
                default:
                    return 0;
            }
        });

        this.filteredVideos = result;
    }

    /**
     * Update loading state in UI
     * @private
     */
    _updateLoadingState() {
        const grid = this.container.querySelector('.video-selector__grid');
        if (grid) {
            grid.innerHTML = '<div class="video-selector__loading">Loading videos...</div>';
        }
    }

    /**
     * Full render of the component
     * @private
     */
    _render() {
        if (this.error) {
            this._renderError();
            return;
        }

        this._renderGrid();
        this._renderStats();
    }

    /**
     * Render error state
     * @private
     */
    _renderError() {
        const grid = this.container.querySelector('.video-selector__grid');
        if (grid) {
            grid.innerHTML = `
                <div class="video-selector__error">
                    <div class="video-selector__error-icon">!</div>
                    <div class="video-selector__error-message">${this._escapeHtml(this.error)}</div>
                    <button class="video-selector__retry-btn" onclick="this.closest('.video-selector').dispatchEvent(new CustomEvent('retry'))">
                        Retry
                    </button>
                </div>
            `;
        }

        // Listen for retry
        this.container.querySelector('.video-selector')?.addEventListener('retry', () => {
            this.loadCatalog();
        }, { once: true });
    }

    /**
     * Render the video grid
     * @private
     */
    _renderGrid() {
        const grid = this.container.querySelector('.video-selector__grid');
        if (!grid) return;

        if (this.filteredVideos.length === 0) {
            grid.innerHTML = `
                <div class="video-selector__empty">
                    <div class="video-selector__empty-message">
                        ${this.currentSearch || this.currentFilter !== 'ALL'
                            ? 'No videos match your criteria'
                            : 'No videos available'}
                    </div>
                </div>
            `;
            return;
        }

        grid.innerHTML = this.filteredVideos.map(video => this._renderVideoCard(video)).join('');

        // Attach click handlers
        const cards = grid.querySelectorAll('.video-card');
        cards.forEach(card => {
            card.addEventListener('click', this._handleVideoClick);
        });
    }

    /**
     * Render a single video card
     * @private
     * @param {Object} video - Video data object
     * @returns {string} HTML string for the card
     */
    _renderVideoCard(video) {
        const isSelected = video.id === this.selectedVideoId;
        const questionCount = video.questions?.length || 0;
        const hasTranscripts = video.has_transcripts;
        const transcriptCount = video.transcript_count || 0;
        const hasMLPredictions = this._hasMLPredictions(video);
        const fileExists = video.file_info?.exists || false;

        // Calculate accuracy if ML predictions exist
        const accuracy = hasMLPredictions ? this._calculateAccuracy(video) : null;

        return `
            <div class="video-card ${isSelected ? 'selected' : ''} ${!fileExists ? 'unavailable' : ''}"
                 data-video-id="${this._escapeHtml(video.id)}"
                 role="button"
                 tabindex="0"
                 aria-pressed="${isSelected}">
                <div class="video-card__header">
                    <h3 class="video-card__title">${this._escapeHtml(video.display_name)}</h3>
                    <span class="video-card__badge video-card__badge--${video.age_group.toLowerCase()}">
                        ${video.age_group === 'PK' ? 'Pre-K' : 'Toddler'}
                    </span>
                </div>
                <div class="video-card__body">
                    <div class="video-card__stats">
                        <div class="video-card__stat">
                            <span class="video-card__stat-value">${questionCount}</span>
                            <span class="video-card__stat-label">Questions</span>
                        </div>
                        <div class="video-card__stat">
                            <span class="video-card__stat-icon ${hasTranscripts ? 'active' : ''}">
                                ${hasTranscripts ? '\u2713' : '\u2717'}
                            </span>
                            <span class="video-card__stat-label">
                                ${hasTranscripts ? `${transcriptCount} Transcripts` : 'No Transcripts'}
                            </span>
                        </div>
                    </div>
                    ${accuracy !== null ? `
                        <div class="video-card__accuracy">
                            <div class="video-card__accuracy-bar">
                                <div class="video-card__accuracy-fill" style="width: ${accuracy}%"></div>
                            </div>
                            <span class="video-card__accuracy-label">${accuracy.toFixed(0)}% ML Accuracy</span>
                        </div>
                    ` : `
                        <div class="video-card__no-ml">
                            <span class="video-card__no-ml-label">ML Pending</span>
                        </div>
                    `}
                </div>
                <div class="video-card__footer">
                    ${fileExists
                        ? `<span class="video-card__file-size">${this._formatFileSize(video.file_info?.file_size_mb)}</span>`
                        : `<span class="video-card__unavailable">Video file unavailable</span>`
                    }
                </div>
            </div>
        `;
    }

    /**
     * Check if video has ML predictions
     * @private
     * @param {Object} video - Video data
     * @returns {boolean}
     */
    _hasMLPredictions(video) {
        if (!video.questions || video.questions.length === 0) return false;
        return video.questions.some(q => q.ml_prediction !== null);
    }

    /**
     * Calculate ML accuracy for a video
     * @private
     * @param {Object} video - Video data
     * @returns {number|null} Accuracy percentage or null if no predictions
     */
    _calculateAccuracy(video) {
        if (!video.questions || video.questions.length === 0) return null;

        const questionsWithPredictions = video.questions.filter(q => q.ml_prediction !== null);
        if (questionsWithPredictions.length === 0) return null;

        const correct = questionsWithPredictions.filter(q =>
            q.ml_prediction === q.ground_truth_type
        ).length;

        return (correct / questionsWithPredictions.length) * 100;
    }

    /**
     * Render statistics footer
     * @private
     */
    _renderStats() {
        const statsEl = this.container.querySelector('.video-selector__stats');
        if (!statsEl) return;

        const totalVideos = this.videos.length;
        const shownVideos = this.filteredVideos.length;
        const pkCount = this.videos.filter(v => v.age_group === 'PK').length;
        const toddlerCount = this.videos.filter(v => v.age_group === 'TODDLER').length;
        const withTranscripts = this.videos.filter(v => v.has_transcripts).length;

        statsEl.innerHTML = `
            <div class="video-selector__stats-content">
                <span>Showing ${shownVideos} of ${totalVideos} videos</span>
                <span class="video-selector__stats-divider">|</span>
                <span>${pkCount} Pre-K</span>
                <span class="video-selector__stats-divider">|</span>
                <span>${toddlerCount} Toddler</span>
                <span class="video-selector__stats-divider">|</span>
                <span>${withTranscripts} with transcripts</span>
            </div>
        `;
    }

    /**
     * Format file size for display
     * @private
     * @param {number} sizeMb - Size in megabytes
     * @returns {string} Formatted size string
     */
    _formatFileSize(sizeMb) {
        if (!sizeMb) return '';
        if (sizeMb >= 1000) {
            return `${(sizeMb / 1000).toFixed(1)} GB`;
        }
        return `${sizeMb.toFixed(0)} MB`;
    }

    /**
     * Escape HTML to prevent XSS
     * @private
     * @param {string} str - String to escape
     * @returns {string} Escaped string
     */
    _escapeHtml(str) {
        if (typeof str !== 'string') return str;
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /**
     * Inject component styles
     * @private
     */
    _injectStyles() {
        // Check if styles already injected
        if (document.getElementById('video-selector-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'video-selector-styles';
        styles.textContent = `
            .video-selector {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                height: 100%;
                display: flex;
                flex-direction: column;
            }

            .video-selector__header {
                margin-bottom: 20px;
            }

            .video-selector__title {
                margin: 0 0 15px 0;
                font-size: 1.5rem;
                color: #333;
            }

            .video-selector__controls {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: center;
            }

            .video-selector__search {
                flex: 1;
                min-width: 200px;
            }

            .video-selector__search-input {
                width: 100%;
                padding: 10px 15px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 0.95rem;
                transition: border-color 0.2s, box-shadow 0.2s;
            }

            .video-selector__search-input:focus {
                outline: none;
                border-color: #4a90d9;
                box-shadow: 0 0 0 3px rgba(74, 144, 217, 0.1);
            }

            .video-selector__filters {
                display: flex;
                gap: 8px;
            }

            .video-selector__filter-btn {
                padding: 8px 16px;
                border: 1px solid #ddd;
                background: white;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: all 0.2s;
            }

            .video-selector__filter-btn:hover {
                background: #f0f0f0;
            }

            .video-selector__filter-btn.active {
                background: #4a90d9;
                color: white;
                border-color: #4a90d9;
            }

            .video-selector__sort {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .video-selector__sort label {
                font-size: 0.9rem;
                color: #666;
            }

            .video-selector__sort-select {
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 0.9rem;
                background: white;
                cursor: pointer;
            }

            .video-selector__content {
                flex: 1;
                overflow-y: auto;
                min-height: 0;
            }

            .video-selector__grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 16px;
                padding: 4px;
            }

            .video-selector__loading,
            .video-selector__empty,
            .video-selector__error {
                grid-column: 1 / -1;
                text-align: center;
                padding: 40px 20px;
                color: #666;
            }

            .video-selector__error {
                color: #dc3545;
            }

            .video-selector__error-icon {
                width: 50px;
                height: 50px;
                margin: 0 auto 15px;
                background: #dc3545;
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                font-weight: bold;
            }

            .video-selector__retry-btn {
                margin-top: 15px;
                padding: 10px 20px;
                background: #4a90d9;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.95rem;
            }

            .video-selector__retry-btn:hover {
                background: #3a7fc8;
            }

            /* Video Card Styles */
            .video-card {
                background: white;
                border-radius: 8px;
                padding: 16px;
                cursor: pointer;
                transition: all 0.2s;
                border: 2px solid transparent;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .video-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transform: translateY(-2px);
            }

            .video-card:focus {
                outline: none;
                border-color: #4a90d9;
            }

            .video-card.selected {
                border-color: #4a90d9;
                background: #f0f7ff;
            }

            .video-card.unavailable {
                opacity: 0.7;
            }

            .video-card__header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 10px;
                margin-bottom: 12px;
            }

            .video-card__title {
                margin: 0;
                font-size: 1rem;
                font-weight: 600;
                color: #333;
                line-height: 1.3;
            }

            .video-card__badge {
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                flex-shrink: 0;
            }

            .video-card__badge--pk {
                background: #e3f2fd;
                color: #1976d2;
            }

            .video-card__badge--toddler {
                background: #fce4ec;
                color: #c2185b;
            }

            .video-card__body {
                margin-bottom: 12px;
            }

            .video-card__stats {
                display: flex;
                gap: 20px;
                margin-bottom: 12px;
            }

            .video-card__stat {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
            }

            .video-card__stat-value {
                font-size: 1.25rem;
                font-weight: 700;
                color: #333;
            }

            .video-card__stat-label {
                font-size: 0.8rem;
                color: #666;
            }

            .video-card__stat-icon {
                font-size: 1rem;
                color: #dc3545;
            }

            .video-card__stat-icon.active {
                color: #28a745;
            }

            .video-card__accuracy {
                margin-top: 8px;
            }

            .video-card__accuracy-bar {
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                overflow: hidden;
            }

            .video-card__accuracy-fill {
                height: 100%;
                background: linear-gradient(90deg, #28a745, #20c997);
                border-radius: 3px;
                transition: width 0.3s ease;
            }

            .video-card__accuracy-label {
                font-size: 0.8rem;
                color: #28a745;
                margin-top: 4px;
                display: block;
            }

            .video-card__no-ml {
                margin-top: 8px;
            }

            .video-card__no-ml-label {
                font-size: 0.8rem;
                color: #6c757d;
                font-style: italic;
            }

            .video-card__footer {
                border-top: 1px solid #eee;
                padding-top: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .video-card__file-size {
                font-size: 0.8rem;
                color: #666;
            }

            .video-card__unavailable {
                font-size: 0.8rem;
                color: #dc3545;
            }

            /* Stats Footer */
            .video-selector__stats {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #ddd;
            }

            .video-selector__stats-content {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                font-size: 0.85rem;
                color: #666;
            }

            .video-selector__stats-divider {
                color: #ddd;
            }

            /* Responsive adjustments */
            @media (max-width: 768px) {
                .video-selector__controls {
                    flex-direction: column;
                    align-items: stretch;
                }

                .video-selector__filters {
                    justify-content: center;
                }

                .video-selector__sort {
                    justify-content: center;
                }

                .video-selector__grid {
                    grid-template-columns: 1fr;
                }
            }
        `;

        document.head.appendChild(styles);
    }

    // Public API methods

    /**
     * Set the selected video by ID
     * @param {string} videoId - The video ID to select
     */
    selectVideo(videoId) {
        const video = this.videos.find(v => v.id === videoId);
        if (!video) {
            console.warn(`[VideoSelector] Video with ID "${videoId}" not found`);
            return;
        }

        this.selectedVideoId = videoId;
        this._renderGrid();

        // Dispatch event
        const event = new CustomEvent('videoSelected', {
            bubbles: true,
            detail: { video }
        });
        this.container.dispatchEvent(event);
    }

    /**
     * Get the currently selected video
     * @returns {Object|null} The selected video data or null
     */
    getSelectedVideo() {
        if (!this.selectedVideoId) return null;
        return this.videos.find(v => v.id === this.selectedVideoId) || null;
    }

    /**
     * Get all videos
     * @returns {Array} Array of all video objects
     */
    getVideos() {
        return [...this.videos];
    }

    /**
     * Get filtered videos
     * @returns {Array} Array of currently filtered video objects
     */
    getFilteredVideos() {
        return [...this.filteredVideos];
    }

    /**
     * Set filter programmatically
     * @param {string} filter - Filter value ('ALL', 'PK', 'TODDLER')
     */
    setFilter(filter) {
        const validFilters = ['ALL', 'PK', 'TODDLER'];
        if (!validFilters.includes(filter)) {
            console.warn(`[VideoSelector] Invalid filter: ${filter}. Valid values: ${validFilters.join(', ')}`);
            return;
        }

        this.currentFilter = filter;

        // Update UI
        const filterBtns = this.container.querySelectorAll('.video-selector__filter-btn');
        filterBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });

        this._applyFiltersAndSort();
        this._renderGrid();
        this._renderStats();
    }

    /**
     * Set search query programmatically
     * @param {string} query - Search query
     */
    setSearch(query) {
        this.currentSearch = (query || '').toLowerCase().trim();

        // Update UI
        const searchInput = this.container.querySelector('.video-selector__search-input');
        if (searchInput) {
            searchInput.value = query || '';
        }

        this._applyFiltersAndSort();
        this._renderGrid();
        this._renderStats();
    }

    /**
     * Set sort option programmatically
     * @param {string} sortBy - Sort option ('name', 'questions', 'age_group')
     */
    setSort(sortBy) {
        const validSorts = ['name', 'questions', 'age_group'];
        if (!validSorts.includes(sortBy)) {
            console.warn(`[VideoSelector] Invalid sort: ${sortBy}. Valid values: ${validSorts.join(', ')}`);
            return;
        }

        this.currentSort = sortBy;

        // Update UI
        const sortSelect = this.container.querySelector('.video-selector__sort-select');
        if (sortSelect) {
            sortSelect.value = sortBy;
        }

        this._applyFiltersAndSort();
        this._renderGrid();
    }

    /**
     * Refresh the component by reloading catalog
     * @returns {Promise<void>}
     */
    async refresh() {
        await this.loadCatalog();
    }

    /**
     * Destroy the component and clean up
     */
    destroy() {
        // Remove event listeners
        const searchInput = this.container.querySelector('.video-selector__search-input');
        if (searchInput) {
            searchInput.removeEventListener('input', this._handleSearchInput);
        }

        const filterBtns = this.container.querySelectorAll('.video-selector__filter-btn');
        filterBtns.forEach(btn => {
            btn.removeEventListener('click', this._handleFilterChange);
        });

        const sortSelect = this.container.querySelector('.video-selector__sort-select');
        if (sortSelect) {
            sortSelect.removeEventListener('change', this._handleSortChange);
        }

        // Clear container
        if (this.container) {
            this.container.innerHTML = '';
        }

        // Clear state
        this.videos = [];
        this.filteredVideos = [];
        this.selectedVideoId = null;

        console.log('[VideoSelector] Component destroyed');
    }
}

// Default export
export default VideoSelector;
