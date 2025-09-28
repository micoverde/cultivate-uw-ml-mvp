/**
 * Unified Professional Header Component
 * Consistent header across all pages
 */

class UnifiedHeader {
    constructor() {
        this.currentPage = this.detectCurrentPage();
        this.init();
    }

    detectCurrentPage() {
        const path = window.location.pathname;
        if (path.includes('demo1')) return 'demo1';
        if (path.includes('demo2')) return 'demo2';
        return 'hub';
    }

    init() {
        this.createHeader();
        this.attachEventListeners();
    }

    createHeader() {
        const header = document.createElement('header');
        header.className = 'unified-header';
        header.innerHTML = `
            <div class="unified-header-content">
                <div class="header-left">
                    <div class="header-brand">
                        <h1 class="header-title">Cultivate Learning</h1>
                        <p class="header-subtitle">ML-Powered Early Childhood Education Platform</p>
                    </div>
                </div>
                <div class="header-nav">
                    <a href="${this.currentPage === 'hub' ? '#' : '../index.html'}"
                       class="nav-link ${this.currentPage === 'hub' ? 'active' : ''}">
                        Hub
                    </a>
                    <a href="${this.currentPage === 'demo1' ? '#' : this.currentPage === 'hub' ? 'demo1/index.html' : '../demo1/index.html'}"
                       class="nav-link ${this.currentPage === 'demo1' ? 'active' : ''}">
                        Demo 1
                    </a>
                    <a href="${this.currentPage === 'demo2' ? '#' : this.currentPage === 'hub' ? 'demo2/index.html' : '../demo2/index.html'}"
                       class="nav-link ${this.currentPage === 'demo2' ? 'active' : ''}">
                        Demo 2
                    </a>
                    <div class="header-separator"></div>
                    <button class="theme-toggle-btn" id="themeToggle">
                        <span id="themeText">Dark Mode</span>
                    </button>
                </div>
            </div>
        `;

        // Insert at the beginning of body
        document.body.insertBefore(header, document.body.firstChild);
    }

    attachEventListeners() {
        const themeToggle = document.getElementById('themeToggle');
        const themeText = document.getElementById('themeText');

        // Load saved theme
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.body.setAttribute('data-theme', savedTheme);
        themeText.textContent = savedTheme === 'dark' ? 'Light Mode' : 'Dark Mode';

        // Theme toggle handler
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            document.body.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            themeText.textContent = newTheme === 'dark' ? 'Light Mode' : 'Dark Mode';
        });
    }
}

// Add separator style
const separatorStyle = document.createElement('style');
separatorStyle.textContent = `
    .header-separator {
        width: 1px;
        height: 24px;
        background: var(--border-light);
        margin: 0 var(--space-md);
    }
`;
document.head.appendChild(separatorStyle);

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.unifiedHeader = new UnifiedHeader();
    });
} else {
    window.unifiedHeader = new UnifiedHeader();
}