/**
 * Professional Footer Component
 * Consistent footer across all pages
 */

class UnifiedFooter {
    constructor() {
        this.currentYear = new Date().getFullYear();
        // Use last 6 digits of git hash for build number
        // In production, this would be injected during CI/CD
        this.buildNumber = localStorage.getItem('buildNumber') || '350888';
        this.init();
    }

    init() {
        this.injectStyles();
        this.createFooter();
    }

    injectStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .unified-footer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: var(--surface-100, #f8f9fa);
                border-top: 1px solid var(--surface-300, #e1e5ea);
                padding: 12px 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 13px;
                color: var(--text-secondary, #4a5568);
                z-index: 100;
            }

            body[data-theme="dark"] .unified-footer {
                background: var(--surface-100, #1a1a1a);
                border-top-color: var(--surface-300, #404040);
                color: var(--text-secondary, #b0b0b0);
            }

            .footer-left {
                display: flex;
                align-items: center;
                gap: 4px;
            }

            .footer-right {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .footer-separator {
                color: var(--text-tertiary, #718096);
                margin: 0 4px;
            }

            body[data-theme="dark"] .footer-separator {
                color: var(--text-tertiary, #808080);
            }

            /* Adjust body padding to account for footer */
            body {
                padding-bottom: 48px;
            }
        `;
        document.head.appendChild(style);
    }

    createFooter() {
        const footer = document.createElement('footer');
        footer.className = 'unified-footer';
        footer.innerHTML = `
            <div class="footer-left">
                <span>&copy; ${this.currentYear} Cultivate Learning</span>
                <span class="footer-separator">|</span>
                <span>All rights reserved</span>
            </div>
            <div class="footer-right">
                <span>Build ${this.buildNumber}</span>
            </div>
        `;
        document.body.appendChild(footer);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.unifiedFooter = new UnifiedFooter();
    });
} else {
    window.unifiedFooter = new UnifiedFooter();
}