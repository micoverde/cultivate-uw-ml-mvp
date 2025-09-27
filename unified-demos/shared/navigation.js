/**
 * Unified Navigation Component for Demo Integration
 * Provides consistent navigation across all demos
 */

function createUnifiedNavigation() {
    const nav = document.createElement('div');
    nav.className = 'unified-nav';
    nav.innerHTML = `
        <div class="nav-content">
            <a href="../index.html" class="nav-back">
                ← Back to Demo Hub
            </a>
            <div class="nav-demos">
                <a href="../demo1/index.html" class="nav-link ${getCurrentDemo() === 'demo1' ? 'active' : ''}">
                    🎭 Child Scenarios
                </a>
                <a href="../demo2/index.html" class="nav-link ${getCurrentDemo() === 'demo2' ? 'active' : ''}">
                    📹 Warren's Video
                </a>
            </div>
        </div>
    `;

    // Add navigation styles
    const style = document.createElement('style');
    style.textContent = `
        .unified-nav {
            background: var(--surface-50, #F4F1EE);
            border-bottom: 1px solid var(--surface-300, #DFD9D4);
            padding: 12px 20px;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .nav-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .nav-back {
            color: var(--text-secondary, #424242);
            text-decoration: none;
            font-size: 14px;
            transition: color 0.3s ease;
        }

        .nav-back:hover {
            color: var(--primary-500, #0078D4);
        }

        .nav-demos {
            display: flex;
            gap: 16px;
        }

        .nav-link {
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            color: var(--text-secondary, #424242);
            font-size: 14px;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }

        .nav-link:hover {
            background: var(--surface-200, #E8E3DF);
            color: var(--text-primary, #1F1F1F);
        }

        .nav-link.active {
            background: var(--primary-500, #0078D4);
            color: white;
            border-color: var(--primary-600, #106EBE);
        }

        @media (max-width: 768px) {
            .nav-content {
                flex-direction: column;
                gap: 12px;
            }

            .nav-demos {
                width: 100%;
                justify-content: center;
            }
        }
    `;

    document.head.appendChild(style);
    document.body.insertBefore(nav, document.body.firstChild);
}

function getCurrentDemo() {
    const path = window.location.pathname;
    if (path.includes('/demo1/')) return 'demo1';
    if (path.includes('/demo2/')) return 'demo2';
    return '';
}

// Auto-initialize navigation when script loads
document.addEventListener('DOMContentLoaded', () => {
    createUnifiedNavigation();
    console.log('🧭 Unified navigation initialized');
});