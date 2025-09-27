/**
 * Professional Theme Management System
 * Elegant dark/light mode toggle with Microsoft Fluent Design
 * Warren's Vision: Professional, elegant, no silly icons
 */

class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        document.body.setAttribute('data-theme', this.theme);
        this.createThemeToggle();
        this.updateToggle();
        this.bindEvents();
    }

    createThemeToggle() {
        // Only create if it doesn't exist
        if (document.getElementById('themeToggle')) return;

        const toggle = document.createElement('div');
        toggle.id = 'themeToggle';
        toggle.className = 'theme-toggle';
        toggle.innerHTML = `
            <svg class="theme-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 18.5C15.59 18.5 18.5 15.59 18.5 12C18.5 8.41 15.59 5.5 12 5.5C8.41 5.5 5.5 8.41 5.5 12C5.5 15.59 8.41 18.5 12 18.5ZM12 2L14.39 5.42C13.65 5.15 12.84 5 12 5C11.16 5 10.35 5.15 9.61 5.42L12 2ZM3.34 7L6.76 4.61C6.15 5.85 5.85 7.15 5.85 8.5C5.85 9.85 6.15 11.15 6.76 12.39L3.34 17ZM3.34 17L6.76 19.39C6.15 18.15 5.85 16.85 5.85 15.5C5.85 14.15 6.15 12.85 6.76 11.61L3.34 7ZM12 22L9.61 18.58C10.35 18.85 11.16 19 12 19C12.84 19 13.65 18.85 14.39 18.58L12 22ZM20.66 17L17.24 19.39C17.85 18.15 18.15 16.85 18.15 15.5C18.15 14.15 17.85 12.85 17.24 11.61L20.66 7ZM20.66 7L17.24 4.61C17.85 5.85 18.15 7.15 18.15 8.5C18.15 9.85 17.85 11.15 17.24 12.39L20.66 17Z"/>
            </svg>
            <div class="theme-toggle-switch" id="themeSwitch">
                <div class="theme-toggle-knob"></div>
            </div>
            <svg class="theme-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M17.75,4.09L15.22,6.03L16.13,9.09L13.5,7.28L10.87,9.09L11.78,6.03L9.25,4.09L12.44,4L13.5,1L14.56,4L17.75,4.09M21.25,11L19.61,12.25L20.2,14.23L18.5,13.06L16.8,14.23L17.39,12.25L15.75,11L17.81,10.95L18.5,9L19.19,10.95L21.25,11M18.97,15.95C19.8,15.87 20.69,17.05 20.16,17.8C19.84,18.25 19.5,18.67 19.08,19.07C15.17,23 8.84,23 4.94,19.07C1.03,15.17 1.03,8.83 4.94,4.93C5.34,4.53 5.76,4.17 6.21,3.85C6.96,3.32 8.14,4.21 8.06,5.04C7.79,7.9 8.75,10.87 10.95,13.06C13.14,15.26 16.1,16.22 18.97,15.95M17.33,17.97C14.5,17.81 11.7,16.64 9.53,14.5C7.36,12.31 6.2,9.5 6.04,6.68C3.23,9.82 3.34,14.4 6.35,17.41C9.37,20.43 14,20.54 17.33,17.97Z"/>
            </svg>
        `;

        document.body.appendChild(toggle);
    }

    toggle() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        document.body.setAttribute('data-theme', this.theme);
        localStorage.setItem('theme', this.theme);
        this.updateToggle();

        console.log(`🎨 Theme switched to: ${this.theme}`);
    }

    updateToggle() {
        const themeSwitch = document.getElementById('themeSwitch');
        if (themeSwitch) {
            if (this.theme === 'dark') {
                themeSwitch.classList.add('active');
            } else {
                themeSwitch.classList.remove('active');
            }
        }
    }

    bindEvents() {
        const toggle = document.getElementById('themeToggle');
        if (toggle) {
            toggle.addEventListener('click', () => {
                this.toggle();
            });
        }
    }
}

// Initialize global theme manager
window.themeManager = new ThemeManager();

console.log('%c🎨 Professional Theme System Loaded', 'font-size: 14px; font-weight: bold; color: #0078D4;');
console.log('✨ Elegant dark/light mode toggle with Microsoft Fluent Design');