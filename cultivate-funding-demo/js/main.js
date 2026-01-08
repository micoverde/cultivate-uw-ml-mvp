/**
 * Cultivate Learning - Main JavaScript
 * General site functionality
 */

document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Animate pipeline nodes on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe pipeline nodes
    document.querySelectorAll('.pipeline-node').forEach(node => {
        observer.observe(node);
    });

    // Observe stat values
    document.querySelectorAll('.stat-value, .impact-value').forEach(stat => {
        observer.observe(stat);
    });

    // Mobile navigation toggle (if implemented)
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
    }

    // Add animation class to elements
    document.querySelectorAll('.feature, .problem-card, .impact-card, .tech-category').forEach(el => {
        observer.observe(el);
    });

    // Console welcome message
    console.log('%c🌱 Cultivate Learning ML Pipeline', 'font-size: 20px; font-weight: bold; color: #0078D4;');
    console.log('%cAI-powered educator support system', 'font-size: 14px; color: #616161;');
    console.log('%cLearn more at: https://cultivatelearning.uw.edu', 'font-size: 12px; color: #107C10;');
});

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    .pipeline-node,
    .feature,
    .problem-card,
    .impact-card,
    .tech-category,
    .stat-value,
    .impact-value {
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.6s ease, transform 0.6s ease;
    }

    .animate-in {
        opacity: 1;
        transform: translateY(0);
    }

    .pipeline-node:nth-child(1) { transition-delay: 0.1s; }
    .pipeline-node:nth-child(2) { transition-delay: 0.2s; }
    .pipeline-node:nth-child(3) { transition-delay: 0.3s; }
    .pipeline-node:nth-child(4) { transition-delay: 0.4s; }
    .pipeline-node:nth-child(5) { transition-delay: 0.5s; }
`;
document.head.appendChild(style);
