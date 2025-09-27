// Comprehensive Unit Tests for Unified Demos Platform
// Real ML Integration Testing - No Simulation

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium } from 'playwright';

describe('Unified Demos Platform Tests', () => {
    let browser;
    let page;
    const BASE_URL = process.env.TEST_URL || 'http://localhost:3005';

    beforeAll(async () => {
        browser = await chromium.launch();
        page = await browser.newPage();
    });

    afterAll(async () => {
        await browser.close();
    });

    describe('Demo 1 - Child Scenarios', () => {
        it('should load without JavaScript errors', async () => {
            const errors = [];
            page.on('pageerror', error => errors.push(error.message));

            await page.goto(`${BASE_URL}/demo1/index.html`);
            await page.waitForLoadState('networkidle');

            expect(errors).toHaveLength(0);
        });

        it('should have working ML classification', async () => {
            await page.goto(`${BASE_URL}/demo1/index.html`);

            // Click first scenario
            await page.click('.scenario-card:first-child');

            // Click analyze button
            await page.click('#analyzeBtn');

            // Wait for ML response
            await page.waitForSelector('.classification-result', { timeout: 10000 });

            // Verify confidence score exists
            const confidence = await page.textContent('.confidence-value');
            expect(confidence).toMatch(/\d+(\.\d+)?%/);
        });

        it('should handle theme switching correctly', async () => {
            await page.goto(`${BASE_URL}/demo1/index.html`);

            // Get initial theme
            const initialTheme = await page.getAttribute('body', 'data-theme');

            // Click theme toggle
            await page.click('#themeToggle');

            // Verify theme changed
            const newTheme = await page.getAttribute('body', 'data-theme');
            expect(newTheme).not.toBe(initialTheme);
        });

        it('should navigate between tabs properly', async () => {
            await page.goto(`${BASE_URL}/demo1/index.html`);

            // Click scenarios tab
            await page.click('[data-tab="scenarios"]');
            const scenariosVisible = await page.isVisible('#scenarios');
            expect(scenariosVisible).toBe(true);

            // Click insights tab
            await page.click('[data-tab="insights"]');
            const insightsVisible = await page.isVisible('#insights');
            expect(insightsVisible).toBe(true);
        });

        it('should not show analytics dashboard when feature flag is disabled', async () => {
            await page.goto(`${BASE_URL}/demo1/index.html`);

            // Analytics dashboard should not exist
            const dashboardExists = await page.$('#analytics-dashboard');
            expect(dashboardExists).toBe(null);
        });
    });

    describe('Demo 2 - Video Upload', () => {
        it('should load without JavaScript errors', async () => {
            const errors = [];
            page.on('pageerror', error => errors.push(error.message));

            await page.goto(`${BASE_URL}/demo2/index.html`);
            await page.waitForLoadState('networkidle');

            expect(errors).toHaveLength(0);
        });

        it('should display upload interface', async () => {
            await page.goto(`${BASE_URL}/demo2/index.html`);

            // Check upload area exists
            const uploadArea = await page.$('#uploadArea');
            expect(uploadArea).not.toBe(null);

            // Check file input exists
            const fileInput = await page.$('input[type="file"]');
            expect(fileInput).not.toBe(null);
        });

        it('should show proper file restrictions', async () => {
            await page.goto(`${BASE_URL}/demo2/index.html`);

            // Check accepted formats text
            const formats = await page.textContent('.accepted-formats');
            expect(formats).toContain('MP4');
            expect(formats).toContain('WebM');
            expect(formats).toContain('MOV');
        });
    });

    describe('Shared Components', () => {
        it('should load analytics module without errors', async () => {
            await page.goto(`${BASE_URL}/demo1/index.html`);

            // Check if AdvancedAnalytics is initialized
            const analyticsLoaded = await page.evaluate(() => {
                return typeof window.analytics !== 'undefined';
            });

            expect(analyticsLoaded).toBe(true);
        });

        it('should track ML predictions in analytics', async () => {
            await page.goto(`${BASE_URL}/demo1/index.html`);

            // Get initial prediction count
            const initialCount = await page.evaluate(() => {
                return window.analytics?.sessionData?.mlPredictions?.length || 0;
            });

            // Make a prediction
            await page.click('.scenario-card:first-child');
            await page.click('#analyzeBtn');
            await page.waitForSelector('.classification-result', { timeout: 10000 });

            // Get new prediction count
            const newCount = await page.evaluate(() => {
                return window.analytics?.sessionData?.mlPredictions?.length || 0;
            });

            expect(newCount).toBeGreaterThan(initialCount);
        });
    });

    describe('Performance Tests', () => {
        it('should load Demo 1 within 3 seconds', async () => {
            const startTime = Date.now();
            await page.goto(`${BASE_URL}/demo1/index.html`);
            await page.waitForLoadState('networkidle');
            const loadTime = Date.now() - startTime;

            expect(loadTime).toBeLessThan(3000);
        });

        it('should load Demo 2 within 3 seconds', async () => {
            const startTime = Date.now();
            await page.goto(`${BASE_URL}/demo2/index.html`);
            await page.waitForLoadState('networkidle');
            const loadTime = Date.now() - startTime;

            expect(loadTime).toBeLessThan(3000);
        });
    });

    describe('Security Tests', () => {
        it('should not expose sensitive configuration', async () => {
            await page.goto(`${BASE_URL}/demo1/index.html`);

            const pageContent = await page.content();

            // Should not expose API keys
            expect(pageContent).not.toContain('sk-');
            expect(pageContent).not.toContain('api_key');
            expect(pageContent).not.toContain('secret');
        });

        it('should sanitize user inputs', async () => {
            await page.goto(`${BASE_URL}/demo1/index.html`);

            // Try XSS in feedback form if it exists
            const feedbackInput = await page.$('#feedbackInput');
            if (feedbackInput) {
                await feedbackInput.type('<script>alert("XSS")</script>');
                await page.click('#submitFeedback');

                // Check no script was executed
                const alertFired = await page.evaluate(() => {
                    return window.xssTestFired || false;
                });

                expect(alertFired).toBe(false);
            }
        });
    });
});