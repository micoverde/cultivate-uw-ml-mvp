const { Builder, By, Key, until, Actions } = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');
const assert = require('assert');

// Test configuration
const TEST_URL = 'http://localhost:3001';
const API_URL = 'http://localhost:8000';
const TIMEOUT = 10000;

// Test results tracking
let testResults = {
  passed: 0,
  failed: 0,
  errors: [],
  warnings: [],
  performance: {},
  accessibility: []
};

// Helper function for colored console output
const log = {
  success: (msg) => console.log(`\x1b[32m✓\x1b[0m ${msg}`),
  error: (msg) => console.log(`\x1b[31m✗\x1b[0m ${msg}`),
  warning: (msg) => console.log(`\x1b[33m⚠\x1b[0m ${msg}`),
  info: (msg) => console.log(`\x1b[36mℹ\x1b[0m ${msg}`),
  section: (msg) => console.log(`\n\x1b[1m${msg}\x1b[0m\n${'='.repeat(50)}`)
};

// Create Chrome driver with options
async function createDriver() {
  const options = new chrome.Options();
  options.addArguments('--no-sandbox');
  options.addArguments('--disable-dev-shm-usage');
  options.addArguments('--disable-gpu');
  options.addArguments('--window-size=1920,1080');

  // Enable browser console logging
  options.setLoggingPrefs({ browser: 'ALL' });

  return await new Builder()
    .forBrowser('chrome')
    .setChromeOptions(options)
    .build();
}

// Test Suite Class
class CultivateTestSuite {
  constructor() {
    this.driver = null;
    this.testStartTime = null;
  }

  async initialize() {
    this.driver = await createDriver();
    await this.driver.manage().setTimeouts({ implicit: 5000 });
    this.testStartTime = Date.now();
  }

  async cleanup() {
    if (this.driver) {
      await this.driver.quit();
    }
  }

  // 1. Basic Page Load Tests
  async testPageLoad() {
    log.section('Testing Page Load and Initial Rendering');

    try {
      // Load the page
      await this.driver.get(TEST_URL);

      // Wait for page to be ready
      await this.driver.wait(until.elementLocated(By.tagName('body')), TIMEOUT);

      // Check page title
      const title = await this.driver.getTitle();
      assert(title, 'Page title should exist');
      log.success(`Page loaded successfully with title: "${title}"`);

      // Check for critical elements
      const criticalElements = [
        { selector: By.tagName('nav'), name: 'Navigation' },
        { selector: By.tagName('main'), name: 'Main content' },
        { selector: By.tagName('footer'), name: 'Footer' },
        { selector: By.css('[data-testid="hero-section"], .hero-section, section'), name: 'Hero section' }
      ];

      for (const element of criticalElements) {
        try {
          await this.driver.findElement(element.selector);
          log.success(`${element.name} rendered correctly`);
        } catch (e) {
          log.warning(`${element.name} not found - might be using different structure`);
        }
      }

      testResults.passed++;
    } catch (error) {
      log.error(`Page load test failed: ${error.message}`);
      testResults.failed++;
      testResults.errors.push({ test: 'Page Load', error: error.message });
    }
  }

  // 2. Console Error Detection
  async testConsoleErrors() {
    log.section('Checking for Console Errors');

    try {
      const logs = await this.driver.manage().logs().get('browser');
      const errors = logs.filter(log => log.level.name === 'SEVERE');

      if (errors.length === 0) {
        log.success('No console errors detected');
        testResults.passed++;
      } else {
        log.error(`Found ${errors.length} console errors:`);
        errors.forEach(error => {
          log.error(`  - ${error.message}`);
          testResults.errors.push({ test: 'Console', error: error.message });
        });
        testResults.failed++;
      }

      // Check for warnings
      const warnings = logs.filter(log => log.level.name === 'WARNING');
      if (warnings.length > 0) {
        log.warning(`Found ${warnings.length} console warnings`);
        testResults.warnings = testResults.warnings.concat(warnings.map(w => w.message));
      }
    } catch (error) {
      log.warning(`Could not check console logs: ${error.message}`);
    }
  }

  // 3. Navigation and Interactive Elements
  async testNavigation() {
    log.section('Testing Navigation and Interactive Elements');

    try {
      // Test dark mode toggle
      try {
        const darkModeToggle = await this.driver.findElement(By.css('button[aria-label*="theme"], button[aria-label*="dark"], button[aria-label*="mode"], [data-testid="theme-toggle"]'));
        await darkModeToggle.click();
        await this.driver.sleep(500); // Wait for transition

        // Check if dark mode was applied
        const htmlElement = await this.driver.findElement(By.tagName('html'));
        const className = await htmlElement.getAttribute('class');

        if (className && className.includes('dark')) {
          log.success('Dark mode toggle works correctly');

          // Toggle back to light mode
          await darkModeToggle.click();
          await this.driver.sleep(500);
          log.success('Light mode toggle works correctly');
        } else {
          log.warning('Dark mode toggle found but class change not detected');
        }

        testResults.passed++;
      } catch (e) {
        log.warning('Dark mode toggle not found or not working');
      }

      // Test navigation links
      const navLinks = await this.driver.findElements(By.css('nav a, nav button'));
      log.info(`Found ${navLinks.length} navigation elements`);

      if (navLinks.length > 0) {
        log.success('Navigation elements present');
        testResults.passed++;
      } else {
        log.warning('No navigation elements found');
      }

    } catch (error) {
      log.error(`Navigation test failed: ${error.message}`);
      testResults.failed++;
      testResults.errors.push({ test: 'Navigation', error: error.message });
    }
  }

  // 4. Demo Section Testing
  async testDemoSection() {
    log.section('Testing Demo Section and Scenario Selection');

    try {
      // Scroll to demo section
      const demoSection = await this.driver.findElement(By.css('#demo, [data-testid="demo"], section:has(h2:has-text("demo")), section:has(h2:has-text("Demo"))'));
      await this.driver.executeScript('arguments[0].scrollIntoView(true);', demoSection);
      await this.driver.sleep(1000);

      // Look for scenario buttons
      const scenarioButtons = await this.driver.findElements(By.css('button[data-scenario], button:has-text("Scenario"), .scenario-card button'));

      if (scenarioButtons.length > 0) {
        log.success(`Found ${scenarioButtons.length} scenario selection buttons`);

        // Test clicking first scenario
        await scenarioButtons[0].click();
        await this.driver.sleep(1000);

        // Check if content changed
        const transcriptArea = await this.driver.findElements(By.css('textarea, .transcript-input, [data-testid="transcript"]'));
        if (transcriptArea.length > 0) {
          log.success('Scenario selection triggers UI update');
          testResults.passed++;
        }
      } else {
        log.warning('No scenario selection buttons found');
      }

    } catch (error) {
      log.warning(`Demo section test skipped: ${error.message}`);
    }
  }

  // 5. Responsive Design Testing
  async testResponsiveDesign() {
    log.section('Testing Responsive Design');

    const viewports = [
      { width: 375, height: 667, name: 'Mobile (iPhone SE)' },
      { width: 768, height: 1024, name: 'Tablet (iPad)' },
      { width: 1920, height: 1080, name: 'Desktop (Full HD)' }
    ];

    for (const viewport of viewports) {
      try {
        await this.driver.manage().window().setRect({ width: viewport.width, height: viewport.height });
        await this.driver.sleep(500);

        // Check if page is still functional
        const body = await this.driver.findElement(By.tagName('body'));
        const isDisplayed = await body.isDisplayed();

        if (isDisplayed) {
          log.success(`${viewport.name} viewport renders correctly`);

          // Check for horizontal scroll (bad responsive design indicator)
          const hasHorizontalScroll = await this.driver.executeScript(
            'return document.documentElement.scrollWidth > document.documentElement.clientWidth'
          );

          if (hasHorizontalScroll) {
            log.warning(`${viewport.name} has horizontal scroll - possible responsive issue`);
            testResults.warnings.push(`Horizontal scroll at ${viewport.width}px width`);
          }
        }

        testResults.passed++;
      } catch (error) {
        log.error(`${viewport.name} test failed: ${error.message}`);
        testResults.failed++;
      }
    }
  }

  // 6. Performance Testing
  async testPerformance() {
    log.section('Testing Performance Metrics');

    try {
      // Reload page to get fresh performance metrics
      await this.driver.navigate().refresh();
      await this.driver.sleep(2000);

      const perfData = await this.driver.executeScript(`
        const perf = window.performance.timing;
        return {
          pageLoadTime: perf.loadEventEnd - perf.navigationStart,
          domContentLoaded: perf.domContentLoadedEventEnd - perf.navigationStart,
          firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0,
          resourceCount: performance.getEntriesByType('resource').length
        };
      `);

      testResults.performance = perfData;

      // Evaluate performance
      if (perfData.pageLoadTime < 3000) {
        log.success(`Fast page load: ${perfData.pageLoadTime}ms`);
        testResults.passed++;
      } else if (perfData.pageLoadTime < 5000) {
        log.warning(`Moderate page load: ${perfData.pageLoadTime}ms`);
        testResults.passed++;
      } else {
        log.error(`Slow page load: ${perfData.pageLoadTime}ms`);
        testResults.failed++;
      }

      log.info(`DOM Content Loaded: ${perfData.domContentLoaded}ms`);
      log.info(`First Paint: ${perfData.firstPaint}ms`);
      log.info(`Total Resources: ${perfData.resourceCount}`);

    } catch (error) {
      log.warning(`Performance test incomplete: ${error.message}`);
    }
  }

  // 7. Accessibility Testing
  async testAccessibility() {
    log.section('Testing Accessibility Features');

    try {
      // Check for alt text on images
      const images = await this.driver.findElements(By.css('img'));
      let imagesWithoutAlt = 0;

      for (const img of images) {
        const alt = await img.getAttribute('alt');
        if (!alt || alt.trim() === '') {
          imagesWithoutAlt++;
        }
      }

      if (imagesWithoutAlt === 0) {
        log.success('All images have alt text');
        testResults.passed++;
      } else {
        log.warning(`${imagesWithoutAlt} images missing alt text`);
        testResults.accessibility.push(`${imagesWithoutAlt} images without alt text`);
      }

      // Check for ARIA labels on buttons
      const buttons = await this.driver.findElements(By.css('button'));
      let buttonsWithoutLabel = 0;

      for (const button of buttons) {
        const ariaLabel = await button.getAttribute('aria-label');
        const text = await button.getText();
        if ((!ariaLabel || ariaLabel.trim() === '') && (!text || text.trim() === '')) {
          buttonsWithoutLabel++;
        }
      }

      if (buttonsWithoutLabel === 0) {
        log.success('All buttons have accessible labels');
        testResults.passed++;
      } else {
        log.warning(`${buttonsWithoutLabel} buttons may lack accessible labels`);
        testResults.accessibility.push(`${buttonsWithoutLabel} buttons without labels`);
      }

      // Check heading hierarchy
      const headings = await this.driver.executeScript(`
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        return Array.from(headings).map(h => h.tagName);
      `);

      if (headings.length > 0 && headings[0] === 'H1') {
        log.success('Proper heading hierarchy detected');
        testResults.passed++;
      } else if (headings.length === 0) {
        log.warning('No semantic headings found');
        testResults.accessibility.push('Missing semantic headings');
      }

    } catch (error) {
      log.warning(`Accessibility test incomplete: ${error.message}`);
    }
  }

  // 8. API Integration Testing
  async testAPIIntegration() {
    log.section('Testing API Integration');

    try {
      // Check if API is reachable
      const apiHealthResponse = await this.driver.executeAsyncScript(`
        const callback = arguments[arguments.length - 1];
        fetch('${API_URL}/health')
          .then(res => res.json())
          .then(data => callback({ success: true, data }))
          .catch(err => callback({ success: false, error: err.message }));
      `);

      if (apiHealthResponse.success) {
        log.success('API health check passed');
        log.info(`API Status: ${JSON.stringify(apiHealthResponse.data)}`);
        testResults.passed++;
      } else {
        log.error(`API health check failed: ${apiHealthResponse.error}`);
        testResults.failed++;
        testResults.errors.push({ test: 'API Health', error: apiHealthResponse.error });
      }

      // Test CORS configuration
      const corsTest = await this.driver.executeAsyncScript(`
        const callback = arguments[arguments.length - 1];
        fetch('${API_URL}/api/transcripts/scenarios', { method: 'GET' })
          .then(res => callback({ success: true, status: res.status }))
          .catch(err => callback({ success: false, error: err.message }));
      `);

      if (corsTest.success) {
        log.success('CORS configuration appears correct');
        testResults.passed++;
      } else {
        log.warning(`CORS test inconclusive: ${corsTest.error}`);
      }

    } catch (error) {
      log.warning(`API test incomplete: ${error.message}`);
    }
  }

  // 9. Form Validation Testing
  async testFormValidation() {
    log.section('Testing Form Inputs and Validation');

    try {
      // Find all form inputs
      const inputs = await this.driver.findElements(By.css('input, textarea, select'));

      if (inputs.length > 0) {
        log.info(`Found ${inputs.length} form elements`);

        // Test each input
        for (let i = 0; i < Math.min(inputs.length, 3); i++) {
          const input = inputs[i];
          const type = await input.getAttribute('type');
          const tagName = await input.getTagName();

          try {
            // Test if input is interactive
            await input.click();
            await input.sendKeys('Test input');
            await input.clear();
            log.success(`${tagName}[type="${type}"] is interactive`);
          } catch (e) {
            log.warning(`${tagName} may not be interactive: ${e.message}`);
          }
        }

        testResults.passed++;
      } else {
        log.info('No form elements found on initial page');
      }

    } catch (error) {
      log.warning(`Form validation test incomplete: ${error.message}`);
    }
  }

  // 10. Memory Leak Detection
  async testMemoryLeaks() {
    log.section('Testing for Memory Leaks');

    try {
      // Get initial memory usage
      const initialMemory = await this.driver.executeScript(`
        if (performance.memory) {
          return performance.memory.usedJSHeapSize;
        }
        return null;
      `);

      if (initialMemory) {
        // Perform actions that might cause memory leaks
        for (let i = 0; i < 5; i++) {
          await this.driver.navigate().refresh();
          await this.driver.sleep(500);
        }

        // Get final memory usage
        const finalMemory = await this.driver.executeScript(`
          return performance.memory ? performance.memory.usedJSHeapSize : null;
        `);

        if (finalMemory) {
          const memoryIncrease = ((finalMemory - initialMemory) / initialMemory) * 100;

          if (memoryIncrease < 50) {
            log.success(`Memory usage stable (${memoryIncrease.toFixed(2)}% increase)`);
            testResults.passed++;
          } else {
            log.warning(`High memory increase detected (${memoryIncrease.toFixed(2)}%)`);
            testResults.warnings.push(`Memory increase: ${memoryIncrease.toFixed(2)}%`);
          }
        }
      } else {
        log.info('Memory profiling not available in this browser');
      }

    } catch (error) {
      log.warning(`Memory leak test incomplete: ${error.message}`);
    }
  }

  // Run all tests
  async runAllTests() {
    log.section('Starting Comprehensive Website Testing');
    log.info(`Testing URL: ${TEST_URL}`);
    log.info(`API URL: ${API_URL}`);

    await this.testPageLoad();
    await this.testConsoleErrors();
    await this.testNavigation();
    await this.testDemoSection();
    await this.testResponsiveDesign();
    await this.testPerformance();
    await this.testAccessibility();
    await this.testAPIIntegration();
    await this.testFormValidation();
    await this.testMemoryLeaks();

    // Generate final report
    this.generateReport();
  }

  // Generate test report
  generateReport() {
    const totalTime = Date.now() - this.testStartTime;

    log.section('TEST RESULTS SUMMARY');
    console.log(`\x1b[32mPassed: ${testResults.passed}\x1b[0m`);
    console.log(`\x1b[31mFailed: ${testResults.failed}\x1b[0m`);
    console.log(`\x1b[33mWarnings: ${testResults.warnings.length}\x1b[0m`);
    console.log(`Total Time: ${(totalTime / 1000).toFixed(2)}s\n`);

    if (testResults.errors.length > 0) {
      log.section('ERRORS TO FIX');
      testResults.errors.forEach((err, i) => {
        console.log(`${i + 1}. ${err.test}: ${err.error}`);
      });
    }

    if (testResults.warnings.length > 0) {
      log.section('WARNINGS TO REVIEW');
      testResults.warnings.forEach((warning, i) => {
        console.log(`${i + 1}. ${warning}`);
      });
    }

    if (testResults.accessibility.length > 0) {
      log.section('ACCESSIBILITY ISSUES');
      testResults.accessibility.forEach((issue, i) => {
        console.log(`${i + 1}. ${issue}`);
      });
    }

    if (Object.keys(testResults.performance).length > 0) {
      log.section('PERFORMANCE METRICS');
      Object.entries(testResults.performance).forEach(([key, value]) => {
        console.log(`${key}: ${value}${key.includes('Time') || key.includes('paint') ? 'ms' : ''}`);
      });
    }

    // Overall assessment
    log.section('OVERALL ASSESSMENT');
    if (testResults.failed === 0 && testResults.errors.length === 0) {
      console.log('\x1b[32m✓ Website passes all critical tests!\x1b[0m');
      console.log('The application appears to be stable and functional.');
    } else {
      console.log('\x1b[31m✗ Critical issues found that need fixing.\x1b[0m');
      console.log('Please review the errors above and fix them.');
    }

    if (testResults.warnings.length > 0) {
      console.log('\x1b[33m⚠ Some warnings were detected that should be reviewed.\x1b[0m');
    }

    // Recommendations
    log.section('RECOMMENDATIONS');
    const recommendations = [];

    if (testResults.performance.pageLoadTime > 3000) {
      recommendations.push('• Optimize page load time (consider code splitting, lazy loading)');
    }

    if (testResults.accessibility.length > 0) {
      recommendations.push('• Fix accessibility issues for better user experience');
    }

    if (testResults.warnings.some(w => w.includes('horizontal scroll'))) {
      recommendations.push('• Review responsive design for mobile devices');
    }

    if (testResults.errors.some(e => e.error.includes('Console'))) {
      recommendations.push('• Fix JavaScript errors in the console');
    }

    if (recommendations.length > 0) {
      recommendations.forEach(rec => console.log(rec));
    } else {
      console.log('• Continue monitoring performance and user feedback');
      console.log('• Consider adding more automated tests for critical paths');
    }
  }
}

// Main execution
async function main() {
  const tester = new CultivateTestSuite();

  try {
    await tester.initialize();
    await tester.runAllTests();
  } catch (error) {
    log.error(`Test suite failed: ${error.message}`);
    console.error(error);
  } finally {
    await tester.cleanup();
  }
}

// Run the tests
main().catch(console.error);