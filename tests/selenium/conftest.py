"""
Selenium Test Configuration and Fixtures
Cultivate Learning ML MVP - End-to-End Test Suite

Provides shared test configuration, browser setup, and utility fixtures
for comprehensive demo testing.

Author: Claude (Partner-Level Microsoft SDE)
Test Suite: End-to-End Demo Validation
"""

import os
import json
import time
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
import logging
from pathlib import Path

# Test Configuration
TEST_CONFIG = {
    'base_url': os.getenv('TEST_BASE_URL', 'http://localhost:3000'),
    'api_base_url': os.getenv('TEST_API_BASE_URL', 'http://cultivate-ml-api-pag.westus2.azurecontainer.io:8000'),
    'timeout': int(os.getenv('TEST_TIMEOUT', '30')),
    'analysis_timeout': int(os.getenv('ANALYSIS_TIMEOUT', '120')),  # 2 minutes for ML analysis
    'browser': os.getenv('TEST_BROWSER', 'chrome'),
    'headless': os.getenv('TEST_HEADLESS', 'false').lower() == 'true',
    'screenshot_dir': Path(__file__).parent / 'screenshots',
    'reports_dir': Path(__file__).parent / 'reports'
}

# Create directories
TEST_CONFIG['screenshot_dir'].mkdir(exist_ok=True)
TEST_CONFIG['reports_dir'].mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration to all tests."""
    return TEST_CONFIG

@pytest.fixture(scope="function")
def browser_setup(request, test_config):
    """
    Setup and teardown browser for each test.
    Supports Chrome and Firefox with configurable options.
    """
    browser_name = test_config['browser'].lower()
    driver = None

    try:
        if browser_name == 'chrome':
            chrome_options = Options()
            if test_config['headless']:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

        elif browser_name == 'firefox':
            firefox_options = FirefoxOptions()
            if test_config['headless']:
                firefox_options.add_argument('--headless')
            firefox_options.add_argument('--width=1920')
            firefox_options.add_argument('--height=1080')

            service = FirefoxService(GeckoDriverManager().install())
            driver = webdriver.Firefox(service=service, options=firefox_options)

        else:
            raise ValueError(f"Unsupported browser: {browser_name}")

        # Configure timeouts
        driver.implicitly_wait(test_config['timeout'])
        driver.set_page_load_timeout(test_config['timeout'])

        logger.info(f"Browser setup complete: {browser_name}")
        yield driver

    except Exception as e:
        logger.error(f"Browser setup failed: {str(e)}")
        raise
    finally:
        if driver:
            # Take screenshot on failure
            if request.node.rep_call.failed:
                timestamp = int(time.time())
                screenshot_path = test_config['screenshot_dir'] / f"failure_{timestamp}.png"
                driver.save_screenshot(str(screenshot_path))
                logger.info(f"Failure screenshot saved: {screenshot_path}")

            driver.quit()
            logger.info("Browser teardown complete")

@pytest.fixture(scope="function")
def demo_page(browser_setup, test_config):
    """Navigate to demo application and provide common page utilities."""
    driver = browser_setup

    class DemoPage:
        def __init__(self, driver, config):
            self.driver = driver
            self.config = config
            self.wait = WebDriverWait(driver, config['timeout'])
            self.analysis_wait = WebDriverWait(driver, config['analysis_timeout'])

        def navigate_to_home(self):
            """Navigate to application home page."""
            self.driver.get(self.config['base_url'])
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            return self

        def navigate_to_scenarios(self):
            """Navigate to professional demo scenarios page."""
            self.driver.get(f"{self.config['base_url']}#scenarios")
            # Wait for scenarios to load
            self.wait.until(EC.presence_of_element_located((By.TEXT, "Professional Demo")))
            return self

        def click_professional_demo(self):
            """Click the Professional Demo button on home page."""
            demo_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Professional Demo')]"))
            )
            demo_button.click()
            return self

        def select_scenario(self, scenario_title):
            """Select a specific scenario by title."""
            # Wait for scenario grid to load
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "scenario-grid")))

            # Click on scenario card with matching title
            scenario_card = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//div[contains(@class, 'scenario-card')]//h3[contains(text(), '{scenario_title}')]"))
            )
            scenario_card.click()

            # Wait for scenario preview to load
            self.wait.until(EC.presence_of_element_located((By.TEXT, "Scenario Preview")))
            return self

        def start_analysis(self):
            """Click the Analyze This Scenario button."""
            analyze_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Analyze This Scenario')]"))
            )
            analyze_button.click()
            return self

        def wait_for_analysis_complete(self):
            """Wait for ML analysis to complete and results to load."""
            # Wait for loading state to appear
            self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Analyzing')]"))
            )

            # Wait for analysis to complete (longer timeout)
            self.analysis_wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
            )
            return self

        def get_analysis_results(self):
            """Extract analysis results data from the results page."""
            results = {}

            # Extract CLASS scores
            try:
                class_scores = self.driver.find_elements(By.CLASS_NAME, "class-score")
                results['class_scores'] = {}
                for score_element in class_scores:
                    label = score_element.find_element(By.CLASS_NAME, "score-label").text
                    value = float(score_element.find_element(By.CLASS_NAME, "score-value").text)
                    results['class_scores'][label] = value
            except Exception as e:
                logger.warning(f"Could not extract CLASS scores: {e}")

            # Extract recommendations
            try:
                recommendations = self.driver.find_elements(By.CLASS_NAME, "recommendation-item")
                results['recommendations'] = [rec.text for rec in recommendations]
            except Exception as e:
                logger.warning(f"Could not extract recommendations: {e}")

            # Extract question quality
            try:
                quality_element = self.driver.find_element(By.CLASS_NAME, "question-quality-score")
                results['question_quality'] = float(quality_element.text)
            except Exception as e:
                logger.warning(f"Could not extract question quality: {e}")

            return results

        def take_screenshot(self, name):
            """Take a screenshot with given name."""
            timestamp = int(time.time())
            screenshot_path = self.config['screenshot_dir'] / f"{name}_{timestamp}.png"
            self.driver.save_screenshot(str(screenshot_path))
            logger.info(f"Screenshot saved: {screenshot_path}")
            return screenshot_path

        def check_page_performance(self):
            """Check basic page performance metrics."""
            # Execute JavaScript to get performance data
            perf_data = self.driver.execute_script("""
                return {
                    loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
                    domContentLoaded: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                    firstPaint: performance.getEntriesByType('paint').find(entry => entry.name === 'first-paint')?.startTime || 0
                };
            """)
            return perf_data

    return DemoPage(driver, test_config)

@pytest.fixture(scope="function")
def api_client(test_config):
    """Provide API client for backend validation."""
    import requests

    class APIClient:
        def __init__(self, base_url):
            self.base_url = base_url
            self.session = requests.Session()

        def health_check(self):
            """Check API health endpoint."""
            response = self.session.get(f"{self.base_url}/api/health")
            return response

        def submit_analysis(self, transcript, metadata=None):
            """Submit transcript for analysis."""
            payload = {
                "transcript": transcript,
                "metadata": metadata or {}
            }
            response = self.session.post(f"{self.base_url}/api/analyze/transcript", json=payload)
            return response

        def get_analysis_status(self, analysis_id):
            """Get analysis status by ID."""
            response = self.session.get(f"{self.base_url}/api/analyze/status/{analysis_id}")
            return response

        def get_analysis_results(self, analysis_id):
            """Get analysis results by ID."""
            response = self.session.get(f"{self.base_url}/api/analyze/results/{analysis_id}")
            return response

    return APIClient(test_config['api_base_url'])

# Pytest hooks for reporting
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test results available to fixtures."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)