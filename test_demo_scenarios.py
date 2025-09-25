#!/usr/bin/env python3
"""
Comprehensive End-to-End Selenium Test Suite for Cultivate Learning ML MVP

Tests both DEMO 1 (Maya Scenario Response Coaching) and DEMO 2 (Video Upload Analysis)
with comprehensive validation of the complete user journey.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #113 - Maya Scenario Response Coaching System
Feature: Continuous E2E Demo Validation
"""

import time
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    WebDriverException
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DemoTestConfig:
    """Configuration for demo tests"""
    FRONTEND_URL = "http://localhost:3002"
    BACKEND_URL = "http://localhost:8000"
    SELENIUM_TIMEOUT = 30
    ANALYSIS_TIMEOUT = 60
    HEADLESS = os.getenv('HEADLESS', 'false').lower() == 'true'
    SCREENSHOT_DIR = "test_screenshots"

class DemoTestSuite:
    """Comprehensive demo test suite with continuous validation"""

    def __init__(self, config: DemoTestConfig):
        self.config = config
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        self.test_results: List[Dict[str, Any]] = []

        # Create screenshot directory
        os.makedirs(self.config.SCREENSHOT_DIR, exist_ok=True)

    def setup_driver(self) -> webdriver.Chrome:
        """Setup Chrome WebDriver with appropriate options"""
        logger.info("Setting up Chrome WebDriver...")

        chrome_options = Options()
        if self.config.HEADLESS:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.wait = WebDriverWait(self.driver, self.config.SELENIUM_TIMEOUT)
            logger.info("Chrome WebDriver setup successful")
            return self.driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome WebDriver: {e}")
            raise

    def teardown_driver(self):
        """Clean up WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("Chrome WebDriver closed")

    def take_screenshot(self, name: str, description: str = ""):
        """Take screenshot for debugging/documentation"""
        if self.driver:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.SCREENSHOT_DIR}/{timestamp}_{name}.png"
            self.driver.save_screenshot(filename)
            logger.info(f"Screenshot saved: {filename} - {description}")
            return filename
        return None

    def wait_for_element(self, locator: tuple, timeout: int = None) -> Any:
        """Wait for element to be present and visible"""
        timeout = timeout or self.config.SELENIUM_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(EC.visibility_of_element_located(locator))

    def wait_for_element_clickable(self, locator: tuple, timeout: int = None) -> Any:
        """Wait for element to be clickable"""
        timeout = timeout or self.config.SELENIUM_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(EC.element_to_be_clickable(locator))

    def safe_click(self, element, max_attempts: int = 3):
        """Safely click element with retry logic"""
        for attempt in range(max_attempts):
            try:
                # Scroll element into view
                self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                time.sleep(0.5)

                # Try regular click
                element.click()
                return True
            except ElementClickInterceptedException:
                logger.warning(f"Click intercepted, attempt {attempt + 1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    # Try JavaScript click as last resort
                    self.driver.execute_script("arguments[0].click();", element)
                    return True
                time.sleep(1)
            except Exception as e:
                logger.error(f"Click failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(1)
        return False

    # ============================================================================
    # DEMO 1: Maya Scenario Response Coaching System Tests
    # ============================================================================

    def test_maya_scenario_complete_flow(self) -> Dict[str, Any]:
        """Test complete Maya scenario flow from start to coaching results"""
        test_name = "maya_scenario_complete_flow"
        logger.info(f"Starting {test_name}")

        result = {
            "test_name": test_name,
            "status": "failed",
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": [],
            "performance_metrics": {}
        }

        try:
            start_time = time.time()

            # Step 1: Load homepage
            logger.info("Step 1: Loading homepage")
            self.driver.get(self.config.FRONTEND_URL)
            self.take_screenshot("01_homepage", "Homepage loaded")

            # Verify homepage loaded correctly
            try:
                header_title = self.wait_for_element((By.XPATH, "//h1[contains(text(), 'AI-Powered Early Education')]"))
                assert "AI-Powered Early Education" in header_title.text
                result["steps_completed"].append("homepage_loaded")
            except Exception as e:
                result["errors"].append(f"Homepage verification failed: {e}")
                raise

            # Step 2: Navigate to Maya scenario
            logger.info("Step 2: Navigating to Maya scenario")
            maya_card = self.wait_for_element_clickable((By.XPATH, "//div[contains(@class, 'card') and .//h3[contains(text(), 'Maya')]]"))
            self.safe_click(maya_card)
            self.take_screenshot("02_maya_scenario", "Maya scenario page loaded")
            result["steps_completed"].append("maya_scenario_navigation")

            # Step 3: Verify scenario content
            logger.info("Step 3: Verifying scenario content")
            scenario_title = self.wait_for_element((By.XPATH, "//h3[contains(text(), 'Maya\\'s Puzzle Frustration')]"))
            audio_transcript = self.wait_for_element((By.XPATH, "//*[contains(text(), 'This is stupid!')]"))
            context_section = self.wait_for_element((By.XPATH, "//*[contains(text(), 'During free play time')]"))

            assert "Maya's Puzzle Frustration" in scenario_title.text
            assert "This is stupid!" in audio_transcript.text
            result["steps_completed"].append("scenario_content_verified")

            # Step 4: Fill educator response
            logger.info("Step 4: Filling educator response")
            test_response = """Maya, I can see you're feeling frustrated with that puzzle. That's okay - puzzles can be tricky! I noticed you got three pieces to fit perfectly. Would you like to try one more piece together, or would you prefer to take a break and come back to it later?"""

            response_textarea = self.wait_for_element((By.XPATH, "//textarea[contains(@placeholder, 'Type your')]"))
            response_textarea.clear()
            response_textarea.send_keys(test_response)

            # Wait for validation
            time.sleep(2)
            self.take_screenshot("03_response_entered", f"Response entered: {len(test_response)} characters")
            result["steps_completed"].append("educator_response_entered")

            # Step 5: Submit for analysis
            logger.info("Step 5: Submitting for analysis")
            submit_button = self.wait_for_element_clickable((By.XPATH, "//button[contains(text(), 'Get AI Coaching Feedback')]"))
            self.safe_click(submit_button)

            # Wait for analysis to start
            analysis_progress = self.wait_for_element((By.XPATH, "//*[contains(text(), 'Analysis in Progress') or contains(text(), 'Analyzing')]"), timeout=10)
            self.take_screenshot("04_analysis_started", "Analysis started")
            result["steps_completed"].append("analysis_submitted")

            # Step 6: Wait for analysis completion
            logger.info("Step 6: Waiting for analysis completion")
            analysis_start_time = time.time()

            # Wait for results page or overall score
            try:
                results_indicator = self.wait_for_element(
                    (By.XPATH, "//*[contains(text(), 'Overall Coaching Score') or contains(text(), 'Your Coaching Feedback')]"),
                    timeout=self.config.ANALYSIS_TIMEOUT
                )
                analysis_end_time = time.time()
                processing_time = analysis_end_time - analysis_start_time
                result["performance_metrics"]["analysis_time"] = processing_time

                self.take_screenshot("05_results_loaded", f"Results loaded in {processing_time:.2f}s")
                result["steps_completed"].append("analysis_completed")

            except TimeoutException:
                result["errors"].append(f"Analysis timeout after {self.config.ANALYSIS_TIMEOUT}s")
                raise

            # Step 7: Validate results content
            logger.info("Step 7: Validating results content")

            # Check for overall score
            overall_score = self.driver.find_element(By.XPATH, "//*[contains(text(), '/10') or contains(@class, 'score')]")

            # Check for category scores
            categories = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Emotional Support') or contains(text(), 'Scaffolding')]")
            assert len(categories) > 0, "No category analysis found"

            # Check for strengths and growth opportunities
            strengths = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Strengths') or contains(text(), 'strength')]")
            growth = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Growth') or contains(text(), 'opportunity')]")

            self.take_screenshot("06_results_validated", "Results content validated")
            result["steps_completed"].append("results_validated")

            # Step 8: Test "Try Another Scenario" functionality
            logger.info("Step 8: Testing navigation back")
            try_another_button = self.wait_for_element_clickable((By.XPATH, "//button[contains(text(), 'Try Another') or contains(text(), 'Start')]"))
            self.safe_click(try_another_button)

            # Verify we're back to scenario input
            scenario_input = self.wait_for_element((By.XPATH, "//textarea[contains(@placeholder, 'Type your')]"))
            result["steps_completed"].append("navigation_tested")

            end_time = time.time()
            result["performance_metrics"]["total_time"] = end_time - start_time
            result["status"] = "passed"
            result["end_time"] = datetime.now().isoformat()

            logger.info(f"✅ {test_name} PASSED in {end_time - start_time:.2f}s")

        except Exception as e:
            result["errors"].append(str(e))
            result["end_time"] = datetime.now().isoformat()
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.take_screenshot("error_maya_scenario", f"Error: {e}")

        self.test_results.append(result)
        return result

    def test_maya_scenario_input_validation(self) -> Dict[str, Any]:
        """Test input validation for Maya scenario"""
        test_name = "maya_scenario_input_validation"
        logger.info(f"Starting {test_name}")

        result = {
            "test_name": test_name,
            "status": "failed",
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": []
        }

        try:
            # Navigate to Maya scenario
            self.driver.get(f"{self.config.FRONTEND_URL}")
            maya_card = self.wait_for_element_clickable((By.XPATH, "//div[contains(@class, 'card') and .//h3[contains(text(), 'Maya')]]"))
            self.safe_click(maya_card)

            # Test minimum character validation
            response_textarea = self.wait_for_element((By.XPATH, "//textarea[contains(@placeholder, 'Type your')]"))
            submit_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Get AI Coaching Feedback')]")

            # Test empty input
            response_textarea.clear()
            assert not submit_button.is_enabled(), "Submit button should be disabled for empty input"
            result["steps_completed"].append("empty_input_validation")

            # Test short input
            response_textarea.send_keys("Too short")
            time.sleep(1)
            assert not submit_button.is_enabled(), "Submit button should be disabled for short input"
            result["steps_completed"].append("short_input_validation")

            # Test valid input
            response_textarea.clear()
            valid_response = "This is a valid response that meets the minimum character requirement for the Maya scenario analysis."
            response_textarea.send_keys(valid_response)
            time.sleep(1)
            assert submit_button.is_enabled(), "Submit button should be enabled for valid input"
            result["steps_completed"].append("valid_input_validation")

            result["status"] = "passed"
            result["end_time"] = datetime.now().isoformat()
            logger.info(f"✅ {test_name} PASSED")

        except Exception as e:
            result["errors"].append(str(e))
            result["end_time"] = datetime.now().isoformat()
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.take_screenshot("error_input_validation", f"Error: {e}")

        self.test_results.append(result)
        return result

    # ============================================================================
    # DEMO 2: Video Upload Analysis System Tests
    # ============================================================================

    def test_video_upload_demo_flow(self) -> Dict[str, Any]:
        """Test video upload demo flow (DEMO 2)"""
        test_name = "video_upload_demo_flow"
        logger.info(f"Starting {test_name}")

        result = {
            "test_name": test_name,
            "status": "failed",
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": []
        }

        try:
            # Navigate to demo page
            self.driver.get(self.config.FRONTEND_URL)

            # Look for demo/upload functionality
            try:
                demo_button = self.wait_for_element_clickable((By.XPATH, "//button[contains(text(), 'Demo') or contains(text(), 'Upload')]"))
                self.safe_click(demo_button)
                result["steps_completed"].append("demo_navigation")

                # Check for file upload or transcript input
                upload_elements = self.driver.find_elements(By.XPATH, "//*[@type='file' or @accept or contains(@class, 'upload') or contains(text(), 'upload')]")
                transcript_elements = self.driver.find_elements(By.XPATH, "//textarea[contains(@placeholder, 'transcript')]")

                if upload_elements:
                    result["steps_completed"].append("upload_interface_found")
                elif transcript_elements:
                    result["steps_completed"].append("transcript_interface_found")
                else:
                    result["errors"].append("No upload or transcript interface found")

                self.take_screenshot("demo_upload_page", "Demo upload page")

            except TimeoutException:
                result["errors"].append("Demo/upload button not found - feature may not be implemented yet")

            result["status"] = "passed" if not result["errors"] else "failed"
            result["end_time"] = datetime.now().isoformat()

            if result["status"] == "passed":
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.warning(f"⚠️ {test_name} INCOMPLETE - Video upload demo not fully implemented")

        except Exception as e:
            result["errors"].append(str(e))
            result["end_time"] = datetime.now().isoformat()
            logger.error(f"❌ {test_name} FAILED: {e}")

        self.test_results.append(result)
        return result

    # ============================================================================
    # Performance and Load Tests
    # ============================================================================

    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks for demo scenarios"""
        test_name = "performance_benchmarks"
        logger.info(f"Starting {test_name}")

        result = {
            "test_name": test_name,
            "status": "failed",
            "start_time": datetime.now().isoformat(),
            "performance_metrics": {},
            "errors": []
        }

        try:
            # Test page load time
            start_time = time.time()
            self.driver.get(self.config.FRONTEND_URL)
            self.wait_for_element((By.XPATH, "//h1"))
            page_load_time = time.time() - start_time
            result["performance_metrics"]["page_load_time"] = page_load_time

            # Test navigation responsiveness
            start_time = time.time()
            maya_card = self.wait_for_element_clickable((By.XPATH, "//div[contains(@class, 'card') and .//h3[contains(text(), 'Maya')]]"))
            self.safe_click(maya_card)
            self.wait_for_element((By.XPATH, "//h3[contains(text(), 'Maya\\'s Puzzle Frustration')]"))
            navigation_time = time.time() - start_time
            result["performance_metrics"]["navigation_time"] = navigation_time

            # Performance benchmarks
            assert page_load_time < 5.0, f"Page load too slow: {page_load_time:.2f}s"
            assert navigation_time < 2.0, f"Navigation too slow: {navigation_time:.2f}s"

            result["status"] = "passed"
            result["end_time"] = datetime.now().isoformat()
            logger.info(f"✅ {test_name} PASSED - Page load: {page_load_time:.2f}s, Navigation: {navigation_time:.2f}s")

        except Exception as e:
            result["errors"].append(str(e))
            result["end_time"] = datetime.now().isoformat()
            logger.error(f"❌ {test_name} FAILED: {e}")

        self.test_results.append(result)
        return result

    # ============================================================================
    # Test Suite Execution
    # ============================================================================

    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("🚀 Starting Comprehensive Demo Test Suite")

        suite_start_time = time.time()
        suite_result = {
            "suite_name": "cultivate_learning_demo_tests",
            "start_time": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_duration": 0,
            "detailed_results": []
        }

        # Setup driver
        try:
            self.setup_driver()
        except Exception as e:
            logger.error(f"Failed to setup driver: {e}")
            return {"error": "Driver setup failed", "details": str(e)}

        # Test suite
        test_functions = [
            self.test_maya_scenario_complete_flow,
            self.test_maya_scenario_input_validation,
            self.test_video_upload_demo_flow,
            self.test_performance_benchmarks
        ]

        try:
            for test_func in test_functions:
                logger.info(f"Running {test_func.__name__}")
                test_result = test_func()
                suite_result["detailed_results"].append(test_result)
                suite_result["tests_run"] += 1

                if test_result["status"] == "passed":
                    suite_result["tests_passed"] += 1
                else:
                    suite_result["tests_failed"] += 1

                # Brief pause between tests
                time.sleep(2)

        finally:
            self.teardown_driver()

        suite_end_time = time.time()
        suite_result["total_duration"] = suite_end_time - suite_start_time
        suite_result["end_time"] = datetime.now().isoformat()

        # Generate summary
        logger.info("=" * 80)
        logger.info("📊 TEST SUITE SUMMARY")
        logger.info(f"Tests Run: {suite_result['tests_run']}")
        logger.info(f"Tests Passed: {suite_result['tests_passed']}")
        logger.info(f"Tests Failed: {suite_result['tests_failed']}")
        logger.info(f"Success Rate: {(suite_result['tests_passed'] / suite_result['tests_run']) * 100:.1f}%")
        logger.info(f"Total Duration: {suite_result['total_duration']:.2f}s")
        logger.info("=" * 80)

        return suite_result

def main():
    """Main execution function"""
    config = DemoTestConfig()
    suite = DemoTestSuite(config)

    try:
        results = suite.run_all_tests()

        # Save results to file
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"📄 Test results saved to: {results_file}")

        # Exit with error code if any tests failed
        if results.get("tests_failed", 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()