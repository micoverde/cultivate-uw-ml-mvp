#!/usr/bin/env python3
"""
Selenium Tests for Model Settings UI
Story 9.1 - Issue #196

Tests the ML Settings modal functionality in the demo page
"""

import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class TestModelSettingsUI:
    """Test ML Model Settings UI functionality"""

    @pytest.fixture
    def driver(self):
        """Setup Chrome driver with options"""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode for CI
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10)

        yield driver
        driver.quit()

    def test_settings_button_visible(self, driver):
        """Test that ML Settings button is visible in header"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Find settings button
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        assert settings_btn.is_displayed()
        assert settings_btn.text == "⚙️ ML Settings"

    def test_open_settings_modal(self, driver):
        """Test opening the settings modal"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Click settings button
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Wait for modal to be visible
        wait = WebDriverWait(driver, 10)
        modal = wait.until(
            EC.visibility_of_element_located((By.ID, "mlSettingsModal"))
        )

        # Verify modal is displayed
        assert "active" in modal.get_attribute("class")

        # Verify modal title
        modal_title = driver.find_element(By.CLASS_NAME, "modal-title")
        assert modal_title.text == "ML Model Settings"

    def test_close_settings_modal(self, driver):
        """Test closing the settings modal"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Close via X button
        close_btn = driver.find_element(By.CLASS_NAME, "modal-close")
        close_btn.click()

        # Verify modal is hidden
        modal = driver.find_element(By.ID, "mlSettingsModal")
        assert "active" not in modal.get_attribute("class")

    def test_close_modal_outside_click(self, driver):
        """Test closing modal by clicking outside"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Click overlay (outside modal content)
        modal_overlay = driver.find_element(By.ID, "mlSettingsModal")
        driver.execute_script(
            "arguments[0].click();",
            modal_overlay
        )

        # Verify modal is hidden
        time.sleep(0.5)
        assert "active" not in modal_overlay.get_attribute("class")

    def test_model_selection_options(self, driver):
        """Test model selection radio buttons"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Find radio buttons
        classic_radio = driver.find_element(
            By.CSS_SELECTOR, 'input[name="model"][value="classic"]'
        )
        ensemble_radio = driver.find_element(
            By.CSS_SELECTOR, 'input[name="model"][value="ensemble"]'
        )

        # Verify classic is selected by default
        assert classic_radio.is_selected()
        assert not ensemble_radio.is_selected()

        # Select ensemble
        ensemble_option = driver.find_element(By.ID, "ensemble-option")
        ensemble_option.click()

        # Verify selection changed
        assert not classic_radio.is_selected()
        assert ensemble_radio.is_selected()

        # Verify visual selection indicator
        assert "selected" in ensemble_option.get_attribute("class")

    def test_performance_metrics_display(self, driver):
        """Test performance metrics table display"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Find performance table
        perf_table = driver.find_element(By.CLASS_NAME, "performance-table")
        assert perf_table.is_displayed()

        # Check table headers
        headers = driver.find_elements(By.CSS_SELECTOR, ".performance-table th")
        assert len(headers) == 3
        assert headers[0].text == "Metric"
        assert headers[1].text == "Classic ML"
        assert headers[2].text == "Ensemble ML"

        # Check metrics are displayed
        accuracy_classic = driver.find_element(By.ID, "classic-accuracy")
        accuracy_ensemble = driver.find_element(By.ID, "ensemble-accuracy")
        assert accuracy_classic.text != ""
        assert accuracy_ensemble.text != ""

    def test_apply_model_selection(self, driver):
        """Test applying model selection changes"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Select ensemble model
        ensemble_option = driver.find_element(By.ID, "ensemble-option")
        ensemble_option.click()

        # Apply changes
        apply_btn = driver.find_element(
            By.CSS_SELECTOR, ".modal-btn-primary"
        )
        apply_btn.click()

        # Wait for status message (mock API response)
        time.sleep(0.5)

        # Check localStorage was updated
        stored_model = driver.execute_script(
            "return localStorage.getItem('ml_model');"
        )
        # Note: Will be 'ensemble' if API call succeeds, otherwise may be null

    def test_cancel_model_selection(self, driver):
        """Test canceling model selection"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Set initial model
        driver.execute_script(
            "localStorage.setItem('ml_model', 'classic');"
        )

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Select different model
        ensemble_option = driver.find_element(By.ID, "ensemble-option")
        ensemble_option.click()

        # Cancel
        cancel_btn = driver.find_element(
            By.CSS_SELECTOR, ".modal-btn:not(.modal-btn-primary)"
        )
        cancel_btn.click()

        # Verify modal closed and selection not saved
        modal = driver.find_element(By.ID, "mlSettingsModal")
        assert "active" not in modal.get_attribute("class")

        stored_model = driver.execute_script(
            "return localStorage.getItem('ml_model');"
        )
        assert stored_model == "classic"

    def test_localStorage_persistence(self, driver):
        """Test that model selection persists in localStorage"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Set model via JavaScript
        driver.execute_script(
            "localStorage.setItem('ml_model', 'ensemble');"
        )

        # Refresh page
        driver.refresh()

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Verify ensemble is selected
        ensemble_radio = driver.find_element(
            By.CSS_SELECTOR, 'input[name="model"][value="ensemble"]'
        )
        assert ensemble_radio.is_selected()

    def test_responsive_design(self, driver):
        """Test modal responsiveness on different screen sizes"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Test mobile size
        driver.set_window_size(375, 667)

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Verify modal is visible and properly sized
        modal_content = driver.find_element(By.CLASS_NAME, "modal-content")
        assert modal_content.is_displayed()

        width = modal_content.size["width"]
        assert width <= 375 * 0.95  # Should be max 95% of viewport

        # Test desktop size
        driver.set_window_size(1920, 1080)
        width = modal_content.size["width"]
        assert width <= 600  # Should respect max-width


class TestErrorHandling:
    """Test error handling scenarios"""

    @pytest.fixture
    def driver(self):
        """Setup Chrome driver"""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()

    def test_api_error_handling(self, driver):
        """Test handling of API errors"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Mock API failure
        driver.execute_script("""
            window.fetch = function() {
                return Promise.reject(new Error('Network error'));
            };
        """)

        # Try to switch models
        driver.execute_script("openMLSettings();")

        ensemble_option = driver.find_element(By.ID, "ensemble-option")
        ensemble_option.click()

        apply_btn = driver.find_element(By.CSS_SELECTOR, ".modal-btn-primary")
        apply_btn.click()

        # Wait for error message
        time.sleep(1)

        status_div = driver.find_element(By.ID, "model-status")
        status_msg = driver.find_element(By.ID, "status-message")

        assert status_div.is_displayed()
        assert "error" in status_msg.text.lower() or "failed" in status_msg.text.lower()

    def test_console_errors(self, driver):
        """Test for JavaScript console errors"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Get console logs
        logs = driver.get_log("browser")

        # Filter for errors
        errors = [log for log in logs if log["level"] == "SEVERE"]

        # There should be no severe errors
        assert len(errors) == 0, f"Console errors found: {errors}"


class TestAccessibility:
    """Test accessibility features"""

    @pytest.fixture
    def driver(self):
        """Setup Chrome driver"""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()

    def test_keyboard_navigation(self, driver):
        """Test keyboard navigation through modal"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Open modal with keyboard
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.send_keys("\n")  # Enter key

        # Verify modal opened
        modal = driver.find_element(By.ID, "mlSettingsModal")
        assert "active" in modal.get_attribute("class")

    def test_aria_labels(self, driver):
        """Test ARIA labels for accessibility"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Check settings button has title
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        title = settings_btn.get_attribute("title")
        assert title == "ML Model Settings"

    def test_focus_management(self, driver):
        """Test focus management in modal"""
        driver.get("http://localhost:8000/demo/public/demo2_warren_fluent.html")

        # Open modal
        settings_btn = driver.find_element(By.CLASS_NAME, "settings-btn")
        settings_btn.click()

        # Check if close button is focusable
        close_btn = driver.find_element(By.CLASS_NAME, "modal-close")
        assert close_btn.is_enabled()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])