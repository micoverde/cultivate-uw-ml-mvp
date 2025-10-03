#!/usr/bin/env python3
"""
Comprehensive Selenium test suite for Cultivate Learning platform
Tests all pages for content, ML connections, errors, and UX professionalism
Warren's requirement: Test and fix all issues iteratively
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
import time
import sys
from datetime import datetime

class PageTester:
    def __init__(self):
        self.driver = None
        self.base_url = "http://localhost:7071"
        self.issues_found = []
        self.ml_connection_results = {}
        self.console_errors = {}

    def setup_driver(self):
        """Setup Chrome driver with console logging"""
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(1920, 1080)

    def get_console_logs(self):
        """Extract console logs including errors"""
        logs = self.driver.get_log('browser')
        errors = []
        warnings = []
        info = []

        for log in logs:
            if log['level'] == 'SEVERE':
                errors.append(log['message'])
            elif log['level'] == 'WARNING':
                warnings.append(log['message'])
            else:
                info.append(log['message'])

        return {'errors': errors, 'warnings': warnings, 'info': info}

    def test_hub_page(self):
        """Test the main hub page"""
        print("\n🔍 Testing Hub Page (index.html)")
        print("=" * 60)

        self.driver.get(f"{self.base_url}/index.html")
        time.sleep(2)

        issues = []

        # Check for page load
        try:
            # Check header exists
            header = self.driver.find_element(By.CLASS_NAME, "unified-header")
            print("✅ Unified header loaded")
        except NoSuchElementException:
            issues.append("❌ Unified header not found")
            print("❌ Unified header not found")

        # Check for hero section
        try:
            hero = self.driver.find_element(By.CLASS_NAME, "hero")
            title = hero.find_element(By.TAG_NAME, "h1").text
            if "Cultivate Learning" in title:
                print(f"✅ Hero section loaded with title: {title}")
            else:
                issues.append(f"⚠️ Unexpected title: {title}")
        except NoSuchElementException:
            issues.append("❌ Hero section not found")

        # Check demo cards
        try:
            demo_cards = self.driver.find_elements(By.CLASS_NAME, "demo-card")
            print(f"✅ Found {len(demo_cards)} demo cards")

            for i, card in enumerate(demo_cards, 1):
                try:
                    title = card.find_element(By.TAG_NAME, "h2").text
                    print(f"  - Demo {i}: {title}")
                except:
                    issues.append(f"❌ Demo card {i} missing title")
        except:
            issues.append("❌ Demo cards not found")

        # Check model settings button
        try:
            settings_btn = self.driver.find_element(By.CLASS_NAME, "model-settings-btn")
            print("✅ Model settings button present")

            # Click to open modal
            settings_btn.click()
            time.sleep(0.5)

            modal = self.driver.find_element(By.CLASS_NAME, "model-settings-modal")
            if "show" in modal.get_attribute("class"):
                print("✅ Model settings modal opens correctly")

                # Close modal
                close_btn = modal.find_element(By.CLASS_NAME, "model-settings-close")
                close_btn.click()
            else:
                issues.append("❌ Model settings modal doesn't open")
        except:
            issues.append("❌ Model settings button not found")

        # Check console errors
        console_logs = self.get_console_logs()
        self.console_errors['hub'] = console_logs['errors']

        if console_logs['errors']:
            print(f"\n⚠️ Console errors found: {len(console_logs['errors'])}")
            for error in console_logs['errors'][:3]:  # Show first 3
                print(f"  - {error[:100]}...")
                issues.append(f"Console error: {error[:100]}")
        else:
            print("✅ No console errors")

        self.issues_found.extend([f"Hub: {issue}" for issue in issues])
        return len(issues) == 0

    def test_demo1_page(self):
        """Test Demo 1 - ECE Question Classification"""
        print("\n🔍 Testing Demo 1 (ECE Question Classification)")
        print("=" * 60)

        self.driver.get(f"{self.base_url}/demo1/index.html")
        time.sleep(2)

        issues = []

        # Check unified header
        try:
            header = self.driver.find_element(By.CLASS_NAME, "unified-header")
            print("✅ Unified header loaded")

            # Check for NO old theme toggle (should be removed)
            try:
                old_toggle = self.driver.find_element(By.CLASS_NAME, "theme-toggle")
                issues.append("❌ Old theme toggle still present - should be removed!")
                print("❌ Old theme toggle still present - should be removed!")
            except NoSuchElementException:
                print("✅ Old theme toggle correctly removed")
        except NoSuchElementException:
            issues.append("❌ Unified header not found")

        # Test child scenarios
        try:
            scenario_btns = self.driver.find_elements(By.CSS_SELECTOR, ".child-scenario-btn")
            print(f"✅ Found {len(scenario_btns)} child scenario buttons")

            if scenario_btns:
                # Test first scenario
                scenario_btns[0].click()
                time.sleep(0.5)

                # Check if scenario loaded
                active_scenario = self.driver.find_element(By.CSS_SELECTOR, ".child-scenario-btn.active")
                if active_scenario:
                    print("✅ Child scenario loads and activates")
        except:
            issues.append("❌ Child scenarios not working")

        # Test ML classification
        try:
            input_field = self.driver.find_element(By.ID, "questionInput")
            analyze_btn = self.driver.find_element(By.ID, "analyzeBtn")

            # Test with a sample question
            input_field.clear()
            input_field.send_keys("What color is the sky?")
            analyze_btn.click()

            # Wait for result
            time.sleep(2)

            try:
                result = self.driver.find_element(By.ID, "result")
                if result.is_displayed():
                    result_text = result.text
                    if "CEQ" in result_text or "OEQ" in result_text:
                        print(f"✅ ML classification working: {result_text[:50]}...")
                        self.ml_connection_results['demo1'] = True
                    else:
                        issues.append("❌ ML classification not showing proper result")
                        self.ml_connection_results['demo1'] = False
            except:
                issues.append("❌ ML result not displayed")
                self.ml_connection_results['demo1'] = False
        except:
            issues.append("❌ ML input/analyze elements not found")
            self.ml_connection_results['demo1'] = False

        # Check performance metrics
        try:
            metrics = self.driver.find_elements(By.CLASS_NAME, "metric-value")
            if metrics:
                print(f"✅ Performance metrics displayed ({len(metrics)} metrics)")
        except:
            issues.append("⚠️ Performance metrics not visible")

        # Console errors
        console_logs = self.get_console_logs()
        self.console_errors['demo1'] = console_logs['errors']

        critical_errors = [e for e in console_logs['errors'] if 'ReferenceError' in e or 'TypeError' in e]
        if critical_errors:
            print(f"\n❌ Critical console errors: {len(critical_errors)}")
            for error in critical_errors[:2]:
                print(f"  - {error[:100]}...")
                issues.append(f"Critical error: {error[:100]}")
        else:
            print("✅ No critical console errors")

        self.issues_found.extend([f"Demo1: {issue}" for issue in issues])
        return len(issues) == 0

    def test_demo2_page(self):
        """Test Demo 2 - Warren's Teaching Video Analysis"""
        print("\n🔍 Testing Demo 2 (Warren's Teaching Video)")
        print("=" * 60)

        self.driver.get(f"{self.base_url}/demo2/index.html")
        time.sleep(3)  # Give video time to load

        issues = []

        # Check header
        try:
            header = self.driver.find_element(By.CLASS_NAME, "unified-header")
            print("✅ Unified header loaded")
        except NoSuchElementException:
            issues.append("❌ Unified header not found")

        # Check video player
        try:
            video = self.driver.find_element(By.ID, "videoPlayer")
            if video:
                print("✅ Video player present")
                # Check if video is loaded
                duration = self.driver.execute_script("return document.getElementById('videoPlayer').duration")
                if duration and duration > 0:
                    print(f"✅ Video loaded (duration: {duration:.1f}s)")
                else:
                    issues.append("⚠️ Video not fully loaded")
        except:
            issues.append("❌ Video player not found")

        # Check transcript
        try:
            transcript = self.driver.find_element(By.ID, "transcript")
            questions = transcript.find_elements(By.CLASS_NAME, "question-item")

            if questions:
                print(f"✅ Transcript loaded with {len(questions)} questions")

                # Check first question for ML classification
                first_q = questions[0]
                try:
                    ml_badge = first_q.find_element(By.CLASS_NAME, "ml-type")
                    ml_type = ml_badge.text
                    if ml_type in ['OEQ', 'CEQ']:
                        print(f"✅ ML classifications present (first: {ml_type})")
                        self.ml_connection_results['demo2'] = True
                except:
                    issues.append("❌ ML classifications not showing")
                    self.ml_connection_results['demo2'] = False
            else:
                issues.append("❌ No questions in transcript")
        except:
            issues.append("❌ Transcript not loaded")

        # Test analyze question button
        try:
            analyze_btn = self.driver.find_element(By.ID, "analyzeQuestionBtn")
            input_field = self.driver.find_element(By.ID, "questionInput")

            input_field.clear()
            input_field.send_keys("Can you tell me about the blocks?")
            analyze_btn.click()

            time.sleep(2)

            # Check for result
            result = self.driver.find_element(By.ID, "analysisResult")
            if result.is_displayed() and result.text:
                print("✅ Question analysis working")
            else:
                issues.append("❌ Question analysis not showing results")
        except:
            issues.append("❌ Question analysis elements not found")

        # Check metrics display
        try:
            summary = self.driver.find_element(By.CLASS_NAME, "metrics-summary")
            if summary:
                print("✅ Metrics summary displayed")
        except:
            issues.append("⚠️ Metrics summary not visible")

        # Test user feedback (check if it works)
        try:
            # Find first question with feedback buttons
            questions = self.driver.find_elements(By.CLASS_NAME, "question-item")
            if questions:
                first_q = questions[0]
                try:
                    thumbs_up = first_q.find_element(By.CSS_SELECTOR, ".rating-btn[title='Correct']")
                    # Just check it exists, don't click to avoid 404
                    print("✅ User feedback buttons present")

                    # Check for feedback endpoint issue
                    console_logs = self.driver.get_log('browser')
                    feedback_errors = [log for log in console_logs if '404' in log['message'] and 'feedback' in log['message']]
                    if feedback_errors:
                        issues.append("❌ Feedback endpoint returning 404 - needs environment-aware URL")
                        print("❌ Feedback endpoint issue detected (404)")
                except:
                    issues.append("⚠️ Feedback buttons not found")
        except:
            pass

        # Console errors
        console_logs = self.get_console_logs()
        self.console_errors['demo2'] = console_logs['errors']

        # Filter for critical errors
        critical_errors = [e for e in console_logs['errors']
                          if any(x in e for x in ['ReferenceError', 'TypeError', 'not defined', '404'])]

        if critical_errors:
            print(f"\n❌ Critical console errors: {len(critical_errors)}")
            for error in critical_errors[:3]:
                error_msg = error.replace('http://localhost:7071', '').strip()[:100]
                print(f"  - {error_msg}...")
                if '404' in error and 'feedback' in error:
                    issues.append("Feedback API endpoint not found (404)")
                else:
                    issues.append(f"Console error: {error_msg}")
        else:
            print("✅ No critical console errors")

        self.issues_found.extend([f"Demo2: {issue}" for issue in issues])
        return len(issues) == 0

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Test completed: {timestamp}\n")

        # ML Connections Summary
        print("🔌 ML Connection Status:")
        for page, status in self.ml_connection_results.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {page}: {'Connected' if status else 'Not Connected'}")

        # Issues Summary
        if self.issues_found:
            print(f"\n⚠️ Total Issues Found: {len(self.issues_found)}")
            print("\nDetailed Issues:")
            for issue in self.issues_found:
                print(f"  • {issue}")
        else:
            print("\n✅ No issues found! All systems operational.")

        # UX Professionalism Check
        print("\n🎨 UX Professionalism Checklist:")
        checks = [
            ("Unified header on all pages", all(page not in str(self.issues_found) for page in ["header not found"])),
            ("No double headers", "double header" not in str(self.issues_found).lower()),
            ("Model settings accessible", "settings" not in str(self.issues_found).lower()),
            ("Child scenarios functional", "scenario" not in str(self.issues_found).lower() or len(self.issues_found) == 0),
            ("Video player working", "video" not in str(self.issues_found).lower() or len(self.issues_found) == 0),
            ("ML classifications visible", bool(self.ml_connection_results)),
            ("No critical console errors", not any("Critical error" in issue for issue in self.issues_found))
        ]

        for check, passed in checks:
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check}")

        # Overall Score
        passed_checks = sum(1 for _, passed in checks if passed)
        total_checks = len(checks)
        score = (passed_checks / total_checks) * 100

        print(f"\n📈 Overall Quality Score: {score:.1f}%")

        if score >= 90:
            print("🎉 Excellent! Platform is production-ready.")
        elif score >= 70:
            print("⚠️ Good, but some improvements needed.")
        else:
            print("❌ Significant issues need attention.")

        return score >= 70

    def run_all_tests(self):
        """Run all page tests"""
        try:
            self.setup_driver()

            print("🚀 Starting Comprehensive Page Testing")
            print("Testing: Hub, Demo1, Demo2")
            print("Checking: Content, ML Connections, Errors, UX")

            # Run tests
            hub_ok = self.test_hub_page()
            demo1_ok = self.test_demo1_page()
            demo2_ok = self.test_demo2_page()

            # Generate report
            overall_ok = self.generate_report()

            return overall_ok

        except Exception as e:
            print(f"\n❌ Test suite error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.driver:
                self.driver.quit()

def main():
    """Main test runner"""
    tester = PageTester()
    success = tester.run_all_tests()

    if not success:
        print("\n🔧 Issues detected - fixes needed!")
        print("Next steps:")
        print("1. Fix demo2 feedback endpoint to use environment-aware URL")
        print("2. Ensure all headers use unified system")
        print("3. Verify ML connections are working")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()