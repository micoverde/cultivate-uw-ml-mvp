#!/usr/bin/env python3
"""
UX Validation Testing with Selenium
Issue #55: Navigate without technical assistance

This test suite validates that educational stakeholders can navigate
the demo intuitively without technical assistance.
"""

import time
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class UXValidationSuite:
    def __init__(self, headless=False, record_interactions=True):
        """Initialize UX validation testing suite."""
        self.headless = headless
        self.record_interactions = record_interactions
        self.interactions = []
        self.errors_encountered = []
        self.success_metrics = {
            'navigation_clarity': 0,
            'error_recovery': 0,
            'task_completion': 0,
            'time_to_value': 0
        }

        # Chrome options for UX testing
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('--disable-web-security')
        self.chrome_options.add_argument('--disable-features=VizDisplayCompositor')

        self.driver = None
        self.wait = None

    def setup_driver(self):
        """Setup Chrome driver for testing."""
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        return self.driver

    def record_interaction(self, action, element, success, duration, context=""):
        """Record user interaction for UX analysis."""
        if self.record_interactions:
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'element': element,
                'success': success,
                'duration_ms': duration * 1000,
                'context': context
            }
            self.interactions.append(interaction)

    def measure_element_clarity(self, selector, expected_label):
        """Test if UI elements are clearly labeled and intuitive."""
        try:
            start_time = time.time()
            element = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))

            # Check if element text matches expected intuitive label
            element_text = element.text.strip()
            is_clear = expected_label.lower() in element_text.lower()

            duration = time.time() - start_time
            self.record_interaction('clarity_check', selector, is_clear, duration,
                                  f"Expected: {expected_label}, Found: {element_text}")

            return is_clear, element_text

        except TimeoutException:
            self.errors_encountered.append(f"Element not found: {selector}")
            return False, "Element not found"

    def test_first_impression_clarity(self):
        """Test 30-second rule: Is value proposition clear immediately?"""
        print("\n🎯 Testing First Impression Clarity...")

        start_time = time.time()
        self.driver.get('http://localhost:8082')

        # Look for clear value proposition within 5 seconds
        try:
            # Check for clear headings
            main_heading = self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'h1, .text-2xl, .text-3xl')
            ))

            # Check for clear call-to-action
            cta_button = self.driver.find_element(By.CSS_SELECTOR,
                'button, .btn, [role="button"]')

            # Measure clarity
            heading_text = main_heading.text
            cta_text = cta_button.text

            # UX Heuristics: Is it immediately clear what this does?
            clear_keywords = ['demo', 'analyze', 'teaching', 'education', 'classroom']
            is_educational_context_clear = any(keyword in heading_text.lower() for keyword in clear_keywords)

            duration = time.time() - start_time

            result = {
                'heading_clarity': is_educational_context_clear,
                'cta_clarity': len(cta_text) > 0,
                'load_time': duration,
                'heading_text': heading_text,
                'cta_text': cta_text
            }

            print(f"✓ Heading: {heading_text}")
            print(f"✓ CTA: {cta_text}")
            print(f"✓ Educational context clear: {is_educational_context_clear}")
            print(f"✓ Load time: {duration:.2f}s")

            return result

        except TimeoutException:
            print("❌ First impression test failed - elements not found")
            return {'success': False, 'error': 'Key elements not found'}

    def test_navigation_intuitiveness(self):
        """Test if users can navigate without getting lost."""
        print("\n🧭 Testing Navigation Intuitiveness...")

        # Test breadcrumb/progress indicators
        try:
            # Look for navigation indicators
            nav_elements = self.driver.find_elements(By.CSS_SELECTOR,
                'nav, .breadcrumb, .progress, .stepper')

            navigation_present = len(nav_elements) > 0

            # Test back button clarity
            back_buttons = self.driver.find_elements(By.CSS_SELECTOR,
                'button:contains("Back"), .back-btn, [aria-label*="back"]')

            # Test next action clarity
            next_buttons = self.driver.find_elements(By.CSS_SELECTOR,
                'button:contains("Next"), .next-btn, .continue-btn')

            result = {
                'navigation_indicators': navigation_present,
                'back_button_present': len(back_buttons) > 0,
                'next_action_clear': len(next_buttons) > 0,
                'nav_elements_count': len(nav_elements)
            }

            print(f"✓ Navigation indicators: {navigation_present}")
            print(f"✓ Back buttons found: {len(back_buttons)}")
            print(f"✓ Next actions clear: {len(next_buttons)}")

            return result

        except Exception as e:
            print(f"❌ Navigation test error: {e}")
            return {'success': False, 'error': str(e)}

    def test_error_handling_friendliness(self):
        """Test error messages are helpful, not intimidating."""
        print("\n🛠️ Testing Error Handling Friendliness...")

        # Simulate common user errors
        error_scenarios = []

        try:
            # Test empty input submission
            textarea = self.driver.find_element(By.CSS_SELECTOR, 'textarea')
            if textarea:
                # Clear any existing text and try to submit empty
                textarea.clear()

                submit_button = self.driver.find_element(By.CSS_SELECTOR,
                    'button[type="submit"], .submit-btn, button:contains("Analyze")')

                submit_button.click()

                # Check for error message
                time.sleep(1)  # Wait for error to appear

                error_elements = self.driver.find_elements(By.CSS_SELECTOR,
                    '.error, .alert-danger, .text-red, [role="alert"]')

                if error_elements:
                    error_text = error_elements[0].text

                    # Analyze error friendliness
                    friendly_indicators = ['help', 'try', 'add', 'please', 'let\'s']
                    technical_indicators = ['error', 'failed', 'invalid', 'required']

                    friendliness_score = sum(1 for word in friendly_indicators if word in error_text.lower())
                    technical_score = sum(1 for word in technical_indicators if word in error_text.lower())

                    is_friendly = friendliness_score > technical_score

                    error_scenarios.append({
                        'scenario': 'empty_input',
                        'error_text': error_text,
                        'is_friendly': is_friendly,
                        'friendliness_score': friendliness_score,
                        'technical_score': technical_score
                    })

                    print(f"✓ Empty input error: {error_text}")
                    print(f"✓ Friendly tone: {is_friendly}")

        except NoSuchElementException:
            print("⚠️ Could not test error handling - elements not found")

        return error_scenarios

    def test_cognitive_load_assessment(self):
        """Assess if interface follows cognitive load principles."""
        print("\n🧠 Testing Cognitive Load...")

        # Count interactive elements visible at once
        interactive_elements = self.driver.find_elements(By.CSS_SELECTOR,
            'button, input, select, textarea, [role="button"], a[href]')

        # Count text blocks
        text_blocks = self.driver.find_elements(By.CSS_SELECTOR,
            'p, h1, h2, h3, h4, h5, h6, .text-lg, .text-xl')

        # Check for overwhelming number of options (7±2 rule)
        primary_actions = self.driver.find_elements(By.CSS_SELECTOR,
            '.btn-primary, .bg-indigo, .bg-blue, button[type="submit"]')

        assessment = {
            'total_interactive_elements': len(interactive_elements),
            'text_blocks': len(text_blocks),
            'primary_actions': len(primary_actions),
            'follows_seven_rule': len(primary_actions) <= 7,
            'cognitive_load_score': len(interactive_elements) + len(text_blocks)
        }

        print(f"✓ Interactive elements: {len(interactive_elements)}")
        print(f"✓ Primary actions: {len(primary_actions)} (should be ≤7)")
        print(f"✓ Follows 7±2 rule: {assessment['follows_seven_rule']}")

        return assessment

    def test_complete_user_journey(self):
        """Test complete user journey from start to export."""
        print("\n🎯 Testing Complete User Journey...")

        journey_start = time.time()
        steps_completed = 0
        total_steps = 5

        try:
            # Step 1: Land on homepage and understand value
            self.driver.get('http://localhost:8082')
            time.sleep(2)
            steps_completed += 1
            print("✓ Step 1: Landed on homepage")

            # Step 2: Find and click scenario selection
            scenario_buttons = self.driver.find_elements(By.CSS_SELECTOR,
                'button:contains("Maya"), .scenario-btn, .demo-scenario')

            if scenario_buttons:
                scenario_buttons[0].click()
                time.sleep(2)
                steps_completed += 1
                print("✓ Step 2: Selected demo scenario")

            # Step 3: Input scenario text
            textarea = self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'textarea')))

            sample_scenario = """
            Teacher: "What do you think will happen if we mix these colors?"
            Student: "Maybe purple?"
            Teacher: "Let's try it and see what happens."
            """

            textarea.clear()
            textarea.send_keys(sample_scenario)
            steps_completed += 1
            print("✓ Step 3: Entered scenario text")

            # Step 4: Submit for analysis
            analyze_button = self.driver.find_element(By.CSS_SELECTOR,
                'button:contains("Analyze"), button[type="submit"], .analyze-btn')
            analyze_button.click()

            # Wait for results with timeout
            try:
                results_section = self.wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '.results, .analysis-results, .ml-predictions')))
                steps_completed += 1
                print("✓ Step 4: Analysis completed")

                # Step 5: Discover export functionality
                export_button = self.driver.find_element(By.CSS_SELECTOR,
                    'button:contains("Export"), .export-btn, .download-btn')

                if export_button.is_displayed():
                    steps_completed += 1
                    print("✓ Step 5: Found export functionality")

            except TimeoutException:
                print("⚠️ Analysis timed out - may indicate UX issue")

        except Exception as e:
            print(f"❌ User journey interrupted: {e}")

        journey_duration = time.time() - journey_start
        completion_rate = steps_completed / total_steps

        result = {
            'steps_completed': steps_completed,
            'total_steps': total_steps,
            'completion_rate': completion_rate,
            'journey_duration': journey_duration,
            'success': completion_rate >= 0.8  # 80% completion threshold
        }

        print(f"✓ Journey completion: {steps_completed}/{total_steps} ({completion_rate:.1%})")
        print(f"✓ Total duration: {journey_duration:.2f}s")

        return result

    def generate_ux_report(self):
        """Generate comprehensive UX assessment report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            'timestamp': timestamp,
            'test_configuration': {
                'headless': self.headless,
                'browser': 'Chrome',
                'viewport': '1920x1080'
            },
            'interactions_recorded': len(self.interactions),
            'errors_encountered': len(self.errors_encountered),
            'success_metrics': self.success_metrics,
            'detailed_interactions': self.interactions,
            'error_log': self.errors_encountered
        }

        # Save report
        report_path = f'/home/warrenjo/src/tmp3/cultivate-uw-ml-mvp/ux_validation_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 UX Report saved: {report_path}")
        return report_path

    def run_full_ux_suite(self):
        """Run complete UX validation test suite."""
        print("🚀 Starting Comprehensive UX Validation Suite")
        print("=" * 60)

        self.setup_driver()

        try:
            # Test Suite Execution
            first_impression = self.test_first_impression_clarity()
            navigation = self.test_navigation_intuitiveness()
            error_handling = self.test_error_handling_friendliness()
            cognitive_load = self.test_cognitive_load_assessment()
            user_journey = self.test_complete_user_journey()

            # Calculate overall UX score
            ux_score = 0
            if first_impression.get('heading_clarity'): ux_score += 20
            if navigation.get('navigation_indicators'): ux_score += 20
            if len(error_handling) > 0 and error_handling[0].get('is_friendly'): ux_score += 20
            if cognitive_load.get('follows_seven_rule'): ux_score += 20
            if user_journey.get('success'): ux_score += 20

            print("\n" + "=" * 60)
            print(f"🎯 Overall UX Score: {ux_score}/100")

            if ux_score >= 80:
                print("🎉 EXCELLENT: Interface ready for non-technical stakeholders")
            elif ux_score >= 60:
                print("✅ GOOD: Minor improvements needed")
            elif ux_score >= 40:
                print("⚠️ NEEDS WORK: Significant UX improvements required")
            else:
                print("❌ POOR: Major redesign needed for stakeholder accessibility")

            # Generate report
            report_path = self.generate_ux_report()

            return {
                'ux_score': ux_score,
                'report_path': report_path,
                'first_impression': first_impression,
                'navigation': navigation,
                'error_handling': error_handling,
                'cognitive_load': cognitive_load,
                'user_journey': user_journey
            }

        finally:
            if self.driver:
                self.driver.quit()

if __name__ == "__main__":
    # Run with browser visible for debugging
    print("Running UX Validation Suite with visible browser...")
    ux_suite = UXValidationSuite(headless=False)
    results = ux_suite.run_full_ux_suite()

    print(f"\n🎯 Final UX Score: {results['ux_score']}/100")
    print(f"📊 Detailed report: {results['report_path']}")