#!/usr/bin/env python3
"""
Test the NaN% fix in Demo 1 production using Selenium
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import json

def test_demo1_production():
    """Test Demo 1 in production for NaN% issue"""

    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.binary_location = '/usr/bin/chromium-browser'

    # Setup console log capture
    chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

    # Use chromium driver
    service = Service('/usr/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        print("\n" + "="*60)
        print("🧪 Testing Demo 1 Production - NaN% Fix Verification")
        print("="*60)

        # Navigate to Demo 1 production
        url = "https://victorious-meadow-0b2278a10.1.azurestaticapps.net/demo1/"
        print(f"\n📍 Navigating to: {url}")
        driver.get(url)

        # Wait for page to load
        wait = WebDriverWait(driver, 10)

        # Wait for scenario tabs to be available
        print("⏳ Waiting for scenario tabs to load...")
        tabs = wait.until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "tab"))
        )
        print(f"✅ Found {len(tabs)} scenario tabs")

        # Click first scenario tab
        if tabs:
            print(f"\n🎯 Clicking first scenario tab...")
            tabs[0].click()
            time.sleep(1)

            # Find and click Analyze Response button
            analyze_btn = wait.until(
                EC.element_to_be_clickable((By.ID, "analyzeBtn"))
            )
            print("🔬 Clicking 'Analyze Response' button...")
            analyze_btn.click()

            # Wait for results
            time.sleep(3)

            # Check for OEQ/CEQ percentage displays
            try:
                oeq_element = driver.find_element(By.ID, "oeqScore")
                ceq_element = driver.find_element(By.ID, "ceqScore")

                oeq_text = oeq_element.text
                ceq_text = ceq_element.text

                print("\n📊 Classification Results:")
                print(f"   OEQ Score: {oeq_text}")
                print(f"   CEQ Score: {ceq_text}")

                # Check if NaN is present
                if "NaN" in oeq_text or "NaN" in ceq_text:
                    print("\n❌ FAILURE: NaN% still appearing in results!")
                    print("   The fix did not work properly.")
                else:
                    print("\n✅ SUCCESS: No NaN% in results!")
                    print("   The fix is working correctly.")

                    # Verify percentages are numeric
                    try:
                        oeq_num = int(oeq_text.replace('%', ''))
                        ceq_num = int(ceq_text.replace('%', ''))
                        print(f"   OEQ: {oeq_num}% (numeric ✓)")
                        print(f"   CEQ: {ceq_num}% (numeric ✓)")
                    except ValueError as e:
                        print(f"   ⚠️ Warning: Could not parse percentages as numbers: {e}")

            except Exception as e:
                print(f"\n⚠️ Could not find score elements: {e}")

        # Capture and check console logs
        print("\n📋 Checking browser console for errors...")
        logs = driver.get_log('browser')

        errors = []
        ml_logs = []

        for log in logs:
            msg = log.get('message', '')
            if 'SEVERE' in log.get('level', ''):
                errors.append(msg)
            if 'ML' in msg or 'probability' in msg or 'NaN' in msg:
                ml_logs.append(msg)

        if errors:
            print(f"\n⚠️ Found {len(errors)} console errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   - {error[:100]}...")
        else:
            print("   ✓ No console errors detected")

        if ml_logs:
            print(f"\n🤖 ML-related console logs:")
            for log in ml_logs[:5]:  # Show first 5 ML logs
                print(f"   - {log[:100]}...")

        # Final summary
        print("\n" + "="*60)
        print("📊 Test Summary:")
        if "NaN" not in oeq_text and "NaN" not in ceq_text:
            print("✅ NaN% fix VERIFIED - Production is working correctly!")
        else:
            print("❌ NaN% issue still present - Fix needs more work")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        driver.quit()
        print("\n🏁 Test completed")

if __name__ == "__main__":
    test_demo1_production()