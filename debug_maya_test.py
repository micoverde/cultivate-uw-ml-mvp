#!/usr/bin/env python3

"""
Debug Test for Maya Scenario
Quick diagnostic to understand what's happening in the browser
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def debug_maya_scenario():
    print("🔍 Starting Maya Scenario Debug Test...")

    # Setup Chrome with visible browser
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        print("1. Loading homepage...")
        driver.get("http://localhost:3002")
        time.sleep(2)

        print(f"   Page title: {driver.title}")
        print(f"   Current URL: {driver.current_url}")

        print("2. Looking for Maya button...")
        maya_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Maya')]")
        print(f"   Found {len(maya_buttons)} Maya buttons")

        for i, button in enumerate(maya_buttons):
            print(f"   Button {i}: '{button.text}' - Visible: {button.is_displayed()}")

        if maya_buttons:
            print("3. Clicking Maya button...")
            maya_button = maya_buttons[0]
            driver.execute_script("arguments[0].scrollIntoView();", maya_button)
            time.sleep(1)

            # Use JavaScript click to bypass overlay issues
            print("   Using JavaScript click to bypass overlay...")
            driver.execute_script("arguments[0].click();", maya_button)

            print("4. Waiting for page to change...")
            time.sleep(5)  # Give React time to render

            print(f"   New URL: {driver.current_url}")
            print(f"   New title: {driver.title}")

            # Check what's in the body
            body = driver.find_element(By.TAG_NAME, "body")
            body_text = body.text[:500] + "..." if len(body.text) > 500 else body.text
            print(f"   Page content sample: {repr(body_text)}")

            # Look for any h3 elements
            h3_elements = driver.find_elements(By.TAG_NAME, "h3")
            print(f"   Found {len(h3_elements)} h3 elements:")
            for i, h3 in enumerate(h3_elements):
                print(f"     H3 {i}: '{h3.text}'")

            # Look for any textarea elements
            textarea_elements = driver.find_elements(By.TAG_NAME, "textarea")
            print(f"   Found {len(textarea_elements)} textarea elements:")
            for i, ta in enumerate(textarea_elements):
                print(f"     Textarea {i}: placeholder='{ta.get_attribute('placeholder')}'")

            # Check for React errors
            try:
                console_logs = driver.get_log('browser')
                if console_logs:
                    print("   Console logs:")
                    for log in console_logs[-5:]:  # Last 5 logs
                        print(f"     {log['level']}: {log['message']}")
            except:
                print("   Could not get console logs")

            # Check React root
            try:
                react_root = driver.find_element(By.ID, "root")
                root_html = react_root.get_attribute("innerHTML")
                print(f"   React root innerHTML length: {len(root_html)}")
                if len(root_html) < 100:
                    print(f"   React root content: {root_html}")
            except Exception as e:
                print(f"   Could not access React root: {e}")

        print("5. Taking final screenshot...")
        driver.save_screenshot("debug_maya_final.png")
        print("   Screenshot saved as debug_maya_final.png")

        input("\nPress Enter to close browser...")

    except Exception as e:
        print(f"❌ Error during debug test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        driver.quit()

if __name__ == "__main__":
    debug_maya_scenario()