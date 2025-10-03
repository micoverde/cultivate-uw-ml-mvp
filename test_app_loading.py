#!/usr/bin/env python3
"""Test if React app is loading properly"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Setup Chrome
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(options=options)

try:
    print("Loading http://localhost:8087/")
    driver.get("http://localhost:8087/")

    # Wait a bit for React to load
    time.sleep(3)

    # Get page source
    page_source = driver.page_source
    print(f"Page length: {len(page_source)} characters")

    # Check if root div has content
    root_element = driver.find_element(By.ID, "root")
    root_html = root_element.get_attribute("innerHTML")
    print(f"Root element content length: {len(root_html)} characters")

    if len(root_html) < 100:
        print("❌ Root element is empty or minimal - React app not loading")

        # Check console logs
        logs = driver.get_log("browser")
        if logs:
            print("\nConsole errors/logs:")
            for log in logs:
                if log['level'] in ['SEVERE', 'ERROR']:
                    print(f"  ERROR: {log['message']}")
                elif 'warning' not in log['message'].lower():
                    print(f"  {log['level']}: {log['message'][:200]}")
    else:
        print("✅ React app loaded successfully")

        # Look for Settings button
        try:
            settings = driver.find_elements(By.CLASS_NAME, "settings")
            if settings:
                print(f"✅ Found {len(settings)} settings-related elements")

            # Look for ML Settings specifically
            page_text = driver.find_element(By.TAG_NAME, "body").text
            if "ML Settings" in page_text or "Settings" in page_text:
                print("✅ Settings text found in page")
            else:
                print("⚠️ Settings text not found")

        except Exception as e:
            print(f"Error checking for settings: {e}")

    # Check for any visible text
    body_text = driver.find_element(By.TAG_NAME, "body").text
    if body_text.strip():
        print(f"\nVisible text preview: {body_text[:200]}...")
    else:
        print("\n⚠️ No visible text on page")

except Exception as e:
    print(f"Error: {e}")
finally:
    driver.quit()