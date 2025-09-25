#!/usr/bin/env python3

"""
Full Maya Flow Test - End-to-End API Integration Test
Tests the complete flow from UI interaction to API response
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_full_maya_flow():
    print("🎯 Testing Full Maya Scenario Flow with API Integration")

    # Setup Chrome with visible browser
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")

    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 15)

    try:
        # Step 1: Load homepage
        print("1. Loading homepage...")
        driver.get("http://localhost:3002")
        time.sleep(2)
        print(f"   Current URL: {driver.current_url}")

        # Step 2: Click Maya scenario button
        print("2. Clicking Maya scenario button...")
        maya_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Maya')]"))
        )
        driver.execute_script("arguments[0].click();", maya_button)
        time.sleep(3)

        # Step 3: Find and fill textarea
        print("3. Finding textarea and entering response...")
        textarea = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "textarea"))
        )

        # Type a comprehensive educator response
        response_text = """Maya, I can see you're feeling really frustrated with this puzzle right now. Those big feelings are completely normal when something feels challenging! I noticed you worked so hard on getting those three pieces to fit together - that shows real persistence and problem-solving skills.

Would you like to take a short break and do some deep breathing with me, or would you prefer to try a different approach with the puzzle? Sometimes when we're feeling overwhelmed, our brains work better after a little pause. What feels right for you?"""

        textarea.clear()
        textarea.send_keys(response_text)
        time.sleep(1)

        # Step 4: Submit the response
        print("4. Submitting response for analysis...")
        submit_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Feedback') or contains(text(), 'Submit') or contains(text(), 'Analyze')]"))
        )
        driver.execute_script("arguments[0].click();", submit_button)

        # Step 5: Wait for analysis to start
        print("5. Waiting for analysis to begin...")
        time.sleep(5)

        # Step 6: Wait for results
        print("6. Waiting for analysis results...")
        max_wait_time = 30  # 30 seconds max
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            # Check if results are visible
            try:
                # Look for coaching score
                score_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '/10') or contains(text(), 'Score') or contains(text(), 'Coaching')]")
                if score_elements:
                    print("   ✅ Analysis results found!")
                    break

                # Look for feedback sections
                feedback_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Strengths') or contains(text(), 'Growth') or contains(text(), 'Recommendations')]")
                if feedback_elements:
                    print("   ✅ Feedback sections found!")
                    break

                time.sleep(2)
                print(f"   Waiting... ({int(time.time() - start_time)}s)")

            except Exception as e:
                print(f"   Checking for results: {e}")
                time.sleep(2)

        # Step 7: Capture final state
        print("7. Capturing final state...")
        driver.save_screenshot("maya_full_flow_final.png")

        # Get page content
        body = driver.find_element(By.TAG_NAME, "body")
        page_text = body.text

        print(f"   Page content length: {len(page_text)} characters")

        # Look for specific coaching elements
        if "Score" in page_text or "/10" in page_text:
            print("   ✅ COACHING SCORE FOUND")
        if "Strengths" in page_text:
            print("   ✅ STRENGTHS SECTION FOUND")
        if "Growth" in page_text or "improvement" in page_text.lower():
            print("   ✅ GROWTH AREAS FOUND")
        if "Recommendation" in page_text:
            print("   ✅ RECOMMENDATIONS FOUND")

        # Print sample of results
        print("\n📊 SAMPLE RESULTS:")
        lines = page_text.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['Score', 'Strengths', 'Growth', 'Recommendation', '/10']):
                print(f"   {line.strip()}")

        print("\n✅ FULL MAYA FLOW TEST COMPLETED!")
        input("\nPress Enter to close browser...")

    except Exception as e:
        print(f"❌ Error during full flow test: {e}")
        import traceback
        traceback.print_exc()
        driver.save_screenshot("maya_full_flow_error.png")

    finally:
        driver.quit()

if __name__ == "__main__":
    test_full_maya_flow()