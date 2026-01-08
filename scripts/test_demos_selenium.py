#!/usr/bin/env python3
"""
Selenium Test Suite for Cultivate Learning Demos
Captures console warnings and errors for all four demos

Run with: python scripts/test_demos_selenium.py
"""

import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

BASE_URL = "https://victorious-meadow-0b2278a10.1.azurestaticapps.net"
DEMO3_PASSWORD = "TURING"

def setup_driver():
    """Setup Chrome driver with console log capture"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # Enable console log capture
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

    driver = webdriver.Chrome(options=options)
    return driver

def get_console_logs(driver, level_filter=None):
    """Get browser console logs, optionally filtered by level"""
    logs = driver.get_log("browser")

    if level_filter:
        # Filter by level (SEVERE = error, WARNING = warn)
        logs = [log for log in logs if log.get("level") in level_filter]

    return logs

def categorize_logs(logs):
    """Categorize logs into errors, warnings, and info"""
    errors = [log for log in logs if log.get("level") == "SEVERE"]
    warnings = [log for log in logs if log.get("level") == "WARNING"]
    info = [log for log in logs if log.get("level") not in ["SEVERE", "WARNING"]]
    return errors, warnings, info

def test_demo1(driver):
    """Test Demo 1 - ECE Question Classification"""
    print("\n" + "="*60)
    print("TESTING DEMO 1: ECE Question Classification")
    print("="*60)

    driver.get(f"{BASE_URL}/demo1/")
    time.sleep(3)  # Wait for page load

    results = {
        "url": f"{BASE_URL}/demo1/",
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "checks": {}
    }

    # Check page title
    try:
        assert "Demo 1" in driver.title or "Classification" in driver.title
        results["checks"]["title"] = "PASS"
        print("✓ Page title correct")
    except AssertionError:
        results["checks"]["title"] = "FAIL"
        print("✗ Page title incorrect")

    # Check for input field
    try:
        input_field = driver.find_element(By.ID, "userResponse")
        results["checks"]["input_field"] = "PASS"
        print("✓ Input field found")
    except:
        results["checks"]["input_field"] = "FAIL"
        print("✗ Input field NOT found")

    # Try to classify a question
    try:
        input_field = driver.find_element(By.ID, "userResponse")
        input_field.clear()
        input_field.send_keys("What do you think will happen next?")

        # Find and click analyze button
        analyze_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Analyze')]")
        analyze_btn.click()
        time.sleep(3)  # Wait for API response

        results["checks"]["api_call"] = "PASS"
        print("✓ API call initiated")
    except Exception as e:
        results["checks"]["api_call"] = f"FAIL: {str(e)}"
        print(f"✗ API call failed: {e}")

    # Get console logs
    logs = get_console_logs(driver)
    errors, warnings, _ = categorize_logs(logs)

    results["errors"] = [{"message": e["message"], "level": e["level"]} for e in errors]
    results["warnings"] = [{"message": w["message"], "level": w["level"]} for w in warnings]

    # Print errors and warnings
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   - {e['message'][:100]}")
    else:
        print("\n✓ No errors")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w['message'][:100]}")
    else:
        print("✓ No warnings")

    results["status"] = "PASS" if not errors else "FAIL"
    return results

def test_demo2(driver):
    """Test Demo 2 - Warren's Teaching Video Analysis"""
    print("\n" + "="*60)
    print("TESTING DEMO 2: Warren's Teaching Video Analysis")
    print("="*60)

    driver.get(f"{BASE_URL}/demo2/")
    time.sleep(5)  # Wait for page and video load

    results = {
        "url": f"{BASE_URL}/demo2/",
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "checks": {}
    }

    # Check page title
    try:
        assert "Demo 2" in driver.title or "Warren" in driver.title
        results["checks"]["title"] = "PASS"
        print("✓ Page title correct")
    except AssertionError:
        results["checks"]["title"] = "FAIL"
        print("✗ Page title incorrect")

    # Check for video player
    try:
        video = driver.find_element(By.ID, "mainVideo")
        results["checks"]["video_player"] = "PASS"
        print("✓ Video player found")

        # Check if video can load
        video_src = video.find_element(By.TAG_NAME, "source").get_attribute("src")
        if "cultivatemlvideos.blob.core.windows.net" in video_src:
            results["checks"]["video_source"] = "PASS"
            print("✓ Video source points to Azure Blob")
        else:
            results["checks"]["video_source"] = "FAIL"
            print("✗ Video source not Azure Blob")
    except Exception as e:
        results["checks"]["video_player"] = f"FAIL: {str(e)}"
        print(f"✗ Video player issue: {e}")

    # Get console logs
    logs = get_console_logs(driver)
    errors, warnings, _ = categorize_logs(logs)

    results["errors"] = [{"message": e["message"], "level": e["level"]} for e in errors]
    results["warnings"] = [{"message": w["message"], "level": w["level"]} for w in warnings]

    # Print errors and warnings
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   - {e['message'][:100]}")
    else:
        print("\n✓ No errors")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w['message'][:100]}")
    else:
        print("✓ No warnings")

    results["status"] = "PASS" if not errors else "FAIL"
    return results

def test_demo3(driver):
    """Test Demo 3 - Multi-Video Analysis Platform"""
    print("\n" + "="*60)
    print("TESTING DEMO 3: Multi-Video Analysis Platform")
    print("="*60)

    driver.get(f"{BASE_URL}/demo3/")
    time.sleep(2)

    results = {
        "url": f"{BASE_URL}/demo3/",
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "checks": {}
    }

    # Check for password overlay
    try:
        overlay = driver.find_element(By.ID, "passwordOverlay")
        if "hidden" not in overlay.get_attribute("class"):
            results["checks"]["password_overlay"] = "PASS"
            print("✓ Password overlay displayed")

            # Enter password
            password_input = driver.find_element(By.ID, "passwordInput")
            password_input.send_keys(DEMO3_PASSWORD)
            submit_btn = driver.find_element(By.ID, "passwordSubmit")
            submit_btn.click()
            time.sleep(2)

            # Check if overlay is now hidden
            if "hidden" in overlay.get_attribute("class"):
                results["checks"]["password_auth"] = "PASS"
                print("✓ Password authentication successful")
            else:
                results["checks"]["password_auth"] = "FAIL"
                print("✗ Password authentication failed")
        else:
            results["checks"]["password_overlay"] = "ALREADY_HIDDEN"
            print("✓ Password overlay already hidden (session active)")
    except Exception as e:
        results["checks"]["password_overlay"] = f"FAIL: {str(e)}"
        print(f"✗ Password overlay issue: {e}")

    time.sleep(3)  # Wait for components to load

    # Check dark mode
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        theme = body.get_attribute("data-theme")
        if theme == "dark":
            results["checks"]["dark_mode"] = "PASS"
            print("✓ Dark mode enabled")
        else:
            results["checks"]["dark_mode"] = f"FAIL: theme={theme}"
            print(f"✗ Dark mode not enabled (theme={theme})")
    except Exception as e:
        results["checks"]["dark_mode"] = f"FAIL: {str(e)}"
        print(f"✗ Dark mode check failed: {e}")

    # Check for video selector container
    try:
        selector = driver.find_element(By.ID, "videoSelectorContainer")
        results["checks"]["video_selector"] = "PASS"
        print("✓ Video selector container found")
    except:
        results["checks"]["video_selector"] = "FAIL"
        print("✗ Video selector container NOT found")

    # Get console logs
    logs = get_console_logs(driver)
    errors, warnings, _ = categorize_logs(logs)

    results["errors"] = [{"message": e["message"], "level": e["level"]} for e in errors]
    results["warnings"] = [{"message": w["message"], "level": w["level"]} for w in warnings]

    # Print errors and warnings
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   - {e['message'][:100]}")
    else:
        print("\n✓ No errors")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w['message'][:100]}")
    else:
        print("✓ No warnings")

    results["status"] = "PASS" if not errors else "FAIL"
    return results

def test_demo4(driver):
    """Test Demo 4 - Feature Extraction Comparison"""
    print("\n" + "="*60)
    print("TESTING DEMO 4: Feature Extraction Comparison")
    print("="*60)

    driver.get(f"{BASE_URL}/demo4/")
    time.sleep(3)  # Wait for page load

    results = {
        "url": f"{BASE_URL}/demo4/",
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "checks": {}
    }

    # Check page title
    try:
        assert "Demo 4" in driver.title or "Feature" in driver.title
        results["checks"]["title"] = "PASS"
        print("✓ Page title correct")
    except AssertionError:
        results["checks"]["title"] = "FAIL"
        print("✗ Page title incorrect")

    # Check for input field
    try:
        input_field = driver.find_element(By.ID, "question-input")
        results["checks"]["input_field"] = "PASS"
        print("✓ Input field found")
    except:
        results["checks"]["input_field"] = "FAIL"
        print("✗ Input field NOT found")

    # Check for analyze button
    try:
        analyze_btn = driver.find_element(By.ID, "analyze-btn")
        results["checks"]["analyze_button"] = "PASS"
        print("✓ Analyze button found")
    except:
        results["checks"]["analyze_button"] = "FAIL"
        print("✗ Analyze button NOT found")

    # Check for extractor cards
    try:
        rule_card = driver.find_element(By.ID, "rule-card")
        semantic_card = driver.find_element(By.ID, "semantic-card")
        hybrid_card = driver.find_element(By.ID, "hybrid-card")
        results["checks"]["extractor_cards"] = "PASS"
        print("✓ All extractor cards found (rule, semantic, hybrid)")
    except:
        results["checks"]["extractor_cards"] = "FAIL"
        print("✗ Some extractor cards NOT found")

    # Try to analyze a question
    try:
        input_field = driver.find_element(By.ID, "question-input")
        input_field.clear()
        input_field.send_keys("What do you think will happen next?")

        analyze_btn = driver.find_element(By.ID, "analyze-btn")
        analyze_btn.click()
        time.sleep(3)  # Wait for API response

        results["checks"]["api_call"] = "PASS"
        print("✓ API call initiated")
    except Exception as e:
        results["checks"]["api_call"] = f"FAIL: {str(e)}"
        print(f"✗ API call failed: {e}")

    # Get console logs
    logs = get_console_logs(driver)
    errors, warnings, _ = categorize_logs(logs)

    results["errors"] = [{"message": e["message"], "level": e["level"]} for e in errors]
    results["warnings"] = [{"message": w["message"], "level": w["level"]} for w in warnings]

    # Print errors and warnings
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   - {e['message'][:100]}")
    else:
        print("\n✓ No errors")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w['message'][:100]}")
    else:
        print("✓ No warnings")

    results["status"] = "PASS" if not errors else "FAIL"
    return results

def main():
    """Run all demo tests"""
    print("\n" + "="*60)
    print("CULTIVATE LEARNING DEMO TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    driver = None
    all_results = {}

    try:
        driver = setup_driver()

        # Test all demos
        all_results["demo1"] = test_demo1(driver)
        all_results["demo2"] = test_demo2(driver)
        all_results["demo3"] = test_demo3(driver)
        all_results["demo4"] = test_demo4(driver)

    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        all_results["error"] = str(e)
    finally:
        if driver:
            driver.quit()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_errors = 0
    total_warnings = 0

    for demo, results in all_results.items():
        if demo == "error":
            continue
        status = results.get("status", "UNKNOWN")
        errors = len(results.get("errors", []))
        warnings = len(results.get("warnings", []))
        total_errors += errors
        total_warnings += warnings

        status_icon = "✓" if status == "PASS" else "✗"
        print(f"{status_icon} {demo.upper()}: {status} ({errors} errors, {warnings} warnings)")

    print(f"\nTOTAL: {total_errors} errors, {total_warnings} warnings")

    # Save results to JSON
    with open("test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to test_results.json")

    return 0 if total_errors == 0 else 1

if __name__ == "__main__":
    exit(main())
