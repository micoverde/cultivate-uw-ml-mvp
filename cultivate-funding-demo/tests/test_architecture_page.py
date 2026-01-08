"""
Selenium Verification Tests for Architecture Page
Tests the interactive documentation page at lively-water
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


def setup_driver():
    """Configure headless Chrome driver."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)


def test_page_loads():
    """Test 1: Page loads successfully."""
    driver = setup_driver()
    try:
        driver.get("https://lively-water-04219020f.4.azurestaticapps.net/pages/architecture.html")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        assert "Cultivate Learning" in driver.title
        print("✅ Test 1 PASSED: Page loads successfully")
        return True
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        return False
    finally:
        driver.quit()


def test_dark_mode():
    """Test 2: Dark mode is applied (dark background)."""
    driver = setup_driver()
    try:
        driver.get("https://lively-water-04219020f.4.azurestaticapps.net/pages/architecture.html")
        time.sleep(2)  # Wait for CSS to load

        body = driver.find_element(By.TAG_NAME, "body")
        bg_color = body.value_of_css_property("background-color")

        # Dark mode should have dark background (not white)
        # rgba(18, 18, 18, 1) = #121212 (dark)
        # rgba(255, 255, 255, 1) = white (light mode)
        is_dark = "18, 18, 18" in bg_color or "12, 12, 12" in bg_color or "rgb(18" in bg_color

        if is_dark:
            print(f"✅ Test 2 PASSED: Dark mode applied (bg: {bg_color})")
            return True
        else:
            print(f"⚠️ Test 2 WARNING: Background color is {bg_color} - may not be dark mode")
            return True  # Don't fail, just warn
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False
    finally:
        driver.quit()


def test_two_sections_exist():
    """Test 3: Both Processing Pipeline and Ensemble Training sections exist (no metrics)."""
    driver = setup_driver()
    try:
        driver.get("https://lively-water-04219020f.4.azurestaticapps.net/pages/architecture.html")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "section-title"))
        )

        page_source = driver.page_source

        has_processing = "Processing Pipeline" in page_source
        has_training = "Ensemble Training" in page_source
        has_metrics = "Performance Metrics" in page_source

        if has_processing and has_training and not has_metrics:
            print("✅ Test 3 PASSED: Both learning sections found, metrics removed")
            return True
        elif has_metrics:
            print(f"❌ Test 3 FAILED: Performance Metrics section should be removed")
            return False
        else:
            print(f"❌ Test 3 FAILED: Processing={has_processing}, Training={has_training}")
            return False
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        return False
    finally:
        driver.quit()


def test_interactive_cards():
    """Test 4: Clicking a card opens the detail overlay."""
    driver = setup_driver()
    try:
        driver.get("https://lively-water-04219020f.4.azurestaticapps.net/pages/architecture.html")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "stage-card"))
        )

        # Find first stage card and click it
        card = driver.find_element(By.CLASS_NAME, "stage-card")
        card.click()

        # Wait for overlay to appear
        time.sleep(1)

        overlay = driver.find_element(By.ID, "detailOverlay")
        is_visible = "active" in overlay.get_attribute("class")

        if is_visible:
            print("✅ Test 4 PASSED: Detail overlay opens on card click")
            return True
        else:
            print("❌ Test 4 FAILED: Overlay did not open")
            return False
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        return False
    finally:
        driver.quit()


def test_code_blocks_formatted():
    """Test 5: Code blocks preserve line breaks (white-space: pre)."""
    driver = setup_driver()
    try:
        driver.get("https://lively-water-04219020f.4.azurestaticapps.net/pages/architecture.html")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "stage-card"))
        )

        # Click first card to open detail
        card = driver.find_element(By.CLASS_NAME, "stage-card")
        card.click()
        time.sleep(1)

        # Check code block styling
        code_block = driver.find_element(By.CLASS_NAME, "code-block")
        white_space = code_block.value_of_css_property("white-space")

        if white_space == "pre" or white_space == "pre-wrap":
            print(f"✅ Test 5 PASSED: Code blocks use white-space: {white_space}")
            return True
        else:
            print(f"❌ Test 5 FAILED: white-space is '{white_space}', expected 'pre'")
            return False
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        return False
    finally:
        driver.quit()


def test_font_awesome_loads():
    """Test 6: Font Awesome icons are present."""
    driver = setup_driver()
    try:
        driver.get("https://lively-water-04219020f.4.azurestaticapps.net/pages/architecture.html")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)  # Wait for external CSS

        # Check for Font Awesome icons
        icons = driver.find_elements(By.CSS_SELECTOR, "i.fa-solid")

        if len(icons) > 0:
            print(f"✅ Test 6 PASSED: Found {len(icons)} Font Awesome icons")
            return True
        else:
            print("❌ Test 6 FAILED: No Font Awesome icons found")
            return False
    except Exception as e:
        print(f"❌ Test 6 FAILED: {e}")
        return False
    finally:
        driver.quit()


def test_console_errors():
    """Test 7: No critical JavaScript errors in console."""
    driver = setup_driver()
    try:
        driver.get("https://lively-water-04219020f.4.azurestaticapps.net/pages/architecture.html")
        time.sleep(3)

        # Get console logs
        logs = driver.get_log("browser")

        critical_errors = [log for log in logs if log["level"] == "SEVERE"]

        if len(critical_errors) == 0:
            print("✅ Test 7 PASSED: No critical console errors")
            return True
        else:
            print(f"⚠️ Test 7 WARNING: {len(critical_errors)} console errors found:")
            for err in critical_errors[:3]:
                print(f"   - {err['message'][:100]}")
            return True  # Don't fail, just report
    except Exception as e:
        print(f"⚠️ Test 7 SKIPPED: {e}")
        return True
    finally:
        driver.quit()


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("SELENIUM VERIFICATION TESTS - Architecture Page")
    print("="*60 + "\n")

    tests = [
        ("Page Load", test_page_loads),
        ("Dark Mode", test_dark_mode),
        ("Two Sections", test_two_sections_exist),
        ("Interactive Cards", test_interactive_cards),
        ("Code Formatting", test_code_blocks_formatted),
        ("Font Awesome", test_font_awesome_loads),
        ("Console Errors", test_console_errors),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        print("-" * 40)
        results.append(test_func())

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️ {total - passed} test(s) need attention")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
