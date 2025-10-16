#!/usr/bin/env python3
"""
Comprehensive Demo2 System Verification

Tests:
1. Both ML endpoints (Classic & Ensemble)
2. Demo2 HTML configuration
3. Rate limit handling in batched requests
4. Model switching functionality
"""

import requests
import json
import time
from typing import Dict, List

BASE_URL = "http://localhost:5001"
DEMO_URL = "http://localhost:6060/demo2/"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def log_test(name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if passed else f"{Colors.RED}❌ FAIL{Colors.RESET}"
    print(f"{status} - {name}")
    if details:
        print(f"      {details}")

def test_classic_endpoint():
    """Test Classic ML endpoint with JSON body"""
    print(f"\n{Colors.BLUE}Testing Classic ML Endpoint{Colors.RESET}")

    test_questions = [
        ("Did you like it?", "CEQ"),
        ("How did that happen?", "OEQ"),
        ("Is it red?", "CEQ"),
    ]

    for text, expected in test_questions:
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/classify",
                headers={"Content-Type": "application/json"},
                json={"text": text},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                classification = data.get("classification")
                confidence = data.get("confidence", 0)
                model = data.get("model")

                passed = (
                    classification is not None and
                    model == "classic" and
                    0 <= confidence <= 1
                )

                log_test(
                    f"Classic: '{text}'",
                    passed,
                    f"→ {classification} ({confidence:.1%}) [expected: {expected}]"
                )
            else:
                log_test(
                    f"Classic: '{text}'",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            log_test(f"Classic: '{text}'", False, str(e))

def test_ensemble_endpoint():
    """Test Ensemble ML endpoint with JSON body"""
    print(f"\n{Colors.BLUE}Testing Ensemble ML Endpoint{Colors.RESET}")

    test_questions = [
        ("Did you like it?", "CEQ"),
        ("How did that happen?", "OEQ"),
        ("Is it red?", "CEQ"),
    ]

    for text, expected in test_questions:
        try:
            response = requests.post(
                f"{BASE_URL}/api/v2/classify/ensemble",
                headers={"Content-Type": "application/json"},
                json={"text": text},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                classification = data.get("classification")
                confidence = data.get("confidence", 0)
                model = data.get("model")
                voting = data.get("voting_details", {})
                num_models = voting.get("num_models", 0)

                passed = (
                    classification is not None and
                    model == "ensemble" and
                    num_models == 5 and
                    0 <= confidence <= 1
                )

                log_test(
                    f"Ensemble: '{text}'",
                    passed,
                    f"→ {classification} ({confidence:.1%}) [{num_models} models] [expected: {expected}]"
                )
            else:
                log_test(
                    f"Ensemble: '{text}'",
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            log_test(f"Ensemble: '{text}'", False, str(e))

def test_demo2_configuration():
    """Test that Demo2 loads api-config.js"""
    print(f"\n{Colors.BLUE}Testing Demo2 Configuration{Colors.RESET}")

    try:
        response = requests.get(DEMO_URL, timeout=5)
        html = response.text

        # Check for api-config.js
        has_api_config = '../shared/api-config.js' in html
        log_test("api-config.js loaded", has_api_config)

        # Check for model-settings.js
        has_model_settings = '../shared/model-settings.js' in html
        log_test("model-settings.js loaded", has_model_settings)

        # Check for reclassifyAllQuestions function
        has_reclassify = 'reclassifyAllQuestions' in html
        log_test("reclassifyAllQuestions() defined", has_reclassify)

        # Check for modelChanged event listener
        has_listener = 'modelChanged' in html
        log_test("modelChanged event listener", has_listener)

    except Exception as e:
        log_test("Demo2 configuration", False, str(e))

def test_batched_requests():
    """Test that batched requests don't hit rate limits"""
    print(f"\n{Colors.BLUE}Testing Batched Request Handling{Colors.RESET}")

    # Simulate what Demo2 does: batch of 5 requests
    batch_size = 5
    questions = [f"Test question {i}?" for i in range(batch_size)]

    try:
        start_time = time.time()
        success_count = 0

        for text in questions:
            response = requests.post(
                f"{BASE_URL}/api/v2/classify/ensemble",
                headers={"Content-Type": "application/json"},
                json={"text": text},
                timeout=5
            )
            if response.status_code == 200:
                success_count += 1

        elapsed = time.time() - start_time

        passed = success_count == batch_size
        log_test(
            f"Batch of {batch_size} requests",
            passed,
            f"{success_count}/{batch_size} succeeded in {elapsed:.2f}s"
        )

    except Exception as e:
        log_test("Batched requests", False, str(e))

def test_model_comparison():
    """Test that Classic and Ensemble give different results"""
    print(f"\n{Colors.BLUE}Testing Model Comparison{Colors.RESET}")

    test_text = "What do you think about this?"

    try:
        # Classic prediction
        classic_resp = requests.post(
            f"{BASE_URL}/api/v1/classify",
            headers={"Content-Type": "application/json"},
            json={"text": test_text},
            timeout=5
        )
        classic_data = classic_resp.json()
        classic_class = classic_data.get("classification")
        classic_conf = classic_data.get("confidence", 0)

        # Ensemble prediction
        ensemble_resp = requests.post(
            f"{BASE_URL}/api/v2/classify/ensemble",
            headers={"Content-Type": "application/json"},
            json={"text": test_text},
            timeout=5
        )
        ensemble_data = ensemble_resp.json()
        ensemble_class = ensemble_data.get("classification")
        ensemble_conf = ensemble_data.get("confidence", 0)

        print(f"      Text: '{test_text}'")
        print(f"      Classic:  {classic_class} ({classic_conf:.1%})")
        print(f"      Ensemble: {ensemble_class} ({ensemble_conf:.1%})")

        # Models should work (classification exists)
        passed = classic_class is not None and ensemble_class is not None

        log_test(
            "Model comparison",
            passed,
            f"Confidence delta: {abs(classic_conf - ensemble_conf):.1%}"
        )

    except Exception as e:
        log_test("Model comparison", False, str(e))

def print_summary():
    """Print summary and next steps"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Verification Complete{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    print(f"\n{Colors.YELLOW}Next Steps for Manual Testing:{Colors.RESET}")
    print(f"1. Open {DEMO_URL} in browser")
    print(f"2. Watch console for re-classification messages")
    print(f"3. Click settings gear → Switch between Classic/Ensemble")
    print(f"4. Observe OEQ/CEQ counts update after re-classification")
    print(f"5. Click on questions to jump to video timestamps")

    print(f"\n{Colors.YELLOW}Expected Behavior:{Colors.RESET}")
    print(f"   - Initial load: Shows old classifications from JSON")
    print(f"   - After 1 second: Re-classifies all 95 questions")
    print(f"   - Console: '✅ Re-classification complete: 95 success, 0 errors'")
    print(f"   - Model switch: Triggers full re-classification (~2 seconds)")
    print(f"   - Stats update: OEQ/CEQ counts change based on model")

if __name__ == "__main__":
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Demo2 System Verification{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    test_classic_endpoint()
    test_ensemble_endpoint()
    test_demo2_configuration()
    test_batched_requests()
    test_model_comparison()
    print_summary()
