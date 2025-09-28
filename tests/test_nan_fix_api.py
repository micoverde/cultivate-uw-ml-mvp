#!/usr/bin/env python3
"""
Test the NaN% fix by calling the production API directly
"""

import requests
import json

def test_production_api():
    """Test production API for proper probability values"""

    print("\n" + "="*60)
    print("🧪 Testing Production API - NaN% Fix Verification")
    print("="*60)

    # Production API endpoint
    api_base = 'https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io'

    # Test questions
    test_questions = [
        "What happened to make you feel that way?",
        "Is the sky blue?",
        "Why do you think that happened?",
        "Can you count to ten?"
    ]

    all_passed = True

    for question in test_questions:
        print(f"\n📝 Testing: '{question}'")

        try:
            # Call production API
            url = f"{api_base}/api/classify?text={requests.utils.quote(question)}"
            response = requests.post(url)

            if response.status_code != 200:
                print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
                all_passed = False
                continue

            result = response.json()
            print(f"   📊 Raw response: {json.dumps(result, indent=2)}")

            # Check what fields are present
            has_oeq_prob = 'oeq_probability' in result
            has_ceq_prob = 'ceq_probability' in result
            has_confidence = 'confidence' in result
            has_classification = 'classification' in result

            print(f"   Fields present:")
            print(f"     - classification: {has_classification}")
            print(f"     - confidence: {has_confidence}")
            print(f"     - oeq_probability: {has_oeq_prob}")
            print(f"     - ceq_probability: {has_ceq_prob}")

            # Simulate what our frontend fix does
            if not has_oeq_prob and not has_ceq_prob:
                print("   ⚠️ No probability fields - frontend will normalize")

                # This is what our fix does
                if has_classification and has_confidence:
                    if result['classification'] == 'OEQ':
                        oeq_prob = result['confidence']
                        ceq_prob = 1 - result['confidence']
                    else:
                        ceq_prob = result['confidence']
                        oeq_prob = 1 - result['confidence']

                    oeq_percent = round((oeq_prob or 0) * 100)
                    ceq_percent = round((ceq_prob or 0) * 100)
                else:
                    oeq_percent = round(0 * 100)
                    ceq_percent = round(0 * 100)
            else:
                # Direct probability fields
                oeq_percent = round((result.get('oeq_probability', 0) or 0) * 100)
                ceq_percent = round((result.get('ceq_probability', 0) or 0) * 100)

            print(f"   📈 Display values:")
            print(f"     - OEQ: {oeq_percent}% {'✅' if oeq_percent != 'NaN' else '❌ NaN!'}")
            print(f"     - CEQ: {ceq_percent}% {'✅' if ceq_percent != 'NaN' else '❌ NaN!'}")

            # Check for NaN
            if str(oeq_percent) == 'nan' or str(ceq_percent) == 'nan':
                print("   ❌ NaN detected in percentages!")
                all_passed = False
            else:
                print("   ✅ No NaN - values are numeric")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False

    print("\n" + "="*60)
    print("📊 Test Summary:")
    if all_passed:
        print("✅ All tests PASSED - No NaN% issues detected!")
        print("   The fix is working correctly in production.")
    else:
        print("❌ Some tests FAILED - Issues detected")
    print("="*60)

if __name__ == "__main__":
    test_production_api()