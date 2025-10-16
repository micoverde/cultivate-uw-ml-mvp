#!/usr/bin/env python3
"""
Comprehensive API testing with ground truth examples.

Tests the /api/v2/classify/ensemble endpoint with validated CEQ/OEQ examples
to measure accuracy, precision, recall, and F1 scores.
"""

import requests
import json
from typing import List, Dict, Tuple
from collections import defaultdict

API_URL = "http://localhost:5001/api/v2/classify/ensemble"

# Ground truth examples from training data and validated sources
CEQ_EXAMPLES = [
    # Yes/No questions (definitively CEQ)
    "Did you like it?",
    "Is this correct?",
    "Can you do it?",
    "Do you want more?",
    "Are you ready?",
    "Did you finish?",
    "Is it big?",
    "Was it fun?",
    "Do you understand?",
    "Are you happy?",
    "Did it work?",
    "Can you see it?",
    "Is it red?",
    "Does it fit?",
    "Will you come?",
    
    # Multiple choice / selection questions (CEQ)
    "Which one is bigger?",
    "What color is it?",
    "How many are there?",
    "Who did this?",
    "Where is it?",
    "When did it happen?",
    
    # Factual recall questions (CEQ)
    "What is the capital of France?",
    "How old are you?",
    "What time is it?",
    "What day is today?",
]

OEQ_EXAMPLES = [
    # How questions requiring explanation (definitively OEQ)
    "How did that happen?",
    "How can we make it better?",
    "How do you think it works?",
    "How would you solve this?",
    "How does this make you feel?",
    
    # Why questions requiring reasoning (definitively OEQ)
    "Why did it fall?",
    "Why do you think so?",
    "Why did you choose that?",
    "Why is this important?",
    "Why does it matter?",
    
    # What think/feel questions (OEQ)
    "What do you think happened?",
    "What do you think about this?",
    "What are your thoughts?",
    "What would you do differently?",
    "What did you learn?",
    "What does this mean to you?",
    
    # Describe/explain questions (OEQ)
    "Can you explain why?",
    "Tell me about your process",
    "Describe what happened",
    "Explain your reasoning",
    "What strategy did you use?",
    
    # Hypothetical/creative questions (OEQ)
    "What if we tried something else?",
    "How else could we do this?",
    "What other solutions are there?",
]

def test_question(text: str, expected_label: str) -> Dict:
    """Test a single question and return results."""
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=5
        )
        response.raise_for_status()
        result = response.json()
        
        predicted_label = result.get("classification")
        confidence = result.get("confidence", 0)
        voting = result.get("voting_details", {})
        
        is_correct = predicted_label == expected_label
        
        return {
            "text": text,
            "expected": expected_label,
            "predicted": predicted_label,
            "correct": is_correct,
            "confidence": confidence,
            "probabilities": result.get("probabilities", {}),
            "vote_tally": voting.get("vote_tally", {}),
            "vote_summary": voting.get("vote_summary", ""),
        }
    except Exception as e:
        return {
            "text": text,
            "expected": expected_label,
            "error": str(e),
            "correct": False
        }

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate precision, recall, F1, and accuracy."""
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {"error": "No valid results"}
    
    # Count true positives, false positives, false negatives, true negatives
    tp = sum(1 for r in valid_results if r["expected"] == "OEQ" and r["predicted"] == "OEQ")
    fp = sum(1 for r in valid_results if r["expected"] == "CEQ" and r["predicted"] == "OEQ")
    fn = sum(1 for r in valid_results if r["expected"] == "OEQ" and r["predicted"] == "CEQ")
    tn = sum(1 for r in valid_results if r["expected"] == "CEQ" and r["predicted"] == "CEQ")
    
    total = len(valid_results)
    correct = sum(1 for r in valid_results if r["correct"])
    
    accuracy = correct / total if total > 0 else 0
    
    precision_oeq = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_oeq = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_oeq = 2 * (precision_oeq * recall_oeq) / (precision_oeq + recall_oeq) if (precision_oeq + recall_oeq) > 0 else 0
    
    precision_ceq = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_ceq = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_ceq = 2 * (precision_ceq * recall_ceq) / (precision_ceq + recall_ceq) if (precision_ceq + recall_ceq) > 0 else 0
    
    balanced_accuracy = (recall_oeq + recall_ceq) / 2
    
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "confusion_matrix": {
            "true_positives_oeq": tp,
            "false_positives_oeq": fp,
            "false_negatives_oeq": fn,
            "true_negatives_ceq": tn,
        },
        "oeq_metrics": {
            "precision": precision_oeq,
            "recall": recall_oeq,
            "f1_score": f1_oeq,
        },
        "ceq_metrics": {
            "precision": precision_ceq,
            "recall": recall_ceq,
            "f1_score": f1_ceq,
        }
    }

def main():
    print("=" * 80)
    print("🧪 COMPREHENSIVE API TESTING WITH GROUND TRUTH")
    print("=" * 80)
    print()
    
    # Test CEQ examples
    print(f"Testing {len(CEQ_EXAMPLES)} CEQ examples...")
    ceq_results = []
    for text in CEQ_EXAMPLES:
        result = test_question(text, "CEQ")
        ceq_results.append(result)
        status = "✅" if result["correct"] else "❌"
        conf = result.get("confidence", 0)
        print(f"  {status} '{text[:50]}...' -> {result.get('predicted', 'ERROR')} ({conf:.1%})")
    
    print()
    
    # Test OEQ examples
    print(f"Testing {len(OEQ_EXAMPLES)} OEQ examples...")
    oeq_results = []
    for text in OEQ_EXAMPLES:
        result = test_question(text, "OEQ")
        oeq_results.append(result)
        status = "✅" if result["correct"] else "❌"
        conf = result.get("confidence", 0)
        print(f"  {status} '{text[:50]}...' -> {result.get('predicted', 'ERROR')} ({conf:.1%})")
    
    print()
    print("=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    # Calculate overall metrics
    all_results = ceq_results + oeq_results
    metrics = calculate_metrics(all_results)
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"  Correct: {metrics['correct']}/{metrics['total_samples']}")
    print(f"  Incorrect: {metrics['incorrect']}/{metrics['total_samples']}")
    
    print(f"\n📈 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                  Predicted CEQ  Predicted OEQ")
    print(f"  Actual CEQ      {cm['true_negatives_ceq']:>13}  {cm['false_positives_oeq']:>13}")
    print(f"  Actual OEQ      {cm['false_negatives_oeq']:>13}  {cm['true_positives_oeq']:>13}")
    
    print(f"\n📊 Per-Class Metrics:")
    print(f"\n  CEQ Metrics:")
    ceq = metrics['ceq_metrics']
    print(f"    Precision: {ceq['precision']:.1%}")
    print(f"    Recall:    {ceq['recall']:.1%}")
    print(f"    F1-Score:  {ceq['f1_score']:.1%}")
    
    print(f"\n  OEQ Metrics:")
    oeq = metrics['oeq_metrics']
    print(f"    Precision: {oeq['precision']:.1%}")
    print(f"    Recall:    {oeq['recall']:.1%}")
    print(f"    F1-Score:  {oeq['f1_score']:.1%}")
    
    # Show misclassifications
    print()
    print("=" * 80)
    print("❌ MISCLASSIFICATIONS (if any)")
    print("=" * 80)
    
    errors = [r for r in all_results if not r["correct"] and "error" not in r]
    if errors:
        for err in errors:
            print(f"\n  Text: '{err['text']}'")
            print(f"  Expected: {err['expected']}")
            print(f"  Predicted: {err['predicted']} ({err['confidence']:.1%} confidence)")
            print(f"  Vote: {err.get('vote_summary', 'N/A')}")
    else:
        print("\n  🎉 No misclassifications! Perfect accuracy!")
    
    print()
    print("=" * 80)
    
    # Return exit code based on accuracy
    if metrics['accuracy'] >= 0.90:
        print("✅ TEST PASSED: Accuracy >= 90%")
        return 0
    else:
        print(f"⚠️  TEST FAILED: Accuracy {metrics['accuracy']:.1%} < 90%")
        return 1

if __name__ == "__main__":
    exit(main())
