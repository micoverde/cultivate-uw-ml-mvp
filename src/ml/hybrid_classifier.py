#!/usr/bin/env python3
"""
Hybrid Classifier: Rule-Based + ML for Better CEQ Detection
Fixes the CEQ misclassification issue by adding deterministic rules
"""

import re
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

class HybridClassifier:
    """
    Combines rule-based detection with ML predictions to improve CEQ classification.
    Addresses the critical issue where "Did the tower fall?" is misclassified as OEQ.
    """

    def __init__(self):
        # Strong CEQ indicators - questions that almost always expect yes/no answers
        self.ceq_starters = [
            'did', 'does', 'do',
            'is', 'are', 'was', 'were',
            'will', 'would', 'could', 'should',
            'can', 'may', 'might',
            'has', 'have', 'had',
            'am'
        ]

        # Strong OEQ indicators - questions that require elaboration
        self.oeq_starters = [
            'what', 'how', 'why', 'where', 'when', 'who',
            'which', 'whose', 'whom'
        ]

        # Patterns that indicate CEQ even if they don't start with CEQ words
        self.ceq_patterns = [
            r'\b(yes|no)\b.*\?$',  # Explicitly asks for yes/no
            r'^.{,30}\?$',  # Very short questions are often CEQ
            r'\bor\b.*\?$',  # Questions with "or" often present choices
            r'\b(right|correct|true|false)\b.*\?$',  # Verification questions
        ]

        # Patterns that indicate OEQ
        self.oeq_patterns = [
            r'\btell\s+me\b',  # "Tell me about..."
            r'\bexplain\b',  # "Explain..."
            r'\bdescribe\b',  # "Describe..."
            r'\bthink\b.*\?$',  # "What do you think?"
            r'\bfeel\b.*\?$',  # "How do you feel?"
        ]

    def apply_rules(self, text: str) -> Tuple[str, float, str]:
        """
        Apply rule-based classification.
        Returns: (classification, confidence, reason)
        """
        text_lower = text.lower().strip()

        # Remove punctuation for word matching
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower).strip()
        first_word = text_clean.split()[0] if text_clean.split() else ''

        # Check for strong CEQ starters
        if first_word in self.ceq_starters:
            # Special cases that might still be OEQ
            if 'do you think' in text_lower or 'do you feel' in text_lower:
                return 'OEQ', 0.7, 'Contains "do you think/feel" pattern'

            # Questions like "Did the tower fall?" are definitely CEQ
            if first_word == 'did':
                return 'CEQ', 0.95, f'Starts with "did" - expects yes/no answer'

            return 'CEQ', 0.85, f'Starts with CEQ indicator: {first_word}'

        # Check for strong OEQ starters
        if first_word in self.oeq_starters:
            # Special case: "What?" alone might be CEQ (clarification)
            if len(text_clean.split()) <= 2:
                return 'CEQ', 0.6, 'Very short what/how question'
            return 'OEQ', 0.85, f'Starts with OEQ indicator: {first_word}'

        # Check CEQ patterns
        for pattern in self.ceq_patterns:
            if re.search(pattern, text_lower):
                return 'CEQ', 0.75, f'Matches CEQ pattern: {pattern}'

        # Check OEQ patterns
        for pattern in self.oeq_patterns:
            if re.search(pattern, text_lower):
                return 'OEQ', 0.75, f'Matches OEQ pattern: {pattern}'

        # No strong rule applies
        return None, 0.0, 'No rule applies'

    def classify(self, text: str, ml_prediction: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Hybrid classification combining rules and ML.

        Args:
            text: Question text to classify
            ml_prediction: Optional ML model prediction to combine with rules

        Returns:
            Classification result with explanation
        """
        # Apply rule-based classification
        rule_class, rule_confidence, rule_reason = self.apply_rules(text)

        # If no ML prediction provided, use rules only
        if ml_prediction is None:
            if rule_class:
                logger.warning(f"⚠️ RULE-BASED OVERRIDE (ML unavailable): '{text[:30]}...' -> {rule_class}")
                logger.warning(f"   This is NOT ideal - ML should handle this properly!")
                return {
                    'classification': rule_class,
                    'confidence': rule_confidence,
                    'method': '⚠️ RULE-BASED (NO ML)',
                    'reason': rule_reason,
                    'warning': 'Using rules because ML is unavailable - this is not ideal!',
                    'oeq_probability': 0.0 if rule_class == 'CEQ' else rule_confidence,
                    'ceq_probability': rule_confidence if rule_class == 'CEQ' else 0.0
                }
            else:
                # No rules apply, default to OEQ (safer default)
                return {
                    'classification': 'OEQ',
                    'confidence': 0.5,
                    'method': 'default',
                    'reason': 'No rules apply, defaulting to OEQ',
                    'oeq_probability': 0.5,
                    'ceq_probability': 0.5
                }

        # Combine rule-based and ML predictions
        ml_class = ml_prediction.get('classification', 'OEQ')
        ml_confidence = ml_prediction.get('confidence', 0.5)
        ml_oeq_prob = ml_prediction.get('oeq_probability', 0.5)
        ml_ceq_prob = ml_prediction.get('ceq_probability', 0.5)

        # If rules are strong (>0.8 confidence), override ML
        if rule_class and rule_confidence > 0.8:
            logger.warning(f"🚨 RULE OVERRIDE TRIGGERED - ML FAILED!")
            logger.warning(f"   Question: '{text[:50]}...'")
            logger.warning(f"   ML said: {ml_class} (confidence: {ml_confidence:.2f})")
            logger.warning(f"   Rule says: {rule_class} (confidence: {rule_confidence:.2f})")
            logger.warning(f"   Reason: {rule_reason}")
            logger.warning(f"   ⚠️ This indicates ML model needs retraining!")
            return {
                'classification': rule_class,
                'confidence': rule_confidence,
                'method': '🚨 RULE-OVERRIDE (ML FAILED)',
                'reason': f'Strong rule: {rule_reason}',
                'ml_classification': ml_class,
                'ml_confidence': ml_confidence,
                'warning': 'ML model failed - using rules as fallback. Model needs retraining!',
                'oeq_probability': 0.1 if rule_class == 'CEQ' else 0.9,
                'ceq_probability': 0.9 if rule_class == 'CEQ' else 0.1
            }

        # If rules and ML agree, boost confidence
        if rule_class == ml_class:
            combined_confidence = min(0.99, (rule_confidence + ml_confidence) / 2 * 1.2)
            return {
                'classification': ml_class,
                'confidence': combined_confidence,
                'method': 'hybrid-agreement',
                'reason': f'Rule and ML agree: {rule_reason}',
                'oeq_probability': ml_oeq_prob,
                'ceq_probability': ml_ceq_prob
            }

        # If they disagree, weight based on confidence
        if rule_class and rule_confidence > 0.6:
            # Rules have moderate confidence, blend predictions
            rule_weight = rule_confidence
            ml_weight = ml_confidence * 0.7  # Slightly reduce ML weight due to known CEQ issues

            if rule_class == 'CEQ':
                ceq_score = rule_weight
                oeq_score = ml_weight if ml_class == 'OEQ' else 0
            else:
                oeq_score = rule_weight
                ceq_score = ml_weight if ml_class == 'CEQ' else 0

            if ceq_score > oeq_score:
                return {
                    'classification': 'CEQ',
                    'confidence': ceq_score / (ceq_score + oeq_score),
                    'method': 'hybrid-weighted',
                    'reason': f'Weighted decision: {rule_reason}',
                    'oeq_probability': oeq_score / (ceq_score + oeq_score),
                    'ceq_probability': ceq_score / (ceq_score + oeq_score)
                }
            else:
                return {
                    'classification': 'OEQ',
                    'confidence': oeq_score / (ceq_score + oeq_score),
                    'method': 'hybrid-weighted',
                    'reason': f'Weighted decision favoring ML',
                    'oeq_probability': oeq_score / (ceq_score + oeq_score),
                    'ceq_probability': ceq_score / (ceq_score + oeq_score)
                }

        # Rules have low confidence, trust ML more
        return {
            'classification': ml_class,
            'confidence': ml_confidence * 0.9,  # Slightly reduce confidence due to disagreement
            'method': 'hybrid-ml-primary',
            'reason': 'Weak rules, trusting ML',
            'rule_suggestion': rule_class,
            'rule_reason': rule_reason,
            'oeq_probability': ml_oeq_prob,
            'ceq_probability': ml_ceq_prob
        }


# Test the classifier
if __name__ == "__main__":
    classifier = HybridClassifier()

    test_questions = [
        "Did the tower fall?",
        "What happened to the tower?",
        "Can you build a tower?",
        "Is the tower tall?",
        "How tall is your tower?",
        "Why did the tower fall down?",
        "Was it fun?",
        "Do you like building?",
        "Tell me about your tower",
        "Explain how you built it"
    ]

    print("\nHybrid Classifier Test Results")
    print("=" * 60)

    for question in test_questions:
        result = classifier.classify(question)
        print(f"\nQ: {question}")
        print(f"   Classification: {result['classification']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Method: {result['method']}")
        print(f"   Reason: {result['reason']}")