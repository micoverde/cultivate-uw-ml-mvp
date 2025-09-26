#!/usr/bin/env python3
"""
Enhanced Feature Extractor for Better OEQ/CEQ Classification
Incorporates educational domain knowledge and linguistic patterns

Warren - This implements the insights from issues #118 and #120!
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional
import pandas as pd

class EnhancedQuestionFeatureExtractor:
    """Extract features specifically designed for OEQ vs CEQ classification"""

    def __init__(self):
        # Educational patterns that strongly indicate question types
        self.oeq_indicators = {
            'why': 3.0,          # Strong OEQ signal
            'explain': 3.0,      # Strong OEQ signal
            'think': 2.5,        # Promotes thinking
            'feel': 2.0,         # Personal response
            'could': 1.5,        # Possibilities
            'would': 1.5,        # Hypothetical
            'should': 1.5,       # Judgment
            'might': 1.5,        # Speculation
            'describe': 2.0,     # Description needed
            'tell me about': 2.5,  # Open narrative
            'what if': 2.5,      # Hypothetical scenario
            'compare': 2.0,      # Analysis required
            'difference': 1.5,   # Comparison (but context matters)
        }

        self.ceq_indicators = {
            'is': 2.0,           # Yes/no likely
            'are': 2.0,          # Yes/no likely
            'do': 1.5,           # Often yes/no
            'does': 1.5,         # Often yes/no
            'can': 1.5,          # Ability check
            'will': 1.5,         # Future certainty
            'how many': 3.0,     # Counting - strong CEQ
            'how much': 3.0,     # Quantity - strong CEQ
            'what color': 2.5,   # Single attribute
            'what shape': 2.5,   # Single attribute
            'where': 2.0,        # Location - usually single answer
            'when': 2.0,         # Time - usually single answer
            'who': 2.0,          # Person - usually single answer
            'which': 2.0,        # Selection from options
            'name': 2.5,         # Naming task
            'what is this': 2.5, # Identification
        }

        # Bloom's Taxonomy levels (1-6)
        self.bloom_keywords = {
            1: ['what', 'when', 'where', 'who', 'which', 'name', 'list'],  # Remember
            2: ['describe', 'explain', 'summarize', 'interpret'],           # Understand
            3: ['how', 'show', 'demonstrate', 'use'],                       # Apply
            4: ['why', 'compare', 'contrast', 'analyze', 'different'],      # Analyze
            5: ['judge', 'evaluate', 'rate', 'recommend'],                  # Evaluate
            6: ['create', 'design', 'invent', 'imagine', 'what if']         # Create
        }

    def extract_features(self, question_text: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Extract comprehensive features for OEQ/CEQ classification

        Args:
            question_text: The question to analyze
            metadata: Optional metadata (wait_time, age_group, etc.)

        Returns:
            Feature vector for ML model
        """
        if not question_text:
            question_text = ""

        question_lower = question_text.lower().strip()
        features = []

        # 1. Basic linguistic features (10 features)
        features.extend(self._extract_basic_features(question_text))

        # 2. OEQ/CEQ indicator scores (2 features)
        oeq_score = self._calculate_oeq_score(question_lower)
        ceq_score = self._calculate_ceq_score(question_lower)
        features.extend([oeq_score, ceq_score])

        # 3. Question starter patterns (15 features)
        features.extend(self._extract_starter_patterns(question_lower))

        # 4. Bloom's Taxonomy level (6 features - one-hot)
        bloom_level = self._estimate_bloom_level(question_lower)
        bloom_onehot = [0] * 6
        if 1 <= bloom_level <= 6:
            bloom_onehot[bloom_level - 1] = 1
        features.extend(bloom_onehot)

        # 5. Complexity metrics (5 features)
        features.extend(self._extract_complexity_features(question_text))

        # 6. Educational context features (8 features)
        features.extend(self._extract_educational_features(question_lower))

        # 7. Metadata features if available (5 features)
        if metadata:
            features.extend([
                float(metadata.get('has_wait_time', False)),
                float(metadata.get('is_yes_no', False)),
                1.0 if metadata.get('age_group') == 'PK' else 0.0,
                1.0 if metadata.get('age_group') == 'TODDLER' else 0.0,
                1.0 if metadata.get('age_group') == 'INFANT' else 0.0
            ])
        else:
            features.extend([0.0] * 5)

        # 8. Special patterns that override classification (5 features)
        features.extend(self._extract_override_patterns(question_lower))

        return np.array(features, dtype=np.float32)

    def _extract_basic_features(self, text: str) -> List[float]:
        """Extract basic linguistic features"""
        return [
            len(text),                                    # Length
            text.count(' ') + 1,                         # Word count
            text.count('?'),                              # Question marks
            text.count(','),                              # Commas (complexity)
            text.count('or'),                             # Options presented
            float(text.endswith('?')),                    # Proper question
            float(text.startswith(('what', 'how', 'why', 'when', 'where', 'who'))),
            float('...' in text),                         # Ellipsis (incomplete)
            float(any(word in text.lower() for word in ['think', 'feel', 'believe'])),
            float(len(text.split()) > 7)                  # Long question
        ]

    def _calculate_oeq_score(self, text: str) -> float:
        """Calculate OEQ indicator score"""
        score = 0.0
        for pattern, weight in self.oeq_indicators.items():
            if pattern in text:
                score += weight
        return min(score, 10.0)  # Cap at 10

    def _calculate_ceq_score(self, text: str) -> float:
        """Calculate CEQ indicator score"""
        score = 0.0
        for pattern, weight in self.ceq_indicators.items():
            if pattern in text:
                score += weight
        return min(score, 10.0)  # Cap at 10

    def _extract_starter_patterns(self, text: str) -> List[float]:
        """Extract question starter patterns (first 1-2 words)"""
        starters = ['what', 'how', 'why', 'when', 'where', 'who', 'which',
                   'is', 'are', 'do', 'does', 'can', 'will', 'would', 'should']

        features = []
        for starter in starters:
            if text.startswith(starter):
                features.append(1.0)
            else:
                features.append(0.0)

        return features

    def _estimate_bloom_level(self, text: str) -> int:
        """Estimate Bloom's Taxonomy level (1-6)"""
        for level in [6, 5, 4, 3, 2, 1]:  # Check from highest to lowest
            keywords = self.bloom_keywords[level]
            if any(keyword in text for keyword in keywords):
                return level
        return 1  # Default to Remember level

    def _extract_complexity_features(self, text: str) -> List[float]:
        """Extract question complexity features"""
        words = text.split()
        return [
            len(set(words)) / max(len(words), 1),        # Vocabulary diversity
            float(any(w in text for w in ['because', 'if', 'when', 'while'])),  # Conditionals
            float('?' in text and ',' in text),          # Multi-part question
            text.count('and') + text.count('or'),        # Conjunctions
            float(len(words) > 10)                       # Very long question
        ]

    def _extract_educational_features(self, text: str) -> List[float]:
        """Extract education-specific features"""
        return [
            float('yes' in text or 'no' in text),        # Explicit yes/no
            float('one' in text or 'single' in text),    # Single answer expected
            float('many' in text or 'all' in text),      # Multiple answers
            float('different' in text or 'same' in text), # Comparison
            float('example' in text or 'such as' in text), # Examples requested
            float('your' in text or 'you' in text),      # Personal response
            float('color' in text or 'shape' in text or 'size' in text),  # Attributes
            float('count' in text or 'number' in text)   # Counting task
        ]

    def _extract_override_patterns(self, text: str) -> List[float]:
        """Extract patterns that strongly override other signals"""
        return [
            float('how many' in text or 'how much' in text),  # Strong CEQ
            float('yes or no' in text or 'yes/no' in text),   # Explicit CEQ
            float('explain why' in text or 'explain how' in text),  # Strong OEQ
            float('what do you think' in text),               # Strong OEQ
            float('tell me about' in text)                    # Strong OEQ
        ]

    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability"""
        names = []

        # Basic features
        names.extend(['length', 'word_count', 'question_marks', 'commas', 'or_count',
                     'ends_with_q', 'starts_with_qword', 'has_ellipsis', 'has_think_feel',
                     'is_long'])

        # Indicator scores
        names.extend(['oeq_score', 'ceq_score'])

        # Starter patterns
        names.extend([f'starts_{s}' for s in ['what', 'how', 'why', 'when', 'where',
                     'who', 'which', 'is', 'are', 'do', 'does', 'can', 'will',
                     'would', 'should']])

        # Bloom's levels
        names.extend([f'bloom_{i}' for i in range(1, 7)])

        # Complexity
        names.extend(['vocab_diversity', 'has_conditionals', 'multipart',
                     'conjunctions', 'very_long'])

        # Educational
        names.extend(['has_yes_no', 'single_answer', 'multiple_answers',
                     'comparison', 'examples', 'personal', 'attributes', 'counting'])

        # Metadata
        names.extend(['has_wait_time', 'is_yes_no_meta', 'age_pk', 'age_toddler',
                     'age_infant'])

        # Override patterns
        names.extend(['how_many_much', 'explicit_yes_no', 'explain_why_how',
                     'what_think', 'tell_about'])

        return names

    def extract_from_csv_data(self, csv_path: str) -> tuple:
        """Extract features from the CSV training data"""
        import json

        # Load the extracted questions
        with open('extracted_questions_for_retraining.json', 'r') as f:
            questions = json.load(f)

        X = []
        y = []
        metadata_list = []

        for q in questions:
            # Get question text or use description
            text = q['question_text'] if q['question_text'] else q['full_description'][:50]

            # Extract metadata
            metadata = {
                'has_wait_time': q['has_wait_time'],
                'is_yes_no': q['is_yes_no'],
                'age_group': 'PK'  # Default, could be extracted from video name
            }

            # Extract features
            features = self.extract_features(text, metadata)
            X.append(features)

            # Label (0 for CEQ, 1 for OEQ)
            y.append(1 if q['label'] == 'OEQ' else 0)

            metadata_list.append({
                'video': q['video'],
                'question_num': q['question_num'],
                'original_text': text,
                'label': q['label']
            })

        return np.array(X), np.array(y), metadata_list

if __name__ == "__main__":
    # Test the enhanced feature extractor
    extractor = EnhancedQuestionFeatureExtractor()

    # Test questions
    test_questions = [
        ("What color is the sky?", "CEQ"),  # Should be CEQ
        ("Why do you think plants need water?", "OEQ"),  # Should be OEQ
        ("How many blocks are there?", "CEQ"),  # Should be CEQ
        ("How do they look different?", "OEQ"),  # Should be OEQ
        ("Can you see the board?", "CEQ"),  # Should be CEQ
        ("What would happen if we mixed these colors?", "OEQ"),  # Should be OEQ
    ]

    print("Testing Enhanced Feature Extraction:")
    print("="*70)

    for question, expected in test_questions:
        features = extractor.extract_features(question)
        oeq_score = features[10]  # OEQ score position
        ceq_score = features[11]  # CEQ score position

        predicted = "OEQ" if oeq_score > ceq_score else "CEQ"

        print(f"\nQuestion: \"{question}\"")
        print(f"  Expected: {expected}")
        print(f"  OEQ Score: {oeq_score:.2f}, CEQ Score: {ceq_score:.2f}")
        print(f"  Predicted: {predicted} {'✅' if predicted == expected else '❌'}")

    print("\n" + "="*70)
    print(f"Feature vector size: {len(features)} features")
    print("Ready for retraining with enhanced features!")