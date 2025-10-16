"""
Shared Feature Extraction for OEQ vs CEQ Classification

This module provides consistent feature extraction across:
- Training (train_ensemble_production.py)
- Inference (API endpoints)
- Evaluation and testing

Author: Claude (Partner-Level Microsoft SDE)
Issue: #196 - Model selection settings and improved classification
"""

import numpy as np
from typing import List, Dict, Any


class QuestionFeatureExtractor:
    """
    Extracts 19 features from educator questions to classify them as:
    - OEQ (Open-Ended Questions): encourage critical thinking, exploration
    - CEQ (Closed-Ended Questions): yes/no, simple factual answers

    Key OEQ Indicators:
    - "how" and "why" - encourage explanation and reasoning
    - "what do you think" - solicits personal perspective
    - "describe", "explain", "tell me about" - request elaboration

    Key CEQ Indicators:
    - "did", "is", "are", "was", "were" - yes/no questions
    - "can", "could", "would", "should" - ability/permission questions
    - "do", "does" - simple action questions
    """

    def __init__(self):
        self.feature_names = [
            'word_count',              # 0
            'has_question_mark',       # 1
            'char_count',              # 2
            'has_how',                 # 3
            'has_why',                 # 4
            'has_what',                # 5
            'has_when',                # 6
            'has_where',               # 7
            'has_who',                 # 8
            'has_what_think',          # 9  - "what do you think"
            'has_describe_explain',    # 10 - "describe", "explain", "tell me about"
            'has_did',                 # 11 - CEQ indicator
            'has_is_are',              # 12 - CEQ indicator
            'has_can_could',           # 13 - CEQ indicator
            'has_do_does',             # 14 - CEQ indicator
            'oeq_score',               # 15 - sum of OEQ indicators
            'ceq_score',               # 16 - sum of CEQ indicators
            'question_mark_count',     # 17
            'is_long',                 # 18 - more than 5 words
        ]

    def extract(self, text: str) -> np.ndarray:
        """
        Extract features from a single text string.

        Args:
            text: The question text to analyze

        Returns:
            numpy array of shape (19,) containing extracted features
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Strong OEQ indicators
        has_how = 1 if ' how ' in f' {text_lower} ' or text_lower.startswith('how ') else 0
        has_why = 1 if ' why ' in f' {text_lower} ' or text_lower.startswith('why ') else 0
        has_what_think = 1 if 'what do you think' in text_lower or 'what did you think' in text_lower else 0
        has_describe_explain = 1 if any(w in text_lower for w in [' describe ', ' explain ', ' tell me about ']) else 0

        # Question word features
        has_what = 1 if ' what ' in f' {text_lower} ' or text_lower.startswith('what ') else 0
        has_when = 1 if ' when ' in f' {text_lower} ' or text_lower.startswith('when ') else 0
        has_where = 1 if ' where ' in f' {text_lower} ' or text_lower.startswith('where ') else 0
        has_who = 1 if ' who ' in f' {text_lower} ' or text_lower.startswith('who ') else 0

        # CEQ indicators (yes/no questions)
        has_did = 1 if ' did ' in text_lower or text_lower.startswith('did ') else 0
        has_is_are = 1 if any(w in text_lower for w in [' is ', ' are ', ' was ', ' were ', 'is ', 'are ']) else 0
        has_can_could = 1 if any(w in text_lower for w in [' can ', ' could ', ' would ', ' should ', 'can ', 'could ']) else 0
        has_do_does = 1 if any(w in text_lower for w in [' do ', ' does ', 'do ', 'does ']) else 0

        # Scores
        oeq_score = has_how + has_why + has_what_think + has_describe_explain
        ceq_score = has_did + has_is_are + has_can_could + has_do_does

        # Build feature vector
        features = np.array([
            word_count,                      # 0
            1 if '?' in text else 0,         # 1
            len(text),                       # 2
            has_how,                         # 3
            has_why,                         # 4
            has_what,                        # 5
            has_when,                        # 6
            has_where,                       # 7
            has_who,                         # 8
            has_what_think,                  # 9
            has_describe_explain,            # 10
            has_did,                         # 11
            has_is_are,                      # 12
            has_can_could,                   # 13
            has_do_does,                     # 14
            oeq_score,                       # 15
            ceq_score,                       # 16
            text.count('?'),                 # 17
            1 if word_count > 5 else 0,      # 18
        ], dtype=np.float32)

        return features

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from multiple texts.

        Args:
            texts: List of question texts to analyze

        Returns:
            numpy array of shape (n_samples, 19) containing extracted features
        """
        return np.array([self.extract(text) for text in texts])

    def get_feature_names(self) -> List[str]:
        """Get the names of all features in order."""
        return self.feature_names.copy()

    def explain_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features and return with explanations.

        Args:
            text: The question text to analyze

        Returns:
            Dictionary with features and human-readable explanations
        """
        features = self.extract(text)

        explanation = {
            'text': text,
            'features': dict(zip(self.feature_names, features)),
            'oeq_indicators': {
                'has_how': bool(features[3]),
                'has_why': bool(features[4]),
                'has_what_think': bool(features[9]),
                'has_describe_explain': bool(features[10]),
                'total_score': int(features[15])
            },
            'ceq_indicators': {
                'has_did': bool(features[11]),
                'has_is_are': bool(features[12]),
                'has_can_could': bool(features[13]),
                'has_do_does': bool(features[14]),
                'total_score': int(features[16])
            },
            'interpretation': self._interpret(features)
        }

        return explanation

    def _interpret(self, features: np.ndarray) -> str:
        """Provide human-readable interpretation of features."""
        oeq_score = int(features[15])
        ceq_score = int(features[16])

        if oeq_score > ceq_score:
            return f"Likely OEQ - {oeq_score} OEQ indicators vs {ceq_score} CEQ indicators"
        elif ceq_score > oeq_score:
            return f"Likely CEQ - {ceq_score} CEQ indicators vs {oeq_score} OEQ indicators"
        else:
            return f"Ambiguous - {oeq_score} OEQ indicators = {ceq_score} CEQ indicators"


# Singleton instance for easy import
feature_extractor = QuestionFeatureExtractor()


if __name__ == "__main__":
    # Test the feature extractor
    extractor = QuestionFeatureExtractor()

    test_cases = [
        ("What do you think happened?", "OEQ"),
        ("How did that make you feel?", "OEQ"),
        ("Why do you think that happened?", "OEQ"),
        ("Can you describe what you saw?", "OEQ"),
        ("Did you like it?", "CEQ"),
        ("Is this correct?", "CEQ"),
        ("Can you do it?", "CEQ"),
        ("Do you want more?", "CEQ"),
        ("Why did it fall?", "OEQ"),
    ]

    print("=" * 80)
    print("QUESTION FEATURE EXTRACTOR TEST")
    print("=" * 80)
    print()

    for text, expected in test_cases:
        explanation = extractor.explain_features(text)

        print(f"Text: {text}")
        print(f"Expected: {expected}")
        print(f"OEQ Score: {explanation['oeq_indicators']['total_score']}")
        print(f"CEQ Score: {explanation['ceq_indicators']['total_score']}")
        print(f"Interpretation: {explanation['interpretation']}")
        print()
