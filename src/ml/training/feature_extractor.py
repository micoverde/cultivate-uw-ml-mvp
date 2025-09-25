#!/usr/bin/env python3
"""
Expert Annotation Feature Extractor for Classical ML Training
Implements Issue #109 design specification for transforming CSV annotations
into scikit-learn compatible feature vectors.

This module converts 26 videos x 119 expert annotations into training data
optimized for RandomForest, SVM, and GradientBoosting models.

Author: Claude-4 (Partner-Level Microsoft SDE)
Issue: #109 - Classical ML Model Training on Expert Annotations
Context: Phase 1 of Issue #76 Comprehensive ML Architecture
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class QuestionAnnotation:
    """Structured representation of a single question annotation."""
    video_title: str
    asset_id: str
    age_group: str
    timestamp: str
    question_time: str
    description: str
    question_type: str  # OEQ, CEQ, Rhetorical
    has_pause: bool
    wait_time_quality: str  # appropriate, insufficient, none
    video_description: str

class ExpertAnnotationFeatureExtractor:
    """
    Converts CSV annotations to scikit-learn compatible feature vectors.
    Optimized for educational domain with 119 expert-labeled examples.

    Features Generated:
    - Linguistic Features (40D): TF-IDF, keywords, question length
    - Temporal Features (15D): Pause patterns, timing information
    - Contextual Features (25D): Age group, video context, activity type
    - Total Dimensionality: 80 features per sample
    """

    def __init__(self):
        self.tfidf_vectorizer = None
        self.age_group_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        self._initialize_feature_extractors()

    def _initialize_feature_extractors(self):
        """Initialize feature extraction components."""
        # TF-IDF for question descriptions (limited vocabulary for small dataset)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20,  # Limited for small dataset
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Keep all terms due to limited data
            max_df=0.8
        )

        # Educational domain keywords
        self.oeq_keywords = [
            'what', 'how', 'why', 'when', 'where', 'which', 'who',
            'describe', 'explain', 'tell me', 'think', 'feel',
            'open', 'opportunity', 'response'
        ]

        self.ceq_keywords = [
            'yes', 'no', 'is', 'are', 'can', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'closed'
        ]

        self.pause_keywords = [
            'pause', 'pauses', 'wait', 'waits', 'listens', 'opportunity',
            'response', 'continues', 'talking', 'sharing', 'rhetorical'
        ]

    def extract_features(self, csv_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Primary feature extraction from CSV annotations.

        Args:
            csv_path: Path to VideosAskingQuestions CSV.csv

        Returns:
            - X: Feature matrix (119, 80)
            - y: Multi-task labels dict
        """
        logger.info(f"🔍 Starting feature extraction from {csv_path}")

        # Load and parse CSV
        questions = self._parse_questions_from_csv(csv_path)
        logger.info(f"📊 Parsed {len(questions)} question annotations")

        # Extract features and labels
        features = []
        labels = {
            'question_type': [],
            'wait_time': [],
            'class_scores': []
        }

        # First pass: collect all descriptions for TF-IDF fitting
        all_descriptions = [q.description for q in questions]
        self.tfidf_vectorizer.fit(all_descriptions)

        # Fit age group encoder
        age_groups = [q.age_group for q in questions]
        self.age_group_encoder.fit(age_groups)

        # Second pass: extract features for each question
        for i, question in enumerate(questions):
            try:
                # Extract multi-modal features
                ling_features = self._extract_linguistic_features(question)
                temp_features = self._extract_temporal_features(question)
                cont_features = self._extract_contextual_features(question)

                # Concatenate all features
                feature_vector = np.concatenate([ling_features, temp_features, cont_features])
                features.append(feature_vector)

                # Extract labels
                labels['question_type'].append(self._classify_question_type(question))
                labels['wait_time'].append(self._assess_wait_time(question))
                labels['class_scores'].append(self._estimate_class_score(question))

            except Exception as e:
                logger.warning(f"⚠️ Error processing question {i}: {e}")
                continue

        # Convert to numpy arrays
        X = np.array(features)
        for key in labels:
            labels[key] = np.array(labels[key])

        # Feature scaling
        X = self.scaler.fit_transform(X)

        logger.info(f"✅ Feature extraction complete: {X.shape[0]} samples × {X.shape[1]} features")
        return X, labels

    def _parse_questions_from_csv(self, csv_path: str) -> List[QuestionAnnotation]:
        """Parse CSV file and extract individual question annotations."""
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        questions = []

        for _, row in df.iterrows():
            # Parse each question column (Q1-Q8)
            for q_num in range(1, 9):
                # Handle inconsistent column naming (some have spaces, some don't)
                possible_q_cols = [f'Question {q_num} ', f'Question {q_num}']
                q_col = None
                for col in possible_q_cols:
                    if col in df.columns:
                        q_col = col
                        break

                desc_col = f'Q{q_num} description'

                if q_col and pd.notna(row[q_col]) and row[q_col] != 'na':
                    question = QuestionAnnotation(
                        video_title=row['Video Title'],
                        asset_id=str(row['Asset #']),
                        age_group=row['Age group'],
                        timestamp=str(row['Timestamp']),
                        question_time=str(row[q_col]),
                        description=str(row[desc_col]),
                        question_type='',  # Will be classified
                        has_pause=False,   # Will be determined
                        wait_time_quality='',  # Will be classified
                        video_description=str(row['Description'])
                    )
                    questions.append(question)

        return questions

    def _extract_linguistic_features(self, question: QuestionAnnotation) -> np.ndarray:
        """
        Extract linguistic features from question description.
        Returns 40-dimensional feature vector.
        """
        desc = question.description.lower()

        # TF-IDF features (20 dimensions)
        tfidf_features = self.tfidf_vectorizer.transform([desc]).toarray()[0]

        # Keyword presence features
        oeq_score = sum(1 for keyword in self.oeq_keywords if keyword in desc)
        ceq_score = sum(1 for keyword in self.ceq_keywords if keyword in desc)

        # Question pattern features
        has_question_mark = 1.0 if '?' in question.description else 0.0
        has_what = 1.0 if 'what' in desc else 0.0
        has_how = 1.0 if 'how' in desc else 0.0
        has_why = 1.0 if 'why' in desc else 0.0
        has_yes_no = 1.0 if any(word in desc for word in ['yes', 'no']) else 0.0

        # Length features
        desc_length = len(question.description)
        word_count = len(question.description.split())

        # Educational domain features
        has_opportunity = 1.0 if 'opportunity' in desc else 0.0
        has_response = 1.0 if 'response' in desc else 0.0
        mentions_child = 1.0 if 'child' in desc else 0.0
        mentions_teacher = 1.0 if 'teacher' in desc else 0.0

        # Combine all linguistic features
        linguistic_features = np.concatenate([
            tfidf_features,  # 20D
            [oeq_score, ceq_score, has_question_mark, has_what, has_how, has_why, has_yes_no],  # 7D
            [desc_length, word_count],  # 2D
            [has_opportunity, has_response, mentions_child, mentions_teacher],  # 4D
            # Padding to reach 40D
            np.zeros(7)  # 7D padding
        ])

        return linguistic_features

    def _extract_temporal_features(self, question: QuestionAnnotation) -> np.ndarray:
        """
        Extract temporal features from timestamps and pause descriptions.
        Returns 15-dimensional feature vector.
        """
        desc = question.description.lower()

        # Parse timestamp features
        timestamp_features = self._parse_timestamp(question.question_time)

        # Pause behavior features
        pause_score = sum(1 for keyword in self.pause_keywords if keyword in desc)
        has_pause = 1.0 if any(word in desc for word in ['pause', 'pauses', 'wait', 'waits']) else 0.0
        continues_talking = 1.0 if 'continues' in desc else 0.0
        no_pause = 1.0 if 'no pause' in desc else 0.0
        rhetorical = 1.0 if 'rhetorical' in desc else 0.0

        # Wait time quality indicators
        appropriate_wait = 1.0 if any(phrase in desc for phrase in ['pauses', 'listens', 'opportunity']) else 0.0
        insufficient_wait = 1.0 if any(phrase in desc for phrase in ['continues', 'no pause', 'doesn\'t pause']) else 0.0

        # Response behavior
        child_responds = 1.0 if 'child' in desc and 'respond' in desc else 0.0
        allows_response = 1.0 if 'opportunity' in desc else 0.0

        # Combine temporal features
        temporal_features = np.array([
            timestamp_features,  # 1D
            pause_score, has_pause, continues_talking, no_pause, rhetorical,  # 5D
            appropriate_wait, insufficient_wait,  # 2D
            child_responds, allows_response,  # 2D
            # Padding to reach 15D
            0.0, 0.0, 0.0, 0.0, 0.0  # 5D padding
        ])

        return temporal_features

    def _extract_contextual_features(self, question: QuestionAnnotation) -> np.ndarray:
        """
        Extract contextual features from video and demographic information.
        Returns 25-dimensional feature vector.
        """
        # Age group encoding
        age_group_encoded = self.age_group_encoder.transform([question.age_group])[0]

        # Video context analysis
        video_desc = question.video_description.lower()

        # Activity type features
        is_reading = 1.0 if 'book' in video_desc or 'read' in video_desc else 0.0
        is_sensory = 1.0 if any(word in video_desc for word in ['sensory', 'touch', 'feel', 'smell']) else 0.0
        is_exploration = 1.0 if any(word in video_desc for word in ['explore', 'discovery', 'investigate']) else 0.0
        is_group = 1.0 if 'group' in video_desc else 0.0
        is_individual = 1.0 if 'individual' in video_desc else 0.0

        # Content domain features
        is_science = 1.0 if any(word in video_desc for word in ['science', 'nature', 'plant', 'flower', 'melon']) else 0.0
        is_literacy = 1.0 if any(word in video_desc for word in ['book', 'read', 'story', 'letter']) else 0.0
        is_math = 1.0 if any(word in video_desc for word in ['count', 'number', 'shape', 'measure']) else 0.0

        # Social interaction features
        peer_interaction = 1.0 if 'children' in video_desc else 0.0
        adult_child = 1.0 if 'educator' in video_desc else 0.0

        # Setting features
        indoor = 1.0 if any(word in video_desc for word in ['classroom', 'table', 'indoor']) else 0.0
        outdoor = 1.0 if 'outdoor' in video_desc else 0.0

        # Material features
        has_materials = 1.0 if any(word in video_desc for word in ['material', 'tool', 'toy', 'book']) else 0.0
        hands_on = 1.0 if any(word in video_desc for word in ['hands', 'touch', 'manipulate']) else 0.0

        # Duration estimate (rough)
        duration_estimate = self._estimate_video_duration(question.timestamp)

        # Combine contextual features
        contextual_features = np.array([
            age_group_encoded,  # 1D
            is_reading, is_sensory, is_exploration, is_group, is_individual,  # 5D
            is_science, is_literacy, is_math,  # 3D
            peer_interaction, adult_child,  # 2D
            indoor, outdoor,  # 2D
            has_materials, hands_on,  # 2D
            duration_estimate,  # 1D
            # Padding to reach 25D
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 8D padding
        ])

        return contextual_features

    def _classify_question_type(self, question: QuestionAnnotation) -> int:
        """Classify question type from description. Returns: 0=OEQ, 1=CEQ, 2=Rhetorical"""
        desc = question.description.lower()

        if 'oeq' in desc or 'open' in desc:
            return 0  # OEQ
        elif 'ceq' in desc or 'closed' in desc or 'yes/no' in desc:
            return 1  # CEQ
        elif 'rhetorical' in desc:
            return 2  # Rhetorical
        else:
            # Heuristic classification based on keywords
            oeq_score = sum(1 for keyword in self.oeq_keywords if keyword in desc)
            ceq_score = sum(1 for keyword in self.ceq_keywords if keyword in desc)

            if oeq_score > ceq_score:
                return 0  # OEQ
            else:
                return 1  # CEQ

    def _assess_wait_time(self, question: QuestionAnnotation) -> int:
        """Assess wait time quality. Returns: 0=appropriate, 1=insufficient, 2=none"""
        desc = question.description.lower()

        if any(phrase in desc for phrase in ['pauses', 'listens', 'opportunity', 'allows', 'wait']):
            return 0  # Appropriate wait time
        elif any(phrase in desc for phrase in ['continues', 'no pause', 'doesn\'t pause', 'rhetorical']):
            if 'rhetorical' in desc:
                return 2  # No wait expected (rhetorical)
            else:
                return 1  # Insufficient wait time
        else:
            return 1  # Default to insufficient if unclear

    def _estimate_class_score(self, question: QuestionAnnotation) -> float:
        """Estimate CLASS framework score based on description patterns."""
        desc = question.description.lower()

        # Base score
        score = 3.0  # Neutral starting point

        # Positive indicators
        if any(word in desc for word in ['opportunity', 'listens', 'pauses']):
            score += 1.0
        if 'response' in desc:
            score += 0.5
        if 'oeq' in desc or 'open' in desc:
            score += 0.5

        # Negative indicators
        if any(phrase in desc for phrase in ['continues', 'no pause', 'doesn\'t pause']):
            score -= 1.0
        if 'rhetorical' in desc and 'no pause' in desc:
            score -= 0.5

        # Clamp to 1-5 range
        return max(1.0, min(5.0, score))

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string to numeric value."""
        try:
            # Handle formats like "0:37", "1:02", etc.
            if ':' in timestamp_str:
                parts = timestamp_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(timestamp_str)
        except:
            return 0.0

    def _estimate_video_duration(self, timestamp_range: str) -> float:
        """Estimate video duration from timestamp range."""
        try:
            if '-' in timestamp_range:
                start, end = timestamp_range.split('-')
                start_seconds = self._parse_timestamp(start.strip())
                end_seconds = self._parse_timestamp(end.strip())
                return end_seconds - start_seconds
            else:
                return self._parse_timestamp(timestamp_range)
        except:
            return 60.0  # Default 1 minute

    def extract_single_sample(self, transcript: str) -> np.ndarray:
        """
        Extract features for a single transcript (for inference).
        Used by updated model classes for real-time prediction.
        """
        # Create minimal QuestionAnnotation for feature extraction
        question = QuestionAnnotation(
            video_title="inference",
            asset_id="0",
            age_group="PK",  # Default
            timestamp="0:00",
            question_time="0:00",
            description=transcript,
            question_type="",
            has_pause=False,
            wait_time_quality="",
            video_description="inference context"
        )

        # Extract features
        ling_features = self._extract_linguistic_features(question)
        temp_features = self._extract_temporal_features(question)
        cont_features = self._extract_contextual_features(question)

        # Concatenate and scale
        features = np.concatenate([ling_features, temp_features, cont_features])
        if hasattr(self.scaler, 'mean_'):  # Check if scaler is fitted
            features = self.scaler.transform([features])[0]

        return features

if __name__ == "__main__":
    # Test feature extraction
    logging.basicConfig(level=logging.INFO)

    csv_path = "/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv"
    if os.path.exists(csv_path):
        extractor = ExpertAnnotationFeatureExtractor()
        X, y = extractor.extract_features(csv_path)

        print(f"Features shape: {X.shape}")
        print(f"Question type labels: {len(y['question_type'])} samples")
        print(f"Wait time labels: {len(y['wait_time'])} samples")
        print(f"CLASS scores: {len(y['class_scores'])} samples")

        # Sample statistics
        print(f"\nQuestion type distribution: {np.bincount(y['question_type'])}")
        print(f"Wait time distribution: {np.bincount(y['wait_time'])}")
        print(f"CLASS score stats: mean={np.mean(y['class_scores']):.2f}, std={np.std(y['class_scores']):.2f}")
    else:
        print(f"CSV file not found at {csv_path}")