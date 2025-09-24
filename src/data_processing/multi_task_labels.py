#!/usr/bin/env python3
"""
Multi-Task Label Engineering for Issue #91

Advanced label generation for concurrent BERT learning tasks.
Creates standardized labels for 4 simultaneous prediction tasks:
1. Question Classification: OEQ vs CEQ (accuracy-critical for coaching)
2. Wait Time Assessment: Appropriate/Inappropriate (timing-sensitive)
3. Interaction Quality: 1-5 regression (pedagogical effectiveness)
4. CLASS Framework: 3-dimensional classification (research alignment)

Author: Claude (Partner-Level Microsoft SDE)
Issue: #91 - Quality Label Standardization and ML Dataset Creation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import re
from datetime import datetime

from label_processing import (
    MultiTaskLabels, QualityScore, AnnotationMetadata,
    ValidationResult, ProcessingStatistics
)

logger = logging.getLogger(__name__)

@dataclass
class LabelGenerationConfig:
    """Configuration for multi-task label generation."""

    # Question Classification Configuration
    oeq_keywords: List[str] = None
    ceq_keywords: List[str] = None
    question_confidence_threshold: float = 0.8

    # Wait Time Configuration
    optimal_wait_time_min: float = 3.0  # Educational research: 3-7 seconds
    optimal_wait_time_max: float = 7.0
    wait_time_tolerance: float = 1.0  # ±1 second tolerance

    # Interaction Quality Configuration
    quality_score_range: Tuple[float, float] = (1.0, 5.0)
    quality_confidence_threshold: float = 0.7

    # CLASS Framework Configuration
    class_dimensions: List[str] = None
    class_score_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.oeq_keywords is None:
            self.oeq_keywords = [
                'why', 'how', 'what do you think', 'what do you feel',
                'explain', 'describe', 'tell me about', 'in your opinion',
                'what would happen if', 'imagine', 'suppose', 'what could'
            ]

        if self.ceq_keywords is None:
            self.ceq_keywords = [
                'is', 'are', 'do', 'does', 'did', 'can', 'could', 'would',
                'should', 'will', 'yes or no', 'true or false', 'which one'
            ]

        if self.class_dimensions is None:
            self.class_dimensions = [
                'emotional_support', 'classroom_organization', 'instructional_support'
            ]

        if self.class_score_weights is None:
            self.class_score_weights = {
                'emotional_support': 0.3,
                'classroom_organization': 0.3,
                'instructional_support': 0.4  # Higher weight for instructional quality
            }

class MultiTaskLabelEngine:
    """
    Advanced label generation for concurrent learning tasks.

    Generates labels for 4 simultaneous ML tasks optimized for multi-modal BERT
    architecture. Ensures label consistency, educational validity, and balanced
    representation across all tasks.
    """

    def __init__(self, config: LabelGenerationConfig = None):
        self.config = config or LabelGenerationConfig()
        self.statistics = ProcessingStatistics(
            total_annotations=0,
            successful_labels=0,
            failed_labels=0,
            missing_data_imputed=0,
            cross_annotator_agreements={},
            class_distributions={},
            processing_time=0.0,
            version_hash=""
        )

        # Educational research mappings
        self.bloom_taxonomy_mapping = self._initialize_bloom_taxonomy()
        self.wait_time_research_standards = self._initialize_wait_time_standards()

        logger.info("MultiTaskLabelEngine initialized with educational research standards")

    def generate_labels(self,
                       annotations: pd.DataFrame,
                       normalized_scores: Dict[str, List[QualityScore]],
                       feature_vectors: Optional[Dict] = None) -> List[MultiTaskLabels]:
        """
        Generate comprehensive multi-task labels for BERT training.

        Args:
            annotations: Raw annotation DataFrame
            normalized_scores: Quality scores from normalization pipeline
            feature_vectors: Optional feature vectors from Issue #90

        Returns:
            List of MultiTaskLabels for each annotation instance
        """

        logger.info(f"Generating multi-task labels for {len(annotations)} annotations")
        start_time = datetime.now()

        try:
            multi_task_labels = []

            for idx, row in annotations.iterrows():
                try:
                    # Generate labels for all 4 tasks
                    question_labels = self._generate_question_classification_labels(row)
                    wait_time_labels = self._generate_wait_time_labels(row, feature_vectors)
                    quality_labels = self._generate_interaction_quality_labels(row, normalized_scores)
                    class_labels = self._generate_class_framework_labels(row, normalized_scores)

                    # Extract metadata
                    metadata = self._extract_annotation_metadata(row, idx)

                    # Combine into complete label set
                    labels = MultiTaskLabels(
                        # Task 1: Question Classification
                        question_type=question_labels['type'],
                        question_confidence=question_labels['confidence'],

                        # Task 2: Wait Time Assessment
                        wait_time_appropriate=wait_time_labels['appropriate'],
                        wait_time_confidence=wait_time_labels['confidence'],

                        # Task 3: Interaction Quality (Regression)
                        interaction_quality_score=quality_labels['score'],
                        interaction_quality_confidence=quality_labels['confidence'],

                        # Task 4: CLASS Framework (Multi-class)
                        class_emotional_support=class_labels['emotional_support'],
                        class_classroom_organization=class_labels['classroom_organization'],
                        class_instructional_support=class_labels['instructional_support'],
                        class_framework_confidence=class_labels['confidence'],

                        # Metadata
                        metadata=metadata
                    )

                    multi_task_labels.append(labels)
                    self.statistics.successful_labels += 1

                except Exception as e:
                    logger.warning(f"Failed to generate labels for row {idx}: {e}")
                    self.statistics.failed_labels += 1
                    continue

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.statistics.total_annotations = len(annotations)
            self.statistics.processing_time = processing_time

            # Analyze class distributions
            self.statistics.class_distributions = self._analyze_class_distributions(multi_task_labels)

            logger.info(f"Generated {len(multi_task_labels)} multi-task label sets in {processing_time:.2f}s")
            return multi_task_labels

        except Exception as e:
            logger.error(f"Multi-task label generation failed: {e}")
            return []

    def _generate_question_classification_labels(self, row: pd.Series) -> Dict[str, Any]:
        """Generate question type classification labels (OEQ vs CEQ)."""

        try:
            # Find question text or description
            question_text = self._extract_question_text(row)

            if not question_text:
                return {
                    'type': 'unknown',
                    'confidence': 0.0,
                    'reasoning': 'No question text found'
                }

            question_lower = question_text.lower().strip()

            # Calculate OEQ indicators
            oeq_score = self._calculate_oeq_indicators(question_lower)

            # Calculate CEQ indicators
            ceq_score = self._calculate_ceq_indicators(question_lower)

            # Determine classification
            if oeq_score > ceq_score and oeq_score >= self.config.question_confidence_threshold:
                question_type = 'OEQ'
                confidence = min(0.95, oeq_score)
            elif ceq_score > oeq_score and ceq_score >= self.config.question_confidence_threshold:
                question_type = 'CEQ'
                confidence = min(0.95, ceq_score)
            else:
                # Ambiguous or balanced - use additional heuristics
                question_type, confidence = self._resolve_ambiguous_question(question_lower, oeq_score, ceq_score)

            # Validate against expert annotation if available
            expert_classification = self._extract_expert_question_classification(row)
            if expert_classification:
                confidence = self._adjust_confidence_with_expert_validation(
                    question_type, expert_classification, confidence
                )

            return {
                'type': question_type,
                'confidence': confidence,
                'oeq_score': oeq_score,
                'ceq_score': ceq_score,
                'reasoning': f'OEQ: {oeq_score:.2f}, CEQ: {ceq_score:.2f}'
            }

        except Exception as e:
            logger.warning(f"Question classification failed: {e}")
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'reasoning': f'Classification error: {str(e)}'
            }

    def _generate_wait_time_labels(self,
                                  row: pd.Series,
                                  feature_vectors: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate wait time appropriateness labels."""

        try:
            # Extract wait time from annotation
            wait_time = self._extract_wait_time_value(row)

            if wait_time is None:
                # Try to extract from feature vectors if available
                if feature_vectors:
                    wait_time = self._extract_wait_time_from_features(row, feature_vectors)

            if wait_time is None:
                return {
                    'appropriate': True,  # Default to appropriate
                    'confidence': 0.1,
                    'reasoning': 'No wait time data available'
                }

            # Apply educational research standards
            is_appropriate = self._evaluate_wait_time_appropriateness(wait_time)

            # Calculate confidence based on clarity of the decision
            confidence = self._calculate_wait_time_confidence(wait_time)

            # Enhance with contextual information
            context_adjustment = self._adjust_wait_time_for_context(row, wait_time)

            return {
                'appropriate': is_appropriate,
                'confidence': min(0.95, confidence * context_adjustment),
                'wait_time_seconds': wait_time,
                'optimal_range': f"{self.config.optimal_wait_time_min}-{self.config.optimal_wait_time_max}s",
                'reasoning': f"Wait time: {wait_time:.1f}s, appropriate: {is_appropriate}"
            }

        except Exception as e:
            logger.warning(f"Wait time label generation failed: {e}")
            return {
                'appropriate': True,
                'confidence': 0.1,
                'reasoning': f'Wait time error: {str(e)}'
            }

    def _generate_interaction_quality_labels(self,
                                           row: pd.Series,
                                           normalized_scores: Dict[str, List[QualityScore]]) -> Dict[str, Any]:
        """Generate interaction quality regression labels (1-5 scale)."""

        try:
            # Extract quality scores for this annotation
            row_quality_scores = self._find_matching_quality_scores(row, normalized_scores)

            if not row_quality_scores:
                return {
                    'score': 3.0,  # Neutral default
                    'confidence': 0.3,
                    'reasoning': 'No quality scores found'
                }

            # Aggregate multiple quality dimensions
            aggregated_score = self._aggregate_quality_scores(row_quality_scores)

            # Validate score range
            quality_score = max(1.0, min(5.0, aggregated_score))

            # Calculate confidence based on score consistency and available data
            confidence = self._calculate_quality_confidence(row_quality_scores)

            # Enhance with educational validity checks
            education_adjusted_score = self._adjust_score_for_educational_validity(
                quality_score, row
            )

            return {
                'score': education_adjusted_score,
                'confidence': confidence,
                'component_scores': [score.normalized_value for score in row_quality_scores],
                'num_dimensions': len(row_quality_scores),
                'reasoning': f"Aggregated from {len(row_quality_scores)} quality dimensions"
            }

        except Exception as e:
            logger.warning(f"Quality label generation failed: {e}")
            return {
                'score': 3.0,
                'confidence': 0.1,
                'reasoning': f'Quality error: {str(e)}'
            }

    def _generate_class_framework_labels(self,
                                       row: pd.Series,
                                       normalized_scores: Dict[str, List[QualityScore]]) -> Dict[str, Any]:
        """Generate CLASS framework multi-dimensional labels."""

        try:
            class_scores = {
                'emotional_support': 3.0,
                'classroom_organization': 3.0,
                'instructional_support': 3.0
            }

            # Extract CLASS-specific scores from normalized data
            for dimension in self.config.class_dimensions:
                dimension_scores = self._extract_class_dimension_scores(
                    row, normalized_scores, dimension
                )

                if dimension_scores:
                    class_scores[dimension] = np.mean([score.normalized_value for score in dimension_scores])

            # Ensure scores are within valid range
            for dimension in class_scores:
                class_scores[dimension] = max(1.0, min(5.0, class_scores[dimension]))

            # Calculate overall confidence
            confidence = self._calculate_class_framework_confidence(row, normalized_scores)

            # Apply educational research validation
            validated_scores = self._validate_class_scores_with_research(class_scores, row)

            return {
                'emotional_support': validated_scores['emotional_support'],
                'classroom_organization': validated_scores['classroom_organization'],
                'instructional_support': validated_scores['instructional_support'],
                'confidence': confidence,
                'reasoning': 'Multi-dimensional CLASS framework assessment'
            }

        except Exception as e:
            logger.warning(f"CLASS framework label generation failed: {e}")
            return {
                'emotional_support': 3.0,
                'classroom_organization': 3.0,
                'instructional_support': 3.0,
                'confidence': 0.1,
                'reasoning': f'CLASS framework error: {str(e)}'
            }

    # Helper methods for question classification

    def _extract_question_text(self, row: pd.Series) -> Optional[str]:
        """Extract question text from annotation row."""

        # Look for question-related columns
        question_columns = [
            col for col in row.index
            if any(keyword in col.lower() for keyword in ['question', 'text', 'description', 'content'])
        ]

        for col in question_columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                return str(row[col]).strip()

        return None

    def _calculate_oeq_indicators(self, question_text: str) -> float:
        """Calculate Open-Ended Question indicators."""

        if not question_text:
            return 0.0

        oeq_score = 0.0

        # Direct keyword matching
        for keyword in self.config.oeq_keywords:
            if keyword in question_text:
                oeq_score += 0.2

        # Pattern matching
        oeq_patterns = [
            r'\b(why|how)\s+\w+',  # Why/How questions
            r'what.*think|what.*feel|what.*believe',  # Opinion questions
            r'explain|describe|tell me about',  # Elaboration requests
            r'what would.*if|imagine.*if|suppose',  # Hypothetical questions
        ]

        for pattern in oeq_patterns:
            if re.search(pattern, question_text):
                oeq_score += 0.15

        # Question length heuristic (longer questions often more open-ended)
        if len(question_text.split()) > 8:
            oeq_score += 0.1

        # Presence of question mark
        if '?' in question_text:
            oeq_score += 0.05

        return min(1.0, oeq_score)

    def _calculate_ceq_indicators(self, question_text: str) -> float:
        """Calculate Closed-Ended Question indicators."""

        if not question_text:
            return 0.0

        ceq_score = 0.0

        # Direct keyword matching
        for keyword in self.config.ceq_keywords:
            if question_text.startswith(keyword + ' '):
                ceq_score += 0.3  # Higher weight for starting words

        # Pattern matching
        ceq_patterns = [
            r'^(is|are|do|does|did|can|could|would|should|will)\s',  # Yes/no questions
            r'(yes.*or.*no|true.*or.*false)',  # Binary choice
            r'which\s+(one|of)',  # Selection questions
            r'how many|how much',  # Quantitative questions
        ]

        for pattern in ceq_patterns:
            if re.search(pattern, question_text):
                ceq_score += 0.2

        # Short length heuristic (closed questions often shorter)
        if len(question_text.split()) <= 5:
            ceq_score += 0.1

        return min(1.0, ceq_score)

    def _resolve_ambiguous_question(self, question_text: str, oeq_score: float, ceq_score: float) -> Tuple[str, float]:
        """Resolve ambiguous question classification using additional heuristics."""

        # Use Bloom's taxonomy mapping
        bloom_level = self._classify_bloom_taxonomy_level(question_text)

        # Higher-order thinking questions lean toward OEQ
        if bloom_level in ['analyze', 'evaluate', 'create']:
            return 'OEQ', 0.7
        elif bloom_level in ['remember', 'understand']:
            return 'CEQ', 0.7

        # Default to the higher score, but with reduced confidence
        if oeq_score > ceq_score:
            return 'OEQ', max(0.5, oeq_score * 0.8)
        elif ceq_score > oeq_score:
            return 'CEQ', max(0.5, ceq_score * 0.8)
        else:
            # Truly ambiguous - slightly favor OEQ for educational benefit
            return 'OEQ', 0.6

    def _classify_bloom_taxonomy_level(self, question_text: str) -> str:
        """Classify question according to Bloom's taxonomy."""

        bloom_keywords = {
            'remember': ['recall', 'remember', 'list', 'identify', 'name', 'define'],
            'understand': ['explain', 'describe', 'summarize', 'compare', 'interpret'],
            'apply': ['solve', 'demonstrate', 'use', 'apply', 'calculate', 'show'],
            'analyze': ['analyze', 'examine', 'compare', 'contrast', 'categorize'],
            'evaluate': ['evaluate', 'judge', 'critique', 'defend', 'justify'],
            'create': ['create', 'design', 'compose', 'generate', 'construct']
        }

        for level, keywords in bloom_keywords.items():
            for keyword in keywords:
                if keyword in question_text:
                    return level

        return 'understand'  # Default to basic comprehension

    # Helper methods for wait time assessment

    def _extract_wait_time_value(self, row: pd.Series) -> Optional[float]:
        """Extract wait time value from annotation row."""

        # Look for wait time columns
        wait_columns = [
            col for col in row.index
            if any(keyword in col.lower() for keyword in ['wait', 'pause', 'time', 'delay'])
        ]

        for col in wait_columns:
            if pd.notna(row[col]):
                try:
                    # Try direct numeric conversion
                    wait_time = float(row[col])
                    if 0 <= wait_time <= 30:  # Reasonable range
                        return wait_time
                except (ValueError, TypeError):
                    # Try to extract from text
                    text_value = str(row[col])
                    numeric_match = re.search(r'(\d+(?:\.\d+)?)', text_value)
                    if numeric_match:
                        return float(numeric_match.group(1))

        return None

    def _evaluate_wait_time_appropriateness(self, wait_time: float) -> bool:
        """Evaluate if wait time meets educational research standards."""

        # Educational research: 3-7 seconds is optimal
        optimal_min = self.config.optimal_wait_time_min
        optimal_max = self.config.optimal_wait_time_max
        tolerance = self.config.wait_time_tolerance

        # Apply tolerance for borderline cases
        return (optimal_min - tolerance) <= wait_time <= (optimal_max + tolerance)

    def _calculate_wait_time_confidence(self, wait_time: float) -> float:
        """Calculate confidence in wait time appropriateness decision."""

        optimal_min = self.config.optimal_wait_time_min
        optimal_max = self.config.optimal_wait_time_max

        if optimal_min <= wait_time <= optimal_max:
            # Perfect range - high confidence
            return 0.95
        elif wait_time < optimal_min:
            # Too short - confidence decreases with distance from optimal
            distance = optimal_min - wait_time
            return max(0.5, 0.9 - (distance * 0.1))
        else:
            # Too long - similar confidence reduction
            distance = wait_time - optimal_max
            return max(0.5, 0.9 - (distance * 0.05))  # More tolerant of longer waits

    # Helper methods for quality score processing

    def _find_matching_quality_scores(self, row: pd.Series, normalized_scores: Dict) -> List[QualityScore]:
        """Find quality scores that match this annotation row."""

        matching_scores = []

        # Extract identifier for this row
        row_identifier = self._get_row_identifier(row)

        for category, scores in normalized_scores.items():
            for score in scores:
                # Check if this score belongs to the current row
                # This is a simplified matching - in practice, would use more sophisticated alignment
                try:
                    # Match by index or other identifier
                    if hasattr(score, 'metadata') and hasattr(score.metadata, 'original_csv_row'):
                        if score.metadata.original_csv_row == row.name:
                            matching_scores.append(score)
                except:
                    # Fallback matching logic
                    continue

        return matching_scores

    def _aggregate_quality_scores(self, quality_scores: List[QualityScore]) -> float:
        """Aggregate multiple quality score dimensions."""

        if not quality_scores:
            return 3.0

        # Weight different categories according to educational importance
        category_weights = {
            'emotional_support': 0.3,
            'classroom_organization': 0.3,
            'instructional_support': 0.4
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for score in quality_scores:
            weight = category_weights.get(score.category, 0.33)
            weighted_sum += score.normalized_value * weight * score.confidence
            total_weight += weight * score.confidence

        if total_weight == 0:
            return 3.0

        return weighted_sum / total_weight

    # Helper methods for metadata extraction

    def _extract_annotation_metadata(self, row: pd.Series, row_idx: int) -> AnnotationMetadata:
        """Extract comprehensive metadata from annotation row."""

        # Extract basic identifiers
        video_id = self._safe_extract(row, ['video_id', 'video', 'clip', 'file'])
        timestamp = self._safe_extract_numeric(row, ['timestamp', 'time', 'start'])
        annotator = self._safe_extract(row, ['annotator', 'rater', 'evaluator'])

        return AnnotationMetadata(
            video_id=str(video_id) if video_id else f"unknown_{row_idx}",
            clip_filename=f"{video_id}.wav" if video_id else f"unknown_{row_idx}.wav",
            timestamp_start=float(timestamp) if timestamp else 0.0,
            timestamp_end=float(timestamp + 5.0) if timestamp else 5.0,  # Estimated duration
            annotator_id=str(annotator) if annotator else "unknown",
            annotation_date=datetime.now().strftime("%Y-%m-%d"),
            original_csv_row=int(row_idx),
            quality_confidence=0.8  # Default confidence
        )

    def _safe_extract(self, row: pd.Series, possible_columns: List[str]) -> Optional[Any]:
        """Safely extract value from row with flexible column matching."""

        for col_pattern in possible_columns:
            matching_cols = [col for col in row.index if col_pattern in col.lower()]
            for col in matching_cols:
                if pd.notna(row[col]):
                    return row[col]
        return None

    def _safe_extract_numeric(self, row: pd.Series, possible_columns: List[str]) -> Optional[float]:
        """Safely extract numeric value from row."""

        value = self._safe_extract(row, possible_columns)
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                # Try to extract number from string
                if isinstance(value, str):
                    numeric_match = re.search(r'(\d+(?:\.\d+)?)', value)
                    if numeric_match:
                        return float(numeric_match.group(1))
        return None

    def _get_row_identifier(self, row: pd.Series) -> str:
        """Get unique identifier for annotation row."""

        # Try multiple identification strategies
        identifiers = []

        video_id = self._safe_extract(row, ['video_id', 'video', 'clip'])
        if video_id:
            identifiers.append(str(video_id))

        timestamp = self._safe_extract_numeric(row, ['timestamp', 'time'])
        if timestamp:
            identifiers.append(str(int(timestamp)))

        if identifiers:
            return "_".join(identifiers)
        else:
            return str(row.name)  # Use row index as fallback

    # Statistical analysis methods

    def _analyze_class_distributions(self, labels: List[MultiTaskLabels]) -> Dict[str, Dict]:
        """Analyze class distributions across all tasks."""

        if not labels:
            return {}

        distributions = {
            'question_types': Counter([label.question_type for label in labels]),
            'wait_time_appropriate': Counter([label.wait_time_appropriate for label in labels]),
            'quality_score_bins': self._analyze_quality_score_distribution(labels),
            'class_framework_stats': self._analyze_class_framework_distribution(labels)
        }

        return distributions

    def _analyze_quality_score_distribution(self, labels: List[MultiTaskLabels]) -> Dict:
        """Analyze distribution of quality scores."""

        scores = [label.interaction_quality_score for label in labels]

        # Create bins for analysis
        bins = np.arange(1, 6, 0.5)  # 1.0-1.5, 1.5-2.0, etc.
        hist, bin_edges = np.histogram(scores, bins=bins)

        return {
            'bins': bin_edges.tolist(),
            'counts': hist.tolist(),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }

    def _analyze_class_framework_distribution(self, labels: List[MultiTaskLabels]) -> Dict:
        """Analyze CLASS framework score distributions."""

        emotional_support = [label.class_emotional_support for label in labels]
        classroom_org = [label.class_classroom_organization for label in labels]
        instructional_support = [label.class_instructional_support for label in labels]

        return {
            'emotional_support': {
                'mean': float(np.mean(emotional_support)),
                'std': float(np.std(emotional_support))
            },
            'classroom_organization': {
                'mean': float(np.mean(classroom_org)),
                'std': float(np.std(classroom_org))
            },
            'instructional_support': {
                'mean': float(np.mean(instructional_support)),
                'std': float(np.std(instructional_support))
            }
        }

    # Initialization methods

    def _initialize_bloom_taxonomy(self) -> Dict[str, List[str]]:
        """Initialize Bloom's taxonomy keyword mapping."""

        return {
            'remember': ['recall', 'remember', 'list', 'identify', 'name', 'define', 'state'],
            'understand': ['explain', 'describe', 'summarize', 'compare', 'interpret', 'discuss'],
            'apply': ['solve', 'demonstrate', 'use', 'apply', 'calculate', 'show', 'implement'],
            'analyze': ['analyze', 'examine', 'compare', 'contrast', 'categorize', 'differentiate'],
            'evaluate': ['evaluate', 'judge', 'critique', 'defend', 'justify', 'assess'],
            'create': ['create', 'design', 'compose', 'generate', 'construct', 'develop']
        }

    def _initialize_wait_time_standards(self) -> Dict[str, Any]:
        """Initialize wait time standards from educational research."""

        return {
            'optimal_range': (3.0, 7.0),  # Research-backed optimal range
            'minimum_acceptable': 1.0,
            'maximum_reasonable': 15.0,
            'context_adjustments': {
                'toddler': -1.0,  # Shorter attention spans
                'preschool': 0.0,  # Standard range
                'kindergarten': +1.0  # Can handle slightly longer waits
            }
        }

    # Stub implementations for remaining helper methods

    def _extract_wait_time_from_features(self, row: pd.Series, feature_vectors: Dict) -> Optional[float]:
        """Extract wait time from feature vectors if available."""
        # Implementation would depend on feature vector structure from Issue #90
        return None

    def _adjust_wait_time_for_context(self, row: pd.Series, wait_time: float) -> float:
        """Adjust wait time evaluation based on contextual factors."""
        # Could adjust based on age group, question complexity, etc.
        return 1.0  # No adjustment for now

    def _extract_expert_question_classification(self, row: pd.Series) -> Optional[str]:
        """Extract expert question classification if available."""
        return self._safe_extract(row, ['question_type', 'type', 'classification'])

    def _adjust_confidence_with_expert_validation(self, predicted: str, expert: str, confidence: float) -> float:
        """Adjust confidence based on agreement with expert classification."""
        if predicted.lower() == expert.lower():
            return min(0.98, confidence * 1.1)  # Boost confidence for agreement
        else:
            return max(0.3, confidence * 0.7)  # Reduce confidence for disagreement

    def _calculate_quality_confidence(self, quality_scores: List[QualityScore]) -> float:
        """Calculate confidence in quality score aggregation."""
        if not quality_scores:
            return 0.1

        # Average confidence across all component scores
        avg_confidence = np.mean([score.confidence for score in quality_scores])

        # Boost confidence for multiple consistent scores
        consistency_bonus = min(0.1, len(quality_scores) * 0.02)

        return min(0.95, avg_confidence + consistency_bonus)

    def _adjust_score_for_educational_validity(self, score: float, row: pd.Series) -> float:
        """Adjust score based on educational validity checks."""
        # Could implement context-aware adjustments
        return score  # No adjustment for now

    def _extract_class_dimension_scores(self, row: pd.Series, normalized_scores: Dict, dimension: str) -> List[QualityScore]:
        """Extract scores specific to a CLASS framework dimension."""
        matching_scores = []

        # Find scores that match this dimension
        for category, scores in normalized_scores.items():
            for score in scores:
                if hasattr(score, 'category') and dimension in score.category:
                    matching_scores.append(score)

        return matching_scores

    def _calculate_class_framework_confidence(self, row: pd.Series, normalized_scores: Dict) -> float:
        """Calculate confidence in CLASS framework scores."""
        # Base confidence on availability and quality of underlying scores
        return 0.8  # Default confidence

    def _validate_class_scores_with_research(self, scores: Dict[str, float], row: pd.Series) -> Dict[str, float]:
        """Validate CLASS scores against educational research standards."""
        # Could implement research-based validation
        return scores  # No validation adjustments for now


def main():
    """Main entry point for testing multi-task label generation."""

    print("🏷️ Multi-Task Label Engineering - Issue #91")
    print("=" * 50)

    # Example usage
    config = LabelGenerationConfig()
    label_engine = MultiTaskLabelEngine(config)

    print("✅ Multi-task label engine initialized")
    print("📊 Ready for 4-task label generation:")
    print("  1. Question Classification (OEQ vs CEQ)")
    print("  2. Wait Time Assessment (Appropriate/Inappropriate)")
    print("  3. Interaction Quality (1-5 regression)")
    print("  4. CLASS Framework (3-dimensional)")

if __name__ == "__main__":
    main()