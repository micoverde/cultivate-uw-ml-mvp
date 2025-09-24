#!/usr/bin/env python3
"""
Basic Unit Tests for Issue #91: Dataset Creation & Validation Framework

Focused test suite for core functionality that can be tested without
heavy dependencies or complex mocking.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #91 - Quality Label Standardization and ML Dataset Creation
"""

import unittest
import numpy as np
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('data_processing')

# Import the modules we can test
from dataset_creation import HuggingFaceDatasetBuilder, DatasetVersionManager
from validation_framework import ComprehensiveValidationFramework, ValidationThresholds
from label_processing import MultiTaskLabels, AnnotationMetadata

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality that doesn't require heavy dependencies."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def test_issue_91_dataset_builder_initialization(self):
        """Test Issue #91: Dataset builder initializes correctly."""
        builder = HuggingFaceDatasetBuilder(output_dir=self.temp_dir, random_seed=42)

        self.assertEqual(builder.random_seed, 42)
        self.assertEqual(str(builder.output_dir), self.temp_dir)
        self.assertIsInstance(builder.version_manager, DatasetVersionManager)

    def test_issue_91_validation_framework_initialization(self):
        """Test Issue #91: Validation framework initializes correctly."""
        validator = ComprehensiveValidationFramework(output_dir=self.temp_dir)

        self.assertIsInstance(validator.thresholds, ValidationThresholds)
        self.assertEqual(validator.thresholds.cohens_kappa_minimum, 0.75)
        self.assertEqual(validator.thresholds.missing_data_threshold, 0.05)

    def test_issue_91_label_creation(self):
        """Test Issue #91: MultiTaskLabels can be created correctly."""
        metadata = AnnotationMetadata(
            video_id="test_video_001",
            clip_filename="clip_001.wav",
            annotator_id="test_annotator",
            timestamp_start=10.5,
            timestamp_end=15.2,
            annotation_date="2024-09-23",
            original_csv_row=1,
            quality_confidence=0.9
        )

        label = MultiTaskLabels(
            question_type="OEQ",
            question_confidence=0.85,
            wait_time_appropriate=True,
            wait_time_confidence=0.90,
            interaction_quality_score=4.2,
            interaction_quality_confidence=0.88,
            class_emotional_support=3.8,
            class_classroom_organization=4.1,
            class_instructional_support=3.9,
            class_framework_confidence=0.86,
            metadata=metadata
        )

        # Verify label properties
        self.assertEqual(label.question_type, "OEQ")
        self.assertAlmostEqual(label.question_confidence, 0.85)
        self.assertTrue(label.wait_time_appropriate)
        self.assertAlmostEqual(label.interaction_quality_score, 4.2)
        self.assertEqual(label.metadata.video_id, "test_video_001")

    def test_issue_91_quality_score_conversion(self):
        """Test Issue #91: Quality score discretization."""
        builder = HuggingFaceDatasetBuilder(output_dir=self.temp_dir)

        # Test quality class conversion
        self.assertEqual(builder._convert_to_quality_class(0.3), 'low')
        self.assertEqual(builder._convert_to_quality_class(0.5), 'medium')
        self.assertEqual(builder._convert_to_quality_class(0.8), 'high')

    def test_issue_91_feature_alignment_ids(self):
        """Test Issue #91: Feature-label ID alignment."""
        builder = HuggingFaceDatasetBuilder(output_dir=self.temp_dir)

        # Test partial ID matching
        feature_id = "test_video_001_segment_10_features"
        label_id = "asset_test_video_001_q_1"

        matches = builder._ids_match_partially(feature_id, label_id)
        self.assertTrue(matches)

        # Test non-matching IDs
        feature_id_2 = "different_video_002_segment_5"
        matches_2 = builder._ids_match_partially(feature_id_2, label_id)
        self.assertFalse(matches_2)

    def test_issue_91_validation_thresholds_customization(self):
        """Test Issue #91: Custom validation thresholds."""
        custom_thresholds = ValidationThresholds(
            cohens_kappa_minimum=0.8,
            missing_data_threshold=0.02,
            class_imbalance_threshold=0.15,
            min_samples_per_task=150
        )

        validator = ComprehensiveValidationFramework(thresholds=custom_thresholds)

        # Verify custom thresholds applied
        self.assertEqual(validator.thresholds.cohens_kappa_minimum, 0.8)
        self.assertEqual(validator.thresholds.missing_data_threshold, 0.02)
        self.assertEqual(validator.thresholds.class_imbalance_threshold, 0.15)
        self.assertEqual(validator.thresholds.min_samples_per_task, 150)

    def test_issue_91_basic_infrastructure_validation(self):
        """Test Issue #91: Basic infrastructure components work correctly."""
        builder = HuggingFaceDatasetBuilder(output_dir=self.temp_dir, random_seed=42)
        validator = ComprehensiveValidationFramework(output_dir=self.temp_dir)

        # Test that both systems are properly initialized
        self.assertTrue(self.temp_dir in str(builder.output_dir))
        self.assertTrue(self.temp_dir in str(validator.output_dir))
        self.assertEqual(builder.random_seed, 42)

    def test_issue_91_basic_quality_discretization(self):
        """Test Issue #91: Quality score discretization works correctly."""
        builder = HuggingFaceDatasetBuilder(output_dir=self.temp_dir)

        # Test boundary cases
        self.assertEqual(builder._convert_to_quality_class(0.0), 'low')
        self.assertEqual(builder._convert_to_quality_class(0.39), 'low')
        self.assertEqual(builder._convert_to_quality_class(0.4), 'medium')
        self.assertEqual(builder._convert_to_quality_class(0.69), 'medium')
        self.assertEqual(builder._convert_to_quality_class(0.7), 'high')
        self.assertEqual(builder._convert_to_quality_class(1.0), 'high')

    def test_issue_91_validation_data_quality_basic(self):
        """Test Issue #91: Basic data quality validation checks."""
        validator = ComprehensiveValidationFramework(output_dir=self.temp_dir)

        # Create test data
        features = {f"sample_{i}": np.random.randn(61) for i in range(20)}
        labels = []

        for i in range(20):
            metadata = AnnotationMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i),
                timestamp_end=float(i + 5),
                annotation_date="2024-09-23",
                original_csv_row=i,
                quality_confidence=0.9
            )

            label = MultiTaskLabels(
                question_type="OEQ" if i % 2 == 0 else "CEQ",
                question_confidence=0.85,
                wait_time_appropriate=True,
                wait_time_confidence=0.90,
                interaction_quality_score=4.0,
                interaction_quality_confidence=0.88,
                class_emotional_support=3.8,
                class_classroom_organization=4.1,
                class_instructional_support=3.9,
                class_framework_confidence=0.86,
                metadata=metadata
            )
            labels.append(label)

        # Run data quality validation
        result = validator._validate_data_quality(labels, features)

        # Verify validation structure
        self.assertIn('component', result)
        self.assertIn('status', result)
        self.assertIn('checks', result)
        self.assertEqual(result['component'], 'Data Quality')

    def test_issue_91_educational_research_standards(self):
        """Test Issue #91: Educational research standards validation."""
        validator = ComprehensiveValidationFramework(output_dir=self.temp_dir)

        # Create adequate sample size for research standards
        labels = []
        for i in range(120):  # Above minimum threshold
            metadata = AnnotationMetadata(
                video_id=f"video_{i % 10}",
                clip_filename=f"clip_{i}.wav",
                annotator_id=f"annotator_{i % 3}",
                timestamp_start=float(i * 5),
                timestamp_end=float(i * 5 + 5),
                annotation_date="2024-09-23",
                original_csv_row=i,
                quality_confidence=0.9
            )

            label = MultiTaskLabels(
                question_type=["OEQ", "CEQ", "SCAFFOLD"][i % 3],
                question_confidence=0.8,
                wait_time_appropriate=bool(i % 2),
                wait_time_confidence=0.85,
                interaction_quality_score=3.0 + np.random.random(),
                interaction_quality_confidence=0.8,
                class_emotional_support=3.5,
                class_classroom_organization=4.0,
                class_instructional_support=3.8,
                class_framework_confidence=0.88,
                metadata=metadata
            )
            labels.append(label)

        # Run research standards validation
        result = validator._validate_research_standards(labels)

        # Verify validation structure
        self.assertEqual(result['component'], 'Research Standards')
        self.assertIn('checks', result)

        # Verify statistical power check exists
        check_names = [check['name'] for check in result['checks']]
        self.assertIn('Statistical Power', check_names)


class TestDatasetVersionManager(unittest.TestCase):
    """Test dataset version management functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_issue_91_version_manager_basic(self):
        """Test Issue #91: Basic version manager functionality."""
        version_manager = DatasetVersionManager(self.temp_dir)

        self.assertEqual(version_manager.versions_dir, self.temp_dir)
        self.assertTrue(self.temp_dir.exists())

        # Test empty version list initially
        versions = version_manager.list_versions()
        self.assertEqual(len(versions), 0)


if __name__ == '__main__':
    print("🧪 Running Issue #91 Basic Unit Tests")
    print("=" * 50)
    print("Testing core functionality without heavy dependencies")
    print()

    unittest.main(verbosity=2, buffer=True)