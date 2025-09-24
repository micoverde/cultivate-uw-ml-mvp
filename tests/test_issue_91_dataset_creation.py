#!/usr/bin/env python3
"""
Unit Tests for Issue #91: HuggingFace Dataset Creation

Comprehensive test suite validating dataset creation, validation framework,
and multi-task learning optimization for the Cultivate Learning ML pipeline.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #91 - Quality Label Standardization and ML Dataset Creation
"""

import unittest
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import modules to test
import sys
sys.path.append('src/data_processing')

from dataset_creation import (
    HuggingFaceDatasetBuilder, DatasetVersion, DatasetMetadata,
    DatasetVersionManager
)
from validation_framework import (
    ComprehensiveValidationFramework, ValidationReport, ValidationThresholds
)
from label_processing import MultiTaskLabels, ValidationResult
from multi_task_labels import LabelMetadata

class TestHuggingFaceDatasetBuilder(unittest.TestCase):
    """Test suite for HuggingFace dataset creation functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.builder = HuggingFaceDatasetBuilder(
            output_dir=self.temp_dir,
            random_seed=42
        )

    def test_builder_initialization(self):
        """Test Issue #91: Dataset builder initializes correctly."""
        self.assertEqual(self.builder.random_seed, 42)
        self.assertEqual(str(self.builder.output_dir), self.temp_dir)
        self.assertIsInstance(self.builder.version_manager, DatasetVersionManager)

    def test_create_label_identifier(self):
        """Test Issue #91: Label identifier creation for feature alignment."""
        # Create test label with metadata
        metadata = LabelMetadata(
            video_id="test_video_001",
            clip_filename="clip_001.wav",
            annotator_id="annotator_1",
            timestamp_start=10.5,
            timestamp_end=15.2,
            original_csv_row=1
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

        identifier = self.builder._create_label_identifier(label)
        self.assertEqual(identifier, "asset_test_video_001_q_unknown")

    def test_ids_match_partially(self):
        """Test Issue #91: Partial ID matching for feature-label alignment."""
        feature_id = "test_video_001_segment_10_features"
        label_id = "asset_test_video_001_q_1"

        matches = self.builder._ids_match_partially(feature_id, label_id)
        self.assertTrue(matches)

        # Test non-matching IDs
        feature_id_2 = "different_video_002_segment_5"
        matches_2 = self.builder._ids_match_partially(feature_id_2, label_id)
        self.assertFalse(matches_2)

    def test_convert_to_quality_class(self):
        """Test Issue #91: Quality score discretization for classification."""
        # Test low quality
        self.assertEqual(self.builder._convert_to_quality_class(0.3), 'low')

        # Test medium quality
        self.assertEqual(self.builder._convert_to_quality_class(0.5), 'medium')

        # Test high quality
        self.assertEqual(self.builder._convert_to_quality_class(0.8), 'high')

    def test_create_stratified_splits(self):
        """Test Issue #91: Stratified data splits for multi-task learning."""
        # Create synthetic data
        n_samples = 100
        features = np.random.randn(n_samples, 61)  # 61-dimensional from Issue #90

        labels = []
        for i in range(n_samples):
            metadata = LabelMetadata(
                video_id=f"video_{i % 10}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i),
                timestamp_end=float(i + 5),
                original_csv_row=i
            )

            label = MultiTaskLabels(
                question_type="OEQ" if i % 2 == 0 else "CEQ",
                question_confidence=0.8 + 0.1 * np.random.random(),
                wait_time_appropriate=bool(i % 3 == 0),
                wait_time_confidence=0.85,
                interaction_quality_score=3.0 + 2.0 * np.random.random(),
                interaction_quality_confidence=0.9,
                class_emotional_support=3.0 + np.random.random(),
                class_classroom_organization=3.0 + np.random.random(),
                class_instructional_support=3.0 + np.random.random(),
                class_framework_confidence=0.88,
                metadata=metadata
            )
            labels.append(label)

        # Test stratified splits
        train_data, val_data, test_data = self.builder._create_stratified_splits(
            features, labels, test_size=0.2, val_size=0.1
        )

        # Verify split sizes
        self.assertAlmostEqual(len(train_data['features']) / n_samples, 0.7, delta=0.1)
        self.assertAlmostEqual(len(val_data['features']) / n_samples, 0.1, delta=0.05)
        self.assertAlmostEqual(len(test_data['features']) / n_samples, 0.2, delta=0.1)

        # Verify feature dimensions maintained
        self.assertEqual(train_data['features'].shape[1], 61)
        self.assertEqual(val_data['features'].shape[1], 61)
        self.assertEqual(test_data['features'].shape[1], 61)

    @patch('datasets.DatasetDict')
    @patch('datasets.Dataset')
    def test_create_huggingface_datasets(self, mock_dataset, mock_dataset_dict):
        """Test Issue #91: HuggingFace dataset conversion."""
        # Mock dataset creation
        mock_dataset.from_dict.return_value = Mock()
        mock_dataset_dict.return_value = Mock()

        # Create test data
        features = np.random.randn(50, 61)
        labels = []

        for i in range(50):
            metadata = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i),
                timestamp_end=float(i + 5),
                original_csv_row=i
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
            labels.append(label)

        train_data = {'features': features[:35], 'labels': labels[:35]}
        val_data = {'features': features[35:42], 'labels': labels[35:42]}
        test_data = {'features': features[42:], 'labels': labels[42:]}

        # Test dataset creation
        result = self.builder._create_huggingface_datasets(train_data, val_data, test_data)

        # Verify dataset creation was called
        self.assertTrue(mock_dataset.from_dict.called)

    def test_convert_to_dataset_examples(self):
        """Test Issue #91: Dataset example conversion for HuggingFace format."""
        # Create test data
        features = np.random.randn(5, 61)
        labels = []

        for i in range(5):
            metadata = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i * 10),
                timestamp_end=float(i * 10 + 5),
                original_csv_row=i
            )

            label = MultiTaskLabels(
                question_type="OEQ" if i % 2 == 0 else "CEQ",
                question_confidence=0.85,
                wait_time_appropriate=bool(i % 2),
                wait_time_confidence=0.90,
                interaction_quality_score=3.0 + i * 0.5,
                interaction_quality_confidence=0.88,
                class_emotional_support=3.0 + i * 0.2,
                class_classroom_organization=4.0,
                class_instructional_support=3.5,
                class_framework_confidence=0.86,
                metadata=metadata
            )
            labels.append(label)

        # Convert to dataset examples
        examples = self.builder._convert_to_dataset_examples(features, labels)

        # Verify structure
        self.assertIn('input_features', examples)
        self.assertIn('question_type', examples)
        self.assertIn('labels', examples)
        self.assertIn('metadata', examples)

        # Verify data integrity
        self.assertEqual(len(examples['input_features']), 5)
        self.assertEqual(len(examples['question_type']), 5)
        self.assertEqual(examples['question_type'][0], 'open_ended')  # OEQ mapped
        self.assertEqual(examples['question_type'][1], 'closed_ended')  # CEQ mapped

    def test_generate_content_hash(self):
        """Test Issue #91: Cryptographic content hashing for dataset integrity."""
        # Create test data
        features = np.random.randn(10, 61)
        labels = []

        for i in range(10):
            metadata = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i),
                timestamp_end=float(i + 5),
                original_csv_row=i
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
            labels.append(label)

        # Mock dataset dict
        mock_dataset_dict = {
            'train': Mock(),
            'validation': Mock(),
            'test': Mock()
        }

        # Generate hash
        content_hash = self.builder._generate_content_hash(mock_dataset_dict, features, labels)

        # Verify hash properties
        self.assertIsInstance(content_hash, str)
        self.assertEqual(len(content_hash), 64)  # SHA256 hex length

        # Verify reproducibility
        content_hash_2 = self.builder._generate_content_hash(mock_dataset_dict, features, labels)
        self.assertEqual(content_hash, content_hash_2)


class TestValidationFramework(unittest.TestCase):
    """Test suite for comprehensive validation framework."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ComprehensiveValidationFramework(output_dir=self.temp_dir)

    def test_framework_initialization(self):
        """Test Issue #91: Validation framework initializes correctly."""
        self.assertIsInstance(self.validator.thresholds, ValidationThresholds)
        self.assertEqual(str(self.validator.output_dir), self.temp_dir)
        self.assertEqual(self.validator.thresholds.cohens_kappa_minimum, 0.75)
        self.assertEqual(self.validator.thresholds.missing_data_threshold, 0.05)

    def test_validate_data_quality(self):
        """Test Issue #91: Data quality validation checks."""
        # Create test data
        features = {f"sample_{i}": np.random.randn(61) for i in range(50)}
        labels = []

        for i in range(50):
            metadata = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i),
                timestamp_end=float(i + 5),
                original_csv_row=i
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
            labels.append(label)

        # Run validation
        result = self.validator._validate_data_quality(labels, features)

        # Verify validation structure
        self.assertIn('component', result)
        self.assertIn('status', result)
        self.assertIn('checks', result)
        self.assertIn('summary', result)

        # Verify checks performed
        check_names = [check['name'] for check in result['checks']]
        self.assertIn('Data Completeness', check_names)
        self.assertIn('Missing Data Rate', check_names)
        self.assertIn('Data Consistency', check_names)

    def test_validate_annotation_quality(self):
        """Test Issue #91: Annotation quality validation."""
        # Create test labels with varying quality
        labels = []

        for i in range(30):
            metadata = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i),
                timestamp_end=float(i + 5),
                original_csv_row=i
            )

            label = MultiTaskLabels(
                question_type="OEQ" if i % 2 == 0 else "CEQ",
                question_confidence=0.7 + 0.2 * np.random.random(),
                wait_time_appropriate=bool(i % 3 == 0),
                wait_time_confidence=0.85,
                interaction_quality_score=2.0 + 3.0 * np.random.random(),
                interaction_quality_confidence=0.8,
                class_emotional_support=3.0 + np.random.random(),
                class_classroom_organization=3.0 + np.random.random(),
                class_instructional_support=3.0 + np.random.random(),
                class_framework_confidence=0.85,
                metadata=metadata
            )
            labels.append(label)

        # Run annotation quality validation
        result = self.validator._validate_annotation_quality(labels)

        # Verify validation results
        self.assertEqual(result['component'], 'Annotation Quality')
        self.assertIn('status', result)
        self.assertIsInstance(result['checks'], list)

        # Check specific validation checks
        check_names = [check['name'] for check in result['checks']]
        self.assertIn('Annotation Completeness', check_names)
        self.assertIn('Annotation Confidence', check_names)

    def test_validate_feature_extraction(self):
        """Test Issue #91: Feature extraction quality validation."""
        # Create 61-dimensional feature vectors (Issue #90 compatibility)
        features = {f"sample_{i}": np.random.randn(61) for i in range(25)}

        # Add some edge cases
        features["nan_sample"] = np.full(61, np.nan)
        features["inf_sample"] = np.full(61, np.inf)

        labels = []
        for i in range(27):  # 25 + 2 edge cases
            metadata = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i),
                timestamp_end=float(i + 5),
                original_csv_row=i
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
            labels.append(label)

        # Run feature validation
        result = self.validator._validate_feature_extraction(features, labels)

        # Verify validation performed
        self.assertEqual(result['component'], 'Feature Extraction')
        self.assertIn('checks', result)

        # Check dimension validation
        dimension_check = next(
            (check for check in result['checks'] if check['name'] == 'Feature Dimensions'),
            None
        )
        self.assertIsNotNone(dimension_check)

    def test_validate_research_standards(self):
        """Test Issue #91: Educational research standards compliance."""
        # Create sufficiently large sample for research standards
        labels = []

        for i in range(120):  # Above minimum threshold
            metadata = LabelMetadata(
                video_id=f"video_{i % 20}",  # 20 different videos
                clip_filename=f"clip_{i}.wav",
                annotator_id=f"annotator_{i % 3}",  # 3 annotators
                timestamp_start=float(i * 5),
                timestamp_end=float(i * 5 + 5),
                original_csv_row=i
            )

            label = MultiTaskLabels(
                question_type=["OEQ", "CEQ", "SCAFFOLD", "REDIRECT"][i % 4],
                question_confidence=0.8 + 0.1 * np.random.random(),
                wait_time_appropriate=bool(i % 2),
                wait_time_confidence=0.85 + 0.1 * np.random.random(),
                interaction_quality_score=2.0 + 3.0 * np.random.random(),
                interaction_quality_confidence=0.8 + 0.15 * np.random.random(),
                class_emotional_support=3.0 + 2.0 * np.random.random(),
                class_classroom_organization=3.0 + 2.0 * np.random.random(),
                class_instructional_support=3.0 + 2.0 * np.random.random(),
                class_framework_confidence=0.8 + 0.15 * np.random.random(),
                metadata=metadata
            )
            labels.append(label)

        # Run research standards validation
        result = self.validator._validate_research_standards(labels)

        # Verify validation structure
        self.assertEqual(result['component'], 'Research Standards')
        self.assertIn('checks', result)

        # Verify statistical power check passes
        power_check = next(
            (check for check in result['checks'] if check['name'] == 'Statistical Power'),
            None
        )
        self.assertIsNotNone(power_check)
        self.assertEqual(power_check['status'], 'PASS')

    def test_cohens_kappa_calculation(self):
        """Test Issue #91: Cohen's kappa calculation for inter-annotator agreement."""
        # Create overlapping annotations with known agreement
        overlapping_pairs = []

        # Perfect agreement pairs
        for i in range(10):
            metadata1 = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="annotator_1",
                timestamp_start=float(i * 5),
                timestamp_end=float(i * 5 + 5),
                original_csv_row=i
            )

            metadata2 = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="annotator_2",
                timestamp_start=float(i * 5),
                timestamp_end=float(i * 5 + 5),
                original_csv_row=i
            )

            # Same quality scores for perfect agreement
            quality_score = 3.0 + i % 3

            label1 = MultiTaskLabels(
                question_type="OEQ",
                question_confidence=0.85,
                wait_time_appropriate=True,
                wait_time_confidence=0.90,
                interaction_quality_score=quality_score,
                interaction_quality_confidence=0.88,
                class_emotional_support=3.8,
                class_classroom_organization=4.1,
                class_instructional_support=3.9,
                class_framework_confidence=0.86,
                metadata=metadata1
            )

            label2 = MultiTaskLabels(
                question_type="OEQ",
                question_confidence=0.85,
                wait_time_appropriate=True,
                wait_time_confidence=0.90,
                interaction_quality_score=quality_score,  # Same score
                interaction_quality_confidence=0.88,
                class_emotional_support=3.8,
                class_classroom_organization=4.1,
                class_instructional_support=3.9,
                class_framework_confidence=0.86,
                metadata=metadata2
            )

            overlapping_pairs.append((label1, label2))

        # Calculate Cohen's kappa
        kappa = self.validator._calculate_cohen_kappa(overlapping_pairs)

        # Perfect agreement should yield kappa = 1.0
        self.assertAlmostEqual(kappa, 1.0, delta=0.1)

    def test_statistical_significance_validation(self):
        """Test Issue #91: Statistical significance testing."""
        # Create test data with known statistical properties
        labels = []

        for i in range(60):
            metadata = LabelMetadata(
                video_id=f"video_{i}",
                clip_filename=f"clip_{i}.wav",
                annotator_id="test_annotator",
                timestamp_start=float(i),
                timestamp_end=float(i + 5),
                original_csv_row=i
            )

            # Create quality difference between question types
            if i % 2 == 0:  # OEQ questions
                quality_score = 4.0 + 0.5 * np.random.random()  # Higher quality
                question_type = "OEQ"
            else:  # CEQ questions
                quality_score = 3.0 + 0.5 * np.random.random()  # Lower quality
                question_type = "CEQ"

            label = MultiTaskLabels(
                question_type=question_type,
                question_confidence=0.85,
                wait_time_appropriate=bool(i % 2),
                wait_time_confidence=0.90,
                interaction_quality_score=quality_score,
                interaction_quality_confidence=0.88,
                class_emotional_support=3.5 + 0.5 * np.random.random(),
                class_classroom_organization=3.5 + 0.5 * np.random.random(),
                class_instructional_support=3.5 + 0.5 * np.random.random(),
                class_framework_confidence=0.86,
                metadata=metadata
            )
            labels.append(label)

        # Mock dataset dict
        mock_dataset_dict = Mock()

        # Run statistical significance validation
        result = self.validator._validate_statistical_significance(labels, mock_dataset_dict)

        # Verify statistical tests performed
        self.assertEqual(result['component'], 'Statistical Significance')
        self.assertIn('tests', result)
        self.assertIsInstance(result['tests'], list)

    def test_validation_thresholds_customization(self):
        """Test Issue #91: Configurable validation thresholds."""
        # Create custom thresholds
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

    def test_generate_recommendations(self):
        """Test Issue #91: Validation recommendation generation."""
        # Create mock validation checks with failures
        validation_checks = [
            {
                'component': 'Data Quality',
                'status': 'FAIL',
                'checks': [
                    {'name': 'Data Completeness', 'status': 'FAIL', 'critical': True}
                ]
            },
            {
                'component': 'Annotation Quality',
                'status': 'WARNING',
                'checks': [
                    {'name': 'Annotation Confidence', 'status': 'WARNING', 'critical': False}
                ]
            }
        ]

        # Generate recommendations
        recommendations = self.validator._generate_recommendations(validation_checks, 'FAIL')

        # Verify recommendations generated
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)
        self.assertTrue(any('CRITICAL' in rec for rec in recommendations))

    def test_critical_issues_extraction(self):
        """Test Issue #91: Critical issue identification."""
        # Create validation checks with critical failures
        validation_checks = [
            {
                'component': 'Feature Extraction',
                'status': 'FAIL',
                'checks': [
                    {
                        'name': 'Feature Dimensions',
                        'status': 'FAIL',
                        'critical': True,
                        'details': 'Expected 61 dimensions, got 30'
                    }
                ]
            }
        ]

        # Extract critical issues
        critical_issues = self.validator._extract_critical_issues(validation_checks)

        # Verify critical issues identified
        self.assertIsInstance(critical_issues, list)
        self.assertTrue(len(critical_issues) > 0)
        self.assertTrue(any('Feature Dimensions' in issue for issue in critical_issues))


class TestDatasetVersionManager(unittest.TestCase):
    """Test suite for dataset version management."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.version_manager = DatasetVersionManager(self.temp_dir)

    def test_version_manager_initialization(self):
        """Test Issue #91: Version manager initializes correctly."""
        self.assertEqual(self.version_manager.versions_dir, self.temp_dir)
        self.assertTrue(self.temp_dir.exists())

    def test_store_and_list_versions(self):
        """Test Issue #91: Dataset version storage and listing."""
        # Create test dataset version
        test_version = DatasetVersion(
            version_id="v20240923_120000_abcd1234",
            content_hash="1234567890abcdef" * 4,  # 64 character hash
            creation_timestamp=datetime.now(),
            feature_vector_shape=(100, 61),
            label_counts={'total': 100, 'by_quality': {'high': 40, 'medium': 35, 'low': 25}},
            split_sizes={'train': 70, 'validation': 15, 'test': 15},
            metadata={'dataset_name': 'test_dataset'},
            dependencies={'feature_extraction': '1.0.0'},
            validation_results=[]
        )

        # Store version
        success = self.version_manager.store_version(test_version)
        self.assertTrue(success)

        # List versions
        versions = self.version_manager.list_versions()
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0].version_id, "v20240923_120000_abcd1234")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete Issue #91 pipeline."""

    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()

    @patch('datasets.DatasetDict')
    @patch('datasets.Dataset')
    def test_end_to_end_dataset_creation(self, mock_dataset, mock_dataset_dict):
        """Test Issue #91: Complete end-to-end dataset creation pipeline."""
        # Mock HuggingFace components
        mock_dataset.from_dict.return_value = Mock()
        mock_dataset_dict.return_value = Mock()

        # Initialize components
        builder = HuggingFaceDatasetBuilder(output_dir=self.temp_dir)

        # Create realistic test data
        n_samples = 150
        feature_vectors = {}
        multi_task_labels = []

        for i in range(n_samples):
            # Create feature vector (61-dimensional from Issue #90)
            feature_vectors[f"sample_{i}"] = np.random.randn(61)

            # Create multi-task label
            metadata = LabelMetadata(
                video_id=f"video_{i % 20}",
                clip_filename=f"clip_{i}.wav",
                annotator_id=f"annotator_{i % 3}",
                timestamp_start=float(i * 5),
                timestamp_end=float(i * 5 + 5),
                original_csv_row=i
            )

            label = MultiTaskLabels(
                question_type=["OEQ", "CEQ", "SCAFFOLD"][i % 3],
                question_confidence=0.7 + 0.2 * np.random.random(),
                wait_time_appropriate=bool(i % 2),
                wait_time_confidence=0.8 + 0.15 * np.random.random(),
                interaction_quality_score=2.0 + 3.0 * np.random.random(),
                interaction_quality_confidence=0.8 + 0.15 * np.random.random(),
                class_emotional_support=3.0 + 2.0 * np.random.random(),
                class_classroom_organization=3.0 + 2.0 * np.random.random(),
                class_instructional_support=3.0 + 2.0 * np.random.random(),
                class_framework_confidence=0.8 + 0.15 * np.random.random(),
                metadata=metadata
            )
            multi_task_labels.append(label)

        # Test dataset creation (would normally create HuggingFace dataset)
        try:
            aligned_features, aligned_labels = builder._align_features_with_labels(
                feature_vectors, multi_task_labels
            )

            # Verify alignment successful
            self.assertGreater(len(aligned_features), 0)
            self.assertEqual(len(aligned_features), len(aligned_labels))

            # Test stratified splits
            train_data, val_data, test_data = builder._create_stratified_splits(
                aligned_features, aligned_labels, test_size=0.2, val_size=0.1
            )

            # Verify split integrity
            total_samples = len(train_data['features']) + len(val_data['features']) + len(test_data['features'])
            self.assertEqual(total_samples, len(aligned_features))

        except Exception as e:
            self.fail(f"End-to-end dataset creation failed: {e}")

    def test_comprehensive_pipeline_validation(self):
        """Test Issue #91: Complete validation pipeline integration."""
        validator = ComprehensiveValidationFramework(output_dir=self.temp_dir)

        # Create comprehensive test data
        n_samples = 80
        feature_vectors = {f"sample_{i}": np.random.randn(61) for i in range(n_samples)}
        multi_task_labels = []

        for i in range(n_samples):
            metadata = LabelMetadata(
                video_id=f"video_{i % 15}",
                clip_filename=f"clip_{i}.wav",
                annotator_id=f"annotator_{i % 2}",  # 2 annotators for agreement
                timestamp_start=float(i * 5),
                timestamp_end=float(i * 5 + 5),
                original_csv_row=i
            )

            label = MultiTaskLabels(
                question_type=["OEQ", "CEQ"][i % 2],
                question_confidence=0.75 + 0.2 * np.random.random(),
                wait_time_appropriate=bool((i + 1) % 3),  # Varied pattern
                wait_time_confidence=0.85 + 0.1 * np.random.random(),
                interaction_quality_score=2.5 + 2.0 * np.random.random(),
                interaction_quality_confidence=0.85 + 0.1 * np.random.random(),
                class_emotional_support=3.0 + 1.5 * np.random.random(),
                class_classroom_organization=3.0 + 1.5 * np.random.random(),
                class_instructional_support=3.0 + 1.5 * np.random.random(),
                class_framework_confidence=0.85 + 0.1 * np.random.random(),
                metadata=metadata
            )
            multi_task_labels.append(label)

        # Test individual validation components
        data_quality = validator._validate_data_quality(multi_task_labels, feature_vectors)
        self.assertIn('status', data_quality)

        annotation_quality = validator._validate_annotation_quality(multi_task_labels)
        self.assertIn('status', annotation_quality)

        feature_quality = validator._validate_feature_extraction(feature_vectors, multi_task_labels)
        self.assertIn('status', feature_quality)


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce test noise

    print("🧪 Running Issue #91 Unit Tests")
    print("=" * 50)
    print("Testing HuggingFace Dataset Creation & Validation Framework")
    print("Enterprise-grade ML pipeline validation")
    print()

    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)