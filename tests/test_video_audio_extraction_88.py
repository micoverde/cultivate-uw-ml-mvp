#!/usr/bin/env python3
"""
Unit tests for Issue #88: Video-to-Audio Extraction Pipeline

Tests the video-to-audio extraction functionality including:
- Video file finding and matching
- Audio extraction and processing
- Timestamp parsing and validation
- Clip generation and quality validation

Author: Claude (Issue #88 Testing)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_processing"))

from video_to_audio_extraction import VideoToAudioExtractor, NumpyEncoder


class TestIssue88VideoAudioExtraction(unittest.TestCase):
    """Test suite for Issue #88: Video-to-Audio Extraction Pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.video_dir = self.test_dir / "videos"
        self.output_dir = self.test_dir / "output"
        self.csv_file = self.test_dir / "test_annotations.csv"

        # Create test directories
        self.video_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        # Create test CSV
        test_data = {
            'Video Title': ['test_video.mp4', 'Test Video 2.MP4'],
            'Asset #': [12345, 67890],
            'Age group': ['3-5', '4-6'],
            'Timestamp': ['', ''],
            'Question 1 ': ['0:30', '1:15'],  # Note trailing space
            'Q1 description': ['Test question 1', 'Test question A'],
            'Question 2 ': ['0:45', 'na'],    # Note trailing space
            'Q2 description': ['Test question 2', ''],
            'Question 3': ['1:00', '2:30'],   # No trailing space
            'Q3 description': ['Test question 3', 'Test question C']
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.csv_file, index=False)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_issue_88_extractor_initialization(self):
        """Issue #88: Test VideoToAudioExtractor initialization with correct parameters"""
        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        self.assertEqual(extractor.video_dir, self.video_dir)
        self.assertEqual(extractor.csv_file, self.csv_file)
        self.assertEqual(extractor.output_dir, self.output_dir)
        self.assertEqual(extractor.target_sample_rate, 16000)
        self.assertEqual(extractor.clip_duration, 5.0)

    def test_issue_88_parse_timestamp_valid_formats(self):
        """Issue #88: Test timestamp parsing for various valid formats"""
        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        # Test valid timestamp formats
        self.assertEqual(extractor.parse_timestamp("0:30"), 30.0)
        self.assertEqual(extractor.parse_timestamp("1:15"), 75.0)
        self.assertEqual(extractor.parse_timestamp("2:00"), 120.0)
        self.assertEqual(extractor.parse_timestamp("10:45"), 645.0)

    def test_issue_88_parse_timestamp_invalid_formats(self):
        """Issue #88: Test timestamp parsing handles invalid formats correctly"""
        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        # Test invalid timestamp formats
        self.assertIsNone(extractor.parse_timestamp("na"))
        self.assertIsNone(extractor.parse_timestamp(""))
        self.assertIsNone(extractor.parse_timestamp("invalid"))
        self.assertIsNone(extractor.parse_timestamp("1:2:3"))  # Too many parts
        self.assertIsNone(extractor.parse_timestamp(None))

    def test_issue_88_find_video_file_exact_match(self):
        """Issue #88: Test video file finding with exact filename matches"""
        # Create test video files
        test_video = self.video_dir / "test_video.mp4"
        test_video.touch()

        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        found_file = extractor.find_video_file("test_video.mp4")
        self.assertEqual(found_file, test_video)

    def test_issue_88_find_video_file_case_variations(self):
        """Issue #88: Test video file finding with case variations"""
        # Create test video file with different case
        test_video = self.video_dir / "test_video.mp4"
        test_video.touch()

        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        # Should find the file even with case mismatch in query
        found_file = extractor.find_video_file("Test_Video.MP4")
        self.assertEqual(found_file, test_video)

    def test_issue_88_find_video_file_not_found(self):
        """Issue #88: Test video file finding returns None for missing files"""
        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        found_file = extractor.find_video_file("nonexistent_video.mp4")
        self.assertIsNone(found_file)

    def test_issue_88_csv_column_name_handling(self):
        """Issue #88: Test CSV parsing handles column name variations (trailing spaces)"""
        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        df = pd.read_csv(self.csv_file)

        # Test that the CSV columns are loaded correctly
        # This tests the column parsing without needing complex mocking
        df = extractor.load_annotations()
        self.assertIn('Question 1 ', df.columns)  # With trailing space
        self.assertIn('Question 3', df.columns)   # Without trailing space

    @patch('video_to_audio_extraction.librosa.load')
    @patch('video_to_audio_extraction.VideoFileClip')
    def test_issue_88_extract_audio_from_video_success(self, mock_video_clip, mock_librosa_load):
        """Issue #88: Test successful audio extraction from video file"""
        # Mock video clip
        mock_audio = Mock()
        mock_video = Mock()
        mock_video.audio = mock_audio
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Mock librosa load
        mock_audio_data = np.random.rand(16000 * 10)  # 10 seconds of audio
        mock_librosa_load.return_value = (mock_audio_data, 16000)

        # Create test video file
        test_video = self.video_dir / "test_video.mp4"
        test_video.touch()

        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        result = extractor.extract_audio_from_video(test_video)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(mock_audio_data))
        mock_video_clip.assert_called_once()
        mock_librosa_load.assert_called_once()

    @patch('video_to_audio_extraction.VideoFileClip')
    def test_issue_88_extract_audio_from_video_no_audio_track(self, mock_video_clip):
        """Issue #88: Test audio extraction handles videos without audio tracks"""
        # Mock video clip with no audio
        mock_video = Mock()
        mock_video.audio = None
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Create test video file
        test_video = self.video_dir / "test_video.mp4"
        test_video.touch()

        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        result = extractor.extract_audio_from_video(test_video)

        self.assertIsNone(result)
        mock_video_clip.assert_called_once()

    def test_issue_88_extract_audio_clip_valid_timestamp(self):
        """Issue #88: Test audio clip extraction with valid timestamp"""
        # Create mock full audio (10 seconds at 16kHz)
        full_audio = np.random.rand(16000 * 10)

        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        # Ensure clip directory exists
        (self.output_dir / "clips").mkdir(parents=True, exist_ok=True)

        # Mock soundfile.write to avoid actual file I/O and create the file
        def mock_write_func(path, audio, samplerate=16000, *args, **kwargs):
            # Create the actual file that validate_audio_clip will check for
            Path(path).touch()

        with patch('video_to_audio_extraction.sf.write', side_effect=mock_write_func) as mock_write:
            with patch.object(extractor, 'validate_audio_clip') as mock_validate:
                mock_validate.return_value = {
                    'is_valid_clip': True,
                    'duration_seconds': 5.0,
                    'file_exists': True
                }

                result = extractor.extract_audio_clip(
                    full_audio, 5.0, "12345", 1
                )

                self.assertIsNotNone(result)
                self.assertEqual(result['asset_number'], "12345")
                self.assertEqual(result['question_number'], 1)
                self.assertEqual(result['question_timestamp'], 5.0)
                mock_write.assert_called_once()

    def test_issue_88_extract_audio_clip_edge_timestamp(self):
        """Issue #88: Test audio clip extraction with timestamp near beginning/end"""
        # Create mock full audio (10 seconds at 16kHz)
        full_audio = np.random.rand(16000 * 10)

        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        # Ensure clip directory exists
        (self.output_dir / "clips").mkdir(parents=True, exist_ok=True)

        # Mock soundfile.write to avoid actual file I/O and create the file
        def mock_write_func(path, audio, samplerate=16000, *args, **kwargs):
            # Create the actual file that validate_audio_clip will check for
            Path(path).touch()

        with patch('video_to_audio_extraction.sf.write', side_effect=mock_write_func) as mock_write:
            with patch.object(extractor, 'validate_audio_clip') as mock_validate:
                mock_validate.return_value = {
                    'is_valid_clip': True,
                    'duration_seconds': 3.0,  # Shorter clip due to edge
                    'file_exists': True
                }

                # Test near beginning (should clip to start)
                result = extractor.extract_audio_clip(
                    full_audio, 1.0, "12345", 1
                )

                self.assertIsNotNone(result)
                mock_write.assert_called()

    def test_issue_88_numpy_encoder_handles_numpy_types(self):
        """Issue #88: Test NumpyEncoder handles numpy data types correctly"""
        encoder = NumpyEncoder()

        # Test numpy integer
        np_int = np.int32(42)
        self.assertEqual(encoder.default(np_int), 42)

        # Test numpy float
        np_float = np.float32(3.14)
        self.assertEqual(encoder.default(np_float), 3.140000104904175)  # float32 precision

        # Test numpy boolean
        np_bool = np.bool_(True)
        self.assertEqual(encoder.default(np_bool), True)

        # Test numpy array
        np_array = np.array([1, 2, 3])
        self.assertEqual(encoder.default(np_array), [1, 2, 3])

    def test_issue_88_json_serialization_with_numpy_encoder(self):
        """Issue #88: Test JSON serialization works with numpy types using NumpyEncoder"""
        test_data = {
            'int_value': np.int32(42),
            'float_value': np.float32(3.14),
            'bool_value': np.bool_(True),
            'array_value': np.array([1, 2, 3])
        }

        # Should not raise an exception
        json_str = json.dumps(test_data, cls=NumpyEncoder)
        parsed_data = json.loads(json_str)

        self.assertEqual(parsed_data['int_value'], 42)
        self.assertAlmostEqual(parsed_data['float_value'], 3.14, places=5)
        self.assertEqual(parsed_data['bool_value'], True)
        self.assertEqual(parsed_data['array_value'], [1, 2, 3])

    @patch('video_to_audio_extraction.librosa.feature.zero_crossing_rate')
    @patch('video_to_audio_extraction.librosa.load')
    def test_issue_88_validate_clip_quality_metrics(self, mock_librosa_load, mock_zcr):
        """Issue #88: Test clip validation includes quality metrics"""
        # Mock librosa functions
        mock_audio = np.random.rand(16000 * 5)  # 5 seconds
        mock_librosa_load.return_value = (mock_audio, 16000)
        mock_zcr.return_value = np.array([[0.1]])

        # Create test clip file
        test_clip = self.output_dir / "test_clip.wav"
        test_clip.touch()

        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        validation = extractor.validate_audio_clip(test_clip)

        # Check that validation includes required metrics
        self.assertIn('is_valid_clip', validation)
        self.assertIn('duration_seconds', validation)
        self.assertIn('sample_rate', validation)
        self.assertIn('rms_energy', validation)
        self.assertIn('zero_crossing_rate', validation)
        self.assertIn('is_silent', validation)
        self.assertIn('has_clipping', validation)

    def test_issue_88_stats_tracking_initialization(self):
        """Issue #88: Test extraction statistics are properly initialized and tracked"""
        extractor = VideoToAudioExtractor(
            video_dir=str(self.video_dir),
            csv_file=str(self.csv_file),
            output_dir=str(self.output_dir)
        )

        # Check initial stats
        expected_keys = [
            'videos_processed', 'clips_extracted', 'failed_extractions',
            'total_duration_extracted', 'errors'
        ]

        for key in expected_keys:
            self.assertIn(key, extractor.stats)

        self.assertEqual(extractor.stats['videos_processed'], 0)
        self.assertEqual(extractor.stats['clips_extracted'], 0)
        self.assertEqual(extractor.stats['failed_extractions'], 0)
        self.assertEqual(extractor.stats['total_duration_extracted'], 0.0)
        self.assertEqual(len(extractor.stats['errors']), 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)