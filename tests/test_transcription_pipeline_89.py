#!/usr/bin/env python3
"""
Unit tests for Issue #89: Transcription & Speaker Diarization Pipeline

Tests the transcription and speaker diarization functionality including:
- Whisper transcription with word-level timestamps
- Pyannote speaker diarization and identification
- Educator vs child speaker classification
- Transcript alignment with quality annotations
- Processing pipeline integration

Author: Claude (Issue #89 Testing)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import asdict

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_processing"))

from transcription_pipeline import (
    TranscriptionPipeline,
    TranscriptWord,
    SpeakerSegment,
    TranscriptSegment,
    ClipTranscript,
    TranscriptionEncoder
)


class TestIssue89TranscriptionPipeline(unittest.TestCase):
    """Test suite for Issue #89: Transcription & Speaker Diarization Pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.clips_dir = self.test_dir / "clips"
        self.output_dir = self.test_dir / "transcripts"
        self.metadata_file = self.test_dir / "clip_manifest.json"

        # Create test directories
        self.clips_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        # Create test metadata
        test_metadata = {
            'clips': [
                {
                    'clip_filename': 'test_12345_q1.wav',
                    'asset_number': '12345',
                    'question_number': 1,
                    'video_title': 'Test Educational Video',
                    'question_description': 'OEQ and teacher gives opportunity to respond',
                    'age_group': '4-5',
                    'clip_start_time': 30.5,
                    'clip_end_time': 35.5,
                    'actual_duration': 5.0
                },
                {
                    'clip_filename': 'test_67890_q2.wav',
                    'asset_number': '67890',
                    'question_number': 2,
                    'video_title': 'Another Test Video',
                    'question_description': 'CEQ wait time analysis',
                    'age_group': '3-4',
                    'clip_start_time': 45.0,
                    'clip_end_time': 48.0,
                    'actual_duration': 3.0
                }
            ]
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(test_metadata, f)

        # Create test audio files (empty files for testing)
        (self.clips_dir / 'test_12345_q1.wav').touch()
        (self.clips_dir / 'test_67890_q2.wav').touch()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_issue_89_pipeline_initialization(self):
        """Issue #89: Test TranscriptionPipeline initialization with correct parameters"""
        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir),
            whisper_model="base"
        )

        self.assertEqual(pipeline.clips_dir, self.clips_dir)
        self.assertEqual(pipeline.metadata_file, self.metadata_file)
        self.assertEqual(pipeline.output_dir, self.output_dir)
        self.assertEqual(pipeline.whisper_model_name, "base")

        # Check that directories are created
        self.assertTrue(pipeline.transcripts_dir.exists())
        self.assertTrue(pipeline.validation_dir.exists())
        self.assertTrue(pipeline.reports_dir.exists())

    def test_issue_89_load_clip_metadata(self):
        """Issue #89: Test loading clip metadata from Issue #88 output"""
        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        metadata = pipeline.load_clip_metadata()

        self.assertIn('clips', metadata)
        self.assertEqual(len(metadata['clips']), 2)

        clip1 = metadata['clips'][0]
        self.assertEqual(clip1['clip_filename'], 'test_12345_q1.wav')
        self.assertEqual(clip1['asset_number'], '12345')
        self.assertEqual(clip1['question_number'], 1)

    def test_issue_89_classify_question_type(self):
        """Issue #89: Test question type classification from descriptions"""
        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        # Test OEQ classification
        self.assertEqual(pipeline.classify_question_type("OEQ and teacher gives opportunity"), "OEQ")
        self.assertEqual(pipeline.classify_question_type("open ended question"), "OEQ")

        # Test CEQ classification
        self.assertEqual(pipeline.classify_question_type("CEQ wait time analysis"), "CEQ")
        self.assertEqual(pipeline.classify_question_type("closed question with response"), "CEQ")

        # Test wait time classification
        self.assertEqual(pipeline.classify_question_type("wait time for student response"), "wait_time")
        self.assertEqual(pipeline.classify_question_type("teacher pauses after question"), "wait_time")

        # Test unknown classification
        self.assertEqual(pipeline.classify_question_type("unclear description"), "unknown")

    @patch('transcription_pipeline.whisper.load_model')
    @patch('transcription_pipeline.librosa.load')
    def test_issue_89_transcribe_audio_clip(self, mock_librosa_load, mock_whisper_load):
        """Issue #89: Test Whisper transcription with word-level timestamps"""
        # Mock Whisper model and result
        mock_whisper_model = Mock()
        mock_whisper_load.return_value = mock_whisper_model

        # Mock transcription result
        mock_transcription_result = {
            'text': 'What do you see in the picture?',
            'segments': [
                {
                    'start': 0.0,
                    'end': 3.0,
                    'text': 'What do you see in the picture?',
                    'avg_logprob': -0.2,
                    'words': [
                        {'word': 'What', 'start': 0.0, 'end': 0.3, 'confidence': 0.95},
                        {'word': 'do', 'start': 0.3, 'end': 0.5, 'confidence': 0.92},
                        {'word': 'you', 'start': 0.5, 'end': 0.7, 'confidence': 0.98},
                        {'word': 'see', 'start': 0.7, 'end': 1.0, 'confidence': 0.96}
                    ]
                }
            ],
            'language': 'en'
        }
        mock_whisper_model.transcribe.return_value = mock_transcription_result

        # Mock librosa audio loading
        mock_audio_data = np.random.rand(16000 * 5)  # 5 seconds at 16kHz
        mock_librosa_load.return_value = (mock_audio_data, 16000)

        # Create test clip file
        test_clip = self.clips_dir / "test_clip.wav"
        test_clip.touch()

        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )
        pipeline.whisper_model = mock_whisper_model

        result = pipeline.transcribe_audio_clip(test_clip)

        self.assertIsNotNone(result)
        self.assertEqual(result['text'], 'What do you see in the picture?')
        self.assertEqual(result['language'], 'en')
        self.assertIn('confidence', result)
        self.assertIn('duration', result)
        self.assertEqual(len(result['segments']), 1)

        # Verify whisper was called correctly
        mock_whisper_model.transcribe.assert_called_once()
        call_args = mock_whisper_model.transcribe.call_args
        self.assertTrue('word_timestamps' in call_args[1])
        self.assertEqual(call_args[1]['language'], 'en')

    @patch('transcription_pipeline.torchaudio.load')
    @patch('transcription_pipeline.Pipeline.from_pretrained')
    def test_issue_89_speaker_diarization(self, mock_pipeline_load, mock_torchaudio_load):
        """Issue #89: Test pyannote speaker diarization functionality"""
        # Mock torchaudio loading
        mock_waveform = torch.randn(1, 16000 * 5)  # 5 seconds
        mock_torchaudio_load.return_value = (mock_waveform, 16000)

        # Mock pyannote pipeline
        mock_diarization_pipeline = Mock()
        mock_pipeline_load.return_value = mock_diarization_pipeline

        # Mock diarization result
        mock_diarization_result = Mock()
        mock_segments = [
            (Mock(start=0.0, end=2.0), None, 'SPEAKER_00'),
            (Mock(start=2.5, end=4.0), None, 'SPEAKER_01'),
            (Mock(start=4.2, end=5.0), None, 'SPEAKER_00')
        ]
        mock_diarization_result.itertracks.return_value = mock_segments
        mock_diarization_pipeline.return_value = mock_diarization_result

        # Create test clip
        test_clip = self.clips_dir / "test_clip.wav"
        test_clip.touch()

        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )
        pipeline.diarization_pipeline = mock_diarization_pipeline

        result = pipeline.perform_speaker_diarization(test_clip)

        self.assertIn('segments', result)
        self.assertIn('num_speakers', result)
        self.assertEqual(result['num_speakers'], 2)  # SPEAKER_00 and SPEAKER_01

        # Check that diarization was called
        mock_diarization_pipeline.assert_called_once()

    @patch('transcription_pipeline.librosa.load')
    @patch('transcription_pipeline.librosa.yin')
    def test_issue_89_identify_educator_vs_child(self, mock_yin, mock_librosa_load):
        """Issue #89: Test educator vs child speaker identification"""
        # Mock audio loading
        mock_audio_data = np.random.rand(16000 * 5)  # 5 seconds
        mock_librosa_load.return_value = (mock_audio_data, 16000)

        # Mock F0 extraction - simulate adult and child voices
        # Adult voice: lower pitch around 120 Hz
        mock_adult_f0 = np.full(100, 120.0)
        # Child voice: higher pitch around 250 Hz
        mock_child_f0 = np.full(100, 250.0)

        def mock_yin_side_effect(*args, **kwargs):
            # Return different F0 values based on audio segment
            segment_start = kwargs.get('segment_start', 0)
            if segment_start < 2.5:  # First segment - adult
                return mock_adult_f0
            else:  # Second segment - child
                return mock_child_f0

        mock_yin.side_effect = [mock_adult_f0, mock_child_f0]

        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        # Test speaker segments
        speaker_segments = [
            {'speaker_id': 'SPEAKER_00', 'start': 0.0, 'end': 2.0, 'duration': 2.0},
            {'speaker_id': 'SPEAKER_01', 'start': 2.5, 'end': 4.0, 'duration': 1.5}
        ]

        test_clip = self.clips_dir / "test_clip.wav"
        test_clip.touch()

        result = pipeline.identify_educator_vs_child(speaker_segments, test_clip)

        self.assertEqual(len(result), 2)

        # First speaker should be classified as educator (lower pitch)
        speaker1 = result[0]
        self.assertEqual(speaker1.age_group, 'educator')
        self.assertIn('educator', speaker1.speaker_id)

        # Second speaker should be classified as child (higher pitch)
        speaker2 = result[1]
        self.assertEqual(speaker2.age_group, 'child')
        self.assertIn('child', speaker2.speaker_id)

    def test_issue_89_align_transcript_speakers(self):
        """Issue #89: Test alignment of transcript segments with speaker information"""
        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        # Mock transcript result
        transcript_result = {
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'What do you see?',
                    'avg_logprob': -0.1,
                    'words': [
                        {'word': 'What', 'start': 0.0, 'end': 0.3, 'confidence': 0.95},
                        {'word': 'do', 'start': 0.3, 'end': 0.5, 'confidence': 0.92},
                        {'word': 'you', 'start': 0.5, 'end': 0.7, 'confidence': 0.98},
                        {'word': 'see', 'start': 0.7, 'end': 1.0, 'confidence': 0.96}
                    ]
                },
                {
                    'start': 2.5,
                    'end': 4.0,
                    'text': 'I see a dog!',
                    'avg_logprob': -0.2,
                    'words': [
                        {'word': 'I', 'start': 2.5, 'end': 2.6, 'confidence': 0.94},
                        {'word': 'see', 'start': 2.6, 'end': 2.9, 'confidence': 0.97},
                        {'word': 'a', 'start': 2.9, 'end': 3.0, 'confidence': 0.93},
                        {'word': 'dog', 'start': 3.0, 'end': 3.3, 'confidence': 0.98}
                    ]
                }
            ]
        }

        # Mock speaker segments
        speaker_segments = [
            SpeakerSegment(
                speaker_id='educator_SPEAKER_00',
                start=0.0,
                end=2.2,
                confidence=0.85,
                age_group='educator'
            ),
            SpeakerSegment(
                speaker_id='child_SPEAKER_01',
                start=2.3,
                end=4.5,
                confidence=0.80,
                age_group='child'
            )
        ]

        result = pipeline.align_transcript_speakers(transcript_result, speaker_segments)

        self.assertEqual(len(result), 2)

        # Check first segment (educator)
        segment1 = result[0]
        self.assertEqual(segment1.text, 'What do you see?')
        self.assertEqual(segment1.speaker, 'educator_SPEAKER_00')
        self.assertEqual(len(segment1.words), 4)

        # Check second segment (child)
        segment2 = result[1]
        self.assertEqual(segment2.text, 'I see a dog!')
        self.assertEqual(segment2.speaker, 'child_SPEAKER_01')
        self.assertEqual(len(segment2.words), 4)

    def test_issue_89_extract_quality_annotation(self):
        """Issue #89: Test extraction of quality annotations from CSV data"""
        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        clip_info = {
            'question_description': 'OEQ and teacher gives opportunity to respond',
            'age_group': '4-5',
            'video_description': 'Educational video about animals',
            'original_timestamp_str': '0:35',
            'asset_number': '12345'
        }

        result = pipeline.extract_quality_annotation(clip_info)

        self.assertEqual(result['question_type'], 'OEQ')
        self.assertEqual(result['description'], 'OEQ and teacher gives opportunity to respond')
        self.assertEqual(result['age_group'], '4-5')
        self.assertEqual(result['video_description'], 'Educational video about animals')
        self.assertEqual(result['original_timestamp'], '0:35')
        self.assertEqual(result['asset_number'], '12345')

    def test_issue_89_transcript_word_dataclass(self):
        """Issue #89: Test TranscriptWord dataclass functionality"""
        word = TranscriptWord(
            word="hello",
            start=1.0,
            end=1.5,
            confidence=0.95,
            speaker="educator_SPEAKER_00"
        )

        self.assertEqual(word.word, "hello")
        self.assertEqual(word.start, 1.0)
        self.assertEqual(word.end, 1.5)
        self.assertEqual(word.confidence, 0.95)
        self.assertEqual(word.speaker, "educator_SPEAKER_00")

    def test_issue_89_speaker_segment_dataclass(self):
        """Issue #89: Test SpeakerSegment dataclass functionality"""
        segment = SpeakerSegment(
            speaker_id="educator_SPEAKER_00",
            start=0.0,
            end=3.0,
            confidence=0.85,
            age_group="educator"
        )

        self.assertEqual(segment.speaker_id, "educator_SPEAKER_00")
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 3.0)
        self.assertEqual(segment.confidence, 0.85)
        self.assertEqual(segment.age_group, "educator")

    def test_issue_89_clip_transcript_dataclass(self):
        """Issue #89: Test ClipTranscript dataclass with complete structure"""
        # Create test data
        words = [
            TranscriptWord("What", 0.0, 0.3, 0.95, "educator_SPEAKER_00"),
            TranscriptWord("do", 0.3, 0.5, 0.92, "educator_SPEAKER_00")
        ]

        segment = TranscriptSegment(
            text="What do you think?",
            start=0.0,
            end=2.0,
            speaker="educator_SPEAKER_00",
            words=words,
            confidence=0.88
        )

        transcript = ClipTranscript(
            clip_id="12345_q1",
            clip_filename="test_12345_q1.wav",
            original_video="Test Educational Video",
            timestamp_range="30.5-35.5",
            duration=5.0,
            transcript_segments=[segment],
            speakers={"educator_SPEAKER_00": ["What do you think?"]},
            quality_annotation={"question_type": "OEQ"},
            processing_metadata={"whisper_model": "medium"}
        )

        # Test all fields
        self.assertEqual(transcript.clip_id, "12345_q1")
        self.assertEqual(transcript.clip_filename, "test_12345_q1.wav")
        self.assertEqual(transcript.original_video, "Test Educational Video")
        self.assertEqual(transcript.duration, 5.0)
        self.assertEqual(len(transcript.transcript_segments), 1)
        self.assertIn("educator_SPEAKER_00", transcript.speakers)

    def test_issue_89_transcription_encoder_handles_numpy_types(self):
        """Issue #89: Test TranscriptionEncoder handles numpy and torch types"""
        encoder = TranscriptionEncoder()

        # Test numpy types
        self.assertEqual(encoder.default(np.int32(42)), 42)
        self.assertEqual(encoder.default(np.float32(3.14)), 3.140000104904175)
        self.assertEqual(encoder.default(np.bool_(True)), True)
        self.assertEqual(encoder.default(np.array([1, 2, 3])), [1, 2, 3])

    def test_issue_89_json_serialization_of_transcript(self):
        """Issue #89: Test JSON serialization of complete transcript structure"""
        # Create complete transcript structure
        word = TranscriptWord("hello", 0.0, 0.5, 0.95, "educator")
        segment = TranscriptSegment("hello world", 0.0, 1.0, "educator", [word], 0.9)

        transcript = ClipTranscript(
            clip_id="test_clip",
            clip_filename="test.wav",
            original_video="Test Video",
            timestamp_range="0-5",
            duration=5.0,
            transcript_segments=[segment],
            speakers={"educator": ["hello world"]},
            quality_annotation={"type": "OEQ"},
            processing_metadata={"model": "whisper"}
        )

        # Convert to dict and serialize
        transcript_dict = asdict(transcript)
        json_str = json.dumps(transcript_dict, cls=TranscriptionEncoder)

        # Should not raise exception
        parsed_data = json.loads(json_str)

        self.assertEqual(parsed_data['clip_id'], "test_clip")
        self.assertEqual(parsed_data['duration'], 5.0)
        self.assertEqual(len(parsed_data['transcript_segments']), 1)

    def test_issue_89_save_transcript_functionality(self):
        """Issue #89: Test transcript saving to JSON file"""
        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        # Create test transcript
        word = TranscriptWord("test", 0.0, 0.5, 0.95, "educator")
        segment = TranscriptSegment("test word", 0.0, 0.5, "educator", [word], 0.9)

        transcript = ClipTranscript(
            clip_id="12345_q1",
            clip_filename="test_12345_q1.wav",
            original_video="Test Video",
            timestamp_range="0-5",
            duration=5.0,
            transcript_segments=[segment],
            speakers={"educator": ["test word"]},
            quality_annotation={"question_type": "OEQ"},
            processing_metadata={"whisper_model": "medium"}
        )

        # Save transcript
        pipeline.save_transcript(transcript)

        # Check file was created
        expected_file = pipeline.transcripts_dir / "12345_q1_transcript.json"
        self.assertTrue(expected_file.exists())

        # Verify content
        with open(expected_file, 'r') as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data['clip_id'], "12345_q1")
        self.assertEqual(saved_data['duration'], 5.0)
        self.assertIn('transcript_segments', saved_data)

    @patch.object(TranscriptionPipeline, 'process_single_clip')
    @patch.object(TranscriptionPipeline, 'initialize_models')
    def test_issue_89_processing_statistics_tracking(self, mock_init_models, mock_process_clip):
        """Issue #89: Test processing statistics are properly tracked"""
        # Mock successful initialization
        mock_init_models.return_value = True

        # Mock successful clip processing
        mock_word = TranscriptWord("test", 0.0, 0.5, 0.95, "educator")
        mock_segment = TranscriptSegment("test", 0.0, 0.5, "educator", [mock_word], 0.9)
        mock_transcript = ClipTranscript(
            clip_id="test",
            clip_filename="test.wav",
            original_video="Test",
            timestamp_range="0-5",
            duration=5.0,
            transcript_segments=[mock_segment],
            speakers={"educator": ["test"]},
            quality_annotation={},
            processing_metadata={}
        )
        mock_process_clip.return_value = mock_transcript

        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        # Process clips
        import asyncio
        success = asyncio.run(pipeline.process_all_clips())

        # Check statistics
        self.assertTrue(success)
        self.assertEqual(pipeline.stats['clips_processed'], 2)  # 2 clips in metadata
        self.assertEqual(pipeline.stats['transcripts_generated'], 2)
        self.assertGreater(pipeline.stats['total_words'], 0)

    def test_issue_89_error_handling_missing_files(self):
        """Issue #89: Test error handling for missing audio files"""
        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        # Test with nonexistent clip
        clip_info = {
            'clip_filename': 'nonexistent_clip.wav',
            'asset_number': '99999',
            'question_number': 1
        }

        result = pipeline.process_single_clip(clip_info)
        self.assertIsNone(result)

    def test_issue_89_educator_pattern_recognition(self):
        """Issue #89: Test educator speech pattern recognition"""
        pipeline = TranscriptionPipeline(
            clips_dir=str(self.clips_dir),
            metadata_file=str(self.metadata_file),
            output_dir=str(self.output_dir)
        )

        # Test educator patterns
        self.assertIn('what', pipeline.educator_patterns['question_words'])
        self.assertIn('let\'s', pipeline.educator_patterns['instruction_phrases'])
        self.assertIn('good', pipeline.educator_patterns['encouragement'])
        self.assertIn('tell me', pipeline.educator_patterns['response_prompts'])

        # These patterns can be used for enhanced speaker identification
        self.assertEqual(len(pipeline.educator_patterns), 4)


if __name__ == '__main__':
    # Add torch import for tests that need it
    try:
        import torch
        globals()['torch'] = torch
    except ImportError:
        # Create a mock torch for tests that need it
        torch = Mock()
        torch.randn = Mock(return_value=np.random.randn(1, 80000))
        globals()['torch'] = torch

    # Run tests with verbose output
    unittest.main(verbosity=2)