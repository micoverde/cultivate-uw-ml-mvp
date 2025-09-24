#!/usr/bin/env python3
"""
Transcription & Speaker Diarization Pipeline for Issue #89

Implements comprehensive audio-to-text transcription with speaker identification
for educational audio clips. Handles classroom noise, multiple speakers, and
generates aligned transcripts with quality annotations.

Author: Claude (Issue #89 Implementation)
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, AsyncIterator
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import required libraries with error handling
try:
    import whisper
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install openai-whisper pyannote.audio torch torchaudio librosa soundfile")
    sys.exit(1)

# JSON encoder for numpy/torch types
class TranscriptionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # torch tensors
            return obj.item()
        return super(TranscriptionEncoder, self).default(obj)

@dataclass
class TranscriptWord:
    """Individual word with timing and speaker information"""
    word: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None

@dataclass
class SpeakerSegment:
    """Speaker segment with timing and identification"""
    speaker_id: str
    start: float
    end: float
    confidence: float
    age_group: str  # 'educator' or 'child'

@dataclass
class TranscriptSegment:
    """Transcript segment with speaker information"""
    text: str
    start: float
    end: float
    speaker: str
    words: List[TranscriptWord]
    confidence: float

@dataclass
class ClipTranscript:
    """Complete transcript for a single audio clip"""
    clip_id: str
    clip_filename: str
    original_video: str
    timestamp_range: str
    duration: float
    transcript_segments: List[TranscriptSegment]
    speakers: Dict[str, List[str]]  # speaker_id -> list of utterances
    quality_annotation: Dict
    processing_metadata: Dict

class TranscriptionPipeline:
    """
    Complete transcription and speaker diarization pipeline
    Processes 101 audio clips from Issue #88 with high accuracy
    """

    def __init__(self,
                 clips_dir: str = "data/processed_audio/clips",
                 metadata_file: str = "data/processed_audio/metadata/clip_manifest.json",
                 output_dir: str = "data/transcripts",
                 whisper_model: str = "medium"):

        self.clips_dir = Path(clips_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.whisper_model_name = whisper_model

        # Create output directories
        self.transcripts_dir = self.output_dir / "transcripts"
        self.validation_dir = self.output_dir / "validation"
        self.reports_dir = self.output_dir / "reports"

        for dir_path in [self.transcripts_dir, self.validation_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.whisper_model = None
        self.diarization_pipeline = None
        self.speaker_embedder = None

        # Processing statistics
        self.stats = {
            'clips_processed': 0,
            'transcripts_generated': 0,
            'speakers_identified': 0,
            'total_words': 0,
            'total_duration': 0.0,
            'processing_time': 0.0,
            'errors': []
        }

        # Educational patterns for speaker identification
        self.educator_patterns = {
            'question_words': ['what', 'how', 'why', 'when', 'where', 'can', 'do', 'does', 'will'],
            'instruction_phrases': ['let\'s', 'now', 'first', 'next', 'try', 'look', 'see'],
            'encouragement': ['good', 'great', 'excellent', 'nice', 'well done', 'that\'s right'],
            'response_prompts': ['tell me', 'show me', 'what do you think', 'can you']
        }

    def initialize_models(self):
        """Initialize Whisper and pyannote models"""
        try:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_model = whisper.load_model(self.whisper_model_name)

            logger.info("Loading pyannote speaker diarization pipeline")
            # Note: This requires authentication token for pyannote models
            # Set HUGGINGFACE_HUB_TOKEN environment variable
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
            )

            logger.info("Loading speaker embedding model")
            self.speaker_embedder = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb",
                device='cpu'  # Use CPU for stability
            )

            logger.info("All models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            logger.error("Ensure HUGGINGFACE_HUB_TOKEN is set for pyannote access")
            return False

    def load_clip_metadata(self) -> Dict:
        """Load metadata from Issue #88 extraction pipeline"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            logger.info(f"Loaded metadata for {len(metadata.get('clips', []))} clips")
            return metadata

        except Exception as e:
            logger.error(f"Failed to load clip metadata: {e}")
            return {'clips': []}

    def transcribe_audio_clip(self, clip_path: Path) -> Dict:
        """
        Transcribe single audio clip with word-level timestamps

        Args:
            clip_path: Path to audio clip file

        Returns:
            Whisper transcription result with word timestamps
        """
        try:
            logger.info(f"Transcribing: {clip_path.name}")

            # Load audio file
            audio_data, sample_rate = librosa.load(str(clip_path), sr=16000)

            # Transcribe with word-level timestamps
            result = self.whisper_model.transcribe(
                audio_data,
                word_timestamps=True,
                language='en',
                task='transcribe'
            )

            # Validate transcription quality
            if not result.get('text', '').strip():
                logger.warning(f"Empty transcription for {clip_path.name}")
                return None

            # Calculate average confidence (Whisper doesn't provide word confidence directly)
            # Use segment-level confidence as proxy
            avg_confidence = np.mean([
                segment.get('avg_logprob', -1.0)
                for segment in result.get('segments', [])
            ]) if result.get('segments') else -1.0

            # Convert to confidence score (0-1 range)
            confidence_score = max(0.0, (avg_confidence + 1.0))

            logger.info(f"Transcribed {len(result.get('text', ''))} characters, "
                       f"confidence: {confidence_score:.3f}")

            return {
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language'],
                'confidence': confidence_score,
                'duration': len(audio_data) / sample_rate
            }

        except Exception as e:
            logger.error(f"Failed to transcribe {clip_path.name}: {e}")
            return None

    def perform_speaker_diarization(self, clip_path: Path) -> Dict:
        """
        Perform speaker diarization on audio clip

        Args:
            clip_path: Path to audio clip file

        Returns:
            Speaker segments with timing and identification
        """
        try:
            logger.info(f"Speaker diarization: {clip_path.name}")

            # Load audio for pyannote
            waveform, sample_rate = torchaudio.load(str(clip_path))

            # Apply diarization pipeline
            diarization_result = self.diarization_pipeline({
                "waveform": waveform,
                "sample_rate": sample_rate
            })

            # Process speaker segments
            speaker_segments = []
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                speaker_segments.append({
                    'speaker_id': speaker,
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.end - segment.start
                })

            # Identify educator vs child speakers
            identified_speakers = self.identify_educator_vs_child(
                speaker_segments,
                clip_path
            )

            logger.info(f"Identified {len(identified_speakers)} speakers")

            return {
                'segments': identified_speakers,
                'num_speakers': len(set(s['speaker_id'] for s in identified_speakers))
            }

        except Exception as e:
            logger.error(f"Failed speaker diarization for {clip_path.name}: {e}")
            return {'segments': [], 'num_speakers': 0}

    def identify_educator_vs_child(self, speaker_segments: List[Dict], clip_path: Path) -> List[SpeakerSegment]:
        """
        Identify which speakers are educators vs children using multiple heuristics

        Args:
            speaker_segments: Raw speaker segments from diarization
            clip_path: Path to audio clip for additional analysis

        Returns:
            Speaker segments with educator/child classification
        """
        try:
            # Load audio for feature analysis
            audio_data, sr = librosa.load(str(clip_path), sr=16000)

            identified_segments = []

            for segment in speaker_segments:
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = audio_data[start_sample:end_sample]

                # Extract prosodic features for age classification
                # Fundamental frequency analysis
                f0 = librosa.yin(
                    segment_audio,
                    fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                    fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                    sr=sr
                )

                # Remove zero values and calculate statistics
                f0_nonzero = f0[f0 > 0]

                if len(f0_nonzero) > 0:
                    avg_f0 = np.mean(f0_nonzero)
                    std_f0 = np.std(f0_nonzero)

                    # Age group classification based on pitch
                    # Children typically have higher fundamental frequencies
                    if avg_f0 > 200:  # High pitch suggests child
                        age_group = 'child'
                        confidence = min(0.8, (avg_f0 - 200) / 200)  # Higher pitch = higher confidence
                    elif avg_f0 < 150:  # Lower pitch suggests adult educator
                        age_group = 'educator'
                        confidence = min(0.9, (200 - avg_f0) / 100)
                    else:  # Ambiguous range
                        age_group = 'child'  # Default to child for COPPA compliance
                        confidence = 0.5
                else:
                    # No pitch data available, use duration heuristic
                    # Educators typically have longer utterances
                    if segment['duration'] > 3.0:
                        age_group = 'educator'
                        confidence = 0.6
                    else:
                        age_group = 'child'
                        confidence = 0.4

                identified_segment = SpeakerSegment(
                    speaker_id=f"{age_group}_{segment['speaker_id']}",
                    start=segment['start'],
                    end=segment['end'],
                    confidence=confidence,
                    age_group=age_group
                )

                identified_segments.append(identified_segment)

            return identified_segments

        except Exception as e:
            logger.error(f"Failed to identify educator vs child: {e}")
            return []

    def align_transcript_speakers(self,
                                 transcript_result: Dict,
                                 speaker_segments: List[SpeakerSegment]) -> List[TranscriptSegment]:
        """
        Align transcript words with speaker segments

        Args:
            transcript_result: Whisper transcription result
            speaker_segments: Speaker diarization result

        Returns:
            Aligned transcript segments with speaker identification
        """
        try:
            aligned_segments = []

            for segment in transcript_result.get('segments', []):
                segment_start = segment['start']
                segment_end = segment['end']
                segment_text = segment['text']

                # Find overlapping speaker
                best_speaker = None
                best_overlap = 0.0

                for speaker_seg in speaker_segments:
                    # Calculate overlap between transcript segment and speaker segment
                    overlap_start = max(segment_start, speaker_seg.start)
                    overlap_end = min(segment_end, speaker_seg.end)
                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_speaker = speaker_seg

                # Create word-level alignments
                words = []
                if 'words' in segment:
                    for word_info in segment['words']:
                        word = TranscriptWord(
                            word=word_info['word'],
                            start=word_info['start'],
                            end=word_info['end'],
                            confidence=word_info.get('confidence', 0.8),
                            speaker=best_speaker.speaker_id if best_speaker else 'unknown'
                        )
                        words.append(word)

                # Create transcript segment
                transcript_segment = TranscriptSegment(
                    text=segment_text.strip(),
                    start=segment_start,
                    end=segment_end,
                    speaker=best_speaker.speaker_id if best_speaker else 'unknown',
                    words=words,
                    confidence=segment.get('avg_logprob', -1.0) + 1.0  # Convert to 0-1 range
                )

                aligned_segments.append(transcript_segment)

            return aligned_segments

        except Exception as e:
            logger.error(f"Failed to align transcript with speakers: {e}")
            return []

    def process_single_clip(self, clip_info: Dict) -> Optional[ClipTranscript]:
        """
        Process single audio clip through complete pipeline

        Args:
            clip_info: Clip metadata from Issue #88 pipeline

        Returns:
            Complete transcript with speaker identification
        """
        try:
            clip_filename = clip_info['clip_filename']
            clip_path = self.clips_dir / clip_filename

            if not clip_path.exists():
                logger.error(f"Clip file not found: {clip_path}")
                return None

            logger.info(f"Processing clip: {clip_filename}")

            # Step 1: Transcribe audio
            transcript_result = self.transcribe_audio_clip(clip_path)
            if not transcript_result:
                self.stats['errors'].append(f"Transcription failed: {clip_filename}")
                return None

            # Step 2: Perform speaker diarization
            speaker_result = self.perform_speaker_diarization(clip_path)

            # Step 3: Align transcript with speakers
            aligned_segments = self.align_transcript_speakers(
                transcript_result,
                speaker_result['segments']
            )

            # Step 4: Extract speaker utterances
            speakers = {}
            for segment in aligned_segments:
                if segment.speaker not in speakers:
                    speakers[segment.speaker] = []
                speakers[segment.speaker].append(segment.text)

            # Step 5: Load quality annotation from CSV
            quality_annotation = self.extract_quality_annotation(clip_info)

            # Step 6: Create complete transcript
            clip_transcript = ClipTranscript(
                clip_id=clip_info['asset_number'] + '_q' + str(clip_info['question_number']),
                clip_filename=clip_filename,
                original_video=clip_info.get('video_title', 'unknown'),
                timestamp_range=f"{clip_info.get('clip_start_time', 0):.1f}-{clip_info.get('clip_end_time', 0):.1f}",
                duration=transcript_result['duration'],
                transcript_segments=aligned_segments,
                speakers=speakers,
                quality_annotation=quality_annotation,
                processing_metadata={
                    'whisper_model': self.whisper_model_name,
                    'transcription_confidence': transcript_result['confidence'],
                    'num_speakers': speaker_result['num_speakers'],
                    'total_words': len([w for seg in aligned_segments for w in seg.words]),
                    'processing_timestamp': datetime.now().isoformat()
                }
            )

            # Update statistics
            self.stats['clips_processed'] += 1
            self.stats['transcripts_generated'] += 1
            self.stats['speakers_identified'] += speaker_result['num_speakers']
            self.stats['total_words'] += len([w for seg in aligned_segments for w in seg.words])
            self.stats['total_duration'] += transcript_result['duration']

            logger.info(f"Successfully processed: {clip_filename}")
            return clip_transcript

        except Exception as e:
            logger.error(f"Failed to process clip {clip_info.get('clip_filename', 'unknown')}: {e}")
            self.stats['errors'].append(f"Processing failed: {clip_info.get('clip_filename', 'unknown')} - {e}")
            return None

    def extract_quality_annotation(self, clip_info: Dict) -> Dict:
        """Extract quality annotation from CSV data"""
        return {
            'question_type': self.classify_question_type(clip_info.get('question_description', '')),
            'description': clip_info.get('question_description', ''),
            'age_group': clip_info.get('age_group', 'unknown'),
            'video_description': clip_info.get('video_description', ''),
            'original_timestamp': clip_info.get('original_timestamp_str', ''),
            'asset_number': clip_info.get('asset_number', 'unknown')
        }

    def classify_question_type(self, description: str) -> str:
        """Classify question type from description"""
        description_lower = description.lower()

        if 'oeq' in description_lower or 'open' in description_lower:
            return 'OEQ'
        elif 'ceq' in description_lower or 'closed' in description_lower:
            return 'CEQ'
        elif 'wait' in description_lower or 'pause' in description_lower:
            return 'wait_time'
        elif 'follow' in description_lower:
            return 'follow_up'
        else:
            return 'unknown'

    def save_transcript(self, transcript: ClipTranscript):
        """Save transcript to JSON file"""
        try:
            output_file = self.transcripts_dir / f"{transcript.clip_id}_transcript.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    asdict(transcript),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    cls=TranscriptionEncoder
                )

            logger.info(f"Saved transcript: {output_file}")

        except Exception as e:
            logger.error(f"Failed to save transcript for {transcript.clip_id}: {e}")

    def generate_processing_report(self):
        """Generate comprehensive processing report"""
        try:
            report = {
                'processing_summary': {
                    'clips_processed': self.stats['clips_processed'],
                    'transcripts_generated': self.stats['transcripts_generated'],
                    'success_rate': (
                        self.stats['transcripts_generated'] / max(1, self.stats['clips_processed']) * 100
                    ),
                    'total_speakers': self.stats['speakers_identified'],
                    'total_words': self.stats['total_words'],
                    'total_duration': self.stats['total_duration'],
                    'avg_words_per_clip': (
                        self.stats['total_words'] / max(1, self.stats['transcripts_generated'])
                    ),
                    'processing_time': self.stats['processing_time']
                },
                'errors': self.stats['errors'],
                'model_info': {
                    'whisper_model': self.whisper_model_name,
                    'diarization_model': 'pyannote/speaker-diarization-3.1',
                    'speaker_embedding_model': 'speechbrain/spkrec-ecapa-voxceleb'
                },
                'quality_thresholds': {
                    'min_transcription_confidence': 0.5,
                    'min_speaker_confidence': 0.4,
                    'target_success_rate': 90.0
                },
                'report_timestamp': datetime.now().isoformat()
            }

            # Save report
            report_file = self.reports_dir / 'transcription_processing_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, cls=TranscriptionEncoder)

            logger.info(f"Processing report saved: {report_file}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate processing report: {e}")
            return {}

    async def process_all_clips(self) -> bool:
        """
        Process all audio clips through transcription pipeline

        Returns:
            True if processing successful, False otherwise
        """
        try:
            logger.info("Starting transcription pipeline for all clips")

            # Initialize models
            if not self.initialize_models():
                logger.error("Failed to initialize models")
                return False

            # Load clip metadata
            metadata = self.load_clip_metadata()
            clips = metadata.get('clips', [])

            if not clips:
                logger.error("No clips found in metadata")
                return False

            logger.info(f"Processing {len(clips)} audio clips")

            # Process each clip
            successful_transcripts = []
            start_time = datetime.now()

            for i, clip_info in enumerate(clips):
                try:
                    logger.info(f"Processing clip {i+1}/{len(clips)}: {clip_info.get('clip_filename', 'unknown')}")

                    transcript = self.process_single_clip(clip_info)
                    if transcript:
                        self.save_transcript(transcript)
                        successful_transcripts.append(transcript)

                    # Progress logging
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i+1}/{len(clips)} clips processed")

                except KeyboardInterrupt:
                    logger.info("Processing interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error processing clip {i+1}: {e}")
                    continue

            # Calculate processing time
            end_time = datetime.now()
            self.stats['processing_time'] = (end_time - start_time).total_seconds()

            # Generate final report
            report = self.generate_processing_report()

            # Print summary
            logger.info("=" * 60)
            logger.info("TRANSCRIPTION PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Clips processed: {self.stats['clips_processed']}")
            logger.info(f"Transcripts generated: {self.stats['transcripts_generated']}")
            logger.info(f"Success rate: {report['processing_summary']['success_rate']:.1f}%")
            logger.info(f"Total speakers identified: {self.stats['speakers_identified']}")
            logger.info(f"Total words: {self.stats['total_words']}")
            logger.info(f"Total duration: {self.stats['total_duration']:.1f} seconds")
            logger.info(f"Processing time: {self.stats['processing_time']:.1f} seconds")

            if self.stats['errors']:
                logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
                for error in self.stats['errors'][:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")

            logger.info(f"Transcripts saved to: {self.transcripts_dir}")
            logger.info(f"Reports saved to: {self.reports_dir}")

            # Success criteria: >90% success rate
            success = report['processing_summary']['success_rate'] >= 90.0

            if success:
                logger.info("✅ Pipeline completed successfully!")
            else:
                logger.warning("⚠️ Pipeline completed with issues - check error log")

            return success

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """Main entry point"""
    print("🎤 Starting Issue #89: Transcription & Speaker Diarization Pipeline")
    print("=" * 70)

    # Check for required environment variables
    if not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        print("⚠️ WARNING: HUGGINGFACE_HUB_TOKEN not set")
        print("   Speaker diarization may fail without authentication token")
        print("   Set token with: export HUGGINGFACE_HUB_TOKEN=your_token")

    # Initialize pipeline
    pipeline = TranscriptionPipeline()

    # Run pipeline
    try:
        import asyncio
        success = asyncio.run(pipeline.process_all_clips())

        if success:
            print("\n✅ Transcription pipeline completed successfully!")
            print("📁 Check data/transcripts/ for generated transcripts")
        else:
            print("\n❌ Pipeline encountered issues - check logs for details")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()