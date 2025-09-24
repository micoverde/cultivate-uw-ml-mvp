#!/usr/bin/env python3
"""
Video-to-Audio Extraction Pipeline for STORY 1.1 (Issue #88)

Extracts high-quality audio segments from 26 annotated video files based on CSV timestamps.
Creates 119 individual audio clips for ML training.

Requirements:
- moviepy: Video-to-audio extraction
- librosa: Audio processing and validation
- pandas: CSV annotation parsing
- pydub: Audio format conversion

Author: Claude (Issue #88 Implementation)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import re

# JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Import required libraries with error handling
try:
    from moviepy import VideoFileClip
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install moviepy librosa soundfile pydub")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoToAudioExtractor:
    """
    Main class for extracting audio segments from expert-annotated videos
    """

    def __init__(self,
                 video_dir: str = "/home/warrenjo/src/tmp2/secure data",
                 csv_file: str = "/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv",
                 output_dir: str = "/home/warrenjo/src/tmp2/cultivate-uw-ml-mvp/data/processed_audio"):

        self.video_dir = Path(video_dir)
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)

        # Audio processing parameters
        self.target_sample_rate = 16000  # 16kHz for ML training
        self.clip_duration = 5.0  # 5 seconds total
        self.pre_question_buffer = 2.0  # 2 seconds before question
        self.post_question_buffer = 3.0  # 3 seconds after question

        # Create output directories
        self.clips_dir = self.output_dir / "clips"
        self.metadata_dir = self.output_dir / "metadata"
        self.validation_dir = self.output_dir / "validation"

        for dir_path in [self.clips_dir, self.metadata_dir, self.validation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Processing statistics
        self.stats = {
            'videos_processed': 0,
            'clips_extracted': 0,
            'failed_extractions': 0,
            'total_duration_extracted': 0.0,
            'errors': []
        }

        logger.info(f"VideoToAudioExtractor initialized")
        logger.info(f"Video directory: {self.video_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def parse_timestamp(self, timestamp_str: str) -> float:
        """
        Parse timestamp string (e.g., "0:37", "1:13") to seconds

        Args:
            timestamp_str: Timestamp in format "M:SS" or "MM:SS"

        Returns:
            float: Timestamp in seconds
        """
        try:
            if pd.isna(timestamp_str) or timestamp_str == 'na':
                return None

            # Handle format like "0:37" or "1:13"
            parts = str(timestamp_str).strip().split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            else:
                logger.warning(f"Invalid timestamp format: {timestamp_str}")
                return None

        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
            return None

    def find_video_file(self, video_title: str) -> Optional[Path]:
        """
        Find video file matching the title, handling filename variations

        Args:
            video_title: Video title from CSV

        Returns:
            Path to video file if found, None otherwise
        """
        # Direct match first
        direct_path = self.video_dir / video_title
        if direct_path.exists():
            return direct_path

        # Try variations - remove spaces, handle special characters
        video_title_clean = video_title.replace(' ', '_').replace('&#39;', "'")
        variations = [
            video_title,
            video_title_clean,
            video_title.replace('.mp4', '.MP4'),
            video_title.replace('.mp4', '.mov'),
            video_title.replace('.mp4', '.MOV'),
            video_title_clean.replace('.mp4', '.MP4'),
            video_title_clean.replace('.mp4', '.mov'),
            video_title_clean.replace('.mp4', '.MOV')
        ]

        for variation in variations:
            file_path = self.video_dir / variation
            if file_path.exists():
                logger.info(f"Found video file: {file_path}")
                return file_path

        # Last resort: search for similar names
        for video_file in self.video_dir.glob("*.mp4"):
            if video_title.split('.')[0].lower() in video_file.name.lower():
                logger.info(f"Found similar video file: {video_file}")
                return video_file

        for video_file in self.video_dir.glob("*.MP4"):
            if video_title.split('.')[0].lower() in video_file.name.lower():
                logger.info(f"Found similar video file: {video_file}")
                return video_file

        for video_file in self.video_dir.glob("*.mov"):
            if video_title.split('.')[0].lower() in video_file.name.lower():
                logger.info(f"Found similar video file: {video_file}")
                return video_file

        for video_file in self.video_dir.glob("*.MOV"):
            if video_title.split('.')[0].lower() in video_file.name.lower():
                logger.info(f"Found similar video file: {video_file}")
                return video_file

        logger.error(f"Video file not found: {video_title}")
        return None

    def extract_audio_from_video(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Extract full audio track from video file using librosa method (more robust)

        Args:
            video_path: Path to video file

        Returns:
            Audio array at target sample rate, or None if failed
        """
        try:
            logger.info(f"Extracting audio from: {video_path.name}")

            # Create temporary audio file
            temp_audio = self.output_dir / "temp_audio.wav"
            temp_audio.parent.mkdir(parents=True, exist_ok=True)

            # Extract audio using moviepy (just for conversion to WAV)
            with VideoFileClip(str(video_path)) as video:
                if video.audio is None:
                    logger.error(f"No audio track found in {video_path.name}")
                    return None

                # Write temporary audio file
                video.audio.write_audiofile(str(temp_audio), logger=None)

            # Load audio using librosa (much more robust)
            audio_array, sample_rate = librosa.load(str(temp_audio), sr=self.target_sample_rate)

            # Clean up temporary file
            if temp_audio.exists():
                temp_audio.unlink()

            logger.info(f"Audio extracted: {len(audio_array)/self.target_sample_rate:.2f} seconds")
            return audio_array

        except Exception as e:
            logger.error(f"Failed to extract audio from {video_path.name}: {e}")
            # Clean up temporary file on error
            temp_audio = self.output_dir / "temp_audio.wav"
            if temp_audio.exists():
                temp_audio.unlink()
            return None

    def extract_audio_clip(self,
                          full_audio: np.ndarray,
                          question_timestamp: float,
                          asset_number: str,
                          question_number: int) -> Optional[Dict]:
        """
        Extract 5-second audio clip centered on question timestamp

        Args:
            full_audio: Full audio array
            question_timestamp: Question timestamp in seconds
            asset_number: Video asset number for naming
            question_number: Question number (1-8)

        Returns:
            Dictionary with clip data and metadata, or None if failed
        """
        try:
            # Calculate clip boundaries
            start_time = max(0, question_timestamp - self.pre_question_buffer)
            end_time = min(len(full_audio) / self.target_sample_rate,
                          question_timestamp + self.post_question_buffer)

            # Convert to sample indices
            start_sample = int(start_time * self.target_sample_rate)
            end_sample = int(end_time * self.target_sample_rate)

            # Extract clip
            audio_clip = full_audio[start_sample:end_sample]

            # Validate clip duration
            actual_duration = len(audio_clip) / self.target_sample_rate
            if actual_duration < 3.0:  # Minimum 3 seconds for meaningful analysis
                logger.warning(f"Clip too short: {actual_duration:.2f}s for asset {asset_number} Q{question_number}")

            # Generate clip filename
            clip_filename = f"{asset_number}_q{question_number}.wav"
            clip_path = self.clips_dir / clip_filename

            # Save audio clip
            sf.write(str(clip_path), audio_clip, self.target_sample_rate)

            # Create metadata
            clip_metadata = {
                'clip_filename': clip_filename,
                'asset_number': asset_number,
                'question_number': question_number,
                'question_timestamp': question_timestamp,
                'clip_start_time': start_time,
                'clip_end_time': end_time,
                'actual_duration': actual_duration,
                'target_duration': self.clip_duration,
                'sample_rate': self.target_sample_rate,
                'file_size_bytes': clip_path.stat().st_size,
                'extraction_timestamp': datetime.now().isoformat()
            }

            logger.info(f"Extracted clip: {clip_filename} ({actual_duration:.2f}s)")
            return clip_metadata

        except Exception as e:
            logger.error(f"Failed to extract clip for asset {asset_number} Q{question_number}: {e}")
            return None

    def validate_audio_clip(self, clip_path: Path) -> Dict:
        """
        Validate audio clip quality and characteristics

        Args:
            clip_path: Path to audio clip

        Returns:
            Validation results dictionary
        """
        try:
            # Load audio for analysis
            audio, sr = librosa.load(str(clip_path), sr=None)

            # Basic quality checks
            validation = {
                'file_exists': clip_path.exists(),
                'file_size_bytes': clip_path.stat().st_size if clip_path.exists() else 0,
                'duration_seconds': len(audio) / sr,
                'sample_rate': sr,
                'is_valid_sample_rate': sr == self.target_sample_rate,
                'max_amplitude': np.max(np.abs(audio)),
                'is_silent': np.max(np.abs(audio)) < 0.001,
                'has_clipping': np.max(np.abs(audio)) > 0.99,
                'rms_energy': np.sqrt(np.mean(audio**2)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
                'validation_timestamp': datetime.now().isoformat()
            }

            # Overall quality assessment
            validation['is_valid_clip'] = (
                validation['file_exists'] and
                validation['duration_seconds'] >= 3.0 and
                validation['is_valid_sample_rate'] and
                not validation['is_silent'] and
                not validation['has_clipping']
            )

            return validation

        except Exception as e:
            logger.error(f"Failed to validate clip {clip_path.name}: {e}")
            return {
                'file_exists': clip_path.exists(),
                'is_valid_clip': False,
                'error': str(e),
                'validation_timestamp': datetime.now().isoformat()
            }

    def process_video_annotations(self, row: pd.Series) -> List[Dict]:
        """
        Process all question annotations for a single video

        Args:
            row: DataFrame row with video annotations

        Returns:
            List of extracted clip metadata
        """
        video_title = row['Video Title']
        asset_number = str(row['Asset #'])

        logger.info(f"Processing video: {video_title} (Asset: {asset_number})")

        # Find video file
        video_path = self.find_video_file(video_title)
        if video_path is None:
            self.stats['errors'].append(f"Video not found: {video_title}")
            return []

        # Extract full audio
        full_audio = self.extract_audio_from_video(video_path)
        if full_audio is None:
            self.stats['errors'].append(f"Audio extraction failed: {video_title}")
            return []

        extracted_clips = []

        # Process each question (Q1-Q8)
        for q_num in range(1, 9):
            # Try both column name variations (with and without trailing space)
            question_col = f'Question {q_num} '  # With trailing space
            if question_col not in row:
                question_col = f'Question {q_num}'  # Without trailing space

            if question_col in row and pd.notna(row[question_col]) and row[question_col] != 'na':
                question_timestamp = self.parse_timestamp(row[question_col])

                if question_timestamp is not None:
                    # Extract audio clip
                    clip_metadata = self.extract_audio_clip(
                        full_audio, question_timestamp, asset_number, q_num
                    )

                    if clip_metadata:
                        # Add annotation data
                        clip_metadata.update({
                            'video_title': video_title,
                            'age_group': row['Age group'],
                            'question_description': row.get(f'Q{q_num} description', ''),
                            'video_description': row.get('Description', ''),
                            'original_timestamp_str': row[question_col]
                        })

                        # Validate clip
                        clip_path = self.clips_dir / clip_metadata['clip_filename']
                        validation_results = self.validate_audio_clip(clip_path)
                        clip_metadata['validation'] = validation_results

                        if validation_results['is_valid_clip']:
                            extracted_clips.append(clip_metadata)
                            self.stats['clips_extracted'] += 1
                            self.stats['total_duration_extracted'] += clip_metadata['actual_duration']
                        else:
                            self.stats['failed_extractions'] += 1
                            logger.warning(f"Invalid clip: {clip_metadata['clip_filename']}")

        self.stats['videos_processed'] += 1
        logger.info(f"Completed processing: {video_title} - {len(extracted_clips)} clips extracted")

        return extracted_clips

    def load_annotations(self) -> pd.DataFrame:
        """
        Load and validate CSV annotations

        Returns:
            DataFrame with annotations
        """
        try:
            logger.info(f"Loading annotations from: {self.csv_file}")

            # Load CSV with proper encoding handling
            df = pd.read_csv(self.csv_file, encoding='utf-8')

            # Clean up any BOM characters
            df.columns = df.columns.str.replace('\ufeff', '')

            logger.info(f"Loaded {len(df)} video annotations")
            logger.info(f"Columns: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            raise

    def save_manifest(self, all_clips_metadata: List[Dict]):
        """
        Save comprehensive manifest of all extracted clips

        Args:
            all_clips_metadata: List of all clip metadata dictionaries
        """
        try:
            manifest = {
                'extraction_summary': {
                    'total_videos': self.stats['videos_processed'],
                    'total_clips_extracted': self.stats['clips_extracted'],
                    'failed_extractions': self.stats['failed_extractions'],
                    'total_duration_seconds': self.stats['total_duration_extracted'],
                    'extraction_timestamp': datetime.now().isoformat(),
                    'audio_parameters': {
                        'sample_rate': self.target_sample_rate,
                        'clip_duration': self.clip_duration,
                        'pre_question_buffer': self.pre_question_buffer,
                        'post_question_buffer': self.post_question_buffer
                    }
                },
                'clips': all_clips_metadata,
                'errors': self.stats['errors']
            }

            # Save manifest
            manifest_path = self.metadata_dir / 'clip_manifest.json'
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            logger.info(f"Manifest saved: {manifest_path}")

            # Save processing log
            log_path = self.metadata_dir / 'processing_log.txt'
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"Video-to-Audio Extraction Log\n")
                f.write(f"==============================\n\n")
                f.write(f"Extraction Date: {datetime.now().isoformat()}\n")
                f.write(f"Total Videos Processed: {self.stats['videos_processed']}\n")
                f.write(f"Total Clips Extracted: {self.stats['clips_extracted']}\n")
                f.write(f"Failed Extractions: {self.stats['failed_extractions']}\n")
                f.write(f"Total Audio Duration: {self.stats['total_duration_extracted']:.2f} seconds\n\n")

                if self.stats['errors']:
                    f.write("Errors Encountered:\n")
                    for error in self.stats['errors']:
                        f.write(f"- {error}\n")

            logger.info(f"Processing log saved: {log_path}")

        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def generate_quality_report(self, all_clips_metadata: List[Dict]):
        """
        Generate comprehensive quality assessment report

        Args:
            all_clips_metadata: List of all clip metadata dictionaries
        """
        try:
            valid_clips = [clip for clip in all_clips_metadata
                          if clip['validation']['is_valid_clip']]

            if not valid_clips:
                logger.warning("No valid clips found for quality report")
                return

            # Calculate quality statistics
            durations = [clip['actual_duration'] for clip in valid_clips]
            rms_energies = [clip['validation']['rms_energy'] for clip in valid_clips]

            quality_stats = {
                'total_clips_analyzed': len(all_clips_metadata),
                'valid_clips': len(valid_clips),
                'invalid_clips': len(all_clips_metadata) - len(valid_clips),
                'success_rate': len(valid_clips) / len(all_clips_metadata) * 100,
                'duration_statistics': {
                    'mean_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'target_duration': self.clip_duration
                },
                'audio_quality_statistics': {
                    'mean_rms_energy': np.mean(rms_energies),
                    'std_rms_energy': np.std(rms_energies),
                    'min_rms_energy': np.min(rms_energies),
                    'max_rms_energy': np.max(rms_energies)
                },
                'age_group_distribution': {},
                'question_type_distribution': {},
                'report_timestamp': datetime.now().isoformat()
            }

            # Age group distribution
            age_groups = [clip['age_group'] for clip in valid_clips]
            for age_group in set(age_groups):
                quality_stats['age_group_distribution'][age_group] = age_groups.count(age_group)

            # Question number distribution
            question_numbers = [clip['question_number'] for clip in valid_clips]
            for q_num in range(1, 9):
                quality_stats['question_type_distribution'][f'Q{q_num}'] = question_numbers.count(q_num)

            # Save quality report
            report_path = self.validation_dir / 'quality_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(quality_stats, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            logger.info(f"Quality report saved: {report_path}")
            logger.info(f"Success rate: {quality_stats['success_rate']:.1f}% ({len(valid_clips)}/{len(all_clips_metadata)} clips)")

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")

    def run_extraction_pipeline(self) -> bool:
        """
        Run the complete video-to-audio extraction pipeline

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting video-to-audio extraction pipeline")

            # Load annotations
            annotations_df = self.load_annotations()

            # Process each video
            all_clips_metadata = []

            for idx, row in annotations_df.iterrows():
                try:
                    clips_metadata = self.process_video_annotations(row)
                    all_clips_metadata.extend(clips_metadata)
                except Exception as e:
                    logger.error(f"Failed to process row {idx}: {e}")
                    self.stats['errors'].append(f"Row {idx} processing failed: {e}")

            # Save manifest and reports
            self.save_manifest(all_clips_metadata)
            self.generate_quality_report(all_clips_metadata)

            # Final summary
            logger.info("=== Extraction Pipeline Complete ===")
            logger.info(f"Videos processed: {self.stats['videos_processed']}")
            logger.info(f"Clips extracted: {self.stats['clips_extracted']}")
            logger.info(f"Failed extractions: {self.stats['failed_extractions']}")
            logger.info(f"Total duration: {self.stats['total_duration_extracted']:.2f} seconds")
            logger.info(f"Output directory: {self.output_dir}")

            # Check if we met the target of 119 clips
            if self.stats['clips_extracted'] >= 119:
                logger.info("✅ SUCCESS: Target of 119 clips achieved!")
                return True
            else:
                logger.warning(f"⚠️  WARNING: Only {self.stats['clips_extracted']}/119 clips extracted")
                return self.stats['clips_extracted'] > 100  # Allow some tolerance

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """
    Main entry point for video-to-audio extraction
    """
    try:
        # Initialize extractor
        extractor = VideoToAudioExtractor()

        # Run extraction pipeline
        success = extractor.run_extraction_pipeline()

        if success:
            print("\n✅ Video-to-audio extraction completed successfully!")
            print(f"📁 Output directory: {extractor.output_dir}")
            print(f"📊 Clips extracted: {extractor.stats['clips_extracted']}")
        else:
            print("\n❌ Extraction pipeline encountered issues")
            print("📝 Check audio_extraction.log for details")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏸️  Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()