"""
Secure data processor for Cultivate Learning educator-child interactions.

This module processes video files from the secure data directory to extract
transcripts and generate training data for the ML model, with focus on:
- Open-ended question detection
- Zone of Proximal Development (ZPD) indicators
- Scaffolding and fading techniques
- CLASS framework alignment

IMPORTANT: This processes private data - never commit outputs to public repo.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import tempfile

logger = logging.getLogger(__name__)

class SecureDataProcessor:
    """Processes secure video data for ML training."""

    def __init__(self, secure_data_path: str = "/home/warrenjo/src/tmp2/secure data"):
        """Initialize processor with secure data directory."""
        self.secure_data_path = Path(secure_data_path)
        self.output_path = Path("data/training_examples")  # Local only, not committed
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()

        # Research-based pattern matching for annotations
        self.open_ended_patterns = [
            r'\b(?:how|why)\s+(?:do|did|does|would|could|might|should)\b',
            r'\bwhat\s+(?:if|would happen|do you think|makes?|causes?)\b',
            r'\btell me (?:about|more about)\b',
            r'\bexplain\s+(?:to me|how|why)\b',
            r'\bdescribe\b',
            r'\bwhat.*notice\b',
            r'\bwhat.*wonder\b',
            r'\bshow me how\b'
        ]

        # ZPD indicators (Zone of Proximal Development)
        self.zpd_patterns = [
            r'\bwhat do you think will happen\b',
            r'\bcan you try\b',
            r'\bwhat if we\b',
            r'\blet\'s see if\b',
            r'\bhow about we\b',
            r'\bwould you like to\b',
            r'\bwhat would happen if\b'
        ]

        # Scaffolding technique indicators
        self.scaffolding_patterns = [
            r'\bgood (?:try|thinking|idea)\b',
            r'\bthat\'s (?:right|interesting|a good point)\b',
            r'\byes,? and\b',
            r'\btell me more\b',
            r'\bwhat else\b',
            r'\bcan you think of another\b',
            r'\bthat reminds me of\b',
            r'\bbuilding on that\b'
        ]

    def extract_audio_from_video(self, video_path: Path) -> Optional[Path]:
        """Extract audio from video file for speech recognition."""
        try:
            with VideoFileClip(str(video_path)) as video:
                # Create temporary audio file
                audio_path = Path(tempfile.mkdtemp()) / f"{video_path.stem}.wav"
                audio = video.audio
                if audio:
                    audio.write_audiofile(str(audio_path), verbose=False, logger=None)
                    return audio_path
                else:
                    logger.warning(f"No audio track found in {video_path}")
                    return None
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            return None

    def transcribe_audio(self, audio_path: Path) -> Optional[str]:
        """Transcribe audio file to text."""
        try:
            with sr.AudioFile(str(audio_path)) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio_data = self.recognizer.record(source)

            # Use Google Speech Recognition (free tier)
            transcript = self.recognizer.recognize_google(audio_data)
            return transcript

        except sr.UnknownValueError:
            logger.warning(f"Could not understand audio in {audio_path}")
            return None
        except sr.RequestError as e:
            logger.error(f"Error with speech recognition service: {e}")
            return None
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return None

    def analyze_transcript(self, transcript: str, video_name: str) -> Dict:
        """Analyze transcript for educational quality indicators."""
        if not transcript:
            return {}

        # Convert to lowercase for pattern matching
        text_lower = transcript.lower()

        # Count pattern matches
        open_ended_count = sum(1 for pattern in self.open_ended_patterns
                              if re.search(pattern, text_lower))

        zpd_count = sum(1 for pattern in self.zpd_patterns
                       if re.search(pattern, text_lower))

        scaffolding_count = sum(1 for pattern in self.scaffolding_patterns
                               if re.search(pattern, text_lower))

        # Count total questions
        total_questions = len(re.findall(r'\?', transcript))

        # Calculate quality metrics
        if total_questions > 0:
            open_ended_ratio = open_ended_count / total_questions
        else:
            open_ended_ratio = 0

        # Determine quality category based on research
        if open_ended_ratio > 0.6 and scaffolding_count > 2:
            quality_category = "exemplary"
            overall_score = 0.85 + (open_ended_ratio * 0.15)
        elif open_ended_ratio > 0.3 and scaffolding_count > 0:
            quality_category = "good"
            overall_score = 0.5 + (open_ended_ratio * 0.35)
        else:
            quality_category = "needs_improvement"
            overall_score = open_ended_ratio * 0.5

        return {
            'video_name': video_name,
            'transcript': transcript,
            'open_ended_questions': open_ended_count,
            'total_questions': total_questions,
            'open_ended_ratio': open_ended_ratio,
            'zpd_indicators': zpd_count,
            'scaffolding_techniques': scaffolding_count,
            'quality_category': quality_category,
            'overall_score': min(overall_score, 1.0),  # Cap at 1.0
            'word_count': len(transcript.split()),
            'sentence_count': len(re.findall(r'[.!?]+', transcript))
        }

    def generate_class_annotations(self, analysis: Dict) -> Dict:
        """Generate CLASS framework annotations."""
        open_ended_ratio = analysis.get('open_ended_ratio', 0)
        scaffolding_count = analysis.get('scaffolding_techniques', 0)
        zpd_count = analysis.get('zpd_indicators', 0)

        # CLASS Language Modeling scores (1-7 scale)
        language_scores = {
            'frequent_conversations': min(7, analysis.get('sentence_count', 0) / 2),
            'open_ended_questions': min(7, open_ended_ratio * 7),
            'repetition_extension': min(7, scaffolding_count),
            'advanced_language': min(7, analysis.get('word_count', 0) / 20)
        }

        # CLASS Quality Feedback scores
        feedback_scores = {
            'scaffolding': min(7, scaffolding_count),
            'encouraging_effort': min(7, scaffolding_count * 1.5),
            'specific_information': min(7, zpd_count * 2),
            'back_and_forth': min(7, open_ended_ratio * 5)
        }

        # CLASS Concept Development scores
        concept_scores = {
            'analysis_reasoning': min(7, open_ended_ratio * 6),
            'creating_suggesting': min(7, zpd_count * 1.5),
            'integration': min(7, (scaffolding_count + zpd_count) / 2),
            'connections_links': min(7, scaffolding_count * 1.2)
        }

        return {
            'language_modeling': language_scores,
            'quality_feedback': feedback_scores,
            'concept_development': concept_scores
        }

    def process_video_file(self, video_path: Path) -> Optional[Dict]:
        """Process a single video file and generate training example."""
        logger.info(f"Processing video: {video_path.name}")

        # Extract audio
        audio_path = self.extract_audio_from_video(video_path)
        if not audio_path:
            return None

        try:
            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            if not transcript:
                return None

            # Analyze transcript
            analysis = self.analyze_transcript(transcript, video_path.name)
            if not analysis:
                return None

            # Generate CLASS annotations
            class_annotations = self.generate_class_annotations(analysis)

            # Combine into training example
            training_example = {
                'id': video_path.stem,
                'source': 'cultivate_learning_secure',
                'analysis': analysis,
                'class_framework': class_annotations,
                'created_at': '2025-09-23',
                'privacy_level': 'secure',  # Never commit to public repo
                'research_basis': [
                    'Vygotsky Zone of Proximal Development',
                    'CLASS Framework (Pianta et al.)',
                    'Hart & Risley Conversation Quality',
                    'Open-ended Question Research'
                ]
            }

            return training_example

        finally:
            # Clean up temporary audio file
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                    audio_path.parent.rmdir()  # Remove temp directory if empty
                except Exception as e:
                    logger.warning(f"Could not clean up temp file {audio_path}: {e}")

    def process_all_videos(self, limit: int = 25) -> List[Dict]:
        """Process all video files in secure directory."""
        video_files = []

        # Find all video files
        for ext in ['.mp4', '.mov', '.MP4', '.MOV']:
            video_files.extend(self.secure_data_path.glob(f'*{ext}'))

        logger.info(f"Found {len(video_files)} video files")

        # Process up to limit
        training_examples = []
        processed_count = 0

        for video_file in video_files:
            if processed_count >= limit:
                break

            # Skip zone identifier files
            if 'Zone.Identifier' in video_file.name:
                continue

            example = self.process_video_file(video_file)
            if example:
                training_examples.append(example)
                processed_count += 1
                logger.info(f"Processed {processed_count}/{limit}: {video_file.name}")
            else:
                logger.warning(f"Failed to process: {video_file.name}")

        return training_examples

    def save_training_data(self, training_examples: List[Dict], filename: str = "training_dataset.json"):
        """Save training examples to local file (NOT committed to repo)."""
        output_file = self.output_path / filename

        with open(output_file, 'w') as f:
            json.dump(training_examples, f, indent=2)

        logger.info(f"Saved {len(training_examples)} training examples to {output_file}")

        # Also save summary statistics
        summary = {
            'total_examples': len(training_examples),
            'quality_distribution': {},
            'avg_scores': {},
            'created_at': '2025-09-23'
        }

        # Calculate quality distribution
        for category in ['exemplary', 'good', 'needs_improvement']:
            count = sum(1 for ex in training_examples
                       if ex['analysis'].get('quality_category') == category)
            summary['quality_distribution'][category] = count

        # Calculate average scores
        if training_examples:
            summary['avg_scores'] = {
                'overall_score': sum(ex['analysis']['overall_score'] for ex in training_examples) / len(training_examples),
                'open_ended_ratio': sum(ex['analysis']['open_ended_ratio'] for ex in training_examples) / len(training_examples),
                'scaffolding_techniques': sum(ex['analysis']['scaffolding_techniques'] for ex in training_examples) / len(training_examples)
            }

        summary_file = self.output_path / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training data summary: {summary}")
        return summary


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Process secure data
    processor = SecureDataProcessor()

    print("Processing secure video data for ML training...")
    print("IMPORTANT: This data will NOT be committed to public repository")

    # Generate training examples
    training_examples = processor.process_all_videos(limit=25)

    if training_examples:
        # Save training data
        summary = processor.save_training_data(training_examples)

        print(f"\n✅ Successfully processed {len(training_examples)} videos")
        print(f"📊 Quality distribution: {summary['quality_distribution']}")
        print(f"📈 Average scores: {summary['avg_scores']}")
        print(f"💾 Data saved to: data/training_examples/")
        print("\n🔒 Remember: This data is private and not committed to repo")
    else:
        print("❌ No training examples generated. Check video files and dependencies.")