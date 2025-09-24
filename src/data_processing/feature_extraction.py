#!/usr/bin/env python3
"""
ML Feature Extraction Pipeline for Issue #90

Extracts acoustic, linguistic, and interaction features from processed audio
and transcript data to create training inputs for BERT-based multi-task learning model.
This bridges raw data processing with ML model training.

Author: Claude (Issue #90 Implementation)
Dependencies: Issues #89 (Transcription & Speaker Diarization)
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime
import re
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import required libraries with error handling
try:
    import librosa
    import soundfile as sf
    import parselmouth
    from parselmouth import praat
    import scipy.stats
    import scipy.signal
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    import spacy
    import nltk
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install librosa soundfile praat-parselmouth scipy scikit-learn spacy nltk transformers torch")
    print("Also run: python -m spacy download en_core_web_sm")
    print("And: python -c \"import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')\"")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class AcousticFeatures:
    """Container for acoustic features extracted from audio."""

    # Prosodic features
    pitch_mean: float
    pitch_std: float
    pitch_range: float
    pitch_contour: List[float]

    # Timing features
    speaking_rate: float
    pause_count: int
    pause_duration_mean: float
    pause_duration_total: float

    # Voice quality features
    jitter: float
    shimmer: float
    harmonics_noise_ratio: float
    voice_quality_score: float

    # Energy/volume features
    energy_mean: float
    energy_std: float
    intensity_mean: float
    intensity_std: float

    # Duration features
    total_duration: float
    speech_duration: float
    silence_ratio: float

@dataclass
class LinguisticFeatures:
    """Container for linguistic features from transcripts."""

    # Question classification
    question_type: str  # 'open-ended', 'closed-ended', 'procedural', 'unknown'
    question_complexity: float  # 0-1 scale

    # Wait time analysis
    pre_question_wait: float
    post_question_wait: float
    total_wait_time: float

    # Turn-taking patterns
    speaker_transitions: int
    educator_talk_ratio: float
    student_response_ratio: float
    overlap_instances: int

    # Linguistic complexity
    sentence_length_mean: float
    vocabulary_diversity: float
    syntactic_complexity: float
    readability_score: float

    # Content analysis
    question_words: List[str]
    key_concepts: List[str]
    emotional_markers: List[str]

@dataclass
class InteractionFeatures:
    """Container for educational interaction features."""

    # CLASS framework alignment
    emotional_support_score: float
    classroom_organization_score: float
    instructional_support_score: float

    # Engagement indicators
    student_engagement_level: float
    participation_rate: float
    response_quality_score: float

    # Quality metrics from CSV annotations
    ground_truth_labels: Dict[str, Any]
    annotation_confidence: float

    # Temporal patterns
    interaction_rhythm: float
    pacing_consistency: float

@dataclass
class CombinedFeatures:
    """Container for all extracted features ready for ML training."""

    video_id: str
    timestamp: datetime
    acoustic: AcousticFeatures
    linguistic: LinguisticFeatures
    interaction: InteractionFeatures
    feature_vector: np.ndarray
    metadata: Dict[str, Any]

class AcousticFeatureExtractor:
    """Extracts acoustic features from audio files using librosa and Parselmouth."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)  # 25ms frames
        self.hop_length = int(0.010 * sample_rate)    # 10ms hop

    def extract_features(self, audio_path: str, transcript_timing: Dict) -> AcousticFeatures:
        """Extract comprehensive acoustic features from audio file."""

        logger.info(f"Extracting acoustic features from {audio_path}")

        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Also load with Parselmouth for Praat analysis
            sound = parselmouth.Sound(audio_path)

            # Extract prosodic features
            prosody = self._extract_prosodic_features(sound, y, sr)

            # Extract timing features
            timing = self._extract_timing_features(y, sr, transcript_timing)

            # Extract voice quality features
            voice_quality = self._extract_voice_quality_features(sound)

            # Extract energy/volume features
            energy = self._extract_energy_features(y, sr)

            # Calculate duration features
            duration = self._extract_duration_features(y, sr, transcript_timing)

            return AcousticFeatures(
                # Prosodic features
                pitch_mean=prosody['pitch_mean'],
                pitch_std=prosody['pitch_std'],
                pitch_range=prosody['pitch_range'],
                pitch_contour=prosody['pitch_contour'],

                # Timing features
                speaking_rate=timing['speaking_rate'],
                pause_count=timing['pause_count'],
                pause_duration_mean=timing['pause_duration_mean'],
                pause_duration_total=timing['pause_duration_total'],

                # Voice quality features
                jitter=voice_quality['jitter'],
                shimmer=voice_quality['shimmer'],
                harmonics_noise_ratio=voice_quality['hnr'],
                voice_quality_score=voice_quality['quality_score'],

                # Energy features
                energy_mean=energy['energy_mean'],
                energy_std=energy['energy_std'],
                intensity_mean=energy['intensity_mean'],
                intensity_std=energy['intensity_std'],

                # Duration features
                total_duration=duration['total_duration'],
                speech_duration=duration['speech_duration'],
                silence_ratio=duration['silence_ratio']
            )

        except Exception as e:
            logger.error(f"Error extracting acoustic features from {audio_path}: {e}")
            return self._get_default_acoustic_features()

    def _extract_prosodic_features(self, sound, y, sr) -> Dict:
        """Extract prosodic features using Praat/Parselmouth."""

        try:
            # Extract pitch using Praat
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']

            # Remove unvoiced frames (0 Hz)
            voiced_pitch = pitch_values[pitch_values > 0]

            if len(voiced_pitch) > 0:
                pitch_mean = np.mean(voiced_pitch)
                pitch_std = np.std(voiced_pitch)
                pitch_range = np.max(voiced_pitch) - np.min(voiced_pitch)

                # Downsample pitch contour for feature vector
                pitch_contour = librosa.resample(
                    voiced_pitch.astype(float),
                    orig_sr=len(voiced_pitch),
                    target_sr=50  # 50 points
                ).tolist()
            else:
                pitch_mean = pitch_std = pitch_range = 0.0
                pitch_contour = [0.0] * 50

            return {
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'pitch_range': pitch_range,
                'pitch_contour': pitch_contour
            }

        except Exception as e:
            logger.warning(f"Error in prosodic feature extraction: {e}")
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_range': 0.0,
                'pitch_contour': [0.0] * 50
            }

    def _extract_timing_features(self, y, sr, transcript_timing) -> Dict:
        """Extract timing and pause features."""

        try:
            # Detect voiced/unvoiced segments using energy threshold
            hop_length = self.hop_length
            frame_length = self.frame_length

            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            # Threshold for voice activity detection
            energy_threshold = np.percentile(rms, 30)  # Bottom 30% considered silence

            # Find silence segments
            silence_frames = rms < energy_threshold
            silence_times = librosa.frames_to_time(np.where(silence_frames)[0], sr=sr, hop_length=hop_length)

            # Group consecutive silence frames into pauses
            pauses = self._group_consecutive_silences(silence_times, min_pause_duration=0.2)

            # Calculate speaking rate from transcript if available
            if transcript_timing and 'word_count' in transcript_timing:
                speaking_rate = transcript_timing['word_count'] / (len(y) / sr * 60)  # words per minute
            else:
                # Estimate from voiced segments
                voiced_duration = len(y) / sr - sum(p['duration'] for p in pauses)
                estimated_syllables = voiced_duration * 3  # Rough estimate
                speaking_rate = estimated_syllables / (len(y) / sr / 60)  # syllables per minute

            return {
                'speaking_rate': speaking_rate,
                'pause_count': len(pauses),
                'pause_duration_mean': np.mean([p['duration'] for p in pauses]) if pauses else 0.0,
                'pause_duration_total': sum(p['duration'] for p in pauses)
            }

        except Exception as e:
            logger.warning(f"Error in timing feature extraction: {e}")
            return {
                'speaking_rate': 0.0,
                'pause_count': 0,
                'pause_duration_mean': 0.0,
                'pause_duration_total': 0.0
            }

    def _extract_voice_quality_features(self, sound) -> Dict:
        """Extract voice quality features using Praat."""

        try:
            # Extract jitter (pitch perturbation)
            pointprocess = praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter = praat.call(pointprocess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

            # Extract shimmer (amplitude perturbation)
            shimmer = praat.call([sound, pointprocess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            # Extract harmonics-to-noise ratio
            harmonicity = praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = praat.call(harmonicity, "Get mean", 0, 0)

            # Calculate overall voice quality score (composite metric)
            # Lower jitter/shimmer and higher HNR indicate better voice quality
            jitter_score = max(0, 1 - (jitter * 1000))  # Convert to 0-1 scale
            shimmer_score = max(0, 1 - (shimmer * 10))
            hnr_score = min(1, max(0, hnr / 25))  # Normalize HNR to 0-1

            quality_score = (jitter_score + shimmer_score + hnr_score) / 3

            return {
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr': hnr,
                'quality_score': quality_score
            }

        except Exception as e:
            logger.warning(f"Error in voice quality feature extraction: {e}")
            return {
                'jitter': 0.0,
                'shimmer': 0.0,
                'hnr': 0.0,
                'quality_score': 0.5
            }

    def _extract_energy_features(self, y, sr) -> Dict:
        """Extract energy and intensity features."""

        try:
            # RMS energy
            rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]

            # Spectral centroid as intensity measure
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]

            return {
                'energy_mean': float(np.mean(rms)),
                'energy_std': float(np.std(rms)),
                'intensity_mean': float(np.mean(spectral_centroid)),
                'intensity_std': float(np.std(spectral_centroid))
            }

        except Exception as e:
            logger.warning(f"Error in energy feature extraction: {e}")
            return {
                'energy_mean': 0.0,
                'energy_std': 0.0,
                'intensity_mean': 0.0,
                'intensity_std': 0.0
            }

    def _extract_duration_features(self, y, sr, transcript_timing) -> Dict:
        """Extract duration and timing-related features."""

        try:
            total_duration = len(y) / sr

            # Estimate speech duration using voice activity detection
            rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            energy_threshold = np.percentile(rms, 30)
            voiced_frames = np.sum(rms > energy_threshold)
            speech_duration = voiced_frames * self.hop_length / sr

            silence_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0

            return {
                'total_duration': total_duration,
                'speech_duration': speech_duration,
                'silence_ratio': silence_ratio
            }

        except Exception as e:
            logger.warning(f"Error in duration feature extraction: {e}")
            return {
                'total_duration': 0.0,
                'speech_duration': 0.0,
                'silence_ratio': 0.0
            }

    def _group_consecutive_silences(self, silence_times, min_pause_duration=0.2) -> List[Dict]:
        """Group consecutive silence frames into meaningful pauses."""

        if len(silence_times) == 0:
            return []

        pauses = []
        current_start = silence_times[0]
        current_end = silence_times[0]

        for i in range(1, len(silence_times)):
            if silence_times[i] - silence_times[i-1] < 0.1:  # Within 100ms, consider consecutive
                current_end = silence_times[i]
            else:
                # End current pause, start new one
                duration = current_end - current_start
                if duration >= min_pause_duration:
                    pauses.append({
                        'start': current_start,
                        'end': current_end,
                        'duration': duration
                    })
                current_start = silence_times[i]
                current_end = silence_times[i]

        # Add final pause
        duration = current_end - current_start
        if duration >= min_pause_duration:
            pauses.append({
                'start': current_start,
                'end': current_end,
                'duration': duration
            })

        return pauses

    def _get_default_acoustic_features(self) -> AcousticFeatures:
        """Return default acoustic features when extraction fails."""

        return AcousticFeatures(
            pitch_mean=0.0, pitch_std=0.0, pitch_range=0.0, pitch_contour=[0.0] * 50,
            speaking_rate=0.0, pause_count=0, pause_duration_mean=0.0, pause_duration_total=0.0,
            jitter=0.0, shimmer=0.0, harmonics_noise_ratio=0.0, voice_quality_score=0.5,
            energy_mean=0.0, energy_std=0.0, intensity_mean=0.0, intensity_std=0.0,
            total_duration=0.0, speech_duration=0.0, silence_ratio=0.0
        )

class LinguisticFeatureExtractor:
    """Extracts linguistic features from transcripts using NLP techniques."""

    def __init__(self):
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("English spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Question classification patterns
        self.open_question_patterns = [
            r'\b(why|how|what.*think|what.*feel|explain|describe|tell me about)\b',
            r'\b(in your opinion|what do you|how would you)\b'
        ]

        self.closed_question_patterns = [
            r'\b(is|are|do|does|did|can|could|would|should|will)\b.*\?',
            r'\b(yes.*or.*no|true.*or.*false)\b'
        ]

        # Download required NLTK data
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except:
            pass

    def extract_features(self, transcript: str, speaker_labels: List[Dict], timing_data: Dict) -> LinguisticFeatures:
        """Extract comprehensive linguistic features from transcript."""

        logger.info("Extracting linguistic features from transcript")

        try:
            # Extract question classification
            question_analysis = self._classify_questions(transcript)

            # Extract wait time analysis
            wait_times = self._analyze_wait_times(speaker_labels, timing_data)

            # Extract turn-taking patterns
            turn_patterns = self._analyze_turn_taking(speaker_labels)

            # Extract linguistic complexity
            complexity = self._analyze_linguistic_complexity(transcript)

            # Extract content analysis
            content = self._analyze_content(transcript)

            return LinguisticFeatures(
                # Question classification
                question_type=question_analysis['type'],
                question_complexity=question_analysis['complexity'],

                # Wait time analysis
                pre_question_wait=wait_times['pre_question'],
                post_question_wait=wait_times['post_question'],
                total_wait_time=wait_times['total'],

                # Turn-taking patterns
                speaker_transitions=turn_patterns['transitions'],
                educator_talk_ratio=turn_patterns['educator_ratio'],
                student_response_ratio=turn_patterns['student_ratio'],
                overlap_instances=turn_patterns['overlaps'],

                # Linguistic complexity
                sentence_length_mean=complexity['sentence_length'],
                vocabulary_diversity=complexity['vocabulary_diversity'],
                syntactic_complexity=complexity['syntactic_complexity'],
                readability_score=complexity['readability'],

                # Content analysis
                question_words=content['question_words'],
                key_concepts=content['key_concepts'],
                emotional_markers=content['emotional_markers']
            )

        except Exception as e:
            logger.error(f"Error extracting linguistic features: {e}")
            return self._get_default_linguistic_features()

    def _classify_questions(self, transcript: str) -> Dict:
        """Classify questions as open-ended vs closed-ended."""

        try:
            questions = [sent for sent in transcript.split('.') if '?' in sent]

            if not questions:
                return {'type': 'unknown', 'complexity': 0.0}

            open_count = 0
            closed_count = 0
            total_complexity = 0

            for question in questions:
                question = question.lower().strip()

                # Check for open-ended patterns
                is_open = any(re.search(pattern, question) for pattern in self.open_question_patterns)

                # Check for closed-ended patterns
                is_closed = any(re.search(pattern, question) for pattern in self.closed_question_patterns)

                if is_open:
                    open_count += 1
                    complexity = min(1.0, len(question.split()) / 10)  # Complexity based on length
                elif is_closed:
                    closed_count += 1
                    complexity = 0.3  # Closed questions generally less complex
                else:
                    complexity = 0.5  # Unknown questions get medium complexity

                total_complexity += complexity

            # Determine primary question type
            if open_count > closed_count:
                question_type = 'open-ended'
            elif closed_count > open_count:
                question_type = 'closed-ended'
            else:
                question_type = 'mixed'

            avg_complexity = total_complexity / len(questions) if questions else 0.0

            return {
                'type': question_type,
                'complexity': avg_complexity
            }

        except Exception as e:
            logger.warning(f"Error in question classification: {e}")
            return {'type': 'unknown', 'complexity': 0.0}

    def _analyze_wait_times(self, speaker_labels: List[Dict], timing_data: Dict) -> Dict:
        """Analyze wait times around questions."""

        try:
            if not speaker_labels or not timing_data:
                return {'pre_question': 0.0, 'post_question': 0.0, 'total': 0.0}

            # Find question segments (simplified - look for question marks in text)
            question_times = []
            for segment in speaker_labels:
                if 'text' in segment and '?' in segment['text']:
                    question_times.append(segment.get('end', 0))

            if not question_times:
                return {'pre_question': 0.0, 'post_question': 0.0, 'total': 0.0}

            # Calculate wait times after questions
            wait_times = []
            for q_time in question_times:
                # Find next speaker segment after question
                next_segments = [s for s in speaker_labels if s.get('start', 0) > q_time]
                if next_segments:
                    next_start = min(s.get('start', float('inf')) for s in next_segments)
                    wait_time = next_start - q_time
                    wait_times.append(wait_time)

            pre_question_wait = 0.0  # Placeholder - would need more sophisticated analysis
            post_question_wait = np.mean(wait_times) if wait_times else 0.0
            total_wait = sum(wait_times)

            return {
                'pre_question': pre_question_wait,
                'post_question': post_question_wait,
                'total': total_wait
            }

        except Exception as e:
            logger.warning(f"Error in wait time analysis: {e}")
            return {'pre_question': 0.0, 'post_question': 0.0, 'total': 0.0}

    def _analyze_turn_taking(self, speaker_labels: List[Dict]) -> Dict:
        """Analyze turn-taking patterns between speakers."""

        try:
            if not speaker_labels:
                return {'transitions': 0, 'educator_ratio': 0.0, 'student_ratio': 0.0, 'overlaps': 0}

            # Count speaker transitions
            transitions = 0
            prev_speaker = None
            educator_duration = 0.0
            student_duration = 0.0
            overlaps = 0

            for segment in speaker_labels:
                current_speaker = segment.get('speaker', 'unknown')
                duration = segment.get('end', 0) - segment.get('start', 0)

                if prev_speaker and prev_speaker != current_speaker:
                    transitions += 1

                # Classify speakers (simplified - assume SPEAKER_00 is educator)
                if current_speaker == 'SPEAKER_00':
                    educator_duration += duration
                else:
                    student_duration += duration

                prev_speaker = current_speaker

            total_duration = educator_duration + student_duration
            educator_ratio = educator_duration / total_duration if total_duration > 0 else 0.0
            student_ratio = student_duration / total_duration if total_duration > 0 else 0.0

            return {
                'transitions': transitions,
                'educator_ratio': educator_ratio,
                'student_ratio': student_ratio,
                'overlaps': overlaps  # Placeholder - would need audio analysis
            }

        except Exception as e:
            logger.warning(f"Error in turn-taking analysis: {e}")
            return {'transitions': 0, 'educator_ratio': 0.0, 'student_ratio': 0.0, 'overlaps': 0}

    def _analyze_linguistic_complexity(self, transcript: str) -> Dict:
        """Analyze linguistic complexity of the transcript."""

        try:
            if not transcript.strip():
                return {
                    'sentence_length': 0.0,
                    'vocabulary_diversity': 0.0,
                    'syntactic_complexity': 0.0,
                    'readability': 0.0
                }

            # Sentence length analysis
            sentences = [s.strip() for s in transcript.split('.') if s.strip()]
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0.0

            # Vocabulary diversity (Type-Token Ratio)
            words = transcript.lower().split()
            unique_words = set(words)
            vocab_diversity = len(unique_words) / len(words) if words else 0.0

            # Syntactic complexity (simplified - based on sentence structure)
            syntactic_complexity = 0.0
            if self.nlp:
                doc = self.nlp(transcript)
                complex_structures = 0
                total_sentences = 0

                for sent in doc.sents:
                    total_sentences += 1
                    # Count subordinate clauses and complex structures
                    complex_structures += len([token for token in sent if token.dep_ in ['advcl', 'acl', 'ccomp']])

                syntactic_complexity = complex_structures / total_sentences if total_sentences > 0 else 0.0

            # Readability score (Flesch-Kincaid approximation)
            if sentence_lengths and words:
                avg_words_per_sentence = np.mean(sentence_lengths)
                avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
                readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
                readability = max(0, min(100, readability)) / 100  # Normalize to 0-1
            else:
                readability = 0.0

            return {
                'sentence_length': avg_sentence_length,
                'vocabulary_diversity': vocab_diversity,
                'syntactic_complexity': syntactic_complexity,
                'readability': readability
            }

        except Exception as e:
            logger.warning(f"Error in linguistic complexity analysis: {e}")
            return {
                'sentence_length': 0.0,
                'vocabulary_diversity': 0.0,
                'syntactic_complexity': 0.0,
                'readability': 0.0
            }

    def _analyze_content(self, transcript: str) -> Dict:
        """Analyze content for key concepts and emotional markers."""

        try:
            # Question words
            question_words = []
            for word in ['what', 'why', 'how', 'when', 'where', 'who', 'which']:
                if word in transcript.lower():
                    question_words.append(word)

            # Key concepts (simplified - high-frequency content words)
            words = transcript.lower().split()
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            content_words = [word for word in words if word not in stop_words and len(word) > 3]

            # Get most frequent content words as key concepts
            word_freq = {}
            for word in content_words:
                word_freq[word] = word_freq.get(word, 0) + 1

            key_concepts = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]

            # Emotional markers (simplified sentiment analysis)
            positive_markers = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'love', 'like', 'enjoy']
            negative_markers = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'difficult', 'hard', 'problem', 'issue']

            emotional_markers = []
            for word in words:
                if word in positive_markers:
                    emotional_markers.append(f"positive:{word}")
                elif word in negative_markers:
                    emotional_markers.append(f"negative:{word}")

            return {
                'question_words': question_words,
                'key_concepts': key_concepts,
                'emotional_markers': emotional_markers
            }

        except Exception as e:
            logger.warning(f"Error in content analysis: {e}")
            return {
                'question_words': [],
                'key_concepts': [],
                'emotional_markers': []
            }

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for readability calculation."""

        word = word.lower()
        if len(word) <= 3:
            return 1

        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1

        return max(1, syllable_count)

    def _get_default_linguistic_features(self) -> LinguisticFeatures:
        """Return default linguistic features when extraction fails."""

        return LinguisticFeatures(
            question_type='unknown', question_complexity=0.0,
            pre_question_wait=0.0, post_question_wait=0.0, total_wait_time=0.0,
            speaker_transitions=0, educator_talk_ratio=0.0, student_response_ratio=0.0, overlap_instances=0,
            sentence_length_mean=0.0, vocabulary_diversity=0.0, syntactic_complexity=0.0, readability_score=0.0,
            question_words=[], key_concepts=[], emotional_markers=[]
        )

class FeatureExtractionPipeline:
    """Main pipeline for extracting all ML features from processed video data."""

    def __init__(self, output_dir: str = "feature_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize feature extractors
        self.acoustic_extractor = AcousticFeatureExtractor()
        self.linguistic_extractor = LinguisticFeatureExtractor()

        # Feature scaling
        self.scaler = StandardScaler()

        logger.info(f"Feature extraction pipeline initialized. Output: {self.output_dir}")

    async def process_video(self, video_id: str, audio_path: str, transcript_data: Dict, csv_annotations: Dict = None) -> CombinedFeatures:
        """Process a single video through the complete feature extraction pipeline."""

        logger.info(f"Processing video {video_id}")

        try:
            # Extract acoustic features
            logger.info(f"Extracting acoustic features for {video_id}")
            acoustic_features = self.acoustic_extractor.extract_features(
                audio_path,
                transcript_data.get('timing', {})
            )

            # Extract linguistic features
            logger.info(f"Extracting linguistic features for {video_id}")
            transcript_text = transcript_data.get('transcript', '')
            speaker_labels = transcript_data.get('speakers', [])
            timing_data = transcript_data.get('timing', {})

            linguistic_features = self.linguistic_extractor.extract_features(
                transcript_text,
                speaker_labels,
                timing_data
            )

            # Extract interaction features (placeholder for now)
            interaction_features = self._extract_interaction_features(
                acoustic_features,
                linguistic_features,
                csv_annotations or {}
            )

            # Create combined feature vector
            feature_vector = self._create_feature_vector(
                acoustic_features,
                linguistic_features,
                interaction_features
            )

            # Create combined features object
            combined_features = CombinedFeatures(
                video_id=video_id,
                timestamp=datetime.now(),
                acoustic=acoustic_features,
                linguistic=linguistic_features,
                interaction=interaction_features,
                feature_vector=feature_vector,
                metadata={
                    'audio_path': str(audio_path),
                    'transcript_length': len(transcript_text),
                    'processing_timestamp': datetime.now().isoformat()
                }
            )

            # Save features
            await self._save_features(combined_features)

            logger.info(f"Successfully processed video {video_id}")
            return combined_features

        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            raise

    def _extract_interaction_features(self, acoustic: AcousticFeatures, linguistic: LinguisticFeatures, csv_annotations: Dict) -> InteractionFeatures:
        """Extract educational interaction features (placeholder implementation)."""

        # This would normally integrate with the CLASS framework and CSV annotations
        # For now, provide basic derived features

        try:
            # Derive CLASS framework scores from acoustic and linguistic features
            # This is a simplified approximation - actual implementation would use trained models

            emotional_support = min(1.0, (acoustic.voice_quality_score + linguistic.readability_score) / 2)
            classroom_organization = min(1.0, linguistic.educator_talk_ratio + (1 - acoustic.silence_ratio))
            instructional_support = min(1.0, linguistic.question_complexity + (acoustic.speaking_rate / 200))

            # Engagement indicators
            student_engagement = min(1.0, linguistic.student_response_ratio * 2)
            participation_rate = min(1.0, linguistic.speaker_transitions / 10)
            response_quality = linguistic.vocabulary_diversity

            # Use CSV annotations if available
            ground_truth_labels = csv_annotations.copy() if csv_annotations else {}
            annotation_confidence = 0.8 if csv_annotations else 0.0

            # Temporal patterns
            interaction_rhythm = acoustic.speaking_rate / 150 if acoustic.speaking_rate > 0 else 0.5
            pacing_consistency = 1.0 - (acoustic.pitch_std / acoustic.pitch_mean) if acoustic.pitch_mean > 0 else 0.5

            return InteractionFeatures(
                emotional_support_score=emotional_support,
                classroom_organization_score=classroom_organization,
                instructional_support_score=instructional_support,
                student_engagement_level=student_engagement,
                participation_rate=participation_rate,
                response_quality_score=response_quality,
                ground_truth_labels=ground_truth_labels,
                annotation_confidence=annotation_confidence,
                interaction_rhythm=interaction_rhythm,
                pacing_consistency=pacing_consistency
            )

        except Exception as e:
            logger.warning(f"Error extracting interaction features: {e}")
            return InteractionFeatures(
                emotional_support_score=0.5, classroom_organization_score=0.5, instructional_support_score=0.5,
                student_engagement_level=0.5, participation_rate=0.5, response_quality_score=0.5,
                ground_truth_labels={}, annotation_confidence=0.0,
                interaction_rhythm=0.5, pacing_consistency=0.5
            )

    def _create_feature_vector(self, acoustic: AcousticFeatures, linguistic: LinguisticFeatures, interaction: InteractionFeatures) -> np.ndarray:
        """Create unified feature vector for ML training."""

        try:
            # Acoustic features (excluding pitch contour which is handled separately)
            acoustic_vector = [
                acoustic.pitch_mean, acoustic.pitch_std, acoustic.pitch_range,
                acoustic.speaking_rate, acoustic.pause_count, acoustic.pause_duration_mean, acoustic.pause_duration_total,
                acoustic.jitter, acoustic.shimmer, acoustic.harmonics_noise_ratio, acoustic.voice_quality_score,
                acoustic.energy_mean, acoustic.energy_std, acoustic.intensity_mean, acoustic.intensity_std,
                acoustic.total_duration, acoustic.speech_duration, acoustic.silence_ratio
            ]

            # Add pitch contour (downsampled to 20 points for feature vector)
            pitch_contour_sample = acoustic.pitch_contour[::max(1, len(acoustic.pitch_contour)//20)][:20]
            while len(pitch_contour_sample) < 20:
                pitch_contour_sample.append(0.0)
            acoustic_vector.extend(pitch_contour_sample)

            # Linguistic features
            linguistic_vector = [
                1.0 if linguistic.question_type == 'open-ended' else 0.0,
                1.0 if linguistic.question_type == 'closed-ended' else 0.0,
                linguistic.question_complexity,
                linguistic.pre_question_wait, linguistic.post_question_wait, linguistic.total_wait_time,
                linguistic.speaker_transitions, linguistic.educator_talk_ratio, linguistic.student_response_ratio, linguistic.overlap_instances,
                linguistic.sentence_length_mean, linguistic.vocabulary_diversity, linguistic.syntactic_complexity, linguistic.readability_score
            ]

            # Interaction features
            interaction_vector = [
                interaction.emotional_support_score, interaction.classroom_organization_score, interaction.instructional_support_score,
                interaction.student_engagement_level, interaction.participation_rate, interaction.response_quality_score,
                interaction.annotation_confidence, interaction.interaction_rhythm, interaction.pacing_consistency
            ]

            # Combine all features
            feature_vector = np.array(acoustic_vector + linguistic_vector + interaction_vector)

            # Handle NaN/inf values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)

            return feature_vector

        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            # Return zero vector of expected size
            return np.zeros(len(acoustic_vector) + len(linguistic_vector) + len(interaction_vector))

    async def _save_features(self, features: CombinedFeatures):
        """Save extracted features to disk."""

        try:
            # Save individual feature components
            output_file = self.output_dir / f"{features.video_id}_features.json"

            # Convert to dictionary for JSON serialization
            features_dict = {
                'video_id': features.video_id,
                'timestamp': features.timestamp.isoformat(),
                'acoustic': asdict(features.acoustic),
                'linguistic': asdict(features.linguistic),
                'interaction': asdict(features.interaction),
                'feature_vector': features.feature_vector.tolist(),
                'metadata': features.metadata
            }

            with open(output_file, 'w') as f:
                json.dump(features_dict, f, indent=2)

            logger.info(f"Saved features to {output_file}")

            # Also save feature vector separately for ML training
            vector_file = self.output_dir / f"{features.video_id}_vector.npy"
            np.save(vector_file, features.feature_vector)

        except Exception as e:
            logger.error(f"Error saving features for {features.video_id}: {e}")

    async def process_batch(self, video_data: List[Dict]) -> List[CombinedFeatures]:
        """Process multiple videos in batch."""

        logger.info(f"Processing batch of {len(video_data)} videos")

        results = []
        for video_info in video_data:
            try:
                features = await self.process_video(
                    video_info['video_id'],
                    video_info['audio_path'],
                    video_info['transcript_data'],
                    video_info.get('csv_annotations')
                )
                results.append(features)
            except Exception as e:
                logger.error(f"Failed to process video {video_info.get('video_id', 'unknown')}: {e}")

        logger.info(f"Successfully processed {len(results)} out of {len(video_data)} videos")
        return results

    def create_training_dataset(self, feature_files: List[str], output_path: str = "training_dataset.pkl"):
        """Create ML-ready training dataset from extracted features."""

        logger.info(f"Creating training dataset from {len(feature_files)} feature files")

        try:
            all_features = []
            all_vectors = []
            all_labels = []

            for feature_file in feature_files:
                with open(feature_file, 'r') as f:
                    features = json.load(f)

                all_features.append(features)
                all_vectors.append(np.array(features['feature_vector']))

                # Extract labels from ground truth annotations
                ground_truth = features['interaction']['ground_truth_labels']
                all_labels.append(ground_truth)

            # Stack feature vectors
            X = np.vstack(all_vectors) if all_vectors else np.array([])

            # Normalize features
            if len(X) > 0:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X

            # Create dataset dictionary
            dataset = {
                'features': X_scaled,
                'labels': all_labels,
                'metadata': all_features,
                'scaler': self.scaler,
                'feature_names': self._get_feature_names(),
                'created_at': datetime.now().isoformat()
            }

            # Save dataset
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(dataset, f)

            logger.info(f"Training dataset saved to {output_path}")
            logger.info(f"Dataset shape: {X_scaled.shape if len(X_scaled) > 0 else 'Empty'}")

            return dataset

        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            raise

    def _get_feature_names(self) -> List[str]:
        """Get descriptive names for all features in the feature vector."""

        acoustic_names = [
            'pitch_mean', 'pitch_std', 'pitch_range',
            'speaking_rate', 'pause_count', 'pause_duration_mean', 'pause_duration_total',
            'jitter', 'shimmer', 'harmonics_noise_ratio', 'voice_quality_score',
            'energy_mean', 'energy_std', 'intensity_mean', 'intensity_std',
            'total_duration', 'speech_duration', 'silence_ratio'
        ]

        # Add pitch contour features
        pitch_contour_names = [f'pitch_contour_{i}' for i in range(20)]
        acoustic_names.extend(pitch_contour_names)

        linguistic_names = [
            'is_open_ended', 'is_closed_ended', 'question_complexity',
            'pre_question_wait', 'post_question_wait', 'total_wait_time',
            'speaker_transitions', 'educator_talk_ratio', 'student_response_ratio', 'overlap_instances',
            'sentence_length_mean', 'vocabulary_diversity', 'syntactic_complexity', 'readability_score'
        ]

        interaction_names = [
            'emotional_support_score', 'classroom_organization_score', 'instructional_support_score',
            'student_engagement_level', 'participation_rate', 'response_quality_score',
            'annotation_confidence', 'interaction_rhythm', 'pacing_consistency'
        ]

        return acoustic_names + linguistic_names + interaction_names

# Example usage and testing functions
async def main():
    """Example usage of the feature extraction pipeline."""

    # Initialize pipeline
    pipeline = FeatureExtractionPipeline()

    # Example video data (replace with actual data paths)
    video_data = [
        {
            'video_id': 'test_video_001',
            'audio_path': 'path/to/audio.wav',
            'transcript_data': {
                'transcript': 'What do you think about this problem? How would you solve it?',
                'speakers': [
                    {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 3.0, 'text': 'What do you think about this problem?'},
                    {'speaker': 'SPEAKER_01', 'start': 4.0, 'end': 6.0, 'text': 'I think we should try...'},
                    {'speaker': 'SPEAKER_00', 'start': 7.0, 'end': 9.0, 'text': 'How would you solve it?'}
                ],
                'timing': {'word_count': 12}
            },
            'csv_annotations': {
                'emotional_support': 0.8,
                'instructional_support': 0.7,
                'classroom_organization': 0.9
            }
        }
    ]

    # Process videos
    results = await pipeline.process_batch(video_data)

    # Create training dataset
    feature_files = list(pipeline.output_dir.glob("*_features.json"))
    if feature_files:
        dataset = pipeline.create_training_dataset([str(f) for f in feature_files])
        print(f"Created training dataset with shape: {dataset['features'].shape}")

    print("Feature extraction pipeline completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())