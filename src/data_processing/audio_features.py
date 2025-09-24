#!/usr/bin/env python3
"""
Audio Feature Extraction Module for Issue #90

Specialized acoustic feature extraction functions optimized for educational
interaction analysis. Focuses on prosody, timing, and voice quality features
that correlate with teaching effectiveness and student engagement.

Author: Claude (Issue #90 Implementation)
"""

import numpy as np
import librosa
import soundfile as sf
import parselmouth
from parselmouth import praat
import scipy.stats
import scipy.signal
from typing import Dict, List, Tuple, Optional
import logging
import warnings

# Suppress parselmouth warnings
warnings.filterwarnings('ignore', category=UserWarning, module='parselmouth')

logger = logging.getLogger(__name__)

class EducationalAudioAnalyzer:
    """
    Specialized audio analysis for educational interactions.

    Focuses on features that research shows correlate with:
    - Teacher effectiveness (wait time, question pacing)
    - Student engagement (response patterns, voice quality)
    - Constructivist pedagogy (open-ended questioning prosody)
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)  # 25ms frames (standard)
        self.hop_length = int(0.010 * sample_rate)    # 10ms hop (high resolution)

        # Educational research-based thresholds
        self.min_wait_time = 0.3  # Minimum wait time for constructivist teaching
        self.ideal_wait_time = 3.0  # Research-backed optimal wait time
        self.min_question_duration = 1.0  # Minimum meaningful question length

        logger.info(f"Educational audio analyzer initialized (sr={sample_rate})")

    def analyze_teaching_prosody(self, audio_path: str, question_timestamps: List[Dict]) -> Dict:
        """
        Analyze prosodic patterns specifically for teaching effectiveness.

        Research focus:
        - Question vs statement prosody differentiation
        - Open-ended vs closed-ended question patterns
        - Engagement-inducing prosodic features
        """

        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            sound = parselmouth.Sound(audio_path)

            # Extract base prosodic features
            pitch = sound.to_pitch()
            intensity = sound.to_intensity()

            prosody_features = {}

            # Analyze each question segment
            question_prosody = []
            for q_info in question_timestamps:
                start_time = q_info.get('start', 0)
                end_time = q_info.get('end', len(y) / sr)
                question_type = q_info.get('type', 'unknown')

                # Extract question-specific prosody
                q_prosody = self._analyze_question_prosody(
                    sound, pitch, intensity, start_time, end_time, question_type
                )
                question_prosody.append(q_prosody)

            # Aggregate question prosody patterns
            prosody_features['question_pitch_rise_ratio'] = np.mean([
                qp.get('pitch_rise_ratio', 0) for qp in question_prosody
            ])

            prosody_features['question_intensity_pattern'] = np.mean([
                qp.get('intensity_consistency', 0) for qp in question_prosody
            ])

            prosody_features['open_ended_prosody_score'] = np.mean([
                qp.get('open_ended_score', 0) for qp in question_prosody
                if qp.get('question_type') == 'open-ended'
            ]) if any(qp.get('question_type') == 'open-ended' for qp in question_prosody) else 0.0

            # Overall teaching prosody effectiveness
            prosody_features['teaching_prosody_effectiveness'] = self._calculate_teaching_effectiveness(
                question_prosody
            )

            return prosody_features

        except Exception as e:
            logger.error(f"Error analyzing teaching prosody: {e}")
            return self._get_default_prosody_features()

    def analyze_wait_time_patterns(self, audio_path: str, transcript_segments: List[Dict]) -> Dict:
        """
        Analyze wait time patterns crucial for constructivist teaching.

        Research focus:
        - Pre-question wait times
        - Post-question wait times
        - Wait time consistency across interaction
        - Optimal wait time achievement (3-5 seconds)
        """

        try:
            # Load audio for silence detection
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Detect silence periods using energy-based VAD
            rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            silence_threshold = np.percentile(rms, 25)  # Bottom 25% as silence

            # Find silence segments
            silence_frames = rms < silence_threshold
            silence_times = librosa.frames_to_time(
                np.where(silence_frames)[0], sr=sr, hop_length=self.hop_length
            )

            # Analyze wait times around questions
            wait_times = []
            question_wait_analysis = []

            for i, segment in enumerate(transcript_segments):
                if '?' in segment.get('text', ''):  # This is a question
                    question_end = segment.get('end', 0)

                    # Find next non-question segment (student response)
                    next_response = None
                    for j in range(i + 1, len(transcript_segments)):
                        if '?' not in transcript_segments[j].get('text', ''):
                            next_response = transcript_segments[j]
                            break

                    if next_response:
                        response_start = next_response.get('start', 0)
                        wait_time = response_start - question_end

                        # Validate wait time with audio silence analysis
                        validated_wait_time = self._validate_wait_time_with_audio(
                            silence_times, question_end, response_start, wait_time
                        )

                        wait_times.append(validated_wait_time)
                        question_wait_analysis.append({
                            'question_text': segment.get('text', ''),
                            'wait_time': validated_wait_time,
                            'is_optimal': self.min_wait_time <= validated_wait_time <= self.ideal_wait_time,
                            'question_type': self._classify_question_type(segment.get('text', ''))
                        })

            # Calculate wait time metrics
            wait_features = {}

            if wait_times:
                wait_features['mean_wait_time'] = float(np.mean(wait_times))
                wait_features['wait_time_std'] = float(np.std(wait_times))
                wait_features['median_wait_time'] = float(np.median(wait_times))
                wait_features['optimal_wait_time_ratio'] = sum(
                    1 for wt in wait_times
                    if self.min_wait_time <= wt <= self.ideal_wait_time
                ) / len(wait_times)

                # Constructivist teaching effectiveness score based on wait times
                wait_features['constructivist_wait_score'] = self._calculate_constructivist_wait_score(
                    wait_times
                )

                # Wait time consistency (important for classroom management)
                wait_features['wait_time_consistency'] = 1.0 / (1.0 + np.std(wait_times))

            else:
                wait_features = self._get_default_wait_time_features()

            wait_features['question_wait_analysis'] = question_wait_analysis
            return wait_features

        except Exception as e:
            logger.error(f"Error analyzing wait time patterns: {e}")
            return self._get_default_wait_time_features()

    def analyze_engagement_acoustics(self, audio_path: str, speaker_segments: List[Dict]) -> Dict:
        """
        Analyze acoustic features that correlate with student engagement.

        Research focus:
        - Voice activation levels (student participation)
        - Response enthusiasm (prosodic energy)
        - Overlapping speech (engagement indicators)
        - Voice quality changes during interaction
        """

        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            sound = parselmouth.Sound(audio_path)

            # Separate teacher and student segments
            teacher_segments = []
            student_segments = []

            for segment in speaker_segments:
                # Assume SPEAKER_00 is teacher, others are students
                if segment.get('speaker') == 'SPEAKER_00':
                    teacher_segments.append(segment)
                else:
                    student_segments.append(segment)

            engagement_features = {}

            # Student voice activation analysis
            if student_segments:
                student_activation = self._analyze_voice_activation(
                    sound, student_segments
                )
                engagement_features.update(student_activation)

            # Response enthusiasm analysis
            enthusiasm_scores = []
            for segment in student_segments:
                enthusiasm = self._analyze_response_enthusiasm(
                    sound, segment.get('start', 0), segment.get('end', 0)
                )
                enthusiasm_scores.append(enthusiasm)

            engagement_features['mean_student_enthusiasm'] = float(np.mean(enthusiasm_scores)) if enthusiasm_scores else 0.0

            # Overlapping speech detection (engagement indicator)
            overlap_analysis = self._detect_speech_overlaps(speaker_segments)
            engagement_features.update(overlap_analysis)

            # Overall engagement score
            engagement_features['overall_engagement_score'] = self._calculate_engagement_score(
                engagement_features
            )

            return engagement_features

        except Exception as e:
            logger.error(f"Error analyzing engagement acoustics: {e}")
            return self._get_default_engagement_features()

    def _analyze_question_prosody(self, sound, pitch, intensity, start_time: float, end_time: float, question_type: str) -> Dict:
        """Analyze prosodic patterns for a specific question."""

        try:
            # Extract pitch and intensity for question segment
            question_pitch = pitch.selected_array['frequency']
            question_times = pitch.xs()

            # Filter to question time range
            time_mask = (question_times >= start_time) & (question_times <= end_time)
            segment_pitch = question_pitch[time_mask]
            segment_pitch = segment_pitch[segment_pitch > 0]  # Remove unvoiced

            if len(segment_pitch) < 2:
                return {'pitch_rise_ratio': 0, 'intensity_consistency': 0.5, 'open_ended_score': 0.5}

            # Pitch rise ratio (questions typically end with rising intonation)
            pitch_slope = np.polyfit(range(len(segment_pitch)), segment_pitch, 1)[0]
            pitch_rise_ratio = 1 / (1 + np.exp(-pitch_slope / 10))  # Sigmoid normalization

            # Intensity consistency (good questions maintain energy)
            intensity_values = intensity.selected_array['intensity']
            intensity_times = intensity.xs()
            intensity_mask = (intensity_times >= start_time) & (intensity_times <= end_time)
            segment_intensity = intensity_values[intensity_mask]

            intensity_consistency = 1.0 - (np.std(segment_intensity) / np.mean(segment_intensity)) if len(segment_intensity) > 0 else 0.5

            # Open-ended question prosodic score
            # Research shows open-ended questions have more varied prosody
            pitch_variation = np.std(segment_pitch) / np.mean(segment_pitch) if np.mean(segment_pitch) > 0 else 0
            open_ended_score = min(1.0, pitch_variation * 2)  # More variation = more open-ended

            return {
                'pitch_rise_ratio': float(pitch_rise_ratio),
                'intensity_consistency': float(max(0, min(1, intensity_consistency))),
                'open_ended_score': float(open_ended_score),
                'question_type': question_type
            }

        except Exception as e:
            logger.warning(f"Error analyzing question prosody: {e}")
            return {'pitch_rise_ratio': 0, 'intensity_consistency': 0.5, 'open_ended_score': 0.5}

    def _calculate_teaching_effectiveness(self, question_prosody: List[Dict]) -> float:
        """Calculate overall teaching effectiveness from prosodic patterns."""

        if not question_prosody:
            return 0.5

        # Weighted combination of prosodic features
        pitch_rise_score = np.mean([qp.get('pitch_rise_ratio', 0) for qp in question_prosody])
        intensity_score = np.mean([qp.get('intensity_consistency', 0) for qp in question_prosody])
        open_ended_score = np.mean([qp.get('open_ended_score', 0) for qp in question_prosody])

        # Research-weighted combination (open-ended questions are crucial)
        effectiveness = (
            0.3 * pitch_rise_score +
            0.3 * intensity_score +
            0.4 * open_ended_score  # Higher weight for open-ended questioning
        )

        return float(max(0, min(1, effectiveness)))

    def _validate_wait_time_with_audio(self, silence_times: np.ndarray, question_end: float, response_start: float, transcript_wait_time: float) -> float:
        """Validate transcript-based wait time with actual audio silence."""

        # Find silence periods between question end and response start
        relevant_silences = silence_times[
            (silence_times >= question_end) & (silence_times <= response_start)
        ]

        if len(relevant_silences) > 0:
            # Use audio-validated wait time if significantly different
            audio_silence_duration = len(relevant_silences) * 0.01  # 10ms hop length

            # If transcript and audio disagree significantly, trust audio
            if abs(audio_silence_duration - transcript_wait_time) > 0.5:
                return audio_silence_duration

        return transcript_wait_time

    def _classify_question_type(self, question_text: str) -> str:
        """Classify question as open-ended or closed-ended based on text."""

        question_lower = question_text.lower()

        # Open-ended indicators
        open_indicators = ['why', 'how', 'what do you think', 'explain', 'describe', 'tell me about']
        closed_indicators = ['is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should']

        open_score = sum(1 for indicator in open_indicators if indicator in question_lower)
        closed_score = sum(1 for indicator in closed_indicators if question_lower.startswith(indicator))

        if open_score > closed_score:
            return 'open-ended'
        elif closed_score > open_score:
            return 'closed-ended'
        else:
            return 'mixed'

    def _calculate_constructivist_wait_score(self, wait_times: List[float]) -> float:
        """Calculate constructivist teaching effectiveness based on wait time patterns."""

        if not wait_times:
            return 0.0

        # Research-based optimal wait time scoring
        optimal_count = sum(1 for wt in wait_times if self.min_wait_time <= wt <= self.ideal_wait_time)
        too_short_count = sum(1 for wt in wait_times if wt < self.min_wait_time)
        too_long_count = sum(1 for wt in wait_times if wt > self.ideal_wait_time)

        # Scoring function based on educational research
        optimal_ratio = optimal_count / len(wait_times)
        penalty_ratio = (too_short_count + too_long_count * 0.5) / len(wait_times)

        constructivist_score = optimal_ratio - penalty_ratio * 0.5
        return float(max(0, min(1, constructivist_score)))

    def _analyze_voice_activation(self, sound, student_segments: List[Dict]) -> Dict:
        """Analyze student voice activation patterns."""

        try:
            total_duration = sound.get_total_duration()
            student_speaking_time = sum(
                segment.get('end', 0) - segment.get('start', 0)
                for segment in student_segments
            )

            student_activation_ratio = student_speaking_time / total_duration if total_duration > 0 else 0

            # Number of student turns (participation frequency)
            student_turns = len(student_segments)
            turn_frequency = student_turns / (total_duration / 60) if total_duration > 0 else 0  # turns per minute

            return {
                'student_activation_ratio': float(student_activation_ratio),
                'student_turn_frequency': float(turn_frequency),
                'total_student_segments': student_turns
            }

        except Exception as e:
            logger.warning(f"Error analyzing voice activation: {e}")
            return {
                'student_activation_ratio': 0.0,
                'student_turn_frequency': 0.0,
                'total_student_segments': 0
            }

    def _analyze_response_enthusiasm(self, sound, start_time: float, end_time: float) -> float:
        """Analyze enthusiasm level of a response based on prosodic features."""

        try:
            # Extract pitch and intensity for response segment
            pitch = sound.to_pitch()
            intensity = sound.to_intensity()

            # Get values for time segment
            pitch_values = pitch.selected_array['frequency']
            pitch_times = pitch.xs()
            intensity_values = intensity.selected_array['intensity']
            intensity_times = intensity.xs()

            # Filter to segment
            pitch_mask = (pitch_times >= start_time) & (pitch_times <= end_time)
            intensity_mask = (intensity_times >= start_time) & (intensity_times <= end_time)

            segment_pitch = pitch_values[pitch_mask]
            segment_pitch = segment_pitch[segment_pitch > 0]  # Remove unvoiced
            segment_intensity = intensity_values[intensity_mask]

            if len(segment_pitch) == 0 or len(segment_intensity) == 0:
                return 0.5  # Neutral enthusiasm

            # Enthusiasm indicators
            pitch_variation = np.std(segment_pitch) / np.mean(segment_pitch) if np.mean(segment_pitch) > 0 else 0
            mean_intensity = np.mean(segment_intensity)
            intensity_variation = np.std(segment_intensity)

            # Combine features for enthusiasm score
            enthusiasm = (
                min(1.0, pitch_variation * 3) * 0.4 +  # Pitch variation indicates engagement
                min(1.0, mean_intensity / 60) * 0.4 +   # Higher intensity indicates enthusiasm
                min(1.0, intensity_variation / 10) * 0.2  # Some intensity variation is good
            )

            return float(max(0, min(1, enthusiasm)))

        except Exception as e:
            logger.warning(f"Error analyzing response enthusiasm: {e}")
            return 0.5

    def _detect_speech_overlaps(self, speaker_segments: List[Dict]) -> Dict:
        """Detect overlapping speech as engagement indicator."""

        try:
            overlaps = []
            total_overlap_time = 0.0

            for i in range(len(speaker_segments) - 1):
                current = speaker_segments[i]
                next_segment = speaker_segments[i + 1]

                current_end = current.get('end', 0)
                next_start = next_segment.get('start', 0)

                # Overlap occurs when next segment starts before current ends
                if next_start < current_end:
                    overlap_duration = current_end - next_start
                    overlaps.append({
                        'start': next_start,
                        'duration': overlap_duration,
                        'speakers': [current.get('speaker'), next_segment.get('speaker')]
                    })
                    total_overlap_time += overlap_duration

            return {
                'overlap_count': len(overlaps),
                'total_overlap_time': float(total_overlap_time),
                'overlap_frequency': len(overlaps) / len(speaker_segments) if speaker_segments else 0.0,
                'overlaps': overlaps
            }

        except Exception as e:
            logger.warning(f"Error detecting speech overlaps: {e}")
            return {
                'overlap_count': 0,
                'total_overlap_time': 0.0,
                'overlap_frequency': 0.0,
                'overlaps': []
            }

    def _calculate_engagement_score(self, engagement_features: Dict) -> float:
        """Calculate overall engagement score from acoustic features."""

        try:
            # Normalize and weight different engagement indicators
            activation_score = engagement_features.get('student_activation_ratio', 0) * 2  # 0-2 range
            enthusiasm_score = engagement_features.get('mean_student_enthusiasm', 0)  # 0-1 range
            turn_score = min(1.0, engagement_features.get('student_turn_frequency', 0) / 5)  # Normalize to 0-1
            overlap_score = min(1.0, engagement_features.get('overlap_frequency', 0) * 10)  # Light overlaps are good

            # Weighted combination
            engagement_score = (
                activation_score * 0.3 +
                enthusiasm_score * 0.3 +
                turn_score * 0.2 +
                overlap_score * 0.2
            ) / 2.0  # Normalize to 0-1

            return float(max(0, min(1, engagement_score)))

        except Exception as e:
            logger.warning(f"Error calculating engagement score: {e}")
            return 0.5

    # Default feature methods
    def _get_default_prosody_features(self) -> Dict:
        """Return default prosody features when analysis fails."""
        return {
            'question_pitch_rise_ratio': 0.5,
            'question_intensity_pattern': 0.5,
            'open_ended_prosody_score': 0.5,
            'teaching_prosody_effectiveness': 0.5
        }

    def _get_default_wait_time_features(self) -> Dict:
        """Return default wait time features when analysis fails."""
        return {
            'mean_wait_time': 0.0,
            'wait_time_std': 0.0,
            'median_wait_time': 0.0,
            'optimal_wait_time_ratio': 0.0,
            'constructivist_wait_score': 0.0,
            'wait_time_consistency': 0.0,
            'question_wait_analysis': []
        }

    def _get_default_engagement_features(self) -> Dict:
        """Return default engagement features when analysis fails."""
        return {
            'student_activation_ratio': 0.0,
            'student_turn_frequency': 0.0,
            'total_student_segments': 0,
            'mean_student_enthusiasm': 0.5,
            'overlap_count': 0,
            'total_overlap_time': 0.0,
            'overlap_frequency': 0.0,
            'overall_engagement_score': 0.5
        }


def extract_educational_audio_features(audio_path: str, transcript_data: Dict) -> Dict:
    """
    Main function to extract all educational audio features from a single audio file.

    Args:
        audio_path: Path to the audio file
        transcript_data: Dictionary containing transcript, speakers, and timing information

    Returns:
        Dictionary of educational audio features
    """

    analyzer = EducationalAudioAnalyzer()

    # Extract question timestamps from transcript
    question_timestamps = []
    for segment in transcript_data.get('speakers', []):
        if '?' in segment.get('text', ''):
            question_timestamps.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', ''),
                'type': analyzer._classify_question_type(segment.get('text', ''))
            })

    # Extract all feature categories
    features = {}

    try:
        # Teaching prosody analysis
        prosody_features = analyzer.analyze_teaching_prosody(audio_path, question_timestamps)
        features['prosody'] = prosody_features

        # Wait time analysis
        wait_time_features = analyzer.analyze_wait_time_patterns(
            audio_path, transcript_data.get('speakers', [])
        )
        features['wait_times'] = wait_time_features

        # Engagement analysis
        engagement_features = analyzer.analyze_engagement_acoustics(
            audio_path, transcript_data.get('speakers', [])
        )
        features['engagement'] = engagement_features

        # Overall educational effectiveness score
        features['educational_effectiveness_score'] = (
            prosody_features.get('teaching_prosody_effectiveness', 0.5) * 0.3 +
            wait_time_features.get('constructivist_wait_score', 0.5) * 0.4 +
            engagement_features.get('overall_engagement_score', 0.5) * 0.3
        )

        logger.info(f"Successfully extracted educational audio features from {audio_path}")

    except Exception as e:
        logger.error(f"Error extracting educational audio features: {e}")
        # Return minimal default features
        features = {
            'prosody': analyzer._get_default_prosody_features(),
            'wait_times': analyzer._get_default_wait_time_features(),
            'engagement': analyzer._get_default_engagement_features(),
            'educational_effectiveness_score': 0.5
        }

    return features