"""
Whisper Audio Analysis Module - Story 7.3 Implementation
Microsoft Partner-Level Audio Processing Architecture

Comprehensive audio analysis pipeline for educational video processing:
- OpenAI Whisper transcription with speaker identification
- Speaker diarization for teacher-child interaction analysis
- Prosodic feature extraction for emotional and engagement analysis
- Audio quality assessment and enhancement

Author: Claude (Partner-Level Microsoft SDE)
Feature: #98 - Video Feature Extraction & Deep Learning Pipeline
Story: 7.3 - Whisper Audio Analysis and Speaker Diarization
"""

import os
import tempfile
import librosa
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import asyncio
import subprocess

# Audio processing imports
try:
    import whisper
    import soundfile as sf
    from pyannote.audio import Pipeline
    WHISPER_AVAILABLE = True
except ImportError as e:
    WHISPER_AVAILABLE = False
    logging.warning(f"Whisper/audio dependencies not available: {e}")

logger = logging.getLogger(__name__)

class WhisperAudioProcessor:
    """
    Comprehensive audio analysis using OpenAI Whisper and advanced audio processing
    Designed for educational video analysis with teacher-child interaction focus
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper audio processor

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper dependencies not available. Install openai-whisper, librosa, soundfile, pyannote.audio")

        self.model_size = model_size
        logger.info(f"Initializing Whisper Audio Processor with model: {model_size}")

        # Load models
        self._load_models()

        # Audio processing parameters
        self.sample_rate = 16000  # Standard for Whisper
        self.chunk_length = 30    # 30-second chunks for processing

        # Educational context parameters
        self.speaker_labels = {
            "SPEAKER_00": "teacher",
            "SPEAKER_01": "child",
            "unknown": "unclear"
        }

    def _load_models(self):
        """Load Whisper and diarization models"""
        try:
            # Load Whisper model
            self.whisper_model = whisper.load_model(self.model_size)
            logger.info(f"✅ Whisper {self.model_size} model loaded successfully")

            # Load speaker diarization model
            try:
                # Try to load pyannote speaker diarization pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")  # Optional HF token
                )
                self.diarization_available = True
                logger.info("✅ Speaker diarization pipeline loaded")
            except Exception as diar_error:
                logger.warning(f"Speaker diarization unavailable: {diar_error}")
                self.diarization_pipeline = None
                self.diarization_available = False

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file using FFmpeg

        Args:
            video_path: Path to input video file

        Returns:
            Path to extracted audio file
        """
        try:
            # Create temporary audio file
            temp_dir = tempfile.mkdtemp(prefix="whisper_audio_")
            audio_path = os.path.join(temp_dir, "extracted_audio.wav")

            # Use FFmpeg to extract audio
            ffmpeg_cmd = [
                "ffmpeg", "-i", video_path,
                "-ar", str(self.sample_rate),  # Set sample rate
                "-ac", "1",                    # Convert to mono
                "-c:a", "pcm_s16le",          # PCM 16-bit encoding
                "-y",                         # Overwrite output
                audio_path
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"✅ Audio extracted successfully: {audio_path}")
            return audio_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg extraction failed: {e.stderr}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper with timestamps

        Args:
            audio_path: Path to audio file

        Returns:
            Comprehensive transcription with segments and metadata
        """
        try:
            logger.info("🎵 Starting Whisper transcription...")

            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",
                language=None,  # Auto-detect language
                temperature=0.0,  # Deterministic output
                word_timestamps=True,  # Enable word-level timestamps
                fp16=False  # Use FP32 for stability
            )

            # Process segments
            processed_segments = []
            total_words = 0

            for segment in result.get("segments", []):
                segment_data = {
                    "id": segment.get("id", 0),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "text": segment.get("text", "").strip(),
                    "words": []
                }

                # Process word-level timestamps if available
                if "words" in segment:
                    for word in segment["words"]:
                        word_data = {
                            "word": word.get("word", "").strip(),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                            "probability": word.get("probability", 0.0)
                        }
                        segment_data["words"].append(word_data)
                        total_words += 1

                processed_segments.append(segment_data)

            # Calculate speaking metrics
            speaking_duration = sum(
                seg["end"] - seg["start"] for seg in processed_segments
            )

            audio_duration = librosa.get_duration(filename=audio_path)
            speaking_ratio = speaking_duration / audio_duration if audio_duration > 0 else 0

            transcription_data = {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "segments": processed_segments,
                "metadata": {
                    "model_size": self.model_size,
                    "total_segments": len(processed_segments),
                    "total_words": total_words,
                    "audio_duration": audio_duration,
                    "speaking_duration": speaking_duration,
                    "speaking_ratio": speaking_ratio,
                    "transcription_time": datetime.utcnow().isoformat()
                }
            }

            logger.info(f"✅ Transcription complete: {total_words} words in {len(processed_segments)} segments")
            return transcription_data

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                "error": str(e),
                "text": "",
                "segments": [],
                "metadata": {"failed": True}
            }

    def perform_speaker_diarization(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform speaker diarization to identify teacher vs child speech

        Args:
            audio_path: Path to audio file

        Returns:
            Speaker diarization results with educational context
        """
        try:
            if not self.diarization_available:
                logger.warning("Speaker diarization unavailable - using fallback")
                return self._fallback_speaker_analysis(audio_path)

            logger.info("👥 Starting speaker diarization...")

            # Run diarization pipeline
            diarization = self.diarization_pipeline(audio_path)

            # Process diarization results
            speaker_segments = []
            speaker_stats = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_label = self.speaker_labels.get(speaker, speaker)

                segment = {
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start,
                    "speaker": speaker_label,
                    "speaker_id": speaker
                }
                speaker_segments.append(segment)

                # Update speaker statistics
                if speaker_label not in speaker_stats:
                    speaker_stats[speaker_label] = {
                        "total_duration": 0,
                        "segment_count": 0,
                        "average_segment_length": 0
                    }

                speaker_stats[speaker_label]["total_duration"] += segment["duration"]
                speaker_stats[speaker_label]["segment_count"] += 1

            # Calculate averages and ratios
            total_audio_duration = librosa.get_duration(filename=audio_path)

            for speaker, stats in speaker_stats.items():
                stats["average_segment_length"] = (
                    stats["total_duration"] / stats["segment_count"]
                    if stats["segment_count"] > 0 else 0
                )
                stats["speaking_ratio"] = (
                    stats["total_duration"] / total_audio_duration
                    if total_audio_duration > 0 else 0
                )

            # Educational analysis
            teacher_time = speaker_stats.get("teacher", {}).get("total_duration", 0)
            child_time = speaker_stats.get("child", {}).get("total_duration", 0)

            interaction_analysis = {
                "teacher_child_ratio": teacher_time / child_time if child_time > 0 else float('inf'),
                "interaction_balance": "teacher_dominated" if teacher_time > child_time * 2 else
                                    "child_dominated" if child_time > teacher_time * 2 else "balanced",
                "turn_taking_frequency": len(speaker_segments) / total_audio_duration if total_audio_duration > 0 else 0
            }

            diarization_data = {
                "speaker_segments": speaker_segments,
                "speaker_statistics": speaker_stats,
                "interaction_analysis": interaction_analysis,
                "metadata": {
                    "total_speakers": len(speaker_stats),
                    "total_segments": len(speaker_segments),
                    "audio_duration": total_audio_duration,
                    "diarization_time": datetime.utcnow().isoformat()
                }
            }

            logger.info(f"✅ Diarization complete: {len(speaker_stats)} speakers, {len(speaker_segments)} segments")
            return diarization_data

        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return self._fallback_speaker_analysis(audio_path)

    def _fallback_speaker_analysis(self, audio_path: str) -> Dict[str, Any]:
        """Fallback speaker analysis when diarization is unavailable"""
        audio_duration = librosa.get_duration(filename=audio_path)

        return {
            "speaker_segments": [
                {
                    "start": 0,
                    "end": audio_duration,
                    "duration": audio_duration,
                    "speaker": "unknown",
                    "speaker_id": "FALLBACK_00"
                }
            ],
            "speaker_statistics": {
                "unknown": {
                    "total_duration": audio_duration,
                    "segment_count": 1,
                    "average_segment_length": audio_duration,
                    "speaking_ratio": 1.0
                }
            },
            "interaction_analysis": {
                "teacher_child_ratio": None,
                "interaction_balance": "unknown",
                "turn_taking_frequency": 0
            },
            "metadata": {
                "total_speakers": 1,
                "total_segments": 1,
                "audio_duration": audio_duration,
                "fallback": True,
                "note": "Speaker diarization unavailable"
            }
        }

    def extract_prosodic_features(self, audio_path: str, segments: List[Dict]) -> Dict[str, Any]:
        """
        Extract prosodic features for emotional and engagement analysis

        Args:
            audio_path: Path to audio file
            segments: Speaker segments from diarization

        Returns:
            Prosodic feature analysis
        """
        try:
            logger.info("🎼 Extracting prosodic features...")

            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Extract global features
            global_features = self._extract_global_audio_features(y, sr)

            # Extract segment-wise features
            segment_features = []

            for segment in segments:
                start_sample = int(segment["start"] * sr)
                end_sample = int(segment["end"] * sr)
                segment_audio = y[start_sample:end_sample]

                if len(segment_audio) > 0:
                    features = self._extract_segment_features(segment_audio, sr)
                    features.update({
                        "start": segment["start"],
                        "end": segment["end"],
                        "speaker": segment.get("speaker", "unknown")
                    })
                    segment_features.append(features)

            # Educational context analysis
            educational_metrics = self._analyze_educational_prosody(segment_features)

            prosodic_data = {
                "global_features": global_features,
                "segment_features": segment_features,
                "educational_metrics": educational_metrics,
                "metadata": {
                    "feature_extraction_time": datetime.utcnow().isoformat(),
                    "total_segments_analyzed": len(segment_features)
                }
            }

            logger.info(f"✅ Prosodic analysis complete: {len(segment_features)} segments analyzed")
            return prosodic_data

        except Exception as e:
            logger.error(f"Prosodic feature extraction failed: {e}")
            return {
                "error": str(e),
                "global_features": {},
                "segment_features": [],
                "educational_metrics": {}
            }

    def _extract_global_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract global audio features"""
        try:
            features = {}

            # Fundamental frequency (pitch)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]

            if len(pitch_values) > 0:
                features["mean_pitch"] = float(np.mean(pitch_values))
                features["std_pitch"] = float(np.std(pitch_values))
                features["min_pitch"] = float(np.min(pitch_values))
                features["max_pitch"] = float(np.max(pitch_values))
            else:
                features.update({"mean_pitch": 0, "std_pitch": 0, "min_pitch": 0, "max_pitch": 0})

            # Energy and dynamics
            rms = librosa.feature.rms(y=y)[0]
            features["mean_energy"] = float(np.mean(rms))
            features["energy_variance"] = float(np.var(rms))

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            features["spectral_centroid_std"] = float(np.std(spectral_centroids))

            # Zero crossing rate (measure of voicing)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["mean_zcr"] = float(np.mean(zcr))

            # Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features["estimated_tempo"] = float(tempo)

            return features

        except Exception as e:
            logger.error(f"Global feature extraction error: {e}")
            return {}

    def _extract_segment_features(self, segment_audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract features for a specific audio segment"""
        try:
            if len(segment_audio) == 0:
                return {}

            features = {}

            # RMS energy
            rms = librosa.feature.rms(y=segment_audio)[0]
            features["segment_energy"] = float(np.mean(rms))

            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr, threshold=0.1)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]

            if len(pitch_values) > 0:
                features["segment_mean_pitch"] = float(np.mean(pitch_values))
                features["pitch_variation"] = float(np.std(pitch_values))
            else:
                features["segment_mean_pitch"] = 0
                features["pitch_variation"] = 0

            # Speaking rate approximation (zero crossings as proxy)
            zcr = librosa.feature.zero_crossing_rate(segment_audio)[0]
            features["speaking_rate_proxy"] = float(np.mean(zcr))

            # Spectral characteristics
            spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0]
            features["segment_spectral_centroid"] = float(np.mean(spectral_centroid))

            return features

        except Exception as e:
            logger.error(f"Segment feature extraction error: {e}")
            return {}

    def _analyze_educational_prosody(self, segment_features: List[Dict]) -> Dict[str, Any]:
        """Analyze prosodic patterns for educational effectiveness"""
        try:
            teacher_segments = [s for s in segment_features if s.get("speaker") == "teacher"]
            child_segments = [s for s in segment_features if s.get("speaker") == "child"]

            metrics = {
                "teacher_metrics": self._calculate_speaker_metrics(teacher_segments),
                "child_metrics": self._calculate_speaker_metrics(child_segments),
                "interaction_quality": {}
            }

            # Interaction quality metrics
            if teacher_segments and child_segments:
                teacher_pitch = np.mean([s.get("segment_mean_pitch", 0) for s in teacher_segments])
                child_pitch = np.mean([s.get("segment_mean_pitch", 0) for s in child_segments])

                metrics["interaction_quality"] = {
                    "pitch_adaptation": abs(teacher_pitch - child_pitch) / max(teacher_pitch, child_pitch) if max(teacher_pitch, child_pitch) > 0 else 0,
                    "energy_balance": self._calculate_energy_balance(teacher_segments, child_segments),
                    "engagement_indicators": self._assess_engagement_indicators(teacher_segments, child_segments)
                }

            return metrics

        except Exception as e:
            logger.error(f"Educational prosody analysis error: {e}")
            return {}

    def _calculate_speaker_metrics(self, segments: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for a specific speaker"""
        if not segments:
            return {}

        energies = [s.get("segment_energy", 0) for s in segments]
        pitches = [s.get("segment_mean_pitch", 0) for s in segments if s.get("segment_mean_pitch", 0) > 0]

        return {
            "average_energy": float(np.mean(energies)) if energies else 0,
            "energy_consistency": float(1 / (1 + np.std(energies))) if energies and np.std(energies) > 0 else 1,
            "average_pitch": float(np.mean(pitches)) if pitches else 0,
            "pitch_variation": float(np.std(pitches)) if pitches else 0,
            "segment_count": len(segments)
        }

    def _calculate_energy_balance(self, teacher_segments: List[Dict], child_segments: List[Dict]) -> float:
        """Calculate energy balance between teacher and child"""
        teacher_energy = np.mean([s.get("segment_energy", 0) for s in teacher_segments]) if teacher_segments else 0
        child_energy = np.mean([s.get("segment_energy", 0) for s in child_segments]) if child_segments else 0

        if teacher_energy + child_energy == 0:
            return 0

        return 1 - abs(teacher_energy - child_energy) / (teacher_energy + child_energy)

    def _assess_engagement_indicators(self, teacher_segments: List[Dict], child_segments: List[Dict]) -> Dict[str, float]:
        """Assess indicators of engagement in the interaction"""
        try:
            teacher_pitch_var = np.mean([s.get("pitch_variation", 0) for s in teacher_segments]) if teacher_segments else 0
            child_pitch_var = np.mean([s.get("pitch_variation", 0) for s in child_segments]) if child_segments else 0

            return {
                "teacher_expressiveness": float(teacher_pitch_var),
                "child_responsiveness": float(child_pitch_var),
                "overall_engagement": float((teacher_pitch_var + child_pitch_var) / 2)
            }
        except Exception:
            return {"teacher_expressiveness": 0, "child_responsiveness": 0, "overall_engagement": 0}

    async def process_video_audio(self, video_path: str) -> Dict[str, Any]:
        """
        Complete audio analysis pipeline for video files

        Args:
            video_path: Path to input video file

        Returns:
            Comprehensive audio analysis results
        """
        try:
            logger.info(f"🎬 Starting complete audio analysis for: {video_path}")
            start_time = datetime.utcnow()

            # Step 1: Extract audio from video
            audio_path = self.extract_audio_from_video(video_path)

            # Step 2: Transcription
            transcription_data = self.transcribe_audio(audio_path)

            # Step 3: Speaker diarization
            diarization_data = self.perform_speaker_diarization(audio_path)

            # Step 4: Prosodic analysis
            prosodic_data = self.extract_prosodic_features(
                audio_path,
                diarization_data.get("speaker_segments", [])
            )

            # Step 5: Combine transcription with speaker information
            enhanced_transcription = self._align_transcription_with_speakers(
                transcription_data, diarization_data
            )

            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            # Compile comprehensive results
            results = {
                "transcription": enhanced_transcription,
                "speaker_diarization": diarization_data,
                "prosodic_analysis": prosodic_data,
                "metadata": {
                    "processing_time_seconds": processing_time,
                    "model_used": self.model_size,
                    "audio_file": audio_path,
                    "processing_timestamp": start_time.isoformat(),
                    "whisper_available": WHISPER_AVAILABLE,
                    "diarization_available": self.diarization_available
                }
            }

            # Cleanup temporary files
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    # Remove temporary directory
                    temp_dir = os.path.dirname(audio_path)
                    os.rmdir(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")

            logger.info(f"✅ Complete audio analysis finished in {processing_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Video audio processing failed: {e}")
            return {
                "error": str(e),
                "transcription": {"text": "", "segments": []},
                "speaker_diarization": {"speaker_segments": []},
                "prosodic_analysis": {"global_features": {}},
                "metadata": {"failed": True}
            }

    def _align_transcription_with_speakers(self, transcription: Dict, diarization: Dict) -> Dict[str, Any]:
        """Align transcription segments with speaker information"""
        try:
            aligned_segments = []
            transcription_segments = transcription.get("segments", [])
            speaker_segments = diarization.get("speaker_segments", [])

            for trans_seg in transcription_segments:
                trans_start = trans_seg.get("start", 0)
                trans_end = trans_seg.get("end", 0)

                # Find overlapping speaker segment
                best_speaker = "unknown"
                best_overlap = 0

                for speaker_seg in speaker_segments:
                    speaker_start = speaker_seg.get("start", 0)
                    speaker_end = speaker_seg.get("end", 0)

                    # Calculate overlap
                    overlap_start = max(trans_start, speaker_start)
                    overlap_end = min(trans_end, speaker_end)
                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_speaker = speaker_seg.get("speaker", "unknown")

                # Create aligned segment
                aligned_segment = trans_seg.copy()
                aligned_segment["speaker"] = best_speaker
                aligned_segment["speaker_confidence"] = best_overlap / (trans_end - trans_start) if trans_end > trans_start else 0
                aligned_segments.append(aligned_segment)

            # Update transcription with speaker alignment
            enhanced_transcription = transcription.copy()
            enhanced_transcription["segments"] = aligned_segments
            enhanced_transcription["metadata"]["speaker_aligned"] = True

            return enhanced_transcription

        except Exception as e:
            logger.error(f"Transcription-speaker alignment failed: {e}")
            return transcription