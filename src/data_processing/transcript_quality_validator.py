#!/usr/bin/env python3
"""
Transcript Quality Validation for Issue #89
Analyzes generated transcripts for quality, completeness, and ML readiness.

Author: Claude (Issue #89 Quality Validation)
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics
import re
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptQualityMetrics:
    """Quality metrics for a single transcript"""
    clip_id: str
    confidence_score: float
    word_count: int
    character_count: int
    has_question: bool
    question_type: str
    speaker_identified: bool
    contains_educator_language: bool
    educational_indicators: List[str]
    quality_score: float

class TranscriptQualityValidator:
    """Validates quality of generated transcripts for ML training"""

    def __init__(self, transcripts_dir: str = "data/transcripts/transcripts"):
        self.transcripts_dir = Path(transcripts_dir)
        self.metrics: List[TranscriptQualityMetrics] = []

        # Educational language indicators
        self.educator_patterns = [
            r'\b(what|how|why|can you|tell me|think about|look at|show me)\b',
            r'\b(let\'s|we can|try|good|great|excellent|wonderful)\b',
            r'\b(because|when|where|which|would you)\b'
        ]

        # Question patterns
        self.question_patterns = [
            r'what\s+(?:do|did|does|can|will|would|is|are)',
            r'how\s+(?:do|did|does|can|will|would|is|are)',
            r'why\s+(?:do|did|does|can|will|would|is|are)',
            r'can\s+you',
            r'tell\s+me',
            r'show\s+me'
        ]

        logger.info(f"Transcript validator initialized for {transcripts_dir}")

    def validate_all_transcripts(self) -> Dict[str, Any]:
        """Validate all available transcripts and generate quality report"""

        logger.info("Starting transcript quality validation")

        transcript_files = list(self.transcripts_dir.glob("*.json"))
        if not transcript_files:
            logger.warning("No transcript files found")
            return {"error": "No transcripts found"}

        logger.info(f"Found {len(transcript_files)} transcript files")

        for transcript_file in transcript_files:
            try:
                metrics = self.validate_transcript(transcript_file)
                if metrics:
                    self.metrics.append(metrics)
            except Exception as e:
                logger.error(f"Failed to validate {transcript_file.name}: {e}")

        report = self.generate_quality_report()
        self.save_quality_report(report)

        logger.info(f"Quality validation complete: {len(self.metrics)} transcripts analyzed")
        return report

    def validate_transcript(self, transcript_file: Path) -> Optional[TranscriptQualityMetrics]:
        """Validate a single transcript file"""

        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract basic info
            clip_id = data.get('clip_id', transcript_file.stem)
            segments = data.get('transcript_segments', [])

            if not segments:
                logger.warning(f"No segments found in {clip_id}")
                return None

            # Combine all text
            full_text = " ".join(segment.get('text', '') for segment in segments)

            # Calculate metrics
            confidence = data.get('processing_metadata', {}).get('transcription_confidence', 0.0)
            word_count = len(full_text.split())
            char_count = len(full_text)

            # Analyze educational content
            has_question = '?' in full_text
            contains_educator_lang = self.detect_educator_language(full_text)
            educational_indicators = self.extract_educational_indicators(full_text)
            question_type = data.get('quality_annotation', {}).get('question_type', 'unknown')

            # Speaker analysis
            speakers = data.get('speakers', {})
            speaker_identified = len(speakers) > 0 and 'unknown' not in speakers

            # Calculate overall quality score
            quality_score = self.calculate_quality_score(
                confidence, word_count, has_question,
                contains_educator_lang, len(educational_indicators)
            )

            return TranscriptQualityMetrics(
                clip_id=clip_id,
                confidence_score=confidence,
                word_count=word_count,
                character_count=char_count,
                has_question=has_question,
                question_type=question_type,
                speaker_identified=speaker_identified,
                contains_educator_language=contains_educator_lang,
                educational_indicators=educational_indicators,
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Error validating {transcript_file}: {e}")
            return None

    def detect_educator_language(self, text: str) -> bool:
        """Detect if text contains educator language patterns"""
        text_lower = text.lower()

        for pattern in self.educator_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def extract_educational_indicators(self, text: str) -> List[str]:
        """Extract educational language indicators from text"""
        indicators = []
        text_lower = text.lower()

        # Question indicators
        for pattern in self.question_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                indicators.append("open_ended_question")
                break

        # Scaffolding indicators
        if re.search(r'\b(good|great|yes|that\'s right|exactly)\b', text_lower):
            indicators.append("positive_reinforcement")

        if re.search(r'\b(try|think|look|show)\b', text_lower):
            indicators.append("cognitive_prompts")

        # Engagement indicators
        if re.search(r'\b(what do you think|how do you feel|tell me more)\b', text_lower):
            indicators.append("engagement_prompts")

        return list(set(indicators))

    def calculate_quality_score(self, confidence: float, word_count: int,
                              has_question: bool, educator_lang: bool,
                              indicator_count: int) -> float:
        """Calculate overall quality score for transcript (0-1)"""

        score = 0.0

        # Confidence score (0-0.4 weight)
        score += min(confidence, 1.0) * 0.4

        # Length appropriateness (0-0.2 weight)
        if 5 <= word_count <= 50:  # Appropriate length for 5-second clips
            score += 0.2
        elif word_count > 0:
            score += 0.1

        # Educational content (0-0.4 weight)
        if has_question:
            score += 0.15
        if educator_lang:
            score += 0.15
        score += min(indicator_count * 0.05, 0.1)

        return min(score, 1.0)

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""

        if not self.metrics:
            return {"error": "No metrics available"}

        # Overall statistics
        confidence_scores = [m.confidence_score for m in self.metrics]
        word_counts = [m.word_count for m in self.metrics]
        quality_scores = [m.quality_score for m in self.metrics]

        # Educational content analysis
        question_clips = sum(1 for m in self.metrics if m.has_question)
        educator_lang_clips = sum(1 for m in self.metrics if m.contains_educator_language)
        speaker_id_clips = sum(1 for m in self.metrics if m.speaker_identified)

        # Question type distribution
        question_types = {}
        for m in self.metrics:
            qt = m.question_type
            question_types[qt] = question_types.get(qt, 0) + 1

        # Educational indicators
        all_indicators = []
        for m in self.metrics:
            all_indicators.extend(m.educational_indicators)

        indicator_counts = {}
        for indicator in all_indicators:
            indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

        # Quality tiers
        high_quality = sum(1 for m in self.metrics if m.quality_score >= 0.7)
        medium_quality = sum(1 for m in self.metrics if 0.4 <= m.quality_score < 0.7)
        low_quality = sum(1 for m in self.metrics if m.quality_score < 0.4)

        return {
            "summary": {
                "total_transcripts": len(self.metrics),
                "validation_timestamp": datetime.utcnow().isoformat(),
                "overall_quality_score": statistics.mean(quality_scores)
            },
            "confidence_analysis": {
                "mean_confidence": statistics.mean(confidence_scores),
                "median_confidence": statistics.median(confidence_scores),
                "min_confidence": min(confidence_scores),
                "max_confidence": max(confidence_scores),
                "std_confidence": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
            },
            "content_analysis": {
                "mean_word_count": statistics.mean(word_counts),
                "median_word_count": statistics.median(word_counts),
                "clips_with_questions": question_clips,
                "clips_with_educator_language": educator_lang_clips,
                "question_percentage": (question_clips / len(self.metrics)) * 100,
                "educator_language_percentage": (educator_lang_clips / len(self.metrics)) * 100
            },
            "speaker_analysis": {
                "clips_with_speaker_id": speaker_id_clips,
                "speaker_identification_rate": (speaker_id_clips / len(self.metrics)) * 100,
                "note": "Low speaker identification expected due to PyAnnote diarization issues"
            },
            "question_type_distribution": question_types,
            "educational_indicators": indicator_counts,
            "quality_tiers": {
                "high_quality_clips": high_quality,
                "medium_quality_clips": medium_quality,
                "low_quality_clips": low_quality,
                "high_quality_percentage": (high_quality / len(self.metrics)) * 100
            },
            "ml_readiness_assessment": {
                "transcription_quality": "Excellent" if statistics.mean(confidence_scores) > 0.6 else "Good",
                "educational_content_coverage": "Excellent" if educator_lang_clips > len(self.metrics) * 0.8 else "Good",
                "dataset_completeness": f"{len(self.metrics)}/105 clips processed",
                "recommended_for_training": statistics.mean(quality_scores) > 0.5
            },
            "detailed_metrics": [
                {
                    "clip_id": m.clip_id,
                    "confidence": m.confidence_score,
                    "quality_score": m.quality_score,
                    "word_count": m.word_count,
                    "has_question": m.has_question,
                    "question_type": m.question_type,
                    "educational_indicators": m.educational_indicators
                }
                for m in sorted(self.metrics, key=lambda x: x.quality_score, reverse=True)
            ][:10]  # Top 10 clips by quality
        }

    def save_quality_report(self, report: Dict[str, Any]) -> None:
        """Save quality report to file"""

        # Create validation directory if it doesn't exist
        validation_dir = Path("data/transcripts/validation")
        validation_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed JSON report
        report_file = validation_dir / "transcript_quality_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save summary text report
        summary_file = validation_dir / "transcript_quality_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            self.write_summary_report(f, report)

        logger.info(f"Quality reports saved: {report_file}, {summary_file}")

    def write_summary_report(self, f, report: Dict[str, Any]) -> None:
        """Write human-readable summary report"""

        f.write("="*70 + "\n")
        f.write("           TRANSCRIPT QUALITY VALIDATION REPORT\n")
        f.write("                     Issue #89 Analysis\n")
        f.write("="*70 + "\n\n")

        summary = report.get('summary', {})
        f.write(f"📊 OVERALL SUMMARY\n")
        f.write(f"  Total Transcripts Analyzed: {summary.get('total_transcripts', 0)}\n")
        f.write(f"  Overall Quality Score: {summary.get('overall_quality_score', 0):.3f}/1.000\n")
        f.write(f"  Analysis Timestamp: {summary.get('validation_timestamp', 'unknown')}\n\n")

        confidence = report.get('confidence_analysis', {})
        f.write(f"🎙️ TRANSCRIPTION CONFIDENCE\n")
        f.write(f"  Mean Confidence: {confidence.get('mean_confidence', 0):.3f}\n")
        f.write(f"  Range: {confidence.get('min_confidence', 0):.3f} - {confidence.get('max_confidence', 0):.3f}\n\n")

        content = report.get('content_analysis', {})
        f.write(f"📚 EDUCATIONAL CONTENT\n")
        f.write(f"  Clips with Questions: {content.get('question_percentage', 0):.1f}%\n")
        f.write(f"  Clips with Educator Language: {content.get('educator_language_percentage', 0):.1f}%\n")
        f.write(f"  Average Words per Clip: {content.get('mean_word_count', 0):.1f}\n\n")

        quality = report.get('quality_tiers', {})
        f.write(f"⭐ QUALITY DISTRIBUTION\n")
        f.write(f"  High Quality (≥0.7): {quality.get('high_quality_percentage', 0):.1f}%\n")
        f.write(f"  Medium Quality (0.4-0.7): {quality.get('medium_quality_clips', 0)} clips\n")
        f.write(f"  Low Quality (<0.4): {quality.get('low_quality_clips', 0)} clips\n\n")

        ml_ready = report.get('ml_readiness_assessment', {})
        f.write(f"🤖 ML TRAINING READINESS\n")
        f.write(f"  Transcription Quality: {ml_ready.get('transcription_quality', 'Unknown')}\n")
        f.write(f"  Educational Coverage: {ml_ready.get('educational_content_coverage', 'Unknown')}\n")
        f.write(f"  Dataset Completeness: {ml_ready.get('dataset_completeness', 'Unknown')}\n")
        f.write(f"  Recommended for Training: {'✅ YES' if ml_ready.get('recommended_for_training') else '❌ NO'}\n\n")

        f.write("="*70 + "\n")
        f.write("Report generated by Transcript Quality Validator\n")
        f.write("Issue #89 - Transcription & Speaker Diarization Pipeline\n")
        f.write("="*70 + "\n")


def main():
    """Main validation execution"""

    print("🔍 Transcript Quality Validation - Issue #89")
    print("=" * 50)

    validator = TranscriptQualityValidator()

    try:
        report = validator.validate_all_transcripts()

        if "error" in report:
            print(f"❌ Error: {report['error']}")
            return 1

        # Display key metrics
        summary = report.get('summary', {})
        confidence = report.get('confidence_analysis', {})
        content = report.get('content_analysis', {})
        quality = report.get('quality_tiers', {})

        print(f"\n📊 VALIDATION RESULTS:")
        print(f"  Transcripts Analyzed: {summary.get('total_transcripts', 0)}")
        print(f"  Overall Quality: {summary.get('overall_quality_score', 0):.3f}/1.000")
        print(f"  Mean Confidence: {confidence.get('mean_confidence', 0):.3f}")
        print(f"  Educational Content: {content.get('educator_language_percentage', 0):.1f}%")
        print(f"  High Quality Clips: {quality.get('high_quality_percentage', 0):.1f}%")

        ml_ready = report.get('ml_readiness_assessment', {})
        print(f"\n🤖 ML TRAINING READINESS:")
        print(f"  Status: {'✅ READY' if ml_ready.get('recommended_for_training') else '⚠️ NEEDS REVIEW'}")
        print(f"  Quality: {ml_ready.get('transcription_quality', 'Unknown')}")

        print(f"\n📄 Reports saved to: data/transcripts/validation/")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ Validation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)