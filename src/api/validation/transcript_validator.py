#!/usr/bin/env python3
"""
Transcript Validation Logic

Advanced validation for educator-child interaction transcripts.
Ensures transcripts meet educational research standards and ML requirements.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #46 - Story 2.1: Submit educator transcripts for analysis
"""

from typing import List, Dict, Set, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    level: ValidationLevel
    code: str
    message: str
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    issues: List[ValidationIssue]
    metadata: Dict[str, Any]
    score: float  # 0-1 quality score

class TranscriptValidator:
    """
    Comprehensive validator for educator interaction transcripts.
    Implements educational research standards and ML data quality requirements.
    """

    # Configuration constants
    MIN_LENGTH = 100
    MAX_LENGTH = 5000
    MIN_TURNS = 2
    RECOMMENDED_MIN_TURNS = 4
    MIN_WORDS_PER_TURN = 2
    MAX_TURN_LENGTH = 500

    # Speaker patterns for recognition
    EDUCATOR_PATTERNS = [
        r'\b(teacher|educator|adult|instructor|miss|mr|mrs|ms)\s*:',
        r'\b[A-Z][a-z]+\s*(teacher|educator)\s*:',
    ]

    CHILD_PATTERNS = [
        r'\b(child|student|kid|learner|boy|girl)\s*:',
        r'\b[A-Z][a-z]+\s*(age\s*\d+?|child|student)\s*:',
    ]

    # PII patterns to detect
    PII_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
        r'\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b',  # Credit card
    ]

    # Educational quality indicators
    QUESTION_INDICATORS = [
        r'\?',  # Question marks
        r'\b(what|who|when|where|why|how|which|can you|do you|will you)\b',
        r'\b(tell me about|describe|explain|think about)\b',
    ]

    SCAFFOLDING_INDICATORS = [
        r'\b(let\'s try|what if|maybe we could|how about|let me help)\b',
        r'\b(good thinking|nice try|almost|close|you\'re right that)\b',
        r'\b(build on that|add to that|expand on)\b',
    ]

    def __init__(self):
        """Initialize validator with compiled regex patterns for performance"""
        self.educator_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.EDUCATOR_PATTERNS]
        self.child_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.CHILD_PATTERNS]
        self.pii_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.PII_PATTERNS]
        self.question_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.QUESTION_INDICATORS]
        self.scaffolding_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.SCAFFOLDING_INDICATORS]

    def validate(self, transcript: str) -> ValidationResult:
        """
        Perform comprehensive transcript validation.
        Returns detailed validation result with errors, warnings, and quality metrics.
        """
        issues = []
        metadata = {}

        # Basic sanitization and preparation
        transcript = transcript.strip()
        lines = [line.strip() for line in transcript.split('\n') if line.strip()]

        # Extract conversational turns
        turns = self._extract_turns(transcript)
        speakers = self._identify_speakers(turns)

        # Core validation checks
        issues.extend(self._validate_length(transcript))
        issues.extend(self._validate_structure(transcript, turns))
        issues.extend(self._validate_speakers(speakers))
        issues.extend(self._validate_turn_quality(turns))
        issues.extend(self._check_privacy(transcript))

        # Educational quality assessment
        quality_metrics = self._assess_educational_quality(transcript, turns)
        metadata.update(quality_metrics)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(issues, quality_metrics)

        # Extract basic statistics
        metadata.update({
            'character_count': len(transcript),
            'word_count': len(transcript.split()),
            'line_count': len(lines),
            'turn_count': len(turns),
            'unique_speakers': len(speakers),
            'avg_turn_length': sum(len(turn['content'].split()) for turn in turns) / max(1, len(turns)),
            'quality_score': quality_score
        })

        # Separate errors and warnings
        errors = [issue.message for issue in issues if issue.level == ValidationLevel.ERROR]
        warnings = [issue.message for issue in issues if issue.level == ValidationLevel.WARNING]

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            issues=issues,
            metadata=metadata,
            score=quality_score
        )

    def _extract_turns(self, transcript: str) -> List[Dict[str, str]]:
        """Extract conversational turns with speaker identification"""
        turns = []

        # Split by lines and look for speaker patterns
        lines = transcript.split('\n')

        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue

            # Split on first colon to separate speaker from content
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                content = parts[1].strip()

                if content:  # Only include turns with actual content
                    turns.append({
                        'speaker': speaker,
                        'content': content,
                        'word_count': len(content.split())
                    })

        return turns

    def _identify_speakers(self, turns: List[Dict[str, str]]) -> Set[str]:
        """Identify unique speakers from turns"""
        return {turn['speaker'] for turn in turns}

    def _validate_length(self, transcript: str) -> List[ValidationIssue]:
        """Validate transcript length requirements"""
        issues = []
        length = len(transcript)

        if length < self.MIN_LENGTH:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="LENGTH_TOO_SHORT",
                message=f"Transcript too short. Minimum {self.MIN_LENGTH} characters required, got {length}",
                suggestion="Add more conversational content between educator and child"
            ))
        elif length > self.MAX_LENGTH:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="LENGTH_TOO_LONG",
                message=f"Transcript too long. Maximum {self.MAX_LENGTH} characters allowed, got {length}",
                suggestion="Consider focusing on a specific interaction segment"
            ))

        return issues

    def _validate_structure(self, transcript: str, turns: List[Dict[str, str]]) -> List[ValidationIssue]:
        """Validate transcript structure and format"""
        issues = []

        # Check minimum number of turns
        if len(turns) < self.MIN_TURNS:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="INSUFFICIENT_TURNS",
                message=f"Too few conversational turns. Minimum {self.MIN_TURNS} required, found {len(turns)}",
                suggestion="Include more back-and-forth conversation between speakers"
            ))
        elif len(turns) < self.RECOMMENDED_MIN_TURNS:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="FEW_TURNS",
                message=f"Few conversational turns. Recommended minimum {self.RECOMMENDED_MIN_TURNS}, found {len(turns)}",
                suggestion="More conversational turns provide richer analysis"
            ))

        # Check for speaker format
        if not turns:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="NO_SPEAKER_FORMAT",
                message="No speaker labels found. Use format 'Speaker: content'",
                suggestion="Format each line as 'Teacher: Hello there!' or 'Child: Hi!'"
            ))

        return issues

    def _validate_speakers(self, speakers: Set[str]) -> List[ValidationIssue]:
        """Validate speaker identification and diversity"""
        issues = []

        if len(speakers) < 2:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                code="INSUFFICIENT_SPEAKERS",
                message="At least 2 different speakers required for interaction analysis",
                suggestion="Include both educator and child voices in the conversation"
            ))

        # Check for educator presence
        has_educator = any(
            any(regex.search(speaker) for regex in self.educator_regex)
            for speaker in speakers
        )

        # Check for child presence
        has_child = any(
            any(regex.search(speaker) for regex in self.child_regex)
            for speaker in speakers
        )

        if not has_educator and not has_child:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                code="UNCLEAR_ROLES",
                message="Could not clearly identify educator and child roles",
                suggestion="Use clear labels like 'Teacher:', 'Educator:', 'Child:', or 'Student:'"
            ))

        return issues

    def _validate_turn_quality(self, turns: List[Dict[str, str]]) -> List[ValidationIssue]:
        """Validate individual turn quality"""
        issues = []

        for i, turn in enumerate(turns):
            word_count = turn['word_count']

            if word_count < self.MIN_WORDS_PER_TURN:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="SHORT_TURN",
                    message=f"Turn {i+1} is very short ({word_count} words)",
                    suggestion="Longer turns provide more context for analysis"
                ))
            elif word_count > self.MAX_TURN_LENGTH:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="LONG_TURN",
                    message=f"Turn {i+1} is very long ({word_count} words)",
                    suggestion="Consider breaking long monologues into shorter exchanges"
                ))

        return issues

    def _check_privacy(self, transcript: str) -> List[ValidationIssue]:
        """Check for potential privacy issues and PII"""
        issues = []

        for pattern_regex in self.pii_regex:
            matches = pattern_regex.findall(transcript)
            if matches:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="POTENTIAL_PII",
                    message=f"Potential personal information detected: {len(matches)} instances",
                    suggestion="Remove or redact personal information before analysis"
                ))
                break

        return issues

    def _assess_educational_quality(self, transcript: str, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """Assess educational quality indicators"""
        metrics = {}

        # Count questions
        question_count = sum(
            len(regex.findall(transcript))
            for regex in self.question_regex
        )

        # Count scaffolding indicators
        scaffolding_count = sum(
            len(regex.findall(transcript))
            for regex in self.scaffolding_regex
        )

        # Calculate ratios
        total_turns = len(turns)
        metrics.update({
            'question_count': question_count,
            'scaffolding_indicators': scaffolding_count,
            'questions_per_turn': question_count / max(1, total_turns),
            'scaffolding_per_turn': scaffolding_count / max(1, total_turns),
            'avg_questions_per_100_words': (question_count / max(1, len(transcript.split()))) * 100
        })

        return metrics

    def _calculate_quality_score(self, issues: List[ValidationIssue], quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1)"""
        base_score = 1.0

        # Deduct for validation issues
        for issue in issues:
            if issue.level == ValidationLevel.ERROR:
                base_score -= 0.3
            elif issue.level == ValidationLevel.WARNING:
                base_score -= 0.1

        # Adjust for educational quality indicators
        question_bonus = min(0.2, quality_metrics.get('questions_per_turn', 0) * 0.1)
        scaffolding_bonus = min(0.1, quality_metrics.get('scaffolding_per_turn', 0) * 0.05)

        final_score = base_score + question_bonus + scaffolding_bonus
        return max(0.0, min(1.0, final_score))

    def get_validation_suggestions(self, result: ValidationResult) -> List[str]:
        """Generate actionable validation suggestions"""
        suggestions = []

        for issue in result.issues:
            if issue.suggestion and issue.level in [ValidationLevel.ERROR, ValidationLevel.WARNING]:
                suggestions.append(issue.suggestion)

        # Add general quality suggestions
        if result.score < 0.7:
            suggestions.append("Consider including more open-ended questions and scaffolding techniques")

        return list(set(suggestions))  # Remove duplicates