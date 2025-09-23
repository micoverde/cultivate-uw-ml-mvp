"""
Data preprocessing module for educator-child interaction analysis.

Handles video and text preprocessing with focus on open-ended question
detection and conversational depth analysis.
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConversationAnalyzer:
    """Analyzes conversation patterns for education quality indicators."""

    def __init__(self):
        """Initialize conversation analyzer with research-based patterns."""
        # Open-ended question patterns based on research
        self.open_ended_patterns = [
            r'\bwhy\s+(?:do|did|does|would|could|should)\b',
            r'\bhow\s+(?:do|did|does|would|could|should)\b',
            r'\bwhat\s+(?:if|would happen|do you think)\b',
            r'\bwhat\s+makes?\b',
            r'\bwhat.*different\b',
            r'\btell me (?:about|more)\b',
            r'\bexplain\b',
            r'\bdescribe\b',
            r'\bwhat.*notice\b',
            r'\bwhat.*wonder\b'
        ]

        # Closed-ended question patterns (to contrast)
        self.closed_ended_patterns = [
            r'\b(?:is|are|was|were|do|does|did|can|could|will|would)\s+\w+\?',
            r'\byes\s+or\s+no\b',
            r'\bwhat\s+color\b',
            r'\bhow\s+many\b',
            r'\bwhere\s+is\b'
        ]

        # Conversation depth indicators
        self.depth_indicators = [
            r'\bbecause\b',
            r'\bso\s+that\b',
            r'\bthat\s+reminds\s+me\b',
            r'\blike\s+when\b',
            r'\bwhat\s+about\b',
            r'\band\s+then\b',
            r'\bbut\s+what\s+if\b'
        ]

    def analyze_transcript(self, transcript: str) -> Dict:
        """Analyze transcript for education quality indicators.

        Args:
            transcript: Conversation transcript

        Returns:
            Dictionary of analysis results
        """
        # Split into speaker turns
        turns = self._parse_speaker_turns(transcript)

        # Analyze each turn
        educator_turns = [turn for turn in turns if turn['speaker'] == 'educator']
        child_turns = [turn for turn in turns if turn['speaker'] == 'child']

        # Calculate metrics
        analysis = {
            'total_turns': len(turns),
            'educator_turns': len(educator_turns),
            'child_turns': len(child_turns),
            'turn_ratio': len(child_turns) / max(len(educator_turns), 1),
            'open_ended_questions': self._count_open_ended_questions(educator_turns),
            'closed_ended_questions': self._count_closed_ended_questions(educator_turns),
            'conversational_depth_score': self._calculate_depth_score(turns),
            'average_turn_length': np.mean([len(turn['text'].split()) for turn in turns]),
            'question_quality_ratio': 0,  # Will calculate below
            'conversation_patterns': self._analyze_patterns(turns)
        }

        # Calculate question quality ratio
        total_questions = analysis['open_ended_questions'] + analysis['closed_ended_questions']
        if total_questions > 0:
            analysis['question_quality_ratio'] = analysis['open_ended_questions'] / total_questions

        return analysis

    def _parse_speaker_turns(self, transcript: str) -> List[Dict]:
        """Parse transcript into speaker turns.

        Args:
            transcript: Raw transcript text

        Returns:
            List of turn dictionaries
        """
        turns = []
        lines = transcript.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for speaker indicators
            if ':' in line:
                speaker_part, text = line.split(':', 1)
                speaker = speaker_part.strip().lower()

                # Normalize speaker identification
                if any(word in speaker for word in ['teacher', 'educator', 'adult']):
                    speaker = 'educator'
                elif any(word in speaker for word in ['child', 'student', 'kid']):
                    speaker = 'child'
                else:
                    speaker = 'unknown'

                turns.append({
                    'speaker': speaker,
                    'text': text.strip(),
                    'word_count': len(text.strip().split())
                })

        return turns

    def _count_open_ended_questions(self, educator_turns: List[Dict]) -> int:
        """Count open-ended questions in educator turns."""
        count = 0
        for turn in educator_turns:
            text = turn['text'].lower()
            for pattern in self.open_ended_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    count += 1
                    break  # Count each turn only once
        return count

    def _count_closed_ended_questions(self, educator_turns: List[Dict]) -> int:
        """Count closed-ended questions in educator turns."""
        count = 0
        for turn in educator_turns:
            text = turn['text'].lower()
            # Count question marks but exclude open-ended patterns
            if '?' in text:
                is_open_ended = any(re.search(pattern, text, re.IGNORECASE)
                                  for pattern in self.open_ended_patterns)
                if not is_open_ended:
                    count += 1
        return count

    def _calculate_depth_score(self, turns: List[Dict]) -> float:
        """Calculate conversational depth score based on research indicators."""
        depth_score = 0
        total_turns = len(turns)

        if total_turns == 0:
            return 0

        # Look for depth indicators
        for turn in turns:
            text = turn['text'].lower()
            for pattern in self.depth_indicators:
                if re.search(pattern, text, re.IGNORECASE):
                    depth_score += 1

        # Normalize by number of turns
        return depth_score / total_turns

    def _analyze_patterns(self, turns: List[Dict]) -> Dict:
        """Analyze conversation patterns for quality indicators."""
        patterns = {
            'back_and_forth_sequences': 0,
            'child_initiated_topics': 0,
            'educator_follow_up_questions': 0,
            'extended_conversations': 0
        }

        # Analyze sequences of turns
        for i in range(len(turns) - 1):
            current_turn = turns[i]
            next_turn = turns[i + 1]

            # Back-and-forth pattern
            if (current_turn['speaker'] == 'child' and
                next_turn['speaker'] == 'educator'):
                patterns['back_and_forth_sequences'] += 1

            # Educator follow-up after child response
            if (current_turn['speaker'] == 'child' and
                next_turn['speaker'] == 'educator' and
                '?' in next_turn['text']):
                patterns['educator_follow_up_questions'] += 1

        # Extended conversations (3+ consecutive turns on same topic)
        # This would require more sophisticated topic analysis

        return patterns


class QualityAnnotator:
    """Annotates interactions with research-based quality indicators."""

    def __init__(self):
        """Initialize quality annotator with CLASS framework indicators."""
        # CLASS Instructional Support indicators
        self.class_indicators = {
            'concept_development': {
                'analysis_reasoning': 0,
                'creating_suggesting': 0,
                'integration': 0,
                'connections_links': 0
            },
            'quality_feedback': {
                'scaffolding': 0,
                'encouraging_effort': 0,
                'specific_information': 0,
                'back_and_forth': 0
            },
            'language_modeling': {
                'frequent_conversations': 0,
                'open_ended_questions': 0,
                'repetition_extension': 0,
                'advanced_language': 0
            }
        }

    def annotate_interaction(self, analysis: Dict, transcript: str) -> Dict:
        """Generate quality annotations based on analysis results.

        Args:
            analysis: Results from conversation analysis
            transcript: Original transcript

        Returns:
            Quality annotation scores and recommendations
        """
        annotations = {
            'overall_score': 0,
            'class_scores': self.class_indicators.copy(),
            'strengths': [],
            'improvement_areas': [],
            'recommendations': [],
            'research_alignment': {}
        }

        # Calculate CLASS scores based on analysis
        self._score_language_modeling(annotations, analysis)
        self._score_quality_feedback(annotations, analysis)
        self._score_concept_development(annotations, analysis)

        # Generate overall score (average of domains)
        domain_scores = []
        for domain, indicators in annotations['class_scores'].items():
            domain_score = np.mean(list(indicators.values()))
            domain_scores.append(domain_score)
        annotations['overall_score'] = np.mean(domain_scores)

        # Generate recommendations
        self._generate_recommendations(annotations, analysis)

        return annotations

    def _score_language_modeling(self, annotations: Dict, analysis: Dict):
        """Score language modeling based on conversation analysis."""
        scores = annotations['class_scores']['language_modeling']

        # Open-ended questions score
        question_ratio = analysis.get('question_quality_ratio', 0)
        scores['open_ended_questions'] = min(question_ratio * 7, 7)  # Scale to 1-7

        # Frequent conversations score
        turn_ratio = analysis.get('turn_ratio', 0)
        scores['frequent_conversations'] = min(turn_ratio * 3.5, 7)

        # Back and forth score
        back_forth = analysis.get('conversation_patterns', {}).get('back_and_forth_sequences', 0)
        total_turns = analysis.get('total_turns', 1)
        back_forth_ratio = back_forth / total_turns
        scores['back_and_forth'] = min(back_forth_ratio * 14, 7)

    def _score_quality_feedback(self, annotations: Dict, analysis: Dict):
        """Score quality feedback indicators."""
        scores = annotations['class_scores']['quality_feedback']

        # Follow-up questions indicate scaffolding
        follow_ups = analysis.get('conversation_patterns', {}).get('educator_follow_up_questions', 0)
        educator_turns = analysis.get('educator_turns', 1)
        follow_up_ratio = follow_ups / educator_turns
        scores['scaffolding'] = min(follow_up_ratio * 14, 7)

        # Back and forth indicates quality feedback
        back_forth = analysis.get('conversation_patterns', {}).get('back_and_forth_sequences', 0)
        total_turns = analysis.get('total_turns', 1)
        scores['back_and_forth'] = min((back_forth / total_turns) * 14, 7)

    def _score_concept_development(self, annotations: Dict, analysis: Dict):
        """Score concept development indicators."""
        scores = annotations['class_scores']['concept_development']

        # Depth score indicates analysis and reasoning
        depth_score = analysis.get('conversational_depth_score', 0)
        scores['analysis_reasoning'] = depth_score * 7

        # Open-ended questions support creating and suggesting
        open_ended = analysis.get('open_ended_questions', 0)
        total_questions = (analysis.get('open_ended_questions', 0) +
                          analysis.get('closed_ended_questions', 0))
        if total_questions > 0:
            scores['creating_suggesting'] = (open_ended / total_questions) * 7

    def _generate_recommendations(self, annotations: Dict, analysis: Dict):
        """Generate evidence-based recommendations for improvement."""
        recommendations = annotations['recommendations']
        strengths = annotations['strengths']
        improvements = annotations['improvement_areas']

        # Analyze open-ended question usage
        question_ratio = analysis.get('question_quality_ratio', 0)
        if question_ratio > 0.6:
            strengths.append("Excellent use of open-ended questions to promote thinking")
        elif question_ratio > 0.3:
            recommendations.append("Increase open-ended questions like 'How do you think...' or 'Why might...'")
        else:
            improvements.append("Limited use of open-ended questions")
            recommendations.append("Replace yes/no questions with 'how' and 'why' questions")

        # Analyze conversation balance
        turn_ratio = analysis.get('turn_ratio', 0)
        if turn_ratio > 0.8:
            strengths.append("Good balance of child and educator talk time")
        elif turn_ratio < 0.3:
            improvements.append("Low child participation in conversation")
            recommendations.append("Use wait time and follow-up questions to encourage child responses")

        # Analyze conversational depth
        depth_score = analysis.get('conversational_depth_score', 0)
        if depth_score > 0.3:
            strengths.append("Conversations show good depth and complexity")
        else:
            recommendations.append("Build on child responses with phrases like 'Tell me more about...'")


if __name__ == "__main__":
    # Example usage
    analyzer = ConversationAnalyzer()
    annotator = QualityAnnotator()

    sample_transcript = """
    Teacher: What are you building with those blocks?
    Child: A castle!
    Teacher: Why did you choose to make a castle?
    Child: Because I like princesses and they live in castles.
    Teacher: That's interesting! How do you think the princess gets to the top of the castle?
    Child: She climbs the stairs!
    Teacher: What else might help her get up high?
    """

    analysis = analyzer.analyze_transcript(sample_transcript)
    annotations = annotator.annotate_interaction(analysis, sample_transcript)

    print("Analysis Results:")
    print(json.dumps(analysis, indent=2))
    print("\nQuality Annotations:")
    print(json.dumps(annotations, indent=2))