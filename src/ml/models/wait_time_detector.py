#!/usr/bin/env python3
"""
Wait Time Detection for Educational Analysis
Analyzes pause appropriateness in educator-child interactions.

Author: Claude-4 (Partner-Level Microsoft SDE)
Issue: #47 - Story 2.2: Question Quality Analysis Implementation
"""

from typing import Dict, Any, List, Tuple, Optional
import re
import logging
import os
import joblib
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseWaitTimeDetector(ABC):
    """Base class for wait time detection models"""

    @abstractmethod
    async def analyze(self, transcript: str) -> Dict[str, Any]:
        """Analyze transcript for wait time patterns"""
        pass

class WaitTimeDetector:
    """
    Wait time detector supporting multiple analysis approaches.

    Classical: Pattern-based analysis using transcript structure and linguistic cues
    Future AR: Real-time audio analysis for actual pause detection
    """

    def __init__(self, model_type: str = 'classical'):
        self.model_type = model_type
        self.version = f"{model_type}_v1.0"

        if model_type == 'classical':
            self.detector = ClassicalWaitTimeDetector()
        elif model_type == 'audio_realtime':
            # TODO: Implement for future AR deployment
            self.detector = AudioRealtimeWaitTimeDetector()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def analyze(self, transcript: str) -> Dict[str, Any]:
        """Main analysis entry point"""
        return await self.detector.analyze(transcript)

class ClassicalWaitTimeDetector(BaseWaitTimeDetector):
    """
    Classical wait time analysis using transcript structure and linguistic patterns.
    Now enhanced with trained SVM model for improved accuracy.

    Analyzes:
    - Question-response timing patterns
    - Child thinking indicators
    - Educator patience markers
    - Conversation flow appropriateness
    """

    def __init__(self, model_path: str = None):
        # Load trained model if available
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'trained', 'wait_time_detector_model.pkl')

        self.trained_model = None
        self.feature_extractor = None

        try:
            if os.path.exists(model_path):
                self.trained_model = joblib.load(model_path)
                # Load feature extractor for trained model
                extractor_path = os.path.join(os.path.dirname(model_path), 'feature_extractor.pkl')
                if os.path.exists(extractor_path):
                    with open(extractor_path, 'rb') as f:
                        import pickle
                        self.feature_extractor = pickle.load(f)
                logger.info(f"Loaded trained wait time detector from {model_path}")
            else:
                logger.warning(f"Trained model not found at {model_path}, using rule-based fallback")
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            self.trained_model = None
        # Educational research indicates 3-5 seconds optimal wait time
        self.optimal_wait_indicators = [
            # Child processing time markers
            r'\b(um+|uh+|hmm+|well)\b',
            r'\blet\s+me\s+think',
            r'\bhold\s+on',
            r'\bi\s+need\s+to\s+think',
            r'\b\.\.\.\s*$',  # Ellipses indicating pause

            # Thoughtful response beginnings
            r'\bi\s+think\s+(that|maybe)',
            r'\bmaybe\s+(it|that|this)',
            r'\bit\s+might\s+be',
        ]

        self.insufficient_wait_indicators = [
            # Immediate educator follow-up
            r'Teacher:\s*.+\?\s*\n\s*Teacher:',  # Teacher asking another question immediately
            r'Educator:\s*.+\?\s*\n\s*Educator:',

            # Rushed responses
            r'Child:\s*(yes|no|okay)\s*\n\s*Teacher:',  # Very short child responses
            r'Student:\s*(i\s+don\'t\s+know)\s*\n\s*Teacher:',

            # Interruption patterns
            r'Child:\s*.+\-\-\s*Teacher:',  # Interrupted child speech
            r'Student:\s*.+\-\-\s*Educator:',
        ]

        self.appropriate_wait_indicators = [
            # Patient educator responses
            r'Child:\s*.+\n\s*Teacher:\s*(great|good|that\'s\s+interesting)',
            r'Student:\s*.+\n\s*Educator:\s*(tell\s+me\s+more|what\s+else)',

            # Building on child responses
            r'Child:\s*.+\n\s*Teacher:\s*(and\s+what|what\s+about)',
            r'Student:\s*.+\n\s*Educator:\s*(how\s+did|why\s+do\s+you\s+think)',

            # Encouraging elaboration
            r'Child:\s*.+\n\s*Teacher:\s*(can\s+you\s+explain|tell\s+me\s+more)',
        ]

        self.complexity_modifiers = {
            # More complex questions need more wait time
            'high_complexity': [r'\bwhy\s+do\s+you\s+think', r'\bhow\s+would\s+you', r'\bwhat\s+if'],
            'medium_complexity': [r'\bwhat\s+do\s+you', r'\bhow\s+does', r'\bwhat\s+happens'],
            'low_complexity': [r'\bis\s+this', r'\bdo\s+you\s+see', r'\bwhat\s+color']
        }

    async def analyze(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze wait time patterns in educational transcript.
        Uses trained SVM model when available, falls back to rule-based analysis.

        Returns comprehensive wait time assessment for demo reliability.
        """
        try:
            # Try trained model first
            if self.trained_model is not None and self.feature_extractor is not None:
                trained_result = self._analyze_with_trained_model(transcript)
                if trained_result is not None:
                    return trained_result

            # Fallback to rule-based analysis
            logger.info("Using rule-based wait time analysis")
            return await self._analyze_rule_based(transcript)

        except Exception as e:
            logger.error(f"Wait time analysis failed: {e}")
            return self._fallback_analysis_sync(transcript)

    def _analyze_with_trained_model(self, transcript: str) -> Optional[Dict[str, Any]]:
        """
        Analyze using trained SVM model.
        Returns None if analysis fails, triggering rule-based fallback.
        """
        try:
            # Extract features for trained model
            features = self.feature_extractor.extract_features_from_transcript(transcript)

            # Get prediction from trained SVM
            prediction = self.trained_model.predict([features])[0]
            probabilities = self.trained_model.predict_proba([features])[0]

            # Map prediction to wait time categories (based on training data)
            # 0: No wait time issues, 1: Some wait time issues, 2: Significant wait time issues
            wait_time_labels = {
                0: "appropriate",
                1: "needs_improvement",
                2: "insufficient"
            }

            # Convert to appropriateness score (higher is better)
            appropriateness_mapping = {0: 0.85, 1: 0.65, 2: 0.35}
            appropriateness_score = appropriateness_mapping[prediction]

            # Add some variance based on confidence
            max_prob = max(probabilities)
            confidence_adjustment = (max_prob - 0.33) * 0.1  # 0.33 = random chance for 3 classes
            appropriateness_score += confidence_adjustment
            appropriateness_score = max(0.0, min(1.0, appropriateness_score))

            return {
                'model_type': 'trained_svm',
                'appropriateness_score': float(appropriateness_score),
                'predicted_category': wait_time_labels[prediction],
                'category_probabilities': {
                    'appropriate': float(probabilities[0]),
                    'needs_improvement': float(probabilities[1]),
                    'insufficient': float(probabilities[2])
                },
                'confidence': float(max_prob),
                'total_questions_analyzed': transcript.count('?'),
                'analysis_method': 'machine_learning'
            }

        except Exception as e:
            logger.warning(f"Trained model analysis failed: {e}")
            return None

    async def _analyze_rule_based(self, transcript: str) -> Dict[str, Any]:
        """
        Original rule-based analysis method.
        """
        # Extract question-response pairs
        qa_pairs = self._extract_qa_pairs(transcript)
        logger.debug(f"Extracted {len(qa_pairs)} question-answer pairs")

        # Analyze each interaction
        wait_time_analyses = []
        for pair in qa_pairs:
            analysis = self._analyze_wait_time_interaction(pair)
            wait_time_analyses.append(analysis)

        # Aggregate analysis
        result = self._aggregate_wait_time_analysis(wait_time_analyses, transcript)
        result['model_type'] = 'rule_based'
        result['analysis_method'] = 'pattern_matching'
        return result

    def _extract_qa_pairs(self, transcript: str) -> List[Dict[str, Any]]:
        """Extract question-answer interaction pairs"""
        lines = transcript.split('\n')
        qa_pairs = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or ':' not in line:
                i += 1
                continue

            speaker, content = line.split(':', 1)
            content = content.strip()

            # Check if this is a question from educator
            if ('teacher' in speaker.lower() or 'educator' in speaker.lower()) and '?' in content:
                # Look for child response
                child_response = None
                next_educator_line = None

                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if ':' in next_line:
                        next_speaker, next_content = next_line.split(':', 1)
                        next_content = next_content.strip()

                        if 'child' in next_speaker.lower() or 'student' in next_speaker.lower():
                            child_response = {
                                'speaker': next_speaker,
                                'content': next_content,
                                'line_number': j
                            }
                            # Look for educator follow-up
                            if j + 1 < len(lines):
                                followup_line = lines[j + 1].strip()
                                if ':' in followup_line:
                                    followup_speaker, followup_content = followup_line.split(':', 1)
                                    if ('teacher' in followup_speaker.lower() or
                                        'educator' in followup_speaker.lower()):
                                        next_educator_line = {
                                            'speaker': followup_speaker,
                                            'content': followup_content.strip(),
                                            'line_number': j + 1
                                        }
                            break
                        elif ('teacher' in next_speaker.lower() or 'educator' in next_speaker.lower()):
                            # No child response before next educator line
                            next_educator_line = {
                                'speaker': next_speaker,
                                'content': next_content,
                                'line_number': j
                            }
                            break

                    j += 1

                qa_pairs.append({
                    'question': {
                        'speaker': speaker,
                        'content': content,
                        'line_number': i
                    },
                    'child_response': child_response,
                    'educator_followup': next_educator_line,
                    'has_response': child_response is not None
                })

            i += 1

        return qa_pairs

    def _analyze_wait_time_interaction(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze wait time for single question-answer interaction"""
        question = qa_pair['question']['content']
        child_response = qa_pair.get('child_response')
        educator_followup = qa_pair.get('educator_followup')

        # Assess question complexity (affects needed wait time)
        complexity = self._assess_question_complexity(question)

        # Check for wait time indicators
        wait_time_score = 0.5  # Base score

        if child_response:
            response_content = child_response['content']

            # Positive indicators (child had time to think)
            thinking_indicators = sum(1 for pattern in self.optimal_wait_indicators
                                     if re.search(pattern, response_content, re.IGNORECASE))

            if thinking_indicators > 0:
                wait_time_score += 0.3

            # Check response quality (longer, more thoughtful responses suggest adequate wait time)
            response_length = len(response_content.split())
            if response_length > 5:  # More elaborate response
                wait_time_score += 0.2

            # Check for interrupted speech patterns
            if '--' in response_content or response_content.endswith('...'):
                wait_time_score -= 0.2

        # Check educator behavior
        if educator_followup:
            followup_content = educator_followup['content']

            # Positive: Building on child's response
            if any(re.search(pattern, f"Child: {child_response['content'] if child_response else ''}\nTeacher: {followup_content}", re.IGNORECASE)
                   for pattern in self.appropriate_wait_indicators):
                wait_time_score += 0.2

        # Negative: Insufficient wait time patterns
        full_interaction = f"Teacher: {question}"
        if child_response:
            full_interaction += f"\nChild: {child_response['content']}"
        if educator_followup:
            full_interaction += f"\nTeacher: {educator_followup['content']}"

        if any(re.search(pattern, full_interaction, re.IGNORECASE)
               for pattern in self.insufficient_wait_indicators):
            wait_time_score -= 0.4

        # Adjust based on question complexity
        complexity_adjustment = {
            'high': 0.1,    # High complexity questions need more wait time
            'medium': 0.0,  # Medium complexity is baseline
            'low': -0.1     # Low complexity questions can have shorter wait time
        }
        wait_time_score += complexity_adjustment.get(complexity, 0)

        return {
            'question_text': question,
            'question_complexity': complexity,
            'wait_time_score': max(0.0, min(1.0, wait_time_score)),  # Clamp to [0, 1]
            'has_child_response': child_response is not None,
            'response_indicators': {
                'thinking_markers': child_response and any(
                    re.search(pattern, child_response['content'], re.IGNORECASE)
                    for pattern in self.optimal_wait_indicators
                ) if child_response else False,
                'interrupted_speech': child_response and ('--' in child_response['content'] or
                                                          child_response['content'].endswith('...')) if child_response else False
            }
        }

    def _assess_question_complexity(self, question: str) -> str:
        """Assess cognitive complexity of question"""
        question_lower = question.lower()

        # High complexity patterns
        if any(re.search(pattern, question_lower) for pattern in self.complexity_modifiers['high_complexity']):
            return 'high'

        # Medium complexity patterns
        if any(re.search(pattern, question_lower) for pattern in self.complexity_modifiers['medium_complexity']):
            return 'medium'

        # Low complexity patterns
        if any(re.search(pattern, question_lower) for pattern in self.complexity_modifiers['low_complexity']):
            return 'low'

        # Default to medium if no clear pattern
        return 'medium'

    def _aggregate_wait_time_analysis(
        self,
        analyses: List[Dict[str, Any]],
        transcript: str
    ) -> Dict[str, Any]:
        """Aggregate individual wait time analyses"""

        if not analyses:
            return self._fallback_analysis_sync(transcript)

        # Calculate overall metrics
        wait_time_scores = [analysis['wait_time_score'] for analysis in analyses]
        overall_appropriateness = sum(wait_time_scores) / len(wait_time_scores)

        # Count patterns
        questions_with_responses = sum(1 for a in analyses if a['has_child_response'])
        total_questions = len(analyses)

        # Analyze complexity distribution
        complexity_counts = {}
        for analysis in analyses:
            complexity = analysis['question_complexity']
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # Identify problematic patterns
        insufficient_wait_count = sum(1 for score in wait_time_scores if score < 0.3)
        excellent_wait_count = sum(1 for score in wait_time_scores if score > 0.8)

        return {
            'appropriateness_score': float(overall_appropriateness),
            'total_questions_analyzed': total_questions,
            'questions_with_responses': questions_with_responses,
            'response_rate': questions_with_responses / max(1, total_questions),
            'complexity_distribution': complexity_counts,
            'pattern_analysis': {
                'insufficient_wait_time': insufficient_wait_count,
                'excellent_wait_time': excellent_wait_count,
                'needs_improvement_percentage': (insufficient_wait_count / max(1, total_questions)) * 100
            },
            'individual_analyses': analyses
        }

    def _fallback_analysis_sync(self, transcript: str) -> Dict[str, Any]:
        """Fallback analysis for demo reliability"""
        logger.warning("Using fallback wait time analysis")

        # Simple heuristic based on question-response patterns
        question_count = transcript.count('?')
        child_responses = transcript.lower().count('child:') + transcript.lower().count('student:')

        response_rate = child_responses / max(1, question_count)
        appropriateness_score = min(0.9, 0.5 + response_rate * 0.3)

        return {
            'appropriateness_score': appropriateness_score,
            'total_questions_analyzed': question_count,
            'questions_with_responses': child_responses,
            'response_rate': response_rate,
            'individual_analyses': []
        }

class AudioRealtimeWaitTimeDetector(BaseWaitTimeDetector):
    """
    Real-time audio-based wait time detection for future AR deployment.

    Foundation for <10ms pause detection on Meta Ray-Ban glasses.
    """

    def __init__(self):
        logger.info("Initializing audio real-time wait time detector (future AR foundation)")
        # TODO: Implement audio processing pipeline for real-time pause detection
        self.audio_processor = None

    async def analyze(self, transcript: str) -> Dict[str, Any]:
        """Audio-based analysis (future implementation)"""
        logger.warning("Audio real-time detector not yet implemented, using classical fallback")

        # Use classical detector as fallback for now
        fallback_detector = ClassicalWaitTimeDetector()
        return await fallback_detector.analyze(transcript)