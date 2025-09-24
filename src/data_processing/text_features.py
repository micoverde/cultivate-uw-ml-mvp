#!/usr/bin/env python3
"""
Text Feature Extraction Module for Issue #90

Specialized linguistic feature extraction focused on educational discourse analysis.
Identifies constructivist pedagogy patterns, question classification, and
interaction quality metrics aligned with educational research.

Author: Claude (Issue #90 Implementation)
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import logging
from collections import Counter
from dataclasses import dataclass
import warnings

# NLP libraries with error handling
try:
    import spacy
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    print(f"Missing NLP libraries: {e}")
    print("Install with: pip install spacy nltk textstat scikit-learn")
    print("Run: python -m spacy download en_core_web_sm")

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

@dataclass
class QuestionAnalysis:
    """Detailed analysis of a single question."""
    text: str
    question_type: str  # 'open-ended', 'closed-ended', 'procedural', 'rhetorical'
    complexity_score: float  # 0-1 scale
    cognitive_demand: str  # 'low', 'medium', 'high'
    wait_time_expected: float  # Seconds expected for this question type
    bloom_taxonomy_level: str  # 'remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'
    zpd_alignment: float  # Zone of Proximal Development alignment score
    scaffolding_indicators: List[str]  # Detected scaffolding techniques

class EducationalTextAnalyzer:
    """
    Specialized text analysis for educational interactions.

    Focuses on pedagogical techniques that research shows improve learning:
    - Constructivist questioning patterns
    - Scaffolding and fading techniques
    - Zone of Proximal Development alignment
    - CLASS framework indicators
    """

    def __init__(self):
        # Initialize NLP models
        self.nlp = None
        self.sentiment_analyzer = None

        try:
            self.nlp = spacy.load("en_core_web_sm")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"NLP initialization warning: {e}")

        # Educational research-based patterns
        self._initialize_pedagogical_patterns()

        logger.info("Educational text analyzer initialized")

    def _initialize_pedagogical_patterns(self):
        """Initialize patterns based on educational research."""

        # Open-ended question patterns (constructivist teaching)
        self.open_ended_patterns = [
            r'\b(why|how|what.*think|what.*feel|what.*believe)\b',
            r'\b(explain|describe|tell me about|share your thoughts)\b',
            r'\b(what would happen if|imagine|suppose)\b',
            r'\b(in your opinion|from your perspective)\b',
            r'\b(how might|what could|what other ways)\b'
        ]

        # Closed-ended question patterns
        self.closed_ended_patterns = [
            r'\b(is|are|do|does|did|can|could|would|should|will)\s+\w+.*\?',
            r'\b(yes.*or.*no|true.*or.*false|which one|either.*or)\b',
            r'\b(who|what|when|where)\s+(?!.*think|.*feel|.*believe)\w+.*\?'
        ]

        # Scaffolding language patterns
        self.scaffolding_patterns = {
            'verbal_prompts': [
                r'\b(think about|consider|remember when)\b',
                r'\b(what do you notice|what stands out)\b',
                r'\b(let me help you|let\'s work together)\b'
            ],
            'cognitive_prompts': [
                r'\b(why do you think|what makes you say that)\b',
                r'\b(how did you figure that out|what\'s your reasoning)\b',
                r'\b(what evidence|what supports that idea)\b'
            ],
            'strategic_prompts': [
                r'\b(what strategy|how could we approach)\b',
                r'\b(what would be another way|try a different approach)\b',
                r'\b(break it down|step by step)\b'
            ]
        }

        # ZPD (Zone of Proximal Development) indicators
        self.zpd_patterns = {
            'assessment': [
                r'\b(what do you already know|what have you learned)\b',
                r'\b(show me how you|walk me through)\b',
                r'\b(what\'s familiar|what\'s new here)\b'
            ],
            'guidance': [
                r'\b(let me show you|watch how I|follow along)\b',
                r'\b(we\'ll do this together|let\'s try together)\b',
                r'\b(I\'ll help you with|let me guide you)\b'
            ],
            'independence': [
                r'\b(try it on your own|you can do this|give it a try)\b',
                r'\b(what do you think you should do next)\b',
                r'\b(you\'re ready to|now you try)\b'
            ]
        }

        # Bloom's Taxonomy patterns
        self.bloom_patterns = {
            'remember': [r'\b(recall|remember|list|identify|define)\b'],
            'understand': [r'\b(explain|describe|summarize|compare)\b'],
            'apply': [r'\b(solve|demonstrate|use|apply|calculate)\b'],
            'analyze': [r'\b(analyze|examine|compare|contrast|categorize)\b'],
            'evaluate': [r'\b(evaluate|judge|critique|defend|justify)\b'],
            'create': [r'\b(create|design|compose|generate|construct)\b']
        }

        # CLASS framework indicators
        self.class_patterns = {
            'emotional_support': [
                r'\b(great job|well done|I\'m proud|that\'s wonderful)\b',
                r'\b(how are you feeling|are you okay|don\'t worry)\b',
                r'\b(you can do it|believe in yourself|keep trying)\b'
            ],
            'classroom_organization': [
                r'\b(first we\'ll|next we need to|let\'s start with)\b',
                r'\b(remember our rule|think about our goal)\b',
                r'\b(time to|we have.*minutes|let\'s wrap up)\b'
            ],
            'instructional_support': [
                r'\b(why do you think|what\'s your reasoning|how did you know)\b',
                r'\b(let\'s think about this differently|another way to look at)\b',
                r'\b(connect this to|this reminds me of|similar to)\b'
            ]
        }

    def analyze_complete_transcript(self, transcript: str, speaker_labels: List[Dict]) -> Dict:
        """
        Comprehensive analysis of educational transcript.

        Args:
            transcript: Complete transcript text
            speaker_labels: List of speaker segments with timing

        Returns:
            Dictionary of educational text features
        """

        logger.info("Analyzing complete educational transcript")

        try:
            # Separate teacher and student utterances
            teacher_utterances, student_utterances = self._separate_speakers(speaker_labels)

            # Analyze questioning patterns
            question_analysis = self._analyze_questioning_patterns(teacher_utterances)

            # Analyze scaffolding and ZPD alignment
            scaffolding_analysis = self._analyze_scaffolding_patterns(teacher_utterances)

            # Analyze discourse complexity
            complexity_analysis = self._analyze_discourse_complexity(transcript)

            # Analyze interaction patterns
            interaction_analysis = self._analyze_interaction_patterns(teacher_utterances, student_utterances)

            # Analyze CLASS framework alignment
            class_analysis = self._analyze_class_framework(teacher_utterances)

            # Calculate overall educational quality scores
            quality_scores = self._calculate_educational_quality_scores(
                question_analysis, scaffolding_analysis, interaction_analysis, class_analysis
            )

            return {
                'questions': question_analysis,
                'scaffolding': scaffolding_analysis,
                'complexity': complexity_analysis,
                'interaction': interaction_analysis,
                'class_framework': class_analysis,
                'quality_scores': quality_scores,
                'summary': self._generate_analysis_summary(
                    question_analysis, scaffolding_analysis, quality_scores
                )
            }

        except Exception as e:
            logger.error(f"Error analyzing transcript: {e}")
            return self._get_default_text_analysis()

    def _separate_speakers(self, speaker_labels: List[Dict]) -> Tuple[List[str], List[str]]:
        """Separate teacher and student utterances."""

        teacher_utterances = []
        student_utterances = []

        for segment in speaker_labels:
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', '')

            if text:
                # Assume SPEAKER_00 is teacher
                if speaker == 'SPEAKER_00':
                    teacher_utterances.append(text)
                else:
                    student_utterances.append(text)

        return teacher_utterances, student_utterances

    def _analyze_questioning_patterns(self, teacher_utterances: List[str]) -> Dict:
        """Analyze teacher questioning patterns for constructivist pedagogy."""

        try:
            questions = [utterance for utterance in teacher_utterances if '?' in utterance]

            if not questions:
                return self._get_default_question_analysis()

            question_analyses = []
            for question in questions:
                analysis = self._analyze_single_question(question)
                question_analyses.append(analysis)

            # Aggregate statistics
            open_ended_count = sum(1 for qa in question_analyses if qa.question_type == 'open-ended')
            closed_ended_count = sum(1 for qa in question_analyses if qa.question_type == 'closed-ended')
            total_questions = len(question_analyses)

            avg_complexity = np.mean([qa.complexity_score for qa in question_analyses])
            avg_zpd_alignment = np.mean([qa.zpd_alignment for qa in question_analyses])

            # Cognitive demand distribution
            cognitive_demand_counts = Counter(qa.cognitive_demand for qa in question_analyses)

            # Bloom's taxonomy distribution
            bloom_counts = Counter(qa.bloom_taxonomy_level for qa in question_analyses)

            return {
                'total_questions': total_questions,
                'open_ended_count': open_ended_count,
                'closed_ended_count': closed_ended_count,
                'open_ended_ratio': open_ended_count / total_questions if total_questions > 0 else 0,
                'average_complexity': float(avg_complexity),
                'average_zpd_alignment': float(avg_zpd_alignment),
                'cognitive_demand_distribution': dict(cognitive_demand_counts),
                'bloom_taxonomy_distribution': dict(bloom_counts),
                'question_analyses': [
                    {
                        'text': qa.text,
                        'type': qa.question_type,
                        'complexity': qa.complexity_score,
                        'cognitive_demand': qa.cognitive_demand,
                        'bloom_level': qa.bloom_taxonomy_level,
                        'zpd_alignment': qa.zpd_alignment,
                        'scaffolding_indicators': qa.scaffolding_indicators
                    } for qa in question_analyses
                ],
                'constructivist_teaching_score': self._calculate_constructivist_score(question_analyses)
            }

        except Exception as e:
            logger.error(f"Error analyzing questioning patterns: {e}")
            return self._get_default_question_analysis()

    def _analyze_single_question(self, question: str) -> QuestionAnalysis:
        """Detailed analysis of a single question."""

        try:
            question_clean = question.strip().lower()

            # Classify question type
            question_type = self._classify_question_type(question_clean)

            # Calculate complexity score
            complexity_score = self._calculate_question_complexity(question)

            # Determine cognitive demand
            cognitive_demand = self._determine_cognitive_demand(question_clean)

            # Estimate wait time expectation
            wait_time_expected = self._estimate_expected_wait_time(question_type, complexity_score)

            # Classify Bloom's taxonomy level
            bloom_level = self._classify_bloom_level(question_clean)

            # Calculate ZPD alignment
            zpd_alignment = self._calculate_zpd_alignment(question_clean)

            # Detect scaffolding indicators
            scaffolding_indicators = self._detect_scaffolding_in_question(question_clean)

            return QuestionAnalysis(
                text=question,
                question_type=question_type,
                complexity_score=complexity_score,
                cognitive_demand=cognitive_demand,
                wait_time_expected=wait_time_expected,
                bloom_taxonomy_level=bloom_level,
                zpd_alignment=zpd_alignment,
                scaffolding_indicators=scaffolding_indicators
            )

        except Exception as e:
            logger.warning(f"Error analyzing question '{question}': {e}")
            return QuestionAnalysis(
                text=question,
                question_type='unknown',
                complexity_score=0.5,
                cognitive_demand='medium',
                wait_time_expected=3.0,
                bloom_taxonomy_level='understand',
                zpd_alignment=0.5,
                scaffolding_indicators=[]
            )

    def _classify_question_type(self, question: str) -> str:
        """Classify question as open-ended, closed-ended, procedural, or rhetorical."""

        # Check for open-ended patterns
        open_score = sum(1 for pattern in self.open_ended_patterns if re.search(pattern, question))

        # Check for closed-ended patterns
        closed_score = sum(1 for pattern in self.closed_ended_patterns if re.search(pattern, question))

        # Check for procedural questions
        procedural_patterns = [r'\b(how do we|what step|what should we do next)\b']
        procedural_score = sum(1 for pattern in procedural_patterns if re.search(pattern, question))

        # Check for rhetorical questions
        rhetorical_patterns = [r'\b(isn\'t that|don\'t you think|wouldn\'t you agree)\b']
        rhetorical_score = sum(1 for pattern in rhetorical_patterns if re.search(pattern, question))

        # Determine primary type
        scores = {
            'open-ended': open_score,
            'closed-ended': closed_score,
            'procedural': procedural_score,
            'rhetorical': rhetorical_score
        }

        max_score = max(scores.values())
        if max_score == 0:
            return 'unknown'

        return max(scores.keys(), key=lambda k: scores[k])

    def _calculate_question_complexity(self, question: str) -> float:
        """Calculate question complexity based on linguistic features."""

        try:
            # Word count complexity
            word_count = len(question.split())
            word_complexity = min(1.0, word_count / 15)  # Normalize to 0-1

            # Syntactic complexity
            syntactic_complexity = 0.0
            if self.nlp:
                doc = self.nlp(question)
                # Count complex syntactic structures
                complex_deps = ['advcl', 'acl', 'ccomp', 'xcomp', 'relcl']
                complex_count = sum(1 for token in doc if token.dep_ in complex_deps)
                syntactic_complexity = min(1.0, complex_count / 3)

            # Vocabulary complexity (simplified)
            complex_words = ['analyze', 'evaluate', 'synthesize', 'compare', 'contrast', 'justify', 'critique']
            vocab_complexity = sum(1 for word in complex_words if word in question.lower()) / len(complex_words)

            # Combined complexity score
            complexity = (word_complexity * 0.3 + syntactic_complexity * 0.4 + vocab_complexity * 0.3)
            return float(max(0, min(1, complexity)))

        except Exception as e:
            logger.warning(f"Error calculating question complexity: {e}")
            return 0.5

    def _determine_cognitive_demand(self, question: str) -> str:
        """Determine cognitive demand level based on question content."""

        high_demand_indicators = [
            'why', 'how', 'analyze', 'evaluate', 'compare', 'justify',
            'what if', 'suppose', 'imagine', 'create', 'design'
        ]

        low_demand_indicators = [
            'what is', 'who is', 'when', 'where', 'list', 'name',
            'recall', 'remember', 'identify'
        ]

        high_score = sum(1 for indicator in high_demand_indicators if indicator in question)
        low_score = sum(1 for indicator in low_demand_indicators if indicator in question)

        if high_score > low_score:
            return 'high'
        elif low_score > high_score:
            return 'low'
        else:
            return 'medium'

    def _estimate_expected_wait_time(self, question_type: str, complexity: float) -> float:
        """Estimate appropriate wait time for question based on type and complexity."""

        base_wait_times = {
            'open-ended': 5.0,
            'closed-ended': 2.0,
            'procedural': 3.0,
            'rhetorical': 1.0,
            'unknown': 3.0
        }

        base_time = base_wait_times.get(question_type, 3.0)
        complexity_modifier = 1 + (complexity * 2)  # 1x to 3x multiplier

        return float(base_time * complexity_modifier)

    def _classify_bloom_level(self, question: str) -> str:
        """Classify question according to Bloom's taxonomy."""

        for level, patterns in self.bloom_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return level

        # Default classification based on question words
        if any(word in question for word in ['why', 'how', 'explain']):
            return 'understand'
        elif any(word in question for word in ['what if', 'create', 'design']):
            return 'create'
        elif any(word in question for word in ['compare', 'analyze']):
            return 'analyze'
        else:
            return 'remember'

    def _calculate_zpd_alignment(self, question: str) -> float:
        """Calculate how well question aligns with ZPD principles."""

        zpd_score = 0.0
        total_categories = len(self.zpd_patterns)

        for category, patterns in self.zpd_patterns.items():
            category_score = sum(1 for pattern in patterns if re.search(pattern, question))
            if category_score > 0:
                zpd_score += 1

        return zpd_score / total_categories if total_categories > 0 else 0.0

    def _detect_scaffolding_in_question(self, question: str) -> List[str]:
        """Detect scaffolding techniques within a question."""

        scaffolding_detected = []

        for technique, patterns in self.scaffolding_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    scaffolding_detected.append(technique)
                    break

        return scaffolding_detected

    def _analyze_scaffolding_patterns(self, teacher_utterances: List[str]) -> Dict:
        """Analyze scaffolding and fading techniques in teacher discourse."""

        try:
            all_text = ' '.join(teacher_utterances).lower()

            scaffolding_counts = {}
            total_scaffolding = 0

            for technique, patterns in self.scaffolding_patterns.items():
                count = sum(len(re.findall(pattern, all_text)) for pattern in patterns)
                scaffolding_counts[technique] = count
                total_scaffolding += count

            # Calculate scaffolding density
            total_words = len(all_text.split())
            scaffolding_density = total_scaffolding / total_words if total_words > 0 else 0

            # Analyze ZPD alignment
            zpd_scores = {}
            for category, patterns in self.zpd_patterns.items():
                score = sum(len(re.findall(pattern, all_text)) for pattern in patterns)
                zpd_scores[category] = score

            return {
                'scaffolding_counts': scaffolding_counts,
                'total_scaffolding_instances': total_scaffolding,
                'scaffolding_density': float(scaffolding_density),
                'zpd_alignment_scores': zpd_scores,
                'zpd_total_score': sum(zpd_scores.values()),
                'constructivist_pedagogy_score': self._calculate_constructivist_pedagogy_score(
                    scaffolding_counts, zpd_scores
                )
            }

        except Exception as e:
            logger.error(f"Error analyzing scaffolding patterns: {e}")
            return self._get_default_scaffolding_analysis()

    def _analyze_discourse_complexity(self, transcript: str) -> Dict:
        """Analyze overall discourse complexity and readability."""

        try:
            if not transcript.strip():
                return self._get_default_complexity_analysis()

            # Basic statistics
            word_count = len(transcript.split())
            sentence_count = len(sent_tokenize(transcript))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

            # Vocabulary diversity (Type-Token Ratio)
            words = [word.lower() for word in word_tokenize(transcript) if word.isalpha()]
            unique_words = set(words)
            ttr = len(unique_words) / len(words) if words else 0

            # Readability scores
            try:
                flesch_score = flesch_reading_ease(transcript)
                fk_grade = flesch_kincaid_grade(transcript)
            except:
                flesch_score = 50.0  # Average
                fk_grade = 8.0

            # Academic vocabulary usage
            academic_words = self._count_academic_vocabulary(words)
            academic_ratio = academic_words / len(words) if words else 0

            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': float(avg_sentence_length),
                'type_token_ratio': float(ttr),
                'flesch_reading_ease': float(flesch_score),
                'flesch_kincaid_grade': float(fk_grade),
                'academic_vocabulary_ratio': float(academic_ratio),
                'complexity_score': self._calculate_overall_complexity(
                    avg_sentence_length, ttr, flesch_score, academic_ratio
                )
            }

        except Exception as e:
            logger.error(f"Error analyzing discourse complexity: {e}")
            return self._get_default_complexity_analysis()

    def _analyze_interaction_patterns(self, teacher_utterances: List[str], student_utterances: List[str]) -> Dict:
        """Analyze interaction patterns between teacher and students."""

        try:
            total_utterances = len(teacher_utterances) + len(student_utterances)

            if total_utterances == 0:
                return self._get_default_interaction_analysis()

            # Basic interaction ratios
            teacher_ratio = len(teacher_utterances) / total_utterances
            student_ratio = len(student_utterances) / total_utterances

            # Word count analysis
            teacher_words = sum(len(utterance.split()) for utterance in teacher_utterances)
            student_words = sum(len(utterance.split()) for utterance in student_utterances)
            total_words = teacher_words + student_words

            teacher_word_ratio = teacher_words / total_words if total_words > 0 else 0
            student_word_ratio = student_words / total_words if total_words > 0 else 0

            # Analyze student response quality
            response_quality = self._analyze_student_response_quality(student_utterances)

            # Analyze turn-taking patterns
            turn_taking = self._analyze_turn_taking_effectiveness(teacher_ratio, student_ratio)

            return {
                'teacher_utterance_ratio': float(teacher_ratio),
                'student_utterance_ratio': float(student_ratio),
                'teacher_word_ratio': float(teacher_word_ratio),
                'student_word_ratio': float(student_word_ratio),
                'total_turns': total_utterances,
                'avg_teacher_utterance_length': teacher_words / len(teacher_utterances) if teacher_utterances else 0,
                'avg_student_utterance_length': student_words / len(student_utterances) if student_utterances else 0,
                'student_response_quality': response_quality,
                'turn_taking_effectiveness': turn_taking,
                'interaction_balance_score': self._calculate_interaction_balance_score(
                    teacher_ratio, student_ratio, response_quality
                )
            }

        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {e}")
            return self._get_default_interaction_analysis()

    def _analyze_class_framework(self, teacher_utterances: List[str]) -> Dict:
        """Analyze alignment with CLASS framework dimensions."""

        try:
            all_text = ' '.join(teacher_utterances).lower()

            class_scores = {}

            for dimension, patterns in self.class_patterns.items():
                score = sum(len(re.findall(pattern, all_text)) for pattern in patterns)
                # Normalize by text length
                total_words = len(all_text.split())
                normalized_score = (score / total_words * 1000) if total_words > 0 else 0
                class_scores[dimension] = float(min(1.0, normalized_score))

            # Calculate overall CLASS score
            overall_class_score = sum(class_scores.values()) / len(class_scores) if class_scores else 0

            return {
                'emotional_support_score': class_scores.get('emotional_support', 0),
                'classroom_organization_score': class_scores.get('classroom_organization', 0),
                'instructional_support_score': class_scores.get('instructional_support', 0),
                'overall_class_score': float(overall_class_score),
                'class_dimension_balance': self._calculate_class_balance(class_scores)
            }

        except Exception as e:
            logger.error(f"Error analyzing CLASS framework: {e}")
            return self._get_default_class_analysis()

    def _calculate_educational_quality_scores(self, questions: Dict, scaffolding: Dict,
                                            interactions: Dict, class_framework: Dict) -> Dict:
        """Calculate overall educational quality scores."""

        try:
            # Constructivist teaching score
            constructivist_score = (
                questions.get('constructivist_teaching_score', 0.5) * 0.4 +
                scaffolding.get('constructivist_pedagogy_score', 0.5) * 0.4 +
                interactions.get('interaction_balance_score', 0.5) * 0.2
            )

            # Student engagement score
            engagement_score = (
                interactions.get('student_response_quality', 0.5) * 0.5 +
                interactions.get('turn_taking_effectiveness', 0.5) * 0.3 +
                (interactions.get('student_utterance_ratio', 0) * 2) * 0.2  # Normalize to 0-1
            )

            # Overall teaching effectiveness
            teaching_effectiveness = (
                constructivist_score * 0.3 +
                engagement_score * 0.3 +
                class_framework.get('overall_class_score', 0.5) * 0.4
            )

            return {
                'constructivist_teaching_score': float(max(0, min(1, constructivist_score))),
                'student_engagement_score': float(max(0, min(1, engagement_score))),
                'teaching_effectiveness_score': float(max(0, min(1, teaching_effectiveness))),
                'zpd_alignment_score': scaffolding.get('zpd_total_score', 0) / 10,  # Normalize
                'question_quality_score': questions.get('average_complexity', 0.5)
            }

        except Exception as e:
            logger.error(f"Error calculating quality scores: {e}")
            return {
                'constructivist_teaching_score': 0.5,
                'student_engagement_score': 0.5,
                'teaching_effectiveness_score': 0.5,
                'zpd_alignment_score': 0.5,
                'question_quality_score': 0.5
            }

    # Helper methods for calculations
    def _calculate_constructivist_score(self, question_analyses: List[QuestionAnalysis]) -> float:
        """Calculate constructivist teaching score based on question analysis."""

        if not question_analyses:
            return 0.5

        # Weight open-ended questions highly
        open_ended_ratio = sum(1 for qa in question_analyses if qa.question_type == 'open-ended') / len(question_analyses)
        avg_complexity = np.mean([qa.complexity_score for qa in question_analyses])
        high_cognitive_ratio = sum(1 for qa in question_analyses if qa.cognitive_demand == 'high') / len(question_analyses)

        score = (open_ended_ratio * 0.4 + avg_complexity * 0.3 + high_cognitive_ratio * 0.3)
        return float(max(0, min(1, score)))

    def _calculate_constructivist_pedagogy_score(self, scaffolding_counts: Dict, zpd_scores: Dict) -> float:
        """Calculate overall constructivist pedagogy score."""

        # Normalize scaffolding counts
        total_scaffolding = sum(scaffolding_counts.values())
        scaffolding_score = min(1.0, total_scaffolding / 10)  # Arbitrary normalization

        # Normalize ZPD scores
        total_zpd = sum(zpd_scores.values())
        zpd_score = min(1.0, total_zpd / 5)  # Arbitrary normalization

        return float((scaffolding_score + zpd_score) / 2)

    def _count_academic_vocabulary(self, words: List[str]) -> int:
        """Count academic vocabulary words."""

        academic_words = {
            'analyze', 'evaluate', 'synthesize', 'compare', 'contrast', 'justify',
            'hypothesize', 'demonstrate', 'illustrate', 'interpret', 'examine',
            'investigate', 'construct', 'develop', 'establish', 'determine'
        }

        return sum(1 for word in words if word in academic_words)

    def _calculate_overall_complexity(self, avg_sentence_length: float, ttr: float,
                                    flesch_score: float, academic_ratio: float) -> float:
        """Calculate overall discourse complexity score."""

        # Normalize components
        sentence_complexity = min(1.0, avg_sentence_length / 20)
        vocab_complexity = min(1.0, ttr * 2)  # TTR is typically 0-0.5
        readability_complexity = (100 - flesch_score) / 100  # Invert Flesch score
        academic_complexity = min(1.0, academic_ratio * 10)

        complexity = (
            sentence_complexity * 0.25 +
            vocab_complexity * 0.25 +
            readability_complexity * 0.25 +
            academic_complexity * 0.25
        )

        return float(max(0, min(1, complexity)))

    def _analyze_student_response_quality(self, student_utterances: List[str]) -> float:
        """Analyze quality of student responses."""

        if not student_utterances:
            return 0.0

        # Length-based quality (longer responses often indicate engagement)
        avg_length = np.mean([len(utterance.split()) for utterance in student_utterances])
        length_score = min(1.0, avg_length / 10)  # Normalize

        # Content-based quality (presence of reasoning words)
        reasoning_words = ['because', 'since', 'therefore', 'however', 'although', 'but', 'so']
        reasoning_count = sum(
            sum(1 for word in reasoning_words if word in utterance.lower())
            for utterance in student_utterances
        )
        reasoning_score = min(1.0, reasoning_count / len(student_utterances))

        return float((length_score + reasoning_score) / 2)

    def _analyze_turn_taking_effectiveness(self, teacher_ratio: float, student_ratio: float) -> float:
        """Analyze effectiveness of turn-taking patterns."""

        # Ideal ratio is roughly 60% teacher, 40% student for guided instruction
        ideal_teacher_ratio = 0.6
        ideal_student_ratio = 0.4

        teacher_deviation = abs(teacher_ratio - ideal_teacher_ratio)
        student_deviation = abs(student_ratio - ideal_student_ratio)

        # Score based on deviation from ideal
        effectiveness = 1.0 - ((teacher_deviation + student_deviation) / 2)
        return float(max(0, effectiveness))

    def _calculate_interaction_balance_score(self, teacher_ratio: float, student_ratio: float,
                                           response_quality: float) -> float:
        """Calculate overall interaction balance score."""

        turn_taking_score = self._analyze_turn_taking_effectiveness(teacher_ratio, student_ratio)

        # Combine turn-taking and response quality
        balance_score = (turn_taking_score * 0.6 + response_quality * 0.4)
        return float(max(0, min(1, balance_score)))

    def _calculate_class_balance(self, class_scores: Dict) -> float:
        """Calculate balance across CLASS framework dimensions."""

        if not class_scores:
            return 0.5

        scores = list(class_scores.values())

        # Balance is good when all dimensions are present (low variance)
        balance = 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.5
        return float(max(0, min(1, balance)))

    def _generate_analysis_summary(self, questions: Dict, scaffolding: Dict, quality_scores: Dict) -> Dict:
        """Generate human-readable analysis summary."""

        try:
            summary = {
                'total_questions': questions.get('total_questions', 0),
                'open_ended_percentage': round(questions.get('open_ended_ratio', 0) * 100, 1),
                'constructivist_score': round(quality_scores.get('constructivist_teaching_score', 0), 2),
                'teaching_effectiveness': round(quality_scores.get('teaching_effectiveness_score', 0), 2),
                'key_strengths': [],
                'areas_for_improvement': [],
                'recommendations': []
            }

            # Identify strengths
            if questions.get('open_ended_ratio', 0) > 0.6:
                summary['key_strengths'].append("High use of open-ended questions")

            if scaffolding.get('constructivist_pedagogy_score', 0) > 0.7:
                summary['key_strengths'].append("Strong scaffolding techniques")

            if quality_scores.get('student_engagement_score', 0) > 0.7:
                summary['key_strengths'].append("Good student engagement")

            # Identify areas for improvement
            if questions.get('open_ended_ratio', 0) < 0.4:
                summary['areas_for_improvement'].append("Increase open-ended questioning")
                summary['recommendations'].append("Try using 'Why do you think...' and 'How would you...' questions")

            if scaffolding.get('zpd_total_score', 0) < 3:
                summary['areas_for_improvement'].append("Enhance ZPD alignment")
                summary['recommendations'].append("Provide more guided practice before independent work")

            return summary

        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return {'error': 'Could not generate summary'}

    # Default return methods
    def _get_default_text_analysis(self) -> Dict:
        """Return default analysis when processing fails."""
        return {
            'questions': self._get_default_question_analysis(),
            'scaffolding': self._get_default_scaffolding_analysis(),
            'complexity': self._get_default_complexity_analysis(),
            'interaction': self._get_default_interaction_analysis(),
            'class_framework': self._get_default_class_analysis(),
            'quality_scores': {
                'constructivist_teaching_score': 0.5,
                'student_engagement_score': 0.5,
                'teaching_effectiveness_score': 0.5,
                'zpd_alignment_score': 0.5,
                'question_quality_score': 0.5
            },
            'summary': {'error': 'Analysis failed'}
        }

    def _get_default_question_analysis(self) -> Dict:
        return {
            'total_questions': 0,
            'open_ended_count': 0,
            'closed_ended_count': 0,
            'open_ended_ratio': 0.0,
            'average_complexity': 0.5,
            'average_zpd_alignment': 0.5,
            'cognitive_demand_distribution': {},
            'bloom_taxonomy_distribution': {},
            'question_analyses': [],
            'constructivist_teaching_score': 0.5
        }

    def _get_default_scaffolding_analysis(self) -> Dict:
        return {
            'scaffolding_counts': {},
            'total_scaffolding_instances': 0,
            'scaffolding_density': 0.0,
            'zpd_alignment_scores': {},
            'zpd_total_score': 0,
            'constructivist_pedagogy_score': 0.5
        }

    def _get_default_complexity_analysis(self) -> Dict:
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0.0,
            'type_token_ratio': 0.0,
            'flesch_reading_ease': 50.0,
            'flesch_kincaid_grade': 8.0,
            'academic_vocabulary_ratio': 0.0,
            'complexity_score': 0.5
        }

    def _get_default_interaction_analysis(self) -> Dict:
        return {
            'teacher_utterance_ratio': 0.0,
            'student_utterance_ratio': 0.0,
            'teacher_word_ratio': 0.0,
            'student_word_ratio': 0.0,
            'total_turns': 0,
            'avg_teacher_utterance_length': 0.0,
            'avg_student_utterance_length': 0.0,
            'student_response_quality': 0.0,
            'turn_taking_effectiveness': 0.5,
            'interaction_balance_score': 0.5
        }

    def _get_default_class_analysis(self) -> Dict:
        return {
            'emotional_support_score': 0.5,
            'classroom_organization_score': 0.5,
            'instructional_support_score': 0.5,
            'overall_class_score': 0.5,
            'class_dimension_balance': 0.5
        }


def extract_educational_text_features(transcript: str, speaker_labels: List[Dict]) -> Dict:
    """
    Main function to extract all educational text features from transcript data.

    Args:
        transcript: Complete transcript text
        speaker_labels: List of speaker segments with timing and text

    Returns:
        Dictionary of educational text features
    """

    analyzer = EducationalTextAnalyzer()
    return analyzer.analyze_complete_transcript(transcript, speaker_labels)