#!/usr/bin/env python3
"""
IMMERSIVE ML PIPELINE LEARNING ASSESSMENT SYSTEM
=================================================
Designed by Dr. Chris Dede, Timothy E. Wirth Professor at Harvard Graduate School of Education

This system transforms traditional assessments into engaging "Mission Control" experiences
where learners are training to become ML Engineers. Assessment feels like achievement,
not evaluation.

The 8 Stages of the ML Pipeline Journey:
1. Data Collection - "The Explorer"
2. Data Cleaning - "The Purifier"
3. Feature Engineering - "The Architect"
4. Model Selection - "The Strategist"
5. Training - "The Trainer"
6. Evaluation - "The Analyst"
7. Hyperparameter Tuning - "The Optimizer"
8. Deployment - "The Commander"

Author: Dr. Chris Dede (Harvard GSE) / Claude-4 Implementation
Context: Cultivate Learning ML MVP - Educational Assessment System
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from pathlib import Path

# Rich terminal formatting for immersive experience
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for enhanced terminal formatting: pip install rich")


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class DifficultyLevel(Enum):
    """Adaptive difficulty levels based on learner performance"""
    NOVICE = 1        # Just starting, needs encouragement
    APPRENTICE = 2    # Getting the hang of it
    PRACTITIONER = 3  # Solid understanding
    EXPERT = 4        # Deep mastery
    MASTER = 5        # Teaching-level understanding


class BadgeType(Enum):
    """Achievement badge types earned through the journey"""
    STAGE_COMPLETE = "stage_complete"
    PERFECT_MISSION = "perfect_mission"
    SPEED_DEMON = "speed_demon"
    COMEBACK_KID = "comeback_kid"
    DEEP_DIVER = "deep_diver"
    HELPER = "helper"
    CERTIFIED_ENGINEER = "certified_engineer"


@dataclass
class Badge:
    """Achievement badge with metadata"""
    badge_type: BadgeType
    name: str
    description: str
    icon: str
    earned_at: str = field(default_factory=lambda: datetime.now().isoformat())
    stage: int = 0


@dataclass
class LearnerProfile:
    """Complete learner profile tracking progress across the ML journey"""
    learner_id: str
    callsign: str  # Mission Control callsign
    current_stage: int = 1
    difficulty_level: DifficultyLevel = DifficultyLevel.NOVICE

    # Stage mastery (0.0 to 1.0 for each stage)
    stage_mastery: Dict[int, float] = field(default_factory=lambda: {i: 0.0 for i in range(1, 9)})

    # Performance tracking
    total_questions_answered: int = 0
    correct_answers: int = 0
    streak: int = 0
    longest_streak: int = 0

    # Badges earned
    badges: List[Badge] = field(default_factory=list)

    # Time tracking
    total_time_spent: float = 0.0
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())

    # Adaptive learning data
    struggle_topics: List[str] = field(default_factory=list)
    strength_topics: List[str] = field(default_factory=list)

    # Certification status
    certified: bool = False
    certification_date: Optional[str] = None


@dataclass
class ConversationalQuestion:
    """A question disguised as a natural conversation in Mission Control"""
    stage: int
    topic: str
    difficulty: DifficultyLevel

    # The conversational prompt (not "Question 1:")
    scenario: str  # The situation/context
    prompt: str    # What Mission Control asks

    # Multiple choice presented as options, not A/B/C/D
    options: List[str]
    correct_index: int

    # Conversational feedback
    success_response: str   # What to say when correct
    struggle_response: str  # What to say when incorrect (encouraging!)

    # Deep dive for curious learners
    deep_dive: str

    # Related topics for adaptive learning
    related_topics: List[str] = field(default_factory=list)


# =============================================================================
# MISSION CONTROL NARRATIVE ENGINE
# =============================================================================

class MissionControlNarrative:
    """
    The narrative engine that transforms assessment into an immersive experience.
    Every interaction feels like you're training at ML Mission Control.
    """

    STAGE_NAMES = {
        1: ("Data Collection", "The Explorer", "Gathering raw materials for our ML mission"),
        2: ("Data Cleaning", "The Purifier", "Ensuring data quality and integrity"),
        3: ("Feature Engineering", "The Architect", "Building the foundation of predictions"),
        4: ("Model Selection", "The Strategist", "Choosing the right tool for the mission"),
        5: ("Training", "The Trainer", "Teaching machines to learn from patterns"),
        6: ("Evaluation", "The Analyst", "Measuring success and finding weaknesses"),
        7: ("Hyperparameter Tuning", "The Optimizer", "Fine-tuning for peak performance"),
        8: ("Deployment", "The Commander", "Launching models into the real world"),
    }

    MISSION_CONTROL_INTROS = [
        "Mission Control here. We've got a situation that needs your expertise.",
        "Attention, Engineer. A new challenge has arrived at the station.",
        "Commander, we need your input on a critical decision.",
        "The team is counting on you for this one, Engineer.",
        "Incoming transmission from the ML Operations Center.",
    ]

    ENCOURAGEMENT_PHRASES = [
        "Outstanding work, Engineer! The team is impressed.",
        "That's exactly right! Your training is paying off.",
        "Excellent analysis! You're thinking like a senior engineer now.",
        "Perfect execution! Mission Control couldn't have done it better.",
        "Brilliant insight! The ML pipeline is in good hands with you.",
    ]

    STRUGGLE_SUPPORT = [
        "Not quite, but you're on the right track. Let me help you see it differently...",
        "Close! Even senior engineers wrestle with this concept. Here's another way to think about it...",
        "That's a common misconception - you're learning exactly what you need to. Let's look at this together...",
        "Good thinking, but there's a twist here. The key insight is...",
        "Almost there! This catches a lot of people. The thing to remember is...",
    ]

    def __init__(self, console: Optional['Console'] = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)

    def get_stage_info(self, stage: int) -> Tuple[str, str, str]:
        """Get stage name, title, and description"""
        return self.STAGE_NAMES.get(stage, ("Unknown", "Mystery", "Exploring new territory"))

    def generate_mission_intro(self, learner: LearnerProfile, stage: int) -> str:
        """Generate an immersive mission introduction"""
        stage_name, title, desc = self.get_stage_info(stage)
        intro = random.choice(self.MISSION_CONTROL_INTROS)

        return f"""
{intro}

Welcome to Stage {stage}: {stage_name}
Your role: {title}
Mission: {desc}

{learner.callsign}, your current mastery level: {learner.difficulty_level.name}
Mission streak: {learner.streak} successful operations
"""

    def format_success(self, message: str) -> str:
        """Format a success message with encouragement"""
        encouragement = random.choice(self.ENCOURAGEMENT_PHRASES)
        return f"{encouragement}\n\n{message}"

    def format_struggle(self, message: str) -> str:
        """Format a struggle response with support (never 'wrong')"""
        support = random.choice(self.STRUGGLE_SUPPORT)
        return f"{support}\n\n{message}"


# =============================================================================
# CONVERSATIONAL QUESTION BANKS
# =============================================================================

class MLPipelineQuestionBank:
    """
    Question bank for all 8 stages of the ML pipeline.
    Questions are designed as natural conversations, not formal quizzes.
    Multiple difficulty levels for adaptive learning.
    """

    def __init__(self):
        self.questions = self._build_question_bank()

    def _build_question_bank(self) -> Dict[int, List[ConversationalQuestion]]:
        """Build the complete question bank for all stages"""
        return {
            1: self._stage1_data_collection(),
            2: self._stage2_data_cleaning(),
            3: self._stage3_feature_engineering(),
            4: self._stage4_model_selection(),
            5: self._stage5_training(),
            6: self._stage6_evaluation(),
            7: self._stage7_hyperparameter_tuning(),
            8: self._stage8_deployment(),
        }

    def get_question(self, stage: int, difficulty: DifficultyLevel,
                     exclude_topics: List[str] = None) -> ConversationalQuestion:
        """Get a question appropriate for the learner's level"""
        stage_questions = self.questions.get(stage, [])

        # Filter by difficulty (allow one level above/below for variety)
        appropriate_questions = [
            q for q in stage_questions
            if abs(q.difficulty.value - difficulty.value) <= 1
        ]

        # Exclude topics already mastered
        if exclude_topics:
            appropriate_questions = [
                q for q in appropriate_questions
                if q.topic not in exclude_topics
            ]

        if not appropriate_questions:
            appropriate_questions = stage_questions

        return random.choice(appropriate_questions) if appropriate_questions else None

    def _stage1_data_collection(self) -> List[ConversationalQuestion]:
        """Stage 1: Data Collection - The Explorer"""
        return [
            ConversationalQuestion(
                stage=1,
                topic="data_sources",
                difficulty=DifficultyLevel.NOVICE,
                scenario="""
Our early childhood education team needs to analyze educator-child interactions.
We have access to video recordings, audio transcripts, and observation notes.
The team is debating which data source to prioritize.
""",
                prompt="Engineer, given our goal of understanding questioning patterns, which data source would you recommend we focus on first?",
                options=[
                    "Video recordings alone - we can see everything happening",
                    "Audio transcripts - they capture the actual words and pauses",
                    "Observation notes - human observers already did the analysis",
                    "Multiple sources combined - each captures different aspects",
                ],
                correct_index=3,
                success_response="Exactly right! Multi-modal data gives us the richest picture. Video shows body language, audio captures prosody and timing, and transcripts give us the words. Together, they tell the complete story.",
                struggle_response="While single sources have value, combining them gives us complementary information. Think of it like a detective using multiple types of evidence - each piece adds to the full picture.",
                deep_dive="In educational ML, multi-modal learning is becoming the gold standard. Research shows that combining audio, video, and text can improve classification accuracy by 15-30% compared to single modality approaches.",
                related_topics=["multimodal_learning", "data_fusion"]
            ),
            ConversationalQuestion(
                stage=1,
                topic="data_quality",
                difficulty=DifficultyLevel.APPRENTICE,
                scenario="""
We've collected 26 video recordings from different classrooms. However,
some videos have poor audio quality, and 3 were recorded with different camera angles.
The team is eager to start training immediately.
""",
                prompt="Before we dive into model training, what concerns should we address about this dataset?",
                options=[
                    "No concerns - 26 videos is enough to start training immediately",
                    "The different camera angles could introduce bias we need to account for",
                    "Poor audio quality could affect transcript accuracy and downstream analysis",
                    "Both the audio quality AND camera angle variations need to be documented and potentially addressed",
                ],
                correct_index=3,
                success_response="Excellent systems thinking! Both issues could introduce noise or bias. Documenting these variations helps us interpret results correctly and decide whether preprocessing can help.",
                struggle_response="Consider how each variation might affect our ML pipeline. Poor audio = noisy transcripts. Different angles = different visual features. Both matter for model reliability.",
                deep_dive="Data quality issues compound through the pipeline. A 10% error in transcription can lead to 15-20% degradation in downstream NLP tasks. Always assess quality BEFORE model development.",
                related_topics=["data_quality", "bias_detection", "preprocessing"]
            ),
            ConversationalQuestion(
                stage=1,
                topic="sampling_bias",
                difficulty=DifficultyLevel.PRACTITIONER,
                scenario="""
Our dataset of 119 expert annotations comes from 26 videos. We notice that 18 of these
videos are from preschool classrooms, 5 from toddler rooms, and only 3 from kindergarten.
The team wants to build a 'universal' question classifier.
""",
                prompt="What sampling issue should we flag before claiming our model works universally?",
                options=[
                    "119 annotations is statistically sufficient for any claim",
                    "The class imbalance between preschool/toddler/kindergarten may cause the model to underperform on underrepresented groups",
                    "The sample size per video matters more than total annotations",
                    "Age group shouldn't affect questioning patterns at all",
                ],
                correct_index=1,
                success_response="Precisely! This is a classic sampling bias issue. A model trained primarily on preschool data may not generalize well to toddler or kindergarten contexts where interaction patterns differ.",
                struggle_response="Think about what the model 'sees' during training. If 69% of examples are preschool, the model learns preschool patterns best. This is called sampling bias - the model reflects the data's distribution.",
                deep_dive="In educational AI, age-appropriate calibration is crucial. Research shows linguistic complexity and interaction patterns vary significantly by developmental stage. Always stratify evaluation by key demographics.",
                related_topics=["sampling_bias", "stratification", "generalization"]
            ),
            ConversationalQuestion(
                stage=1,
                topic="privacy_considerations",
                difficulty=DifficultyLevel.EXPERT,
                scenario="""
A partner organization wants to share their classroom video dataset with us.
The videos show children's faces and include audio of real conversations.
They're offering 500 additional videos that could dramatically improve our model.
""",
                prompt="Before accepting this data, what ethical and legal considerations must we address?",
                options=[
                    "Accept immediately - more data always improves models",
                    "Require parental consent verification, IRB approval, and data handling agreements",
                    "Just anonymize by blurring faces in post-processing",
                    "Decline all external data to avoid complications",
                ],
                correct_index=1,
                success_response="Outstanding ethical awareness! Working with children's data requires IRB approval, verified parental consent, proper data handling agreements, and secure storage protocols. Ethical AI starts with ethical data collection.",
                struggle_response="When working with minors, consent and privacy aren't optional - they're legal requirements. FERPA, COPPA, and institutional IRB policies all apply. More data isn't worth ethical or legal violations.",
                deep_dive="Educational AI carries special responsibilities. Beyond legal compliance (FERPA, COPPA), consider: data minimization, purpose limitation, retention policies, and the right to be forgotten. Build ethics into your pipeline from day one.",
                related_topics=["privacy", "ethics", "consent", "FERPA", "COPPA"]
            ),
        ]

    def _stage2_data_cleaning(self) -> List[ConversationalQuestion]:
        """Stage 2: Data Cleaning - The Purifier"""
        return [
            ConversationalQuestion(
                stage=2,
                topic="missing_values",
                difficulty=DifficultyLevel.NOVICE,
                scenario="""
We're preparing our educator response dataset for training. Looking at the spreadsheet,
I notice that 15 out of 119 rows are missing the 'wait_time' measurement.
The team is unsure how to proceed.
""",
                prompt="How should we handle these missing wait_time values?",
                options=[
                    "Delete all 15 rows - we can't use incomplete data",
                    "Fill them all with zeros - no wait time recorded means no wait",
                    "Investigate why they're missing and choose an appropriate strategy based on the reason",
                    "Use the average wait time to fill all missing values",
                ],
                correct_index=2,
                success_response="Perfect approach! Missing data has a story. Maybe the audio was unclear, maybe the educator interrupted, maybe it wasn't applicable. Understanding the 'why' guides the right solution.",
                struggle_response="Different types of 'missingness' need different solutions. Data missing because of audio issues (random) is different from data missing because the educator never paused (informative). Investigate first!",
                deep_dive="Missing data mechanisms: MCAR (completely random), MAR (random given other variables), MNAR (not random - the missingness tells you something). Your imputation strategy should match the mechanism.",
                related_topics=["missing_data", "imputation", "data_investigation"]
            ),
            ConversationalQuestion(
                stage=2,
                topic="outlier_detection",
                difficulty=DifficultyLevel.APPRENTICE,
                scenario="""
Looking at our question length feature, most values are between 5-30 words.
However, we have three entries with 150+ words and two with just 1 word.
These extremes are pulling our statistics.
""",
                prompt="These extreme values could be outliers. What's your recommendation?",
                options=[
                    "Remove all outliers automatically - they'll confuse the model",
                    "Keep all data - real data has variation",
                    "Examine each extreme value to determine if it's an error, edge case, or valid but rare example",
                    "Cap all values at the 95th percentile",
                ],
                correct_index=2,
                success_response="Excellent judgment! A 150-word 'question' might be an annotation error, or it might be a valuable example of extended discourse. A 1-word entry might be 'What?' - perfectly valid! Context matters.",
                struggle_response="Automatic outlier removal can delete your most interesting data points. That unusually long exchange might be a master teacher scaffolding complex thinking. That one-word question might be 'Why?' - deceptively simple but open-ended.",
                deep_dive="In educational data, 'outliers' often represent expert behavior or challenging situations - exactly what we want to model! Statistical outliers aren't always data quality issues. Always investigate before removing.",
                related_topics=["outlier_analysis", "edge_cases", "data_integrity"]
            ),
            ConversationalQuestion(
                stage=2,
                topic="text_normalization",
                difficulty=DifficultyLevel.PRACTITIONER,
                scenario="""
Our transcripts contain various formats: 'dont', 'don't', 'DO NOT', 'do not'.
There are also annotations like '[laughs]', '[inaudible]', and timestamps.
We need consistent text for our NLP pipeline.
""",
                prompt="What text normalization steps would you recommend for our transcript data?",
                options=[
                    "Lowercase everything and remove all punctuation",
                    "Keep text exactly as transcribed for authenticity",
                    "Standardize contractions, lowercase, preserve paralinguistic markers like [laughs] as potential features, and document decisions",
                    "Remove all special annotations since we only care about words",
                ],
                correct_index=2,
                success_response="Thoughtful approach! Standardizing helps consistency, but '[laughs]' might indicate a warm interaction, and '[inaudible]' tells us about audio quality. Preserve what might matter, document what you change.",
                struggle_response="Text normalization is about balance. Too much cleaning loses information (prosodic markers, emphasis). Too little creates inconsistency. Think about what features your model needs downstream.",
                deep_dive="For educational interaction analysis, paralinguistic features matter! A question followed by [pause] or [laughter] carries different meaning. Consider creating separate feature columns for these markers rather than discarding them.",
                related_topics=["text_preprocessing", "feature_preservation", "documentation"]
            ),
            ConversationalQuestion(
                stage=2,
                topic="duplicate_detection",
                difficulty=DifficultyLevel.EXPERT,
                scenario="""
While auditing our dataset, we found that 8 annotations appear twice - same text,
same labels, same video source. Additionally, 12 annotations have identical text
but different labels from different annotators (inter-rater disagreement).
""",
                prompt="How should we handle these two types of duplicates?",
                options=[
                    "Remove all duplicates - duplicates always indicate errors",
                    "Keep identical duplicates for training (more weight), and for disagreements, use majority vote or expert adjudication",
                    "Remove true duplicates (likely data entry errors), but preserve disagreements as they reveal labeling ambiguity that's informative",
                    "Average the labels for all duplicates",
                ],
                correct_index=2,
                success_response="Sophisticated distinction! True duplicates are likely errors. But annotator disagreement is gold - it tells us which examples are genuinely ambiguous and might need special handling or additional annotation guidelines.",
                struggle_response="Not all 'duplicates' are the same. Copy-paste errors should be removed. But when two experts label the same text differently, that ambiguity is real - it reflects the inherent difficulty of the task.",
                deep_dive="Inter-annotator disagreement is a feature, not a bug. These cases help us calibrate model confidence, identify edge cases for additional training, and set realistic expectations for model accuracy. Cohen's Kappa on disagreement cases is highly informative.",
                related_topics=["inter_rater_reliability", "annotation_quality", "ambiguity"]
            ),
        ]

    def _stage3_feature_engineering(self) -> List[ConversationalQuestion]:
        """Stage 3: Feature Engineering - The Architect"""
        return [
            ConversationalQuestion(
                stage=3,
                topic="linguistic_features",
                difficulty=DifficultyLevel.NOVICE,
                scenario="""
We're building features to classify open-ended vs closed-ended questions.
The junior engineer suggests just counting words like 'what' and 'why'.
""",
                prompt="Word counting is a start, but what additional linguistic features might help distinguish question types?",
                options=[
                    "Word counts are sufficient - keep it simple",
                    "Add sentence structure features like whether the question starts with a question word vs auxiliary verb",
                    "Only use deep learning - it finds features automatically",
                    "Focus on the response rather than the question",
                ],
                correct_index=1,
                success_response="Great insight! 'What color is it?' (closed) vs 'What do you think about it?' (open) both start with 'what'. The auxiliary verb pattern (is/do/did) helps distinguish them. Linguistic structure matters!",
                struggle_response="Consider: 'Is it red?' vs 'What makes you say that?' Both are questions, but structure differs. Open-ended questions often use 'do you think/feel' patterns while closed ones use 'is/are/can' + noun.",
                deep_dive="Effective question classification combines lexical (word-level), syntactic (structure), and semantic (meaning) features. Research shows that dependency parsing features add 5-10% accuracy over bag-of-words alone.",
                related_topics=["linguistic_features", "syntax", "question_patterns"]
            ),
            ConversationalQuestion(
                stage=3,
                topic="temporal_features",
                difficulty=DifficultyLevel.APPRENTICE,
                scenario="""
Our model classifies individual questions well, but misses the conversation flow.
A 'why?' after a child's story is different from 'why?' after a closed question.
Context seems to matter.
""",
                prompt="How could we capture this conversational context in our features?",
                options=[
                    "Just classify each question independently - context is too complex",
                    "Add features about the previous turn: speaker, question type, response length, topic continuity",
                    "Use a rule-based system for context",
                    "Transcribe entire conversations as one long text",
                ],
                correct_index=1,
                success_response="Exactly! Discourse context is crucial. A follow-up question after a child's extended response is often scaffolding deeper thinking. Previous turn features capture this pedagogical intent.",
                struggle_response="Think about how conversation flows. 'Tell me more' after a child shares something = follow-up (good!). After a yes/no answer = prompting elaboration. Same words, different context = different function.",
                deep_dive="Temporal and contextual features are key in educational dialogue analysis. Memory-augmented networks can track conversation state, but simpler n-gram history features often work well for classification tasks.",
                related_topics=["context_features", "dialogue_modeling", "discourse_analysis"]
            ),
            ConversationalQuestion(
                stage=3,
                topic="audio_features",
                difficulty=DifficultyLevel.PRACTITIONER,
                scenario="""
We have audio recordings along with transcripts. Our current model uses only text.
The team wonders if audio features could help, especially for detecting wait time patterns.
""",
                prompt="What audio features might be valuable for our educator interaction analysis?",
                options=[
                    "Audio quality is too variable to be useful",
                    "Just use speech-to-text confidence scores",
                    "Extract prosodic features: speech rate, pause duration, pitch variation, and energy levels",
                    "Only use audio if text features fail",
                ],
                correct_index=2,
                success_response="Perfect! Prosodic features are pedagogically meaningful. A rising pitch at the end signals genuine inquiry. Pause duration after questions is literally 'wait time'. Energy levels can indicate enthusiasm or engagement.",
                struggle_response="Audio carries information beyond words. A teacher who asks quickly and answers their own question (short pause) is different from one who asks, waits 5 seconds, and genuinely expects an answer. Pause analysis directly measures wait time!",
                deep_dive="Prosodic features from audio using tools like OpenSMILE or wav2vec2 can capture pedagogical style. Research shows that educators rated as 'warm' have specific pitch contours, and wait time is directly measurable from pause detection.",
                related_topics=["audio_features", "prosody", "multimodal_fusion"]
            ),
            ConversationalQuestion(
                stage=3,
                topic="feature_interaction",
                difficulty=DifficultyLevel.EXPERT,
                scenario="""
Our individual features perform okay, but we suspect there are meaningful interactions.
For example, the combination of 'starts with what' + 'contains think/feel' might be
more predictive than either feature alone.
""",
                prompt="How should we approach capturing feature interactions?",
                options=[
                    "Deep learning will find interactions automatically - don't engineer them",
                    "Create all possible pairwise interactions",
                    "Identify theoretically motivated interactions based on educational research, and validate with feature importance analysis",
                    "Feature interactions add too much complexity - keep features independent",
                ],
                correct_index=2,
                success_response="Wise approach! Domain knowledge guides interaction design. 'What + think/feel' = open-ended questioning pattern. 'Is + color/number' = closed recall. Theory-driven interactions are interpretable AND effective.",
                struggle_response="Random interaction exploration is computationally expensive and risks overfitting. But ignoring interactions misses signal. The sweet spot: use educational research to hypothesize which combinations matter, then validate empirically.",
                deep_dive="For 56 features, all pairwise interactions = 1,540 new features. Most are noise. Domain-guided interaction terms (e.g., question_word * metacognitive_verb) are sparse but powerful. Combine with feature selection for best results.",
                related_topics=["feature_interactions", "domain_knowledge", "interpretability"]
            ),
        ]

    def _stage4_model_selection(self) -> List[ConversationalQuestion]:
        """Stage 4: Model Selection - The Strategist"""
        return [
            ConversationalQuestion(
                stage=4,
                topic="model_comparison",
                difficulty=DifficultyLevel.NOVICE,
                scenario="""
We're choosing between three approaches for question classification:
- Rule-based system (if starts with 'what' and contains 'think' -> open-ended)
- Random Forest classifier
- Deep neural network
The team is debating which to try first.
""",
                prompt="With 119 annotated examples, which approach would you recommend starting with?",
                options=[
                    "Deep neural network - it's the most powerful",
                    "Rule-based system - it's the most interpretable",
                    "Random Forest - good balance of performance and data efficiency for small datasets",
                    "Try all three simultaneously",
                ],
                correct_index=2,
                success_response="Strategic choice! With 119 examples, neural networks risk overfitting. Random Forests are robust to small data, provide feature importance, and establish a strong baseline. Start simple, add complexity if needed.",
                struggle_response="Match your model to your data size. Deep learning shines with 10,000+ examples. With 119, classical ML often wins. Random Forests also give you interpretable feature importance - valuable for educational applications.",
                deep_dive="The 'start simple' principle: establish a baseline with interpretable models, then add complexity only if needed. In educational AI, interpretability often matters as much as accuracy - stakeholders need to understand predictions.",
                related_topics=["model_selection", "data_efficiency", "interpretability"]
            ),
            ConversationalQuestion(
                stage=4,
                topic="ensemble_strategy",
                difficulty=DifficultyLevel.APPRENTICE,
                scenario="""
Our Random Forest achieves 78% accuracy. Adding XGBoost gives 80%.
A neural network gives 76% but catches different examples than the tree models.
The team is considering an ensemble.
""",
                prompt="How might we combine these models effectively?",
                options=[
                    "Just use the best single model (XGBoost at 80%)",
                    "Average all predictions equally",
                    "Use voting ensemble, weighted by individual model accuracy, allowing diverse models to cover each other's weaknesses",
                    "Train a neural network on top of all predictions",
                ],
                correct_index=2,
                success_response="Excellent ensemble thinking! The neural network catching 'different' examples is exactly why ensembles work - diverse models make diverse errors. Weighted voting leverages each model's strengths.",
                struggle_response="Ensemble value comes from diversity, not just accuracy. If two models agree on easy cases but disagree on hard ones, their combination can outperform either. Weight models by their reliability, and let them vote.",
                deep_dive="Ensemble diversity is key. Models with different inductive biases (trees vs neural nets) make uncorrelated errors. Research shows that a 78% and 76% model ensemble can beat an 80% model if their errors don't overlap.",
                related_topics=["ensemble_methods", "model_diversity", "voting_strategies"]
            ),
            ConversationalQuestion(
                stage=4,
                topic="deployment_constraints",
                difficulty=DifficultyLevel.PRACTITIONER,
                scenario="""
Our project has three deployment targets:
- Edge devices in classrooms (real-time feedback, <10ms inference)
- Mobile app for educators (<50ms inference, limited battery)
- Cloud dashboard for administrators (no latency requirement)
""",
                prompt="How should our model selection differ across these deployment scenarios?",
                options=[
                    "Use the same model everywhere for consistency",
                    "Only build the cloud model - others can call it via API",
                    "Design a model hierarchy: lightweight classical ML for edge, optimized neural net for mobile, full ensemble for cloud",
                    "Prioritize accuracy for all scenarios regardless of constraints",
                ],
                correct_index=2,
                success_response="Brilliant three-tier architecture! Edge needs speed (Random Forest gives <1ms inference). Mobile balances speed and accuracy (distilled models). Cloud can run the full ensemble. Same predictions, right tool for each context.",
                struggle_response="Different deployment contexts have different constraints. A 200ms neural net is unusable for real-time classroom feedback but perfect for overnight batch analysis. Design your model portfolio for the full spectrum of use cases.",
                deep_dive="Three-tier ML deployment is common in production systems. Edge: quantized models, pruned trees. Mobile: knowledge distillation, MobileBERT. Cloud: full ensembles, expensive models. Use model cascading where edge handles easy cases, escalating to cloud for ambiguous ones.",
                related_topics=["deployment_constraints", "model_optimization", "edge_computing"]
            ),
            ConversationalQuestion(
                stage=4,
                topic="transfer_learning",
                difficulty=DifficultyLevel.EXPERT,
                scenario="""
We're considering fine-tuning a pre-trained language model (BERT/DistilBERT) on our
119 educational examples. The model was trained on general web text, not educational
discourse. Some team members worry about domain mismatch.
""",
                prompt="What's your recommendation for using pre-trained models in our educational domain?",
                options=[
                    "Don't use pre-trained models - domain mismatch will hurt performance",
                    "Use pre-trained models as-is without fine-tuning",
                    "Fine-tune with careful strategies: freeze lower layers, use small learning rates, augment with educational domain knowledge",
                    "Train an educational language model from scratch",
                ],
                correct_index=2,
                success_response="Nuanced approach! Pre-trained models capture general language understanding valuable for any domain. Careful fine-tuning (lower layers frozen, low learning rate) adapts them without losing that foundation. 119 examples is enough for the top layers.",
                struggle_response="Transfer learning is about leveraging general knowledge while adapting to specifics. BERT knows grammar and meaning from billions of examples. Your 119 examples teach it 'educational questioning patterns' specifically. Both contribute.",
                deep_dive="For small educational datasets, fine-tuning strategies matter: (1) Gradual unfreezing prevents catastrophic forgetting, (2) Low learning rate (2e-5) preserves pre-trained knowledge, (3) Educational domain continued pre-training can bridge the gap before fine-tuning.",
                related_topics=["transfer_learning", "fine_tuning", "domain_adaptation"]
            ),
        ]

    def _stage5_training(self) -> List[ConversationalQuestion]:
        """Stage 5: Training - The Trainer"""
        return [
            ConversationalQuestion(
                stage=5,
                topic="train_test_split",
                difficulty=DifficultyLevel.NOVICE,
                scenario="""
We have 119 annotated examples ready for training. The junior engineer suggests using
all 119 for training to maximize learning, then testing on the same data.
""",
                prompt="What's problematic about training and testing on the same data?",
                options=[
                    "Nothing - more training data is always better",
                    "The model will memorize the training data and fail on new examples (overfitting)",
                    "We should use 90% for training since we have few examples",
                    "We only need a test set if we have over 1000 examples",
                ],
                correct_index=1,
                success_response="Exactly right! If we test on training data, we measure memorization, not generalization. It's like giving students the exact same test they studied - they might score 100% but not actually understand the material.",
                struggle_response="Think of it like a student who memorizes answers vs understands concepts. Testing on training data tells us how well the model 'memorized'. We need held-out data to know if it truly 'learned' the patterns.",
                deep_dive="With 119 examples, use stratified k-fold cross-validation (k=5) to maximize both training data and evaluation reliability. Each fold trains on 95 examples and tests on 24. Report mean and standard deviation across folds.",
                related_topics=["train_test_split", "overfitting", "cross_validation"]
            ),
            ConversationalQuestion(
                stage=5,
                topic="class_imbalance",
                difficulty=DifficultyLevel.APPRENTICE,
                scenario="""
Our dataset has 85 open-ended questions and 34 closed-ended questions (71%/29% split).
After training, the model predicts open-ended for almost everything and gets 75% accuracy.
""",
                prompt="Why is 75% accuracy misleading here, and how should we address it?",
                options=[
                    "75% is good - that's above chance",
                    "The model learned to always predict the majority class; use F1-score and address imbalance through class weighting or resampling",
                    "Just collect more closed-ended examples",
                    "Accuracy is the only metric that matters",
                ],
                correct_index=1,
                success_response="Astute observation! A model predicting 'open-ended' always would get 71% accuracy but be useless. Class-weighted loss or SMOTE resampling helps, and F1-score/confusion matrix reveal the true performance.",
                struggle_response="If 71% of data is class A, predicting always-A gives 71% accuracy with zero learning! Look at per-class metrics. A model that catches 90% of open-ended but only 20% of closed-ended isn't balanced.",
                deep_dive="For imbalanced educational data: (1) Stratified sampling preserves class ratios, (2) Class weights (1/class_frequency) balances gradients, (3) SMOTE creates synthetic minority examples, (4) Focal loss emphasizes hard examples. Evaluate with macro-F1 and per-class recall.",
                related_topics=["class_imbalance", "evaluation_metrics", "resampling"]
            ),
            ConversationalQuestion(
                stage=5,
                topic="overfitting_detection",
                difficulty=DifficultyLevel.PRACTITIONER,
                scenario="""
Our neural network shows:
- Training accuracy: 98%
- Validation accuracy: 72%
Each epoch, training accuracy increases but validation accuracy started decreasing at epoch 15.
""",
                prompt="What's happening here and how should we respond?",
                options=[
                    "The model is doing great - 98% training accuracy is excellent",
                    "The 26-point gap indicates severe overfitting; implement early stopping, regularization, or reduce model complexity",
                    "Just train longer - validation will eventually catch up",
                    "Increase model size to capture more patterns",
                ],
                correct_index=1,
                success_response="Classic overfitting detection! The model is memorizing training data (98%) but failing to generalize (72%). Early stopping at epoch 15, dropout regularization, or a simpler model architecture would help.",
                struggle_response="When training accuracy climbs but validation plateaus or drops, the model is memorizing, not learning. It's like a student who aces practice tests by memorization but fails the real exam with new questions.",
                deep_dive="Overfitting remedies: (1) Early stopping (save best validation model), (2) Regularization (L2 weight decay, dropout), (3) Data augmentation, (4) Reduce model capacity, (5) Collect more diverse training data. With 119 examples, expect some overfitting - regularize accordingly.",
                related_topics=["overfitting", "regularization", "early_stopping"]
            ),
            ConversationalQuestion(
                stage=5,
                topic="reproducibility",
                difficulty=DifficultyLevel.EXPERT,
                scenario="""
Our colleague ran the same training code and got different results: 78% vs our 82%.
We can't reproduce each other's findings, which makes our paper submission risky.
""",
                prompt="What factors affect reproducibility and how do we ensure consistent results?",
                options=[
                    "ML is inherently random - just report a range",
                    "Set random seeds, log all hyperparameters, version data and code, and document hardware/software environments",
                    "Run training multiple times and pick the best result",
                    "Use the same laptop for all experiments",
                ],
                correct_index=1,
                success_response="Comprehensive reproducibility approach! Random seeds control initialization, hyperparameter logging enables replication, versioned data prevents drift, and environment documentation (Python version, GPU, CUDA) ensures exact recreation.",
                struggle_response="Reproducibility requires controlling ALL sources of randomness and variation. Random seeds, data shuffling, weight initialization, dropout patterns - each can cause variation. Document everything to replicate exactly.",
                deep_dive="For ML reproducibility: (1) Set seeds: random.seed(42), np.random.seed(42), torch.manual_seed(42), (2) Use deterministic algorithms where possible, (3) Version datasets with hashes, (4) Container-based environments (Docker) for exact replication, (5) Log everything with tools like MLflow or Weights & Biases.",
                related_topics=["reproducibility", "experiment_tracking", "version_control"]
            ),
        ]

    def _stage6_evaluation(self) -> List[ConversationalQuestion]:
        """Stage 6: Evaluation - The Analyst"""
        return [
            ConversationalQuestion(
                stage=6,
                topic="metric_selection",
                difficulty=DifficultyLevel.NOVICE,
                scenario="""
Our model classifies questions as open-ended or closed-ended. The stakeholders ask:
'How good is it?' We have accuracy, precision, recall, and F1-score available.
""",
                prompt="Which metric best captures model performance for educational stakeholders?",
                options=[
                    "Accuracy - it's the most intuitive",
                    "Precision - we want to be sure when we predict open-ended",
                    "Recall - we don't want to miss any open-ended questions",
                    "It depends on what matters more: false positives or false negatives. Ask stakeholders which errors are more costly.",
                ],
                correct_index=3,
                success_response="Perfect stakeholder-centered thinking! If falsely flagging closed questions as open (false positive) means unnecessary coaching, precision matters. If missing open-ended opportunities (false negative) is worse, emphasize recall. Align metrics with impact.",
                struggle_response="Metrics should reflect real-world costs. Missing an expert's open-ended question (FN) might mean missed feedback opportunity. Incorrectly praising a closed question (FP) might undermine trust. Which matters more depends on the application.",
                deep_dive="In educational AI, involve stakeholders in metric selection. Confusion matrix analysis in human terms: 'Of 100 questions flagged as open-ended, 85 actually are (precision). Of 100 actual open-ended questions, we catch 78 (recall).' Makes trade-offs concrete.",
                related_topics=["evaluation_metrics", "stakeholder_communication", "error_analysis"]
            ),
            ConversationalQuestion(
                stage=6,
                topic="confidence_calibration",
                difficulty=DifficultyLevel.APPRENTICE,
                scenario="""
Our model predicts 90% confidence on some questions but is wrong 30% of the time on those.
On other predictions with 70% confidence, it's wrong only 10% of the time.
""",
                prompt="What issue does this reveal, and why does it matter?",
                options=[
                    "Random variation - confidence scores are just estimates",
                    "The model is poorly calibrated - its confidence doesn't match actual accuracy, which affects trust and decision-making",
                    "High confidence always means high accuracy",
                    "We should ignore confidence and just use the prediction",
                ],
                correct_index=1,
                success_response="Critical insight for deployed systems! A well-calibrated model should be wrong 10% of the time on 90%-confidence predictions. Poor calibration erodes user trust. Temperature scaling or Platt calibration can fix this.",
                struggle_response="Imagine: model says '90% sure it's open-ended', but it's wrong 30% of the time. Users learn to distrust the model. Good calibration means confidence = reliability. When it says 90%, it should be right 90% of the time.",
                deep_dive="Calibration evaluation: plot predicted confidence vs actual accuracy (reliability diagram). Perfect calibration = diagonal line. Methods to improve: temperature scaling (simple), Platt scaling (logistic recalibration), isotonic regression (non-parametric). Essential for human-AI collaboration.",
                related_topics=["calibration", "confidence_scores", "reliability"]
            ),
            ConversationalQuestion(
                stage=6,
                topic="error_analysis",
                difficulty=DifficultyLevel.PRACTITIONER,
                scenario="""
Our model achieves 82% overall accuracy but performs poorly on certain question patterns:
- Rhetorical questions (treated as open-ended but aren't)
- Follow-up questions ('Tell me more' - short but open)
- Questions with conditional phrasing ('If you could...')
""",
                prompt="What should we do with this error analysis information?",
                options=[
                    "82% is good enough - accept some errors",
                    "Create specific rules to override model on these patterns",
                    "Use error analysis to guide data augmentation, feature engineering, or create specialized sub-models for tricky patterns",
                    "Report per-category accuracy and stop there",
                ],
                correct_index=2,
                success_response="Error analysis should drive improvement! Rhetorical questions might need additional features (are they followed by educator's own answer?). Follow-ups might need context features. Conditionals might warrant a specialized classifier.",
                struggle_response="Error patterns reveal where to invest effort. Instead of improving overall accuracy blindly, targeted interventions on specific failure modes are more efficient. Each error type suggests a different solution.",
                deep_dive="Systematic error analysis: (1) Categorize errors (boundary cases, annotation errors, feature gaps, fundamental ambiguity), (2) Prioritize by impact and fixability, (3) For each category, design targeted intervention: new features, more data, rule-based post-processing, or accept inherent ambiguity.",
                related_topics=["error_analysis", "targeted_improvement", "failure_modes"]
            ),
            ConversationalQuestion(
                stage=6,
                topic="statistical_significance",
                difficulty=DifficultyLevel.EXPERT,
                scenario="""
Model A achieves 81.2% accuracy. Model B achieves 82.8% accuracy.
With only 119 test examples, the team is debating whether B is truly better.
""",
                prompt="Is a 1.6% difference significant with 119 examples?",
                options=[
                    "Yes - 82.8% > 81.2% so Model B wins",
                    "Run a statistical test (McNemar's test for paired comparisons) to determine if the difference is significant given sample size",
                    "The difference is too small to matter",
                    "Collect more data until the difference is larger",
                ],
                correct_index=1,
                success_response="Statistical rigor! With 119 examples, 1.6% difference is ~2 examples. McNemar's test for paired comparisons tells us if model B's wins on specific examples are significant or could be chance. Report confidence intervals, not just point estimates.",
                struggle_response="With small test sets, random variation can cause apparent differences. Two models might 'switch places' with a different random split. Statistical tests protect us from over-interpreting noise as signal.",
                deep_dive="For ML model comparison: (1) Use paired tests since both models see same examples (McNemar's for classification), (2) Report confidence intervals (bootstrap), (3) Consider Bayesian comparisons for small samples, (4) Cross-validation variance tells you expected variation.",
                related_topics=["statistical_testing", "significance", "confidence_intervals"]
            ),
        ]

    def _stage7_hyperparameter_tuning(self) -> List[ConversationalQuestion]:
        """Stage 7: Hyperparameter Tuning - The Optimizer"""
        return [
            ConversationalQuestion(
                stage=7,
                topic="search_strategies",
                difficulty=DifficultyLevel.NOVICE,
                scenario="""
We need to tune our Random Forest: number of trees (10-500), max depth (3-20),
and min samples per leaf (1-20). The junior engineer suggests trying every combination.
""",
                prompt="What's the issue with exhaustive grid search here?",
                options=[
                    "Nothing - exhaustive search finds the best combination",
                    "With 3 parameters and many values, grid search could take thousands of runs. Random search or Bayesian optimization is more efficient for initial exploration.",
                    "We should only tune one parameter at a time",
                    "Hyperparameters don't affect performance much anyway",
                ],
                correct_index=1,
                success_response="Computational awareness! If we try 50 values per parameter, that's 50x50x50 = 125,000 combinations. Random search with 100 trials often finds near-optimal settings by exploring the space more efficiently.",
                struggle_response="The curse of dimensionality applies to hyperparameter search too. As we add parameters, exhaustive search becomes exponentially expensive. Smarter search strategies get close to optimal with far fewer evaluations.",
                deep_dive="Hyperparameter search strategies: Grid (exhaustive, exponential cost), Random (surprisingly effective, Bergstra & Bengio 2012), Bayesian (learns which regions are promising), Hyperband (early stopping for bad configs). Start with random, refine with Bayesian.",
                related_topics=["hyperparameter_tuning", "search_strategies", "computational_efficiency"]
            ),
            ConversationalQuestion(
                stage=7,
                topic="validation_strategy",
                difficulty=DifficultyLevel.APPRENTICE,
                scenario="""
We're using 5-fold cross-validation for hyperparameter tuning. After many iterations,
we find settings that achieve 85% CV accuracy. The team celebrates and prepares to deploy.
""",
                prompt="Before celebrating, what concern should we raise about this tuning process?",
                options=[
                    "85% is great - ready to deploy",
                    "We've potentially overfit to the cross-validation folds; we need a held-out test set that was never used during tuning",
                    "We should have used 10-fold instead of 5-fold",
                    "Cross-validation always gives reliable estimates",
                ],
                correct_index=1,
                success_response="Sharp catch! By tuning on CV performance, we may have 'leaked' information from all folds. A final held-out test set (never touched during tuning) gives unbiased performance estimate for deployment.",
                struggle_response="When we tune hyperparameters to maximize CV score over hundreds of trials, we're indirectly 'fitting' to the entire training set. Keep a final test set sacred - only evaluate once after all tuning is complete.",
                deep_dive="Nested cross-validation: outer loop for final evaluation, inner loop for hyperparameter tuning. With 119 examples, this is expensive but necessary for unbiased estimates. Alternatively, hold out 20% before any tuning.",
                related_topics=["nested_cv", "data_leakage", "evaluation_protocol"]
            ),
            ConversationalQuestion(
                stage=7,
                topic="learning_rate",
                difficulty=DifficultyLevel.PRACTITIONER,
                scenario="""
Our neural network with learning rate 0.1 trains fast but oscillates wildly.
Learning rate 0.0001 is stable but converges too slowly. We're looking for the sweet spot.
""",
                prompt="What approach would help find an optimal learning rate?",
                options=[
                    "Pick something in between like 0.01",
                    "Use a learning rate finder (gradually increase LR, plot loss) to find the steepest descent region, then use learning rate scheduling",
                    "Learning rate doesn't matter much with Adam optimizer",
                    "Use the default learning rate for the framework",
                ],
                correct_index=1,
                success_response="Modern optimization approach! A learning rate finder (Leslie Smith's technique) empirically identifies good ranges. Then, scheduling (cosine annealing, warm restarts) dynamically adjusts LR during training for best convergence.",
                struggle_response="Learning rate is often the most impactful hyperparameter. Too high = instability, too low = slow/stuck. LR finders plot loss vs. LR during a short run to find the 'sweet spot' where loss decreases fastest.",
                deep_dive="Learning rate strategies: (1) LR finder for initial range, (2) Warm-up (start low, ramp up) for transformers, (3) Cosine annealing (high to low cyclically), (4) One-cycle policy (up then down). AdamW + cosine annealing is a strong default for deep learning.",
                related_topics=["learning_rate", "optimization", "scheduling"]
            ),
            ConversationalQuestion(
                stage=7,
                topic="regularization_tuning",
                difficulty=DifficultyLevel.EXPERT,
                scenario="""
We're tuning regularization for our neural network: dropout rate (0.0-0.7) and
L2 weight decay (1e-6 to 1e-2). The optimal settings seem to depend on each other.
""",
                prompt="How should we handle this interdependence between regularization hyperparameters?",
                options=[
                    "Tune them independently and combine the best settings",
                    "Use maximum regularization to prevent overfitting",
                    "Tune them jointly since they interact; higher dropout may need less weight decay. Use 2D grid search or Bayesian optimization over both.",
                    "Pick standard values from literature and don't tune",
                ],
                correct_index=2,
                success_response="Sophisticated understanding of hyperparameter interactions! Dropout and weight decay both prevent overfitting but do so differently. Heavy dropout + heavy decay might be too much regularization. Joint optimization captures their synergy.",
                struggle_response="Hyperparameters often interact. Two regularization methods at maximum strength might over-regularize, hurting performance more than either alone. Think of total regularization 'budget' distributed across methods.",
                deep_dive="Regularization interactions: dropout (stochastic), weight decay (L2 penalty), early stopping (implicit). For transformers, high dropout + AdamW weight decay + warmup is standard. For small datasets, tune the balance carefully - total regularization matters more than any single component.",
                related_topics=["regularization", "hyperparameter_interaction", "model_capacity"]
            ),
        ]

    def _stage8_deployment(self) -> List[ConversationalQuestion]:
        """Stage 8: Deployment - The Commander"""
        return [
            ConversationalQuestion(
                stage=8,
                topic="model_serving",
                difficulty=DifficultyLevel.NOVICE,
                scenario="""
Our trained model sits in a Python file. The web application team needs to use it.
They're asking how to 'call' our model from their Node.js application.
""",
                prompt="How should we make our model accessible to the web application?",
                options=[
                    "Share the Python file and they can run it",
                    "Wrap the model in a REST API (like FastAPI) that accepts text and returns predictions",
                    "Convert the model to JavaScript",
                    "They should switch to Python",
                ],
                correct_index=1,
                success_response="Standard production pattern! A REST API creates a language-agnostic interface. Send JSON with text, receive JSON with prediction. FastAPI gives you automatic documentation and async support for high throughput.",
                struggle_response="Web services communicate via APIs, regardless of underlying language. Your model becomes a 'microservice' - input comes in via HTTP, prediction goes out. Any client (JavaScript, mobile app, other services) can use it.",
                deep_dive="Model serving options: FastAPI (Python, async, great for ML), Flask (simpler but synchronous), TorchServe (PyTorch-specific), TensorFlow Serving (TF-specific), Triton (multi-framework, GPU optimization). For 119-example model, FastAPI is perfect.",
                related_topics=["api_design", "model_serving", "microservices"]
            ),
            ConversationalQuestion(
                stage=8,
                topic="monitoring",
                difficulty=DifficultyLevel.APPRENTICE,
                scenario="""
Our model has been running in production for a month. The team assumes 'it's working'
because there are no error messages. But are we sure the predictions are still good?
""",
                prompt="What production monitoring should we implement?",
                options=[
                    "Error logging is sufficient - if it's not crashing, it's working",
                    "Monitor prediction distributions, confidence scores, input characteristics, and periodically validate against expert labels",
                    "Just monitor response latency",
                    "Production monitoring is only needed for critical systems",
                ],
                correct_index=1,
                success_response="Production ML requires continuous monitoring! If input distribution shifts (different question patterns emerge), predictions may degrade without crashing. Monitor: prediction distribution, confidence calibration, latency, and periodic expert validation.",
                struggle_response="A model can 'work' (no errors) while producing bad predictions if inputs change from training distribution. Monitor for distribution drift, decreasing confidence, and validate a sample of predictions regularly.",
                deep_dive="ML monitoring layers: (1) Technical (latency, errors, resource usage), (2) Data quality (missing values, format changes), (3) Model health (prediction distribution, confidence), (4) Business impact (are predictions driving good outcomes?). Tools: Prometheus, Grafana, WhyLabs, Evidently.",
                related_topics=["monitoring", "observability", "drift_detection"]
            ),
            ConversationalQuestion(
                stage=8,
                topic="model_updates",
                difficulty=DifficultyLevel.PRACTITIONER,
                scenario="""
After 6 months, we have 200 new annotated examples. The team wants to update the model
but worries about breaking what works. How do we safely improve?
""",
                prompt="What's the safest approach to updating a production model?",
                options=[
                    "Replace the old model with the new one immediately",
                    "Keep the old model forever - don't risk breaking it",
                    "Train new model, compare on held-out set, deploy via canary release (small traffic %), monitor, then full rollout",
                    "A/B test forever without committing to either model",
                ],
                correct_index=2,
                success_response="Robust deployment process! Evaluate new model rigorously before production, canary release to small traffic percentage allows monitoring in real conditions, full rollout only after confirming improvement. Keep old model as fallback.",
                struggle_response="Model updates need gradual validation. Training improvements don't always translate to production. Canary releases catch issues before they affect all users. Monitor closely during transition.",
                deep_dive="Safe ML deployment: (1) Offline evaluation on held-out data, (2) Shadow deployment (run alongside production, compare outputs, no user impact), (3) Canary release (5% traffic), (4) Gradual rollout (25%, 50%, 100%), (5) Instant rollback capability. Blue-green deployment enables fast switchback.",
                related_topics=["deployment_strategies", "canary_release", "rollback"]
            ),
            ConversationalQuestion(
                stage=8,
                topic="edge_deployment",
                difficulty=DifficultyLevel.EXPERT,
                scenario="""
We want to deploy our model to tablets in classrooms for real-time feedback.
Network connectivity is unreliable, latency requirements are <100ms,
and the tablets have limited memory.
""",
                prompt="How do we adapt our model for this edge deployment scenario?",
                options=[
                    "Our model requires cloud - edge is impossible",
                    "Just use a smaller model and accept lower accuracy",
                    "Apply model compression (quantization, pruning, distillation), optimize for mobile runtime (ONNX/TFLite), implement graceful degradation for connectivity issues",
                    "Require better tablets with more resources",
                ],
                correct_index=2,
                success_response="Comprehensive edge ML strategy! Quantization (int8) reduces model size 4x with ~1% accuracy loss. Knowledge distillation trains a smaller student model. ONNX runtime optimizes inference. Graceful degradation caches last known state during connectivity loss.",
                struggle_response="Edge deployment requires optimization across the stack: model size (quantization, pruning), runtime (ONNX, TFLite), and resilience (offline capability, caching). Each technique contributes to a deployable solution.",
                deep_dive="Edge ML optimization chain: (1) Quantization (FP32 -> INT8, 4x smaller), (2) Pruning (remove unimportant weights, 50-90% sparse), (3) Knowledge distillation (transfer to smaller architecture), (4) Mobile runtimes (ONNX Runtime, TFLite, CoreML), (5) On-device caching for offline capability.",
                related_topics=["edge_deployment", "model_compression", "mobile_optimization"]
            ),
        ]


# =============================================================================
# ADAPTIVE DIFFICULTY ENGINE
# =============================================================================

class AdaptiveDifficultyEngine:
    """
    Adjusts difficulty based on learner performance, ensuring engagement
    without frustration or boredom.
    """

    def __init__(self):
        self.window_size = 5  # Number of recent questions to consider

    def calculate_next_difficulty(self, learner: LearnerProfile,
                                   recent_results: List[bool]) -> DifficultyLevel:
        """
        Calculate appropriate difficulty based on recent performance.

        Rules:
        - 4/5 correct at current level -> increase difficulty
        - 2/5 or fewer correct -> decrease difficulty
        - Otherwise, stay at current level
        """
        if not recent_results:
            return learner.difficulty_level

        window = recent_results[-self.window_size:]
        success_rate = sum(window) / len(window)

        current = learner.difficulty_level.value

        if success_rate >= 0.8 and current < 5:
            return DifficultyLevel(current + 1)
        elif success_rate <= 0.4 and current > 1:
            return DifficultyLevel(current - 1)
        else:
            return learner.difficulty_level

    def get_encouragement_message(self, learner: LearnerProfile,
                                   is_correct: bool) -> str:
        """Generate appropriate encouragement based on context"""
        if is_correct:
            if learner.streak >= 5:
                return f"Incredible {learner.streak}-answer streak! You're on fire!"
            elif learner.streak >= 3:
                return f"That's {learner.streak} in a row! Keep it going!"
            else:
                return "Correct! Nice work, Engineer."
        else:
            if learner.streak > 3:
                return f"Streak broken at {learner.streak}, but what a run! Let's learn from this one."
            else:
                return "Not quite, but this is how we learn. Let's explore this together."


# =============================================================================
# BADGE SYSTEM
# =============================================================================

class BadgeSystem:
    """
    Achievement system that celebrates progress and mastery.
    Badges are earned through various accomplishments.
    """

    BADGE_DEFINITIONS = {
        "explorer_badge": Badge(
            badge_type=BadgeType.STAGE_COMPLETE,
            name="Data Explorer",
            description="Completed Stage 1: Data Collection",
            icon="[EXPLORER]",
            stage=1
        ),
        "purifier_badge": Badge(
            badge_type=BadgeType.STAGE_COMPLETE,
            name="Data Purifier",
            description="Completed Stage 2: Data Cleaning",
            icon="[PURIFIER]",
            stage=2
        ),
        "architect_badge": Badge(
            badge_type=BadgeType.STAGE_COMPLETE,
            name="Feature Architect",
            description="Completed Stage 3: Feature Engineering",
            icon="[ARCHITECT]",
            stage=3
        ),
        "strategist_badge": Badge(
            badge_type=BadgeType.STAGE_COMPLETE,
            name="Model Strategist",
            description="Completed Stage 4: Model Selection",
            icon="[STRATEGIST]",
            stage=4
        ),
        "trainer_badge": Badge(
            badge_type=BadgeType.STAGE_COMPLETE,
            name="ML Trainer",
            description="Completed Stage 5: Training",
            icon="[TRAINER]",
            stage=5
        ),
        "analyst_badge": Badge(
            badge_type=BadgeType.STAGE_COMPLETE,
            name="Performance Analyst",
            description="Completed Stage 6: Evaluation",
            icon="[ANALYST]",
            stage=6
        ),
        "optimizer_badge": Badge(
            badge_type=BadgeType.STAGE_COMPLETE,
            name="Hyperparameter Optimizer",
            description="Completed Stage 7: Hyperparameter Tuning",
            icon="[OPTIMIZER]",
            stage=7
        ),
        "commander_badge": Badge(
            badge_type=BadgeType.STAGE_COMPLETE,
            name="Deployment Commander",
            description="Completed Stage 8: Deployment",
            icon="[COMMANDER]",
            stage=8
        ),
        "perfect_mission": Badge(
            badge_type=BadgeType.PERFECT_MISSION,
            name="Perfect Mission",
            description="Completed a stage with 100% accuracy",
            icon="[PERFECT]"
        ),
        "streak_master": Badge(
            badge_type=BadgeType.SPEED_DEMON,
            name="Streak Master",
            description="Achieved a 10-answer streak",
            icon="[STREAK]"
        ),
        "comeback_kid": Badge(
            badge_type=BadgeType.COMEBACK_KID,
            name="Comeback Kid",
            description="Recovered from 3 wrong answers to complete a stage",
            icon="[COMEBACK]"
        ),
        "certified_engineer": Badge(
            badge_type=BadgeType.CERTIFIED_ENGINEER,
            name="Certified ML Engineer",
            description="Passed the final certification exam",
            icon="[CERTIFIED]"
        ),
    }

    def check_for_badges(self, learner: LearnerProfile,
                         stage_just_completed: Optional[int] = None) -> List[Badge]:
        """Check if learner has earned any new badges"""
        new_badges = []
        earned_badge_names = [b.name for b in learner.badges]

        # Check stage completion badges
        if stage_just_completed:
            stage_badge_key = [
                "explorer_badge", "purifier_badge", "architect_badge",
                "strategist_badge", "trainer_badge", "analyst_badge",
                "optimizer_badge", "commander_badge"
            ][stage_just_completed - 1]

            badge = self.BADGE_DEFINITIONS[stage_badge_key]
            if badge.name not in earned_badge_names:
                new_badge = Badge(
                    badge_type=badge.badge_type,
                    name=badge.name,
                    description=badge.description,
                    icon=badge.icon,
                    stage=badge.stage
                )
                new_badges.append(new_badge)

        # Check streak badge
        if learner.longest_streak >= 10:
            streak_badge = self.BADGE_DEFINITIONS["streak_master"]
            if streak_badge.name not in earned_badge_names:
                new_badges.append(Badge(
                    badge_type=streak_badge.badge_type,
                    name=streak_badge.name,
                    description=streak_badge.description,
                    icon=streak_badge.icon
                ))

        return new_badges


# =============================================================================
# CERTIFICATION EXPERIENCE
# =============================================================================

class CertificationExperience:
    """
    The final certification experience that synthesizes learning
    across all 8 stages of the ML pipeline.
    """

    def __init__(self, question_bank: MLPipelineQuestionBank):
        self.question_bank = question_bank
        self.passing_threshold = 0.75  # 75% to pass

    def generate_certification_exam(self, learner: LearnerProfile) -> List[ConversationalQuestion]:
        """Generate a certification exam with questions from all stages"""
        exam_questions = []

        # 2 questions per stage, adapted to learner's demonstrated level
        for stage in range(1, 9):
            stage_mastery = learner.stage_mastery.get(stage, 0.5)

            # Adjust difficulty based on demonstrated mastery
            if stage_mastery >= 0.8:
                target_difficulty = DifficultyLevel.EXPERT
            elif stage_mastery >= 0.6:
                target_difficulty = DifficultyLevel.PRACTITIONER
            else:
                target_difficulty = DifficultyLevel.APPRENTICE

            # Get 2 questions from this stage
            for _ in range(2):
                question = self.question_bank.get_question(stage, target_difficulty)
                if question:
                    exam_questions.append(question)

        random.shuffle(exam_questions)
        return exam_questions

    def evaluate_certification(self, results: List[bool]) -> Tuple[bool, float, Dict[int, float]]:
        """
        Evaluate certification exam results.
        Returns: (passed, overall_score, per_stage_scores)
        """
        if not results:
            return False, 0.0, {}

        overall_score = sum(results) / len(results)
        passed = overall_score >= self.passing_threshold

        # Calculate per-stage performance (assuming 2 questions per stage)
        per_stage = {}
        for i, stage in enumerate(range(1, 9)):
            stage_results = results[i*2:(i+1)*2] if i*2 < len(results) else []
            if stage_results:
                per_stage[stage] = sum(stage_results) / len(stage_results)
            else:
                per_stage[stage] = 0.0

        return passed, overall_score, per_stage


# =============================================================================
# PROGRESS DASHBOARD
# =============================================================================

class ProgressDashboard:
    """
    Rich terminal dashboard showing learner progress across all stages.
    """

    def __init__(self, console: Optional['Console'] = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.narrative = MissionControlNarrative(console)

    def render_dashboard(self, learner: LearnerProfile) -> str:
        """Render a comprehensive progress dashboard"""
        if self.console and RICH_AVAILABLE:
            return self._render_rich_dashboard(learner)
        else:
            return self._render_text_dashboard(learner)

    def _render_text_dashboard(self, learner: LearnerProfile) -> str:
        """Text-based dashboard for terminals without rich support"""
        lines = []
        lines.append("=" * 70)
        lines.append("       ML MISSION CONTROL - ENGINEER PROGRESS DASHBOARD")
        lines.append("=" * 70)
        lines.append(f"  Callsign: {learner.callsign}")
        lines.append(f"  Rank: {learner.difficulty_level.name}")
        lines.append(f"  Current Mission: Stage {learner.current_stage}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("  STAGE MASTERY LEVELS")
        lines.append("-" * 70)

        stage_names = {
            1: "Data Collection",
            2: "Data Cleaning",
            3: "Feature Engineering",
            4: "Model Selection",
            5: "Training",
            6: "Evaluation",
            7: "Hyperparameter Tuning",
            8: "Deployment"
        }

        for stage in range(1, 9):
            mastery = learner.stage_mastery.get(stage, 0.0)
            bar_length = int(mastery * 20)
            bar = "#" * bar_length + "-" * (20 - bar_length)
            status = "COMPLETE" if mastery >= 0.8 else "IN PROGRESS" if mastery > 0 else "LOCKED"
            lines.append(f"  Stage {stage}: {stage_names[stage]:25} [{bar}] {mastery*100:.0f}% {status}")

        lines.append("")
        lines.append("-" * 70)
        lines.append("  PERFORMANCE STATS")
        lines.append("-" * 70)
        accuracy = (learner.correct_answers / learner.total_questions_answered * 100) if learner.total_questions_answered > 0 else 0
        lines.append(f"  Total Questions: {learner.total_questions_answered}")
        lines.append(f"  Accuracy: {accuracy:.1f}%")
        lines.append(f"  Current Streak: {learner.streak}")
        lines.append(f"  Best Streak: {learner.longest_streak}")

        lines.append("")
        lines.append("-" * 70)
        lines.append("  BADGES EARNED")
        lines.append("-" * 70)
        if learner.badges:
            for badge in learner.badges:
                lines.append(f"  {badge.icon} {badge.name}: {badge.description}")
        else:
            lines.append("  No badges earned yet. Complete missions to earn badges!")

        if learner.certified:
            lines.append("")
            lines.append("*" * 70)
            lines.append("       *** CERTIFIED ML PIPELINE ENGINEER ***")
            lines.append(f"       Certification Date: {learner.certification_date}")
            lines.append("*" * 70)

        lines.append("=" * 70)
        return "\n".join(lines)

    def _render_rich_dashboard(self, learner: LearnerProfile) -> None:
        """Rich terminal dashboard with colors and formatting"""
        # Create main layout
        layout = Layout()

        # Header
        header_text = Text()
        header_text.append("ML MISSION CONTROL", style="bold cyan")
        header_text.append(" - ENGINEER PROGRESS DASHBOARD", style="white")

        # Engineer info panel
        info_text = f"""
[bold cyan]Callsign:[/bold cyan] {learner.callsign}
[bold cyan]Rank:[/bold cyan] {learner.difficulty_level.name}
[bold cyan]Current Mission:[/bold cyan] Stage {learner.current_stage}
[bold cyan]Certification:[/bold cyan] {"[green]CERTIFIED[/green]" if learner.certified else "[yellow]In Progress[/yellow]"}
"""

        # Create mastery table
        mastery_table = Table(title="Stage Mastery", box=box.ROUNDED)
        mastery_table.add_column("Stage", style="cyan")
        mastery_table.add_column("Name", style="white")
        mastery_table.add_column("Progress", justify="center")
        mastery_table.add_column("Status", justify="center")

        stage_names = {
            1: "Data Collection",
            2: "Data Cleaning",
            3: "Feature Engineering",
            4: "Model Selection",
            5: "Training",
            6: "Evaluation",
            7: "Hyperparameter Tuning",
            8: "Deployment"
        }

        for stage in range(1, 9):
            mastery = learner.stage_mastery.get(stage, 0.0)
            bar = self._create_progress_bar(mastery)

            if mastery >= 0.8:
                status = "[green]COMPLETE[/green]"
            elif mastery > 0:
                status = "[yellow]IN PROGRESS[/yellow]"
            else:
                status = "[dim]LOCKED[/dim]"

            mastery_table.add_row(
                str(stage),
                stage_names[stage],
                bar,
                status
            )

        # Stats panel
        accuracy = (learner.correct_answers / learner.total_questions_answered * 100) if learner.total_questions_answered > 0 else 0
        stats_text = f"""
[bold]Total Questions:[/bold] {learner.total_questions_answered}
[bold]Accuracy:[/bold] {accuracy:.1f}%
[bold]Current Streak:[/bold] {learner.streak}
[bold]Best Streak:[/bold] {learner.longest_streak}
"""

        # Badges panel
        badges_text = ""
        if learner.badges:
            for badge in learner.badges:
                badges_text += f"{badge.icon} [bold]{badge.name}[/bold]\n   {badge.description}\n\n"
        else:
            badges_text = "[dim]No badges earned yet. Complete missions to earn badges![/dim]"

        # Render everything
        self.console.print(Panel(header_text, style="bold"))
        self.console.print(Panel(Markdown(info_text), title="Engineer Profile"))
        self.console.print(mastery_table)
        self.console.print(Panel(Markdown(stats_text), title="Performance Stats"))
        self.console.print(Panel(Markdown(badges_text), title="Badges Earned"))

        if learner.certified:
            cert_panel = Panel(
                "[bold green]*** CERTIFIED ML PIPELINE ENGINEER ***[/bold green]\n" +
                f"Certification Date: {learner.certification_date}",
                style="green"
            )
            self.console.print(cert_panel)

    def _create_progress_bar(self, value: float, width: int = 20) -> str:
        """Create a simple progress bar string"""
        filled = int(value * width)
        empty = width - filled
        return f"[green]{'=' * filled}[/green][dim]{'-' * empty}[/dim] {value*100:.0f}%"


# =============================================================================
# MAIN ASSESSMENT ENGINE
# =============================================================================

class ImmersiveMLAssessment:
    """
    Main orchestrator for the immersive ML pipeline learning experience.
    Coordinates all components to deliver engaging, adaptive assessment.
    """

    def __init__(self, learner_id: str, callsign: str):
        self.console = Console() if RICH_AVAILABLE else None
        self.narrative = MissionControlNarrative(self.console)
        self.question_bank = MLPipelineQuestionBank()
        self.adaptive_engine = AdaptiveDifficultyEngine()
        self.badge_system = BadgeSystem()
        self.certification = CertificationExperience(self.question_bank)
        self.dashboard = ProgressDashboard(self.console)

        # Initialize or load learner profile
        self.learner = LearnerProfile(
            learner_id=learner_id,
            callsign=callsign
        )

        # Session tracking
        self.session_questions: List[ConversationalQuestion] = []
        self.session_results: List[bool] = []

    def start_mission(self, stage: Optional[int] = None) -> None:
        """Start a learning mission for a specific stage"""
        stage = stage or self.learner.current_stage

        # Display mission intro
        intro = self.narrative.generate_mission_intro(self.learner, stage)
        self._display(intro)

    def present_challenge(self) -> Tuple[ConversationalQuestion, int]:
        """Present a challenge (question) to the learner"""
        stage = self.learner.current_stage
        difficulty = self.learner.difficulty_level

        # Get an appropriate question
        question = self.question_bank.get_question(
            stage,
            difficulty,
            exclude_topics=self.learner.strength_topics[-5:]  # Avoid recently mastered topics
        )

        if not question:
            self._display("Mission Control: No more challenges available at this level. Try advancing to the next stage!")
            return None, -1

        # Display the scenario and prompt
        self._display("\n" + "=" * 60)
        self._display(f"SITUATION BRIEFING:")
        self._display(question.scenario)
        self._display("\n" + question.prompt)
        self._display("-" * 60)

        # Display options naturally
        for i, option in enumerate(question.options):
            self._display(f"  [{i+1}] {option}")

        self._display("-" * 60)

        return question, len(question.options)

    def evaluate_response(self, question: ConversationalQuestion,
                          chosen_index: int) -> Tuple[bool, str]:
        """Evaluate learner's response and provide feedback"""
        is_correct = (chosen_index == question.correct_index)

        # Update learner stats
        self.learner.total_questions_answered += 1
        self.session_results.append(is_correct)

        if is_correct:
            self.learner.correct_answers += 1
            self.learner.streak += 1
            if self.learner.streak > self.learner.longest_streak:
                self.learner.longest_streak = self.learner.streak

            # Track mastered topic
            if question.topic not in self.learner.strength_topics:
                self.learner.strength_topics.append(question.topic)

            feedback = self.narrative.format_success(question.success_response)
        else:
            self.learner.streak = 0

            # Track struggle topic
            if question.topic not in self.learner.struggle_topics:
                self.learner.struggle_topics.append(question.topic)

            feedback = self.narrative.format_struggle(question.struggle_response)

        # Update stage mastery
        stage = question.stage
        current_mastery = self.learner.stage_mastery.get(stage, 0.0)
        # Simple exponential moving average for mastery
        new_mastery = current_mastery * 0.8 + (1.0 if is_correct else 0.0) * 0.2
        self.learner.stage_mastery[stage] = new_mastery

        # Check for adaptive difficulty adjustment
        new_difficulty = self.adaptive_engine.calculate_next_difficulty(
            self.learner, self.session_results
        )
        if new_difficulty != self.learner.difficulty_level:
            self._display(f"\n[Mission Control: Adjusting challenge level to {new_difficulty.name}]")
            self.learner.difficulty_level = new_difficulty

        # Get encouragement message
        encouragement = self.adaptive_engine.get_encouragement_message(
            self.learner, is_correct
        )

        self._display("\n" + encouragement)
        self._display(feedback)

        # Offer deep dive for curious learners
        if is_correct:
            self._display(f"\n[DEEP DIVE available: {question.deep_dive[:100]}...]")

        return is_correct, feedback

    def complete_stage(self, stage: int) -> List[Badge]:
        """Mark a stage as complete and award badges"""
        self.learner.stage_mastery[stage] = max(self.learner.stage_mastery.get(stage, 0.0), 0.8)

        # Check for new badges
        new_badges = self.badge_system.check_for_badges(self.learner, stage)

        for badge in new_badges:
            self.learner.badges.append(badge)
            self._display(f"\n*** BADGE EARNED: {badge.icon} {badge.name} ***")
            self._display(f"    {badge.description}")

        # Advance to next stage
        if stage < 8:
            self.learner.current_stage = stage + 1
            self._display(f"\nAdvancing to Stage {self.learner.current_stage}!")

        return new_badges

    def run_certification_exam(self) -> Tuple[bool, float, Dict[int, float]]:
        """Run the final certification exam"""
        self._display("\n" + "=" * 70)
        self._display("       ML PIPELINE ENGINEER CERTIFICATION EXAM")
        self._display("=" * 70)
        self._display("""
Welcome to the final certification exam, Engineer.

This exam covers all 8 stages of the ML pipeline:
- 16 questions total (2 per stage)
- You need 75% to pass
- Questions are calibrated to your demonstrated mastery

Take your time. The team believes in you.
""")

        exam_questions = self.certification.generate_certification_exam(self.learner)
        exam_results = []

        for i, question in enumerate(exam_questions):
            self._display(f"\n--- Question {i+1} of {len(exam_questions)} ---")
            self._display(f"[Stage {question.stage}: {self.narrative.get_stage_info(question.stage)[0]}]")

            self._display(question.scenario)
            self._display(question.prompt)

            for j, option in enumerate(question.options):
                self._display(f"  [{j+1}] {option}")

            # In real usage, would get user input here
            # For now, return the exam setup

        return exam_questions

    def show_dashboard(self) -> None:
        """Display the progress dashboard"""
        dashboard_output = self.dashboard.render_dashboard(self.learner)
        if not RICH_AVAILABLE:
            self._display(dashboard_output)

    def save_progress(self, filepath: str) -> None:
        """Save learner progress to file"""
        progress = asdict(self.learner)
        with open(filepath, 'w') as f:
            json.dump(progress, f, indent=2)

    def load_progress(self, filepath: str) -> None:
        """Load learner progress from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.learner = LearnerProfile(**data)

    def _display(self, message: str) -> None:
        """Display message with appropriate formatting"""
        if self.console and RICH_AVAILABLE:
            self.console.print(message)
        else:
            print(message)


# =============================================================================
# DEMO AND TESTING FUNCTIONS
# =============================================================================

def demo_immersive_assessment():
    """Demonstrate the immersive assessment system"""
    print("\n" + "=" * 70)
    print("   CULTIVATE LEARNING ML PIPELINE - IMMERSIVE ASSESSMENT DEMO")
    print("   Designed by Dr. Chris Dede, Harvard Graduate School of Education")
    print("=" * 70)

    # Create assessment instance
    assessment = ImmersiveMLAssessment(
        learner_id="demo_learner_001",
        callsign="Phoenix"
    )

    # Show initial dashboard
    print("\n--- INITIAL PROGRESS DASHBOARD ---")
    assessment.show_dashboard()

    # Start a mission
    print("\n--- STARTING MISSION ---")
    assessment.start_mission(stage=1)

    # Present a challenge
    print("\n--- PRESENTING CHALLENGE ---")
    question, num_options = assessment.present_challenge()

    if question:
        # Simulate correct answer
        print("\n--- EVALUATING RESPONSE (simulated correct answer) ---")
        is_correct, feedback = assessment.evaluate_response(question, question.correct_index)

        # Present another challenge with wrong answer
        print("\n--- PRESENTING ANOTHER CHALLENGE ---")
        question2, _ = assessment.present_challenge()
        if question2:
            print("\n--- EVALUATING RESPONSE (simulated incorrect answer) ---")
            wrong_index = (question2.correct_index + 1) % len(question2.options)
            is_correct2, feedback2 = assessment.evaluate_response(question2, wrong_index)

    # Complete the stage
    print("\n--- COMPLETING STAGE ---")
    badges = assessment.complete_stage(1)

    # Show updated dashboard
    print("\n--- UPDATED PROGRESS DASHBOARD ---")
    assessment.show_dashboard()

    # Show certification exam preview
    print("\n--- CERTIFICATION EXAM PREVIEW ---")
    exam_questions = assessment.run_certification_exam()
    print(f"\nGenerated {len(exam_questions)} certification exam questions.")

    print("\n" + "=" * 70)
    print("   DEMO COMPLETE")
    print("=" * 70)


def get_sample_question_for_stage(stage: int) -> ConversationalQuestion:
    """Get a sample question for any stage - useful for testing"""
    bank = MLPipelineQuestionBank()
    return bank.get_question(stage, DifficultyLevel.PRACTITIONER)


def create_learner_session(learner_id: str, callsign: str) -> ImmersiveMLAssessment:
    """Factory function to create a new assessment session"""
    return ImmersiveMLAssessment(learner_id, callsign)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    demo_immersive_assessment()
