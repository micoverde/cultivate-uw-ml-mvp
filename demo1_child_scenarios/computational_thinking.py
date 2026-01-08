#!/usr/bin/env python3
"""
COMPUTATIONAL THINKING PROGRESSIONS FOR ML PIPELINE LEARNING
=============================================================
Designed by Dr. Hal Abelson's principles for computational thinking education.

This module implements the four pillars of computational thinking:
1. Pattern Recognition - What patterns do you see across stages?
2. Abstraction - If we had to explain this stage in one sentence...
3. Decomposition - Let's break down this complex process...
4. Algorithm Design - How would YOU design this step?
5. Debugging/Troubleshooting - Something went wrong - let's figure it out together

Warren - This builds transferable thinking skills, not just ML knowledge.
"""

import sys
import time
import random
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# RICH TERMINAL FORMATTING (No external dependencies)
# =============================================================================

class Colors:
    """ANSI color codes for rich terminal output"""
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    # Background colors
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'

    # Reset
    RESET = '\033[0m'


def styled(text: str, *styles) -> str:
    """Apply multiple styles to text"""
    style_codes = ''.join(styles)
    return f"{style_codes}{text}{Colors.RESET}"


def header(text: str, width: int = 70, char: str = "=") -> str:
    """Create a styled header"""
    padding = (width - len(text) - 2) // 2
    line = char * width
    centered = f"{char}{' ' * padding}{text}{' ' * padding}{char}"
    if len(centered) < width:
        centered = centered[:-1] + char
    return f"\n{styled(line, Colors.BRIGHT_CYAN, Colors.BOLD)}\n{styled(centered, Colors.BRIGHT_CYAN, Colors.BOLD)}\n{styled(line, Colors.BRIGHT_CYAN, Colors.BOLD)}\n"


def subheader(text: str) -> str:
    """Create a styled subheader"""
    return f"\n{styled('>>> ' + text, Colors.BRIGHT_YELLOW, Colors.BOLD)}\n"


def thinking_prompt(text: str) -> str:
    """Create a thinking prompt box"""
    lines = text.split('\n')
    max_len = max(len(line) for line in lines) + 4
    border = styled("+" + "-" * max_len + "+", Colors.BRIGHT_MAGENTA)
    result = [border]
    for line in lines:
        padded = line + " " * (max_len - len(line) - 2)
        result.append(styled(f"| {padded} |", Colors.BRIGHT_MAGENTA))
    result.append(border)
    return '\n'.join(result)


def success_box(text: str) -> str:
    """Create a success message box"""
    return styled(f"[SUCCESS] {text}", Colors.BRIGHT_GREEN, Colors.BOLD)


def error_box(text: str) -> str:
    """Create an error message box"""
    return styled(f"[ERROR] {text}", Colors.BRIGHT_RED, Colors.BOLD)


def info_box(text: str) -> str:
    """Create an info message box"""
    return styled(f"[INFO] {text}", Colors.BRIGHT_CYAN)


def warning_box(text: str) -> str:
    """Create a warning message box"""
    return styled(f"[WARNING] {text}", Colors.BRIGHT_YELLOW)


def progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a simple progress bar"""
    filled = int(width * current / total)
    bar = "=" * filled + ">" + " " * (width - filled - 1)
    percent = int(100 * current / total)
    return styled(f"[{bar}] {percent}%", Colors.BRIGHT_GREEN)


# =============================================================================
# ML PIPELINE STAGES DEFINITION
# =============================================================================

class MLStage(Enum):
    """The 8 stages of the ML pipeline"""
    DATA_COLLECTION = 1
    DATA_PREPROCESSING = 2
    FEATURE_ENGINEERING = 3
    MODEL_SELECTION = 4
    TRAINING = 5
    EVALUATION = 6
    DEPLOYMENT = 7
    MONITORING = 8


@dataclass
class StageInfo:
    """Information about each ML pipeline stage"""
    stage: MLStage
    name: str
    description: str
    one_sentence: str  # Abstraction exercise
    key_patterns: List[str]
    sub_components: List[str]  # Decomposition
    common_errors: List[str]  # Debugging scenarios
    algorithm_prompt: str  # Algorithm design exercise


ML_PIPELINE_STAGES: Dict[MLStage, StageInfo] = {
    MLStage.DATA_COLLECTION: StageInfo(
        stage=MLStage.DATA_COLLECTION,
        name="Data Collection",
        description="Gathering raw data from various sources for training the ML model.",
        one_sentence="We find and gather the information our model will learn from.",
        key_patterns=[
            "All data has a source (sensors, surveys, logs, APIs)",
            "More data often leads to better models",
            "Data quality matters more than quantity",
            "Raw data is rarely ready to use"
        ],
        sub_components=[
            "Source Identification (Where does data come from?)",
            "Data Extraction (How do we get it?)",
            "Data Storage (Where do we keep it?)",
            "Data Validation (Is it what we expected?)"
        ],
        common_errors=[
            "Missing values in critical fields",
            "Inconsistent date formats",
            "Duplicate entries",
            "Encoding issues (UTF-8 vs ASCII)"
        ],
        algorithm_prompt="Design a system to collect classroom interaction data from video recordings."
    ),

    MLStage.DATA_PREPROCESSING: StageInfo(
        stage=MLStage.DATA_PREPROCESSING,
        name="Data Preprocessing",
        description="Cleaning and transforming raw data into a format suitable for ML.",
        one_sentence="We clean and organize the messy data so machines can understand it.",
        key_patterns=[
            "Garbage in = garbage out",
            "Preprocessing steps should be reproducible",
            "Keep the original data intact",
            "Document all transformations"
        ],
        sub_components=[
            "Missing Value Handling (Fill, drop, or flag?)",
            "Normalization (Scale values to similar ranges)",
            "Encoding (Convert text to numbers)",
            "Outlier Detection (Find unusual values)"
        ],
        common_errors=[
            "Data leakage from test to training set",
            "Losing important information when filling missing values",
            "Over-normalizing and losing distribution characteristics",
            "Incorrect handling of categorical variables"
        ],
        algorithm_prompt="Design a cleaning process for educator transcript data with typos and missing words."
    ),

    MLStage.FEATURE_ENGINEERING: StageInfo(
        stage=MLStage.FEATURE_ENGINEERING,
        name="Feature Engineering",
        description="Creating meaningful features that help the model learn patterns.",
        one_sentence="We create new measurements that help the model see what matters.",
        key_patterns=[
            "Good features capture domain knowledge",
            "Simple features often work best",
            "Features should be independent of each other",
            "The same data can yield many different features"
        ],
        sub_components=[
            "Feature Extraction (Pull information from raw data)",
            "Feature Selection (Choose the most useful ones)",
            "Feature Transformation (Combine or modify features)",
            "Feature Validation (Do they actually help?)"
        ],
        common_errors=[
            "Creating features that leak future information",
            "Too many correlated features",
            "Features that don't generalize",
            "Ignoring domain expertise"
        ],
        algorithm_prompt="Design 5 features to detect open-ended questions vs closed questions in speech."
    ),

    MLStage.MODEL_SELECTION: StageInfo(
        stage=MLStage.MODEL_SELECTION,
        name="Model Selection",
        description="Choosing the right algorithm for the problem at hand.",
        one_sentence="We pick the best tool for the specific job we're trying to do.",
        key_patterns=[
            "No single model is best for all problems",
            "Simpler models are easier to understand",
            "Complex models need more data",
            "The best model depends on the use case"
        ],
        sub_components=[
            "Problem Type Analysis (Classification? Regression?)",
            "Baseline Model (Start simple)",
            "Model Comparison (Try multiple approaches)",
            "Complexity vs Interpretability Trade-off"
        ],
        common_errors=[
            "Using a complex model when simple would work",
            "Not establishing a baseline first",
            "Ignoring inference time requirements",
            "Choosing models you don't understand"
        ],
        algorithm_prompt="Design a decision tree to choose between 3 models for question classification."
    ),

    MLStage.TRAINING: StageInfo(
        stage=MLStage.TRAINING,
        name="Model Training",
        description="Teaching the model to recognize patterns in the data.",
        one_sentence="The model practices until it learns to recognize patterns.",
        key_patterns=[
            "Training is iterative (many rounds)",
            "Too much training leads to overfitting",
            "Learning rate controls how fast we learn",
            "Validation data tells us when to stop"
        ],
        sub_components=[
            "Data Splitting (Train/Validation/Test)",
            "Forward Pass (Make predictions)",
            "Loss Calculation (Measure errors)",
            "Backward Pass (Adjust weights)"
        ],
        common_errors=[
            "Training on test data",
            "Not shuffling data between epochs",
            "Learning rate too high or too low",
            "Not saving checkpoints"
        ],
        algorithm_prompt="Design the training loop for a neural network with early stopping."
    ),

    MLStage.EVALUATION: StageInfo(
        stage=MLStage.EVALUATION,
        name="Model Evaluation",
        description="Measuring how well the model performs on unseen data.",
        one_sentence="We test the model on new examples it has never seen before.",
        key_patterns=[
            "Use data the model hasn't seen",
            "Different metrics for different goals",
            "Accuracy isn't always the best metric",
            "Compare to a meaningful baseline"
        ],
        sub_components=[
            "Metric Selection (What matters for this problem?)",
            "Test Set Evaluation (Final performance)",
            "Error Analysis (Where does it fail?)",
            "Confidence Estimation (How sure is it?)"
        ],
        common_errors=[
            "Optimizing for the wrong metric",
            "Peeking at test data during development",
            "Ignoring class imbalance",
            "Not doing error analysis"
        ],
        algorithm_prompt="Design an evaluation system that shows both accuracy AND fairness across groups."
    ),

    MLStage.DEPLOYMENT: StageInfo(
        stage=MLStage.DEPLOYMENT,
        name="Model Deployment",
        description="Making the model available for real-world use.",
        one_sentence="We put the trained model somewhere it can actually help people.",
        key_patterns=[
            "Production environment differs from development",
            "Latency and throughput matter",
            "Models need to handle edge cases",
            "Versioning is essential"
        ],
        sub_components=[
            "Model Serialization (Save the model)",
            "API Design (How to access it)",
            "Scaling Strategy (Handle many requests)",
            "Rollback Plan (What if it fails?)"
        ],
        common_errors=[
            "Different preprocessing in production",
            "No error handling for bad inputs",
            "Not monitoring response times",
            "Missing version control"
        ],
        algorithm_prompt="Design an API that serves predictions with proper error handling and logging."
    ),

    MLStage.MONITORING: StageInfo(
        stage=MLStage.MONITORING,
        name="Model Monitoring",
        description="Watching model performance over time and detecting issues.",
        one_sentence="We keep watching to make sure the model keeps working well.",
        key_patterns=[
            "Data distribution can change over time",
            "Model performance can degrade",
            "Early detection prevents disasters",
            "Feedback loops improve models"
        ],
        sub_components=[
            "Drift Detection (Data changing?)",
            "Performance Tracking (Still accurate?)",
            "Alerting (Notify when issues arise)",
            "Retraining Triggers (When to update?)"
        ],
        common_errors=[
            "Not logging predictions",
            "Ignoring slow degradation",
            "No feedback collection mechanism",
            "Missing alerting thresholds"
        ],
        algorithm_prompt="Design an alert system that detects when model accuracy drops by 5%."
    )
}


# =============================================================================
# 1. PATTERN RECOGNITION EXERCISES
# =============================================================================

class PatternRecognitionExercise:
    """
    Pattern Recognition: The ability to notice similarities, differences,
    and regularities in data. Core computational thinking skill.

    "What patterns do you see across stages?"
    """

    def __init__(self):
        self.patterns_discovered: List[str] = []
        self.comparison_insights: List[Tuple[MLStage, MLStage, str]] = []

    def cross_stage_pattern_analysis(self) -> str:
        """
        Interactive exercise: Find patterns that appear across multiple stages.
        Returns formatted analysis and prompts.
        """
        output = []
        output.append(header("PATTERN RECOGNITION: Cross-Stage Analysis"))

        output.append(styled(
            "Dr. Abelson's Insight: The same patterns appear again and again in computing.\n"
            "Learning to see these patterns is the heart of computational thinking.\n",
            Colors.ITALIC, Colors.DIM
        ))

        # Present the cross-cutting patterns
        cross_cutting_patterns = [
            {
                "pattern": "Input -> Process -> Output",
                "stages": [MLStage.DATA_COLLECTION, MLStage.TRAINING, MLStage.DEPLOYMENT],
                "description": "Every stage takes something IN, does something to it, and produces something OUT",
                "examples": [
                    "Collection: Raw sources -> Extract -> Clean data",
                    "Training: Data + Model -> Learn -> Trained weights",
                    "Deployment: Request -> Predict -> Response"
                ]
            },
            {
                "pattern": "Validate Before Proceeding",
                "stages": [MLStage.DATA_PREPROCESSING, MLStage.EVALUATION, MLStage.MONITORING],
                "description": "We always check our work before moving to the next step",
                "examples": [
                    "Preprocessing: Verify no missing values before training",
                    "Evaluation: Test on held-out data before deployment",
                    "Monitoring: Check performance before serving new users"
                ]
            },
            {
                "pattern": "Iteration and Refinement",
                "stages": [MLStage.FEATURE_ENGINEERING, MLStage.TRAINING, MLStage.MONITORING],
                "description": "We try, measure, adjust, and repeat",
                "examples": [
                    "Features: Create -> Test -> Improve -> Repeat",
                    "Training: Epoch 1 -> Measure loss -> Adjust -> Epoch 2...",
                    "Monitoring: Detect drift -> Retrain -> Deploy -> Monitor..."
                ]
            },
            {
                "pattern": "Trade-offs and Decisions",
                "stages": [MLStage.MODEL_SELECTION, MLStage.FEATURE_ENGINEERING, MLStage.DEPLOYMENT],
                "description": "Every choice has costs and benefits",
                "examples": [
                    "Model: Accuracy vs Interpretability",
                    "Features: More features vs Overfitting risk",
                    "Deployment: Latency vs Accuracy"
                ]
            }
        ]

        for i, pattern_info in enumerate(cross_cutting_patterns, 1):
            output.append(subheader(f"Pattern {i}: {pattern_info['pattern']}"))
            output.append(styled(pattern_info['description'], Colors.WHITE))
            output.append("")

            output.append(styled("Appears in stages:", Colors.BRIGHT_YELLOW))
            for stage in pattern_info['stages']:
                stage_info = ML_PIPELINE_STAGES[stage]
                output.append(styled(f"  - {stage_info.name}", Colors.BRIGHT_CYAN))

            output.append("")
            output.append(styled("Examples:", Colors.BRIGHT_YELLOW))
            for example in pattern_info['examples']:
                output.append(styled(f"    {example}", Colors.WHITE))
            output.append("")

        # Interactive prompt
        output.append(thinking_prompt(
            "THINKING CHALLENGE:\n"
            "Can you find another pattern that appears in multiple stages?\n"
            "Hint: Think about how errors are handled, or how data flows..."
        ))

        return '\n'.join(output)

    def compare_stages(self, stage_a: MLStage, stage_b: MLStage) -> str:
        """
        Compare two stages to identify similarities and differences.
        """
        info_a = ML_PIPELINE_STAGES[stage_a]
        info_b = ML_PIPELINE_STAGES[stage_b]

        output = []
        output.append(header(f"PATTERN COMPARISON: {info_a.name} vs {info_b.name}"))

        # Side by side comparison table
        output.append(styled("SIMILARITIES:", Colors.BRIGHT_GREEN, Colors.BOLD))

        # Find common patterns
        common_patterns = []
        for pattern_a in info_a.key_patterns:
            for pattern_b in info_b.key_patterns:
                if self._patterns_similar(pattern_a, pattern_b):
                    common_patterns.append((pattern_a, pattern_b))

        if common_patterns:
            for pa, pb in common_patterns[:3]:
                output.append(styled(f"  - Both emphasize: {self._extract_common_theme(pa, pb)}", Colors.GREEN))
        else:
            output.append(styled("  - Both are essential pipeline stages", Colors.GREEN))
            output.append(styled("  - Both require careful validation", Colors.GREEN))

        output.append("")
        output.append(styled("DIFFERENCES:", Colors.BRIGHT_YELLOW, Colors.BOLD))
        output.append(styled(f"  {info_a.name}: {info_a.one_sentence}", Colors.CYAN))
        output.append(styled(f"  {info_b.name}: {info_b.one_sentence}", Colors.MAGENTA))

        output.append("")
        output.append(thinking_prompt(
            f"YOUR TURN: What's one thing {info_a.name} teaches us\n"
            f"that we can use when doing {info_b.name}?"
        ))

        return '\n'.join(output)

    def _patterns_similar(self, p1: str, p2: str) -> bool:
        """Check if two patterns share common themes"""
        keywords = ['data', 'quality', 'validate', 'iterate', 'simple', 'document']
        p1_lower = p1.lower()
        p2_lower = p2.lower()
        return any(k in p1_lower and k in p2_lower for k in keywords)

    def _extract_common_theme(self, p1: str, p2: str) -> str:
        """Extract the common theme from two patterns"""
        keywords = {
            'data': 'data quality and integrity',
            'quality': 'quality over quantity',
            'validate': 'validation at each step',
            'iterate': 'iterative improvement',
            'simple': 'starting simple',
            'document': 'documentation'
        }
        for k, theme in keywords.items():
            if k in p1.lower() and k in p2.lower():
                return theme
        return "attention to detail"

    def pattern_matching_game(self) -> str:
        """
        Interactive game: Match patterns to their stages.
        """
        output = []
        output.append(header("PATTERN MATCHING GAME"))

        output.append(styled(
            "Can you match each pattern to the correct ML stage?\n",
            Colors.WHITE
        ))

        # Shuffle some patterns and stages
        challenges = [
            ("This is iterative and may run for thousands of rounds", MLStage.TRAINING),
            ("Garbage in equals garbage out", MLStage.DATA_PREPROCESSING),
            ("No single approach is best for all problems", MLStage.MODEL_SELECTION),
            ("Data distribution can change over time", MLStage.MONITORING),
            ("Good solutions capture domain knowledge", MLStage.FEATURE_ENGINEERING),
        ]

        random.shuffle(challenges)

        for i, (pattern, correct_stage) in enumerate(challenges, 1):
            output.append(styled(f"\nPattern {i}: ", Colors.BRIGHT_YELLOW) +
                         styled(f'"{pattern}"', Colors.WHITE))
            output.append(styled(f"  Answer: {ML_PIPELINE_STAGES[correct_stage].name}",
                                Colors.DIM))

        output.append("")
        output.append(thinking_prompt(
            "WHY does each pattern belong to its stage?\n"
            "Can you think of a stage where the pattern ALSO applies?"
        ))

        return '\n'.join(output)


# =============================================================================
# 2. ABSTRACTION EXERCISES
# =============================================================================

class AbstractionExercise:
    """
    Abstraction: Reducing complexity by focusing on essential details
    and ignoring irrelevant ones.

    "If we had to explain this stage in one sentence..."
    """

    def __init__(self):
        self.abstraction_levels: Dict[MLStage, List[str]] = {}

    def one_sentence_challenge(self, stage: MLStage) -> str:
        """
        Challenge: Explain a complex stage in one sentence.
        Tests ability to identify essential concepts.
        """
        info = ML_PIPELINE_STAGES[stage]

        output = []
        output.append(header(f"ONE SENTENCE ABSTRACTION: {info.name}"))

        output.append(styled(
            "Dr. Abelson says: 'The power of abstraction is that it lets us\n"
            "think about the WHAT without worrying about the HOW.'\n",
            Colors.ITALIC, Colors.DIM
        ))

        # Show the full complexity first
        output.append(subheader("The Full Picture (Complex Version)"))
        output.append(styled(info.description, Colors.WHITE))
        output.append("")
        output.append(styled("Components involved:", Colors.BRIGHT_YELLOW))
        for component in info.sub_components:
            output.append(styled(f"  - {component}", Colors.CYAN))

        output.append("")

        # Challenge them to abstract
        output.append(subheader("The Abstraction Challenge"))
        output.append(styled(
            "Your goal: Capture the ESSENCE in ONE sentence.\n"
            "What would a 10-year-old understand?\n",
            Colors.BRIGHT_GREEN
        ))

        output.append("")
        output.append(styled("Expert abstraction:", Colors.BRIGHT_YELLOW))
        output.append(styled(f'  "{info.one_sentence}"', Colors.BRIGHT_GREEN, Colors.BOLD))

        output.append("")
        output.append(thinking_prompt(
            f"Can you create a DIFFERENT one-sentence abstraction\n"
            f"for {info.name} that captures a different aspect?"
        ))

        return '\n'.join(output)

    def abstraction_ladder(self, stage: MLStage) -> str:
        """
        Show different levels of abstraction for the same concept.
        From concrete details to high-level principles.
        """
        info = ML_PIPELINE_STAGES[stage]

        output = []
        output.append(header(f"ABSTRACTION LADDER: {info.name}"))

        output.append(styled(
            "We can describe the same thing at different levels of detail.\n"
            "Moving UP the ladder = more abstract, broader, simpler.\n"
            "Moving DOWN = more concrete, specific, detailed.\n",
            Colors.DIM, Colors.ITALIC
        ))

        # Define abstraction levels for each stage
        abstraction_levels = self._get_abstraction_levels(stage)

        # Visual ladder
        output.append(styled("\n                     MOST ABSTRACT", Colors.BRIGHT_MAGENTA))
        output.append(styled("                          ^", Colors.DIM))

        for i, (level_name, description) in enumerate(abstraction_levels):
            indent = "  " * (4 - i)
            bar = "=" * (50 - len(indent))
            output.append(styled(f"{indent}+{bar}+", Colors.BRIGHT_CYAN))
            output.append(styled(f"{indent}| Level {5-i}: {level_name}", Colors.BRIGHT_YELLOW))
            output.append(styled(f"{indent}| {description[:45]}...", Colors.WHITE))
            output.append(styled(f"{indent}+{bar}+", Colors.BRIGHT_CYAN))

        output.append(styled("                          v", Colors.DIM))
        output.append(styled("                     MOST CONCRETE", Colors.BRIGHT_GREEN))

        output.append("")
        output.append(thinking_prompt(
            "At which level would you explain this to:\n"
            "1. A fellow ML engineer?\n"
            "2. A business investor?\n"
            "3. A curious 8-year-old?"
        ))

        return '\n'.join(output)

    def _get_abstraction_levels(self, stage: MLStage) -> List[Tuple[str, str]]:
        """Get abstraction levels for a specific stage"""
        levels = {
            MLStage.DATA_COLLECTION: [
                ("Philosophy", "Information is the foundation of knowledge"),
                ("Principle", "Better inputs lead to better outputs"),
                ("Process", "Gather data from sources for ML training"),
                ("Method", "Extract data via APIs, scraping, or sensors"),
                ("Code", "requests.get(url).json()['data']")
            ],
            MLStage.TRAINING: [
                ("Philosophy", "Learning is adjusting based on feedback"),
                ("Principle", "Models improve through iteration"),
                ("Process", "Feed data, calculate errors, update weights"),
                ("Method", "Forward pass, loss computation, backpropagation"),
                ("Code", "optimizer.zero_grad(); loss.backward(); optimizer.step()")
            ],
            MLStage.EVALUATION: [
                ("Philosophy", "Testing reveals truth"),
                ("Principle", "Measure on unseen examples"),
                ("Process", "Compare predictions to ground truth"),
                ("Method", "Calculate precision, recall, F1 score"),
                ("Code", "f1_score(y_true, y_pred, average='weighted')")
            ]
        }

        # Default levels for stages not specifically defined
        default = [
            ("Philosophy", f"The essence of {ML_PIPELINE_STAGES[stage].name}"),
            ("Principle", ML_PIPELINE_STAGES[stage].key_patterns[0] if ML_PIPELINE_STAGES[stage].key_patterns else "Core principle"),
            ("Process", ML_PIPELINE_STAGES[stage].one_sentence),
            ("Method", ML_PIPELINE_STAGES[stage].sub_components[0] if ML_PIPELINE_STAGES[stage].sub_components else "Key method"),
            ("Code", "# Implementation details here")
        ]

        return levels.get(stage, default)

    def interface_design_exercise(self) -> str:
        """
        Design exercise: Create clean interfaces that hide complexity.
        """
        output = []
        output.append(header("ABSTRACTION IN ACTION: Interface Design"))

        output.append(styled(
            "The best abstractions create a CLEAN INTERFACE.\n"
            "Users interact with the simple interface while\n"
            "complex implementation details stay hidden.\n",
            Colors.DIM, Colors.ITALIC
        ))

        # Show example
        output.append(subheader("Example: ML Classification Interface"))

        code = '''
# Complex reality (hidden):
class ComplexMLPipeline:
    def __init__(self):
        self.load_tokenizer()
        self.load_feature_extractor()
        self.load_neural_network()
        self.load_normalization_params()
        self.setup_gpu_acceleration()
        self.load_ensemble_models()
        # ... 50 more lines of setup ...

    def classify_with_all_details(self, text, use_gpu, batch_size,
                                   threshold, ensemble_mode, ...):
        # ... 200 lines of implementation ...
        pass

# Clean abstraction (what users see):
class MLClassifier:
    def classify(self, text: str) -> str:
        """Returns 'OEQ' or 'CEQ'"""
        return self._internal_classify(text)
'''

        output.append(styled(code, Colors.WHITE))

        output.append("")
        output.append(thinking_prompt(
            "Design YOUR clean interface for one of these:\n"
            "1. A data preprocessing module\n"
            "2. A model training system\n"
            "3. A performance monitoring dashboard\n\n"
            "What 3-5 methods would you expose to users?"
        ))

        return '\n'.join(output)


# =============================================================================
# 3. DECOMPOSITION VISUALIZATION
# =============================================================================

class DecompositionExercise:
    """
    Decomposition: Breaking complex problems into smaller,
    manageable pieces.

    "Let's break down this complex process..."
    """

    def __init__(self):
        self.decomposition_trees: Dict[MLStage, Dict] = {}

    def break_down_stage(self, stage: MLStage, depth: int = 2) -> str:
        """
        Visualize the breakdown of a stage into sub-components.
        """
        info = ML_PIPELINE_STAGES[stage]

        output = []
        output.append(header(f"DECOMPOSITION: {info.name}"))

        output.append(styled(
            "Dr. Abelson: 'The way to solve a complex problem is to\n"
            "break it into smaller problems that you know how to solve.'\n",
            Colors.ITALIC, Colors.DIM
        ))

        # Build tree visualization
        output.append(subheader("Component Tree"))

        output.append(styled(f"[{info.name}]", Colors.BRIGHT_CYAN, Colors.BOLD))
        output.append(styled("    |", Colors.DIM))

        for i, component in enumerate(info.sub_components):
            is_last = i == len(info.sub_components) - 1
            prefix = "    +--" if is_last else "    +--"
            continuation = "       " if is_last else "    |  "

            output.append(styled(f"{prefix} {component.split('(')[0].strip()}", Colors.BRIGHT_YELLOW))

            # Add sub-sub-components for richer decomposition
            sub_tasks = self._get_sub_tasks(component)
            for j, task in enumerate(sub_tasks):
                is_last_task = j == len(sub_tasks) - 1
                task_prefix = f"{continuation}+--" if is_last_task else f"{continuation}+--"
                output.append(styled(f"{task_prefix} {task}", Colors.CYAN))

            if not is_last:
                output.append(styled("    |", Colors.DIM))

        output.append("")
        output.append(thinking_prompt(
            "YOUR TURN: Pick one sub-component and break it down\n"
            "into 3-4 even smaller steps. How small can you go?"
        ))

        return '\n'.join(output)

    def _get_sub_tasks(self, component: str) -> List[str]:
        """Get sub-tasks for a component"""
        sub_tasks = {
            "Source Identification": ["List all possible sources", "Evaluate source reliability", "Check legal/ethical access"],
            "Data Extraction": ["Set up connections", "Handle authentication", "Implement extraction logic"],
            "Missing Value Handling": ["Detect missing values", "Choose fill strategy", "Apply and validate"],
            "Forward Pass": ["Load input batch", "Compute layer outputs", "Generate predictions"],
            "Metric Selection": ["Identify business goals", "Map to technical metrics", "Define thresholds"],
            "API Design": ["Define endpoints", "Document parameters", "Implement error codes"],
            "Drift Detection": ["Collect recent data", "Compare distributions", "Trigger alerts"]
        }

        # Find matching sub-tasks
        component_key = component.split('(')[0].strip()
        return sub_tasks.get(component_key, ["Define inputs", "Process", "Validate outputs"])

    def pipeline_flow_visualization(self) -> str:
        """
        Show how all 8 stages connect in a flow.
        """
        output = []
        output.append(header("ML PIPELINE FLOW: The Big Picture"))

        output.append(styled(
            "Before diving deep into any stage, let's see how they all connect.\n",
            Colors.DIM
        ))

        # ASCII flow diagram
        flow = '''
    +-------------------+
    | 1. DATA COLLECTION|
    +--------+----------+
             |
             v
    +-------------------+
    | 2. PREPROCESSING  |
    +--------+----------+
             |
             v
    +-------------------+
    | 3. FEATURE ENG.   |<--------+
    +--------+----------+         |
             |                    |
             v                    |
    +-------------------+         |
    | 4. MODEL SELECT   |         |
    +--------+----------+         |
             |                    |
             v                    | FEEDBACK LOOP
    +-------------------+         | (Iterate to improve)
    | 5. TRAINING       |         |
    +--------+----------+         |
             |                    |
             v                    |
    +-------------------+         |
    | 6. EVALUATION     |---------+
    +--------+----------+
             |
             v (If good enough)
    +-------------------+
    | 7. DEPLOYMENT     |
    +--------+----------+
             |
             v
    +-------------------+
    | 8. MONITORING     |----> [Triggers retraining if needed]
    +-------------------+
'''

        output.append(styled(flow, Colors.BRIGHT_CYAN))

        output.append("")
        output.append(thinking_prompt(
            "Notice the FEEDBACK LOOP between Evaluation and Feature Engineering.\n"
            "Why is this loop essential? What happens without it?"
        ))

        return '\n'.join(output)

    def divide_and_conquer_exercise(self, problem: str) -> str:
        """
        Practice divide-and-conquer problem solving.
        """
        output = []
        output.append(header("DIVIDE AND CONQUER EXERCISE"))

        output.append(styled(
            f'Problem: "{problem}"\n',
            Colors.BRIGHT_YELLOW
        ))

        output.append(styled(
            "Let's break this down using the D&C approach:\n",
            Colors.WHITE
        ))

        # Framework for breaking down
        steps = [
            ("1. UNDERSTAND", "What exactly are we trying to solve?"),
            ("2. DIVIDE", "What are the 2-3 main sub-problems?"),
            ("3. CONQUER", "How do we solve each sub-problem?"),
            ("4. COMBINE", "How do we put the solutions together?"),
            ("5. VERIFY", "Does the combined solution work?")
        ]

        for step, question in steps:
            output.append(styled(f"\n{step}", Colors.BRIGHT_CYAN, Colors.BOLD))
            output.append(styled(f"  {question}", Colors.WHITE))
            output.append(styled(f"  Your answer: _____________________", Colors.DIM))

        output.append("")
        output.append(thinking_prompt(
            "Example Decomposition:\n"
            "'Build an ML classifier' breaks down into:\n"
            "  1. Prepare training data\n"
            "  2. Design model architecture\n"
            "  3. Train and evaluate\n"
            "  4. Deploy and monitor\n\n"
            "Now try it with your problem!"
        ))

        return '\n'.join(output)


# =============================================================================
# 4. ALGORITHM DESIGN MOMENTS
# =============================================================================

class AlgorithmDesignExercise:
    """
    Algorithm Design: Creating step-by-step solutions to problems.

    "How would YOU design this step?"
    """

    def __init__(self):
        self.design_history: List[Dict] = []

    def design_challenge(self, stage: MLStage) -> str:
        """
        Present an algorithm design challenge for a specific stage.
        """
        info = ML_PIPELINE_STAGES[stage]

        output = []
        output.append(header(f"ALGORITHM DESIGN: {info.name}"))

        output.append(styled(
            "Dr. Abelson: 'An algorithm is a recipe for computation.\n"
            "The art is making it clear, correct, and efficient.'\n",
            Colors.ITALIC, Colors.DIM
        ))

        output.append(subheader("Your Design Challenge"))
        output.append(styled(f'"{info.algorithm_prompt}"', Colors.BRIGHT_YELLOW, Colors.BOLD))

        output.append("")
        output.append(styled("Design Framework:", Colors.BRIGHT_CYAN))

        framework = '''
Step 1: Define INPUTS
    - What information do you start with?
    - What format is it in?

Step 2: Define OUTPUTS
    - What should the algorithm produce?
    - How will you know if it's correct?

Step 3: Define the PROCESS
    - What are the main steps?
    - In what order do they happen?
    - What decisions need to be made?

Step 4: Consider EDGE CASES
    - What could go wrong?
    - How do you handle unusual inputs?
'''

        output.append(styled(framework, Colors.WHITE))

        output.append("")
        output.append(thinking_prompt(
            "Write your algorithm in PSEUDOCODE:\n\n"
            "ALGORITHM: [Your Name Here]\n"
            "INPUT: [What you receive]\n"
            "OUTPUT: [What you produce]\n\n"
            "1. [First step]\n"
            "2. [Second step]\n"
            "   IF [condition] THEN\n"
            "      [action]\n"
            "3. [Third step]\n"
            "4. RETURN [result]"
        ))

        return '\n'.join(output)

    def pseudocode_to_python(self) -> str:
        """
        Exercise: Convert pseudocode to actual Python.
        Shows the translation from design to implementation.
        """
        output = []
        output.append(header("FROM DESIGN TO CODE"))

        output.append(styled(
            "Great algorithms start as ideas, then become pseudocode,\n"
            "and finally become working programs.\n",
            Colors.DIM
        ))

        # Example: Feature extraction algorithm
        output.append(subheader("Example: Question Classification Feature Extractor"))

        pseudocode = '''
ALGORITHM: ExtractQuestionFeatures
INPUT: question_text (string)
OUTPUT: feature_vector (list of numbers)

1. CONVERT text to lowercase
2. SPLIT text into words
3. FOR each word IN oeq_keywords:
      COUNT occurrences in text
      ADD count to features
4. FOR each word IN ceq_keywords:
      COUNT occurrences in text
      ADD count to features
5. ADD word_count to features
6. ADD question_mark_count to features
7. IF starts_with('what' OR 'how' OR 'why'):
      ADD 1 to features
   ELSE:
      ADD 0 to features
8. RETURN features
'''

        output.append(styled("PSEUDOCODE:", Colors.BRIGHT_YELLOW))
        output.append(styled(pseudocode, Colors.WHITE))

        python_code = '''
def extract_question_features(question_text: str) -> list:
    """Extract features from a question for classification."""
    # Step 1
    text_lower = question_text.lower()

    # Step 2
    words = text_lower.split()

    # Step 3 & 4
    features = []
    oeq_keywords = ['what', 'how', 'why', 'explain', 'describe']
    ceq_keywords = ['is', 'are', 'do', 'does', 'can', 'will']

    for keyword in oeq_keywords:
        features.append(text_lower.count(keyword))

    for keyword in ceq_keywords:
        features.append(text_lower.count(keyword))

    # Step 5 & 6
    features.append(len(words))
    features.append(question_text.count('?'))

    # Step 7
    question_starters = ['what', 'how', 'why']
    starts_with_q = 1 if any(text_lower.startswith(q) for q in question_starters) else 0
    features.append(starts_with_q)

    # Step 8
    return features
'''

        output.append(styled("\nPYTHON CODE:", Colors.BRIGHT_GREEN))
        output.append(styled(python_code, Colors.WHITE))

        output.append("")
        output.append(thinking_prompt(
            "Notice how each pseudocode step maps to Python code.\n"
            "YOUR CHALLENGE: Add a new feature to detect if the\n"
            "question contains emotion words (happy, sad, excited, worried)."
        ))

        return '\n'.join(output)

    def algorithm_comparison(self) -> str:
        """
        Compare different algorithm approaches for the same problem.
        """
        output = []
        output.append(header("ALGORITHM COMPARISON: Multiple Solutions"))

        output.append(styled(
            "There's often more than one way to solve a problem.\n"
            "Good computational thinkers can evaluate trade-offs.\n",
            Colors.DIM
        ))

        output.append(subheader("Problem: Detect if input is an open-ended question"))

        # Approach 1: Rule-based
        output.append(styled("\nAPPROACH 1: Rule-Based", Colors.BRIGHT_YELLOW, Colors.BOLD))
        approach1 = '''
def is_open_ended_rules(text):
    """Use explicit rules to detect open-ended questions."""
    text = text.lower()

    # Open-ended indicators
    if text.startswith(('what', 'how', 'why', 'describe', 'explain')):
        if '?' in text:
            return True

    # Closed-ended indicators
    if text.startswith(('is', 'are', 'do', 'does', 'can', 'will')):
        return False

    return None  # Uncertain

PROS: Fast, interpretable, no training needed
CONS: Misses subtle cases, brittle to variations
'''
        output.append(styled(approach1, Colors.WHITE))

        # Approach 2: ML-based
        output.append(styled("\nAPPROACH 2: Machine Learning", Colors.BRIGHT_CYAN, Colors.BOLD))
        approach2 = '''
def is_open_ended_ml(text):
    """Use trained ML model to classify."""
    features = extract_features(text)
    probability = model.predict_proba([features])[0]
    return probability[1] > 0.5  # OEQ class

PROS: Handles subtle cases, improves with data
CONS: Needs training data, less interpretable
'''
        output.append(styled(approach2, Colors.WHITE))

        # Approach 3: Hybrid
        output.append(styled("\nAPPROACH 3: Hybrid", Colors.BRIGHT_MAGENTA, Colors.BOLD))
        approach3 = '''
def is_open_ended_hybrid(text):
    """Combine rules and ML for best results."""
    # First, try clear rules
    rule_result = is_open_ended_rules(text)
    if rule_result is not None:
        return rule_result

    # Fall back to ML for uncertain cases
    return is_open_ended_ml(text)

PROS: Fast for clear cases, accurate for edge cases
CONS: More complex to maintain
'''
        output.append(styled(approach3, Colors.WHITE))

        output.append("")
        output.append(thinking_prompt(
            "Which approach would you choose for:\n"
            "1. A prototype demo? (Approach ___)\n"
            "2. Production system with lots of data? (Approach ___)\n"
            "3. A system that needs to explain its decisions? (Approach ___)\n\n"
            "There's no single right answer - it depends on context!"
        ))

        return '\n'.join(output)


# =============================================================================
# 5. DEBUGGING/TROUBLESHOOTING SCENARIOS
# =============================================================================

class DebuggingExercise:
    """
    Debugging: Systematic problem-solving when things go wrong.

    "Something went wrong - let's figure it out together"
    """

    def __init__(self):
        self.debug_sessions: List[Dict] = []

    def debug_scenario(self, stage: MLStage) -> str:
        """
        Present a debugging scenario for a specific stage.
        """
        info = ML_PIPELINE_STAGES[stage]

        output = []
        output.append(header(f"DEBUGGING SCENARIO: {info.name}"))

        output.append(styled(
            "Dr. Abelson: 'Debugging is twice as hard as writing code.\n"
            "So if you write code as cleverly as you can, you are by\n"
            "definition not smart enough to debug it.'\n",
            Colors.ITALIC, Colors.DIM
        ))

        # Pick a random error for this stage
        error = random.choice(info.common_errors)

        output.append(subheader("THE SITUATION"))
        output.append(styled(f"Stage: {info.name}", Colors.CYAN))
        output.append(styled(f"Error observed: {error}", Colors.BRIGHT_RED, Colors.BOLD))

        # Create scenario details
        scenario = self._create_scenario(stage, error)

        output.append("")
        output.append(styled("What you see:", Colors.BRIGHT_YELLOW))
        output.append(styled(scenario['symptom'], Colors.WHITE))

        output.append("")
        output.append(styled("Error message:", Colors.BRIGHT_RED))
        output.append(styled(f"  {scenario['error_message']}", Colors.RED))

        output.append("")
        output.append(subheader("DEBUGGING FRAMEWORK"))

        framework = '''
1. REPRODUCE: Can you make the error happen again?
   Your check: _____________________

2. ISOLATE: What's the smallest input that causes the error?
   Your check: _____________________

3. HYPOTHESIZE: What might be causing this?
   Your hypothesis: _____________________

4. TEST: How can you test your hypothesis?
   Your test: _____________________

5. FIX: What change would solve the problem?
   Your fix: _____________________

6. VERIFY: Does the fix work? Any side effects?
   Your verification: _____________________
'''

        output.append(styled(framework, Colors.WHITE))

        output.append("")
        output.append(styled("Hint:", Colors.BRIGHT_GREEN))
        output.append(styled(f"  {scenario['hint']}", Colors.GREEN))

        output.append("")
        output.append(thinking_prompt(
            "Work through the debugging framework above.\n"
            "What's your diagnosis? What would you try first?"
        ))

        return '\n'.join(output)

    def _create_scenario(self, stage: MLStage, error: str) -> Dict[str, str]:
        """Create a detailed scenario for a specific error"""
        scenarios = {
            "Missing values in critical fields": {
                "symptom": "Model training fails after 10 epochs with NaN loss",
                "error_message": "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')",
                "hint": "Check if all required columns exist in your CSV. Use df.isnull().sum() to find missing values."
            },
            "Inconsistent date formats": {
                "symptom": "Some records parse correctly, others show wrong dates",
                "error_message": "Warning: Parsed date 2025-01-12 as 2012-01-25 (day/month/year format detected)",
                "hint": "Check if dates are sometimes MM/DD/YYYY and sometimes DD/MM/YYYY"
            },
            "Data leakage from test to training set": {
                "symptom": "Training accuracy is 99%, but production accuracy is 60%",
                "error_message": "No error - but suspicious performance gap",
                "hint": "Check if any test data was accidentally used during training or validation"
            },
            "Training on test data": {
                "symptom": "Perfect test accuracy, terrible real-world performance",
                "error_message": "No error - metrics look great but something is wrong",
                "hint": "Verify your data splitting happened BEFORE any preprocessing"
            },
            "Learning rate too high or too low": {
                "symptom": "Loss explodes to infinity OR decreases extremely slowly",
                "error_message": "Loss: NaN (if too high) or Loss: 0.8532 -> 0.8531 -> 0.8530 (if too low)",
                "hint": "Try learning rates of 0.01, 0.001, and 0.0001 and compare"
            },
            "Not saving checkpoints": {
                "symptom": "Training crashed after 4 hours and you have to start over",
                "error_message": "RuntimeError: CUDA out of memory",
                "hint": "Save model state every N epochs: torch.save(model.state_dict(), 'checkpoint.pth')"
            },
            "Optimizing for the wrong metric": {
                "symptom": "High accuracy but users complain the model is wrong",
                "error_message": "No error - 95% accuracy achieved!",
                "hint": "If 95% of data is one class, predicting that class always gives 95% accuracy"
            },
            "Different preprocessing in production": {
                "symptom": "Model works in testing but fails in production",
                "error_message": "ValueError: Could not convert string to float: 'What do you think?'",
                "hint": "Ensure the same tokenizer/scaler/encoder is used in production as training"
            },
            "No error handling for bad inputs": {
                "symptom": "API crashes when user sends unexpected data",
                "error_message": "TypeError: 'NoneType' object has no attribute 'lower'",
                "hint": "Add input validation: if text is None: return error_response()"
            },
            "Not logging predictions": {
                "symptom": "Can't figure out why some users get wrong answers",
                "error_message": "No error - but no way to investigate issues",
                "hint": "Log: timestamp, input, output, confidence for every prediction"
            }
        }

        # Default scenario if error not found
        default = {
            "symptom": f"Something isn't working correctly in {ML_PIPELINE_STAGES[stage].name}",
            "error_message": f"Unexpected behavior: {error}",
            "hint": "Start by checking your inputs and outputs at each step"
        }

        return scenarios.get(error, default)

    def systematic_debugging_tutorial(self) -> str:
        """
        Teach the systematic approach to debugging.
        """
        output = []
        output.append(header("SYSTEMATIC DEBUGGING: The Scientific Method"))

        output.append(styled(
            "Debugging IS the scientific method applied to code:\n"
            "Observe -> Hypothesize -> Experiment -> Conclude\n",
            Colors.DIM
        ))

        # The process
        steps = [
            ("OBSERVE",
             "What exactly is happening?",
             "Document the exact error message, the inputs that cause it, and when it occurs.",
             Colors.BRIGHT_CYAN),

            ("REPRODUCE",
             "Can you make it happen again?",
             "If you can't reproduce it, you can't fix it. Find the minimal steps to trigger the bug.",
             Colors.BRIGHT_YELLOW),

            ("HYPOTHESIZE",
             "What might be causing this?",
             "Based on the symptoms, list 2-3 possible causes. Start with the most likely.",
             Colors.BRIGHT_MAGENTA),

            ("EXPERIMENT",
             "How can you test your hypothesis?",
             "Add print statements, check intermediate values, simplify the input.",
             Colors.BRIGHT_GREEN),

            ("CONCLUDE",
             "What did you learn?",
             "Either confirm the cause and fix it, or eliminate a possibility and try the next hypothesis.",
             Colors.BRIGHT_BLUE),
        ]

        for name, question, description, color in steps:
            output.append("")
            output.append(styled(f"Step: {name}", color, Colors.BOLD))
            output.append(styled(f"  Question: {question}", Colors.WHITE))
            output.append(styled(f"  Action: {description}", Colors.DIM))

        output.append("")
        output.append(subheader("DEBUG TOOLS CHEAT SHEET"))

        tools = '''
+------------------------+------------------------------------------+
| Tool                   | When to Use                              |
+------------------------+------------------------------------------+
| print(variable)        | Quick check of a value                   |
| type(variable)         | When types might be wrong                |
| len(collection)        | When data sizes seem off                 |
| assert condition       | To verify assumptions                    |
| try/except             | To catch and examine exceptions          |
| breakpoint()           | To pause and inspect state (Python 3.7+) |
| logging.debug()        | For production debugging                 |
+------------------------+------------------------------------------+
'''

        output.append(styled(tools, Colors.CYAN))

        output.append("")
        output.append(thinking_prompt(
            "Next time you hit a bug, RESIST the urge to randomly\n"
            "change things. Instead, follow the scientific method:\n"
            "Observe -> Hypothesize -> Experiment -> Conclude"
        ))

        return '\n'.join(output)

    def rubber_duck_debugging(self) -> str:
        """
        Teach the rubber duck debugging technique.
        """
        output = []
        output.append(header("RUBBER DUCK DEBUGGING"))

        output.append(styled(
            "Sometimes the best debugger is explaining your code to a rubber duck.\n"
            "When you explain step-by-step, you often find the bug yourself!\n",
            Colors.DIM
        ))

        output.append(styled('''
    ,~~.
   (  6 6 )
   /`--'\\
  //|  |\\\\
 (/ |  | \\)
    `""`

"Tell me about your bug. I'm listening."
''', Colors.BRIGHT_YELLOW))

        prompts = [
            "1. What are you trying to accomplish?",
            "2. What does your code do step by step?",
            "3. What did you expect to happen?",
            "4. What actually happened?",
            "5. At which step does expected diverge from actual?"
        ]

        output.append(styled("\nThe Duck's Questions:", Colors.BRIGHT_CYAN, Colors.BOLD))
        for prompt in prompts:
            output.append(styled(f"  {prompt}", Colors.WHITE))

        output.append("")
        output.append(thinking_prompt(
            "Try it now! Explain your current code or problem\n"
            "out loud as if the duck is listening.\n\n"
            "You'll be amazed how often the act of explaining\n"
            "reveals the bug you couldn't see before."
        ))

        return '\n'.join(output)


# =============================================================================
# INTEGRATED LEARNING SESSION
# =============================================================================

class ComputationalThinkingSession:
    """
    A complete computational thinking learning session that integrates
    all five pillars: Pattern Recognition, Abstraction, Decomposition,
    Algorithm Design, and Debugging.
    """

    def __init__(self):
        self.pattern_exercises = PatternRecognitionExercise()
        self.abstraction_exercises = AbstractionExercise()
        self.decomposition_exercises = DecompositionExercise()
        self.algorithm_exercises = AlgorithmDesignExercise()
        self.debugging_exercises = DebuggingExercise()

        self.session_history: List[str] = []
        self.skills_practiced: Dict[str, int] = {
            "pattern_recognition": 0,
            "abstraction": 0,
            "decomposition": 0,
            "algorithm_design": 0,
            "debugging": 0
        }

    def welcome(self) -> str:
        """Display welcome message and session overview"""
        output = []
        output.append(header("COMPUTATIONAL THINKING FOR ML PIPELINES", char="*"))

        output.append(styled(
            "Welcome to your computational thinking journey!\n\n"
            "You'll develop five core skills that transfer far beyond ML:\n",
            Colors.WHITE
        ))

        skills = [
            ("1. Pattern Recognition", "See what's similar across different contexts", Colors.BRIGHT_CYAN),
            ("2. Abstraction", "Focus on what matters, ignore what doesn't", Colors.BRIGHT_YELLOW),
            ("3. Decomposition", "Break big problems into small pieces", Colors.BRIGHT_GREEN),
            ("4. Algorithm Design", "Create step-by-step solutions", Colors.BRIGHT_MAGENTA),
            ("5. Debugging", "Systematically find and fix problems", Colors.BRIGHT_RED),
        ]

        for skill, desc, color in skills:
            output.append(styled(f"  {skill}", color, Colors.BOLD))
            output.append(styled(f"    {desc}", Colors.DIM))

        output.append("")
        output.append(styled(
            "These skills will help you understand ML pipelines AND make you\n"
            "a better problem-solver in any domain.\n",
            Colors.WHITE
        ))

        output.append(styled(
            "\"The essence of computer science is abstraction and automation.\"\n"
            "  - Dr. Hal Abelson, MIT\n",
            Colors.ITALIC, Colors.DIM
        ))

        return '\n'.join(output)

    def run_exercise(self, skill: str, stage: Optional[MLStage] = None) -> str:
        """Run a specific exercise"""

        if stage is None:
            stage = random.choice(list(MLStage))

        if skill == "pattern":
            self.skills_practiced["pattern_recognition"] += 1
            return self.pattern_exercises.cross_stage_pattern_analysis()

        elif skill == "abstraction":
            self.skills_practiced["abstraction"] += 1
            return self.abstraction_exercises.one_sentence_challenge(stage)

        elif skill == "decomposition":
            self.skills_practiced["decomposition"] += 1
            return self.decomposition_exercises.break_down_stage(stage)

        elif skill == "algorithm":
            self.skills_practiced["algorithm_design"] += 1
            return self.algorithm_exercises.design_challenge(stage)

        elif skill == "debug":
            self.skills_practiced["debugging"] += 1
            return self.debugging_exercises.debug_scenario(stage)

        else:
            return error_box(f"Unknown skill: {skill}")

    def show_progress(self) -> str:
        """Show progress across all skills"""
        output = []
        output.append(header("YOUR COMPUTATIONAL THINKING PROGRESS"))

        total = sum(self.skills_practiced.values())

        if total == 0:
            output.append(styled("You haven't practiced any skills yet. Let's get started!", Colors.WHITE))
            return '\n'.join(output)

        for skill, count in self.skills_practiced.items():
            skill_display = skill.replace("_", " ").title()
            bar = progress_bar(count, max(total, 5), width=30)
            output.append(f"{skill_display:20} {bar} ({count} exercises)")

        output.append("")
        output.append(styled(f"Total exercises completed: {total}", Colors.BRIGHT_GREEN, Colors.BOLD))

        return '\n'.join(output)

    def interactive_menu(self) -> str:
        """Show the interactive menu"""
        output = []
        output.append(subheader("CHOOSE YOUR EXERCISE"))

        menu = '''
+---+----------------------+----------------------------------------+
| # | Exercise Type        | What You'll Practice                   |
+---+----------------------+----------------------------------------+
| 1 | Pattern Recognition  | Find patterns across ML pipeline stages|
| 2 | Abstraction          | Explain complex concepts simply        |
| 3 | Decomposition        | Break down stages into components      |
| 4 | Algorithm Design     | Create step-by-step solutions          |
| 5 | Debugging            | Troubleshoot ML pipeline errors        |
| 6 | Pipeline Overview    | See all 8 stages and how they connect  |
| 7 | Show Progress        | View your skill development            |
| 0 | Exit                 | End the session                        |
+---+----------------------+----------------------------------------+
'''

        output.append(styled(menu, Colors.CYAN))

        return '\n'.join(output)


# =============================================================================
# MAIN DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_pattern_recognition():
    """Demo of pattern recognition exercises"""
    exercise = PatternRecognitionExercise()

    print(exercise.cross_stage_pattern_analysis())
    print("\n" + "=" * 70 + "\n")
    print(exercise.compare_stages(MLStage.TRAINING, MLStage.MONITORING))
    print("\n" + "=" * 70 + "\n")
    print(exercise.pattern_matching_game())


def demonstrate_abstraction():
    """Demo of abstraction exercises"""
    exercise = AbstractionExercise()

    print(exercise.one_sentence_challenge(MLStage.TRAINING))
    print("\n" + "=" * 70 + "\n")
    print(exercise.abstraction_ladder(MLStage.EVALUATION))
    print("\n" + "=" * 70 + "\n")
    print(exercise.interface_design_exercise())


def demonstrate_decomposition():
    """Demo of decomposition exercises"""
    exercise = DecompositionExercise()

    print(exercise.break_down_stage(MLStage.DATA_PREPROCESSING))
    print("\n" + "=" * 70 + "\n")
    print(exercise.pipeline_flow_visualization())
    print("\n" + "=" * 70 + "\n")
    print(exercise.divide_and_conquer_exercise("Build an ML system that detects student engagement from video"))


def demonstrate_algorithm_design():
    """Demo of algorithm design exercises"""
    exercise = AlgorithmDesignExercise()

    print(exercise.design_challenge(MLStage.FEATURE_ENGINEERING))
    print("\n" + "=" * 70 + "\n")
    print(exercise.pseudocode_to_python())
    print("\n" + "=" * 70 + "\n")
    print(exercise.algorithm_comparison())


def demonstrate_debugging():
    """Demo of debugging exercises"""
    exercise = DebuggingExercise()

    print(exercise.debug_scenario(MLStage.TRAINING))
    print("\n" + "=" * 70 + "\n")
    print(exercise.systematic_debugging_tutorial())
    print("\n" + "=" * 70 + "\n")
    print(exercise.rubber_duck_debugging())


def run_full_session():
    """Run a complete computational thinking session"""
    session = ComputationalThinkingSession()

    print(session.welcome())

    # Demo each skill
    for skill in ["pattern", "abstraction", "decomposition", "algorithm", "debug"]:
        print("\n" + "=" * 70 + "\n")
        stage = random.choice(list(MLStage))
        print(session.run_exercise(skill, stage))

    print("\n" + "=" * 70 + "\n")
    print(session.show_progress())


def main():
    """Main entry point with menu"""
    session = ComputationalThinkingSession()

    print(session.welcome())

    while True:
        print(session.interactive_menu())

        try:
            choice = input(styled("\nEnter your choice (0-7): ", Colors.BRIGHT_GREEN))
        except (KeyboardInterrupt, EOFError):
            print("\n" + styled("Session ended. Keep thinking computationally!", Colors.BRIGHT_YELLOW))
            break

        stage = random.choice(list(MLStage))

        if choice == "1":
            print(session.run_exercise("pattern", stage))
        elif choice == "2":
            print(session.run_exercise("abstraction", stage))
        elif choice == "3":
            print(session.run_exercise("decomposition", stage))
        elif choice == "4":
            print(session.run_exercise("algorithm", stage))
        elif choice == "5":
            print(session.run_exercise("debug", stage))
        elif choice == "6":
            exercise = DecompositionExercise()
            print(exercise.pipeline_flow_visualization())
        elif choice == "7":
            print(session.show_progress())
        elif choice == "0":
            print(styled("\nSession ended. Keep thinking computationally!", Colors.BRIGHT_YELLOW))
            break
        else:
            print(error_box("Invalid choice. Please enter 0-7."))

        print("\n" + "-" * 70)
        input(styled("\nPress Enter to continue...", Colors.DIM))


if __name__ == "__main__":
    # Run demo of all exercises
    print(styled("\n" + "=" * 70, Colors.BRIGHT_CYAN))
    print(styled("COMPUTATIONAL THINKING PROGRESSIONS - DEMO MODE", Colors.BRIGHT_CYAN, Colors.BOLD))
    print(styled("=" * 70 + "\n", Colors.BRIGHT_CYAN))

    # Quick demo of each exercise type
    run_full_session()
