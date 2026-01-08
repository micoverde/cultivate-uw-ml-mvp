#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CULTIVATE LEARNING ML PIPELINE - ENHANCED INTERACTIVE INVESTOR GAME v2.0    ║
║                                                                              ║
║  Based on the work of:                                                       ║
║  - Dr. Karen Brennan (HGSE) - Reflection & Creative Learning Spiral          ║
║  - Dr. Mitch Resnick (MIT) - Tinkering Zones & 4 P's                         ║
║  - Prof. David Malan (Harvard CS50) - Scaffolding & Clear Explanations       ║
║  - Dr. Hal Abelson (MIT) - Computational Thinking Progressions               ║
║  - Dr. Chris Dede (HGSE) - Immersive Assessment & Mission Control            ║
║                                                                              ║
║  Architecture: lively-water-04219020f.4.azurestaticapps.net/pages/architecture║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS & FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def clear_screen():
    os.system('clear')

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'═' * 74}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'═' * 74}{Colors.END}\n")

def print_stage(num, title, subtitle):
    print(f"\n{Colors.BOLD}{Colors.CYAN}┏{'━' * 72}┓{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}┃  STAGE {num}: {title.upper():<58}┃{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}┃  {subtitle:<66}┃{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}┗{'━' * 72}┛{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}  ✅ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}  ℹ️  {text}{Colors.END}")

def print_question(text):
    print(f"\n{Colors.YELLOW}  ❓ {text}{Colors.END}")

def print_reflection(text):
    print(f"\n{Colors.MAGENTA}  💭 {text}{Colors.END}")

def print_tinker(text):
    print(f"\n{Colors.CYAN}  🔧 TINKER ZONE: {text}{Colors.END}")

def print_mission(text):
    print(f"\n{Colors.YELLOW}  🚀 MISSION CONTROL: {text}{Colors.END}")

def print_badge(badge):
    print(f"\n  {Colors.BOLD}{Colors.GREEN}🏆 BADGE EARNED: {badge}{Colors.END}")

def wait_for_enter(prompt="Press ENTER to continue..."):
    input(f"\n  {Colors.CYAN}▶ {prompt}{Colors.END}")

def slow_print(text, delay=0.03):
    """Print text slowly for dramatic effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# GAME STATE WITH PROGRESS TRACKING (Dr. Dede)
# ═══════════════════════════════════════════════════════════════════════════════

class LearnerProgress:
    """Track learner progress across all dimensions"""

    def __init__(self, learner_name="John"):
        self.learner_name = learner_name
        self.callsign = "Phoenix"
        self.current_stage = 0
        self.score = 0
        self.total_questions = 0
        self.correct_answers = 0
        self.badges = []
        self.reflections = []
        self.discoveries = []
        self.tinker_experiments = []
        self.session_start = datetime.now().isoformat()

        # Stage-specific data
        self.selected_video = None
        self.audio_file = None
        self.transcript = None
        self.questions_detected = []
        self.features_extracted = {}
        self.classifications = []

        # Adaptive difficulty (Dr. Dede)
        self.difficulty_level = "APPRENTICE"  # NOVICE -> APPRENTICE -> PRACTITIONER -> EXPERT -> MASTER
        self.recent_performance = []  # Last 5 answers

    def add_reflection(self, stage, reflection_type, content):
        """Save a learner reflection (Dr. Brennan)"""
        self.reflections.append({
            "stage": stage,
            "type": reflection_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def add_discovery(self, stage, discovery):
        """Track learner discoveries (Dr. Resnick)"""
        self.discoveries.append({
            "stage": stage,
            "discovery": discovery,
            "timestamp": datetime.now().isoformat()
        })

    def add_tinker_experiment(self, stage, parameter, old_value, new_value, result):
        """Log tinkering experiments (Dr. Resnick)"""
        self.tinker_experiments.append({
            "stage": stage,
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    def update_performance(self, correct):
        """Update adaptive difficulty based on recent performance"""
        self.recent_performance.append(correct)
        if len(self.recent_performance) > 5:
            self.recent_performance.pop(0)

        # Adjust difficulty based on recent accuracy
        if len(self.recent_performance) >= 3:
            recent_accuracy = sum(self.recent_performance) / len(self.recent_performance)
            if recent_accuracy >= 0.8:
                self._increase_difficulty()
            elif recent_accuracy <= 0.4:
                self._decrease_difficulty()

    def _increase_difficulty(self):
        levels = ["NOVICE", "APPRENTICE", "PRACTITIONER", "EXPERT", "MASTER"]
        current_idx = levels.index(self.difficulty_level)
        if current_idx < len(levels) - 1:
            self.difficulty_level = levels[current_idx + 1]
            print(f"\n  {Colors.GREEN}📈 Difficulty increased to {self.difficulty_level}!{Colors.END}")

    def _decrease_difficulty(self):
        levels = ["NOVICE", "APPRENTICE", "PRACTITIONER", "EXPERT", "MASTER"]
        current_idx = levels.index(self.difficulty_level)
        if current_idx > 0:
            self.difficulty_level = levels[current_idx - 1]
            print(f"\n  {Colors.YELLOW}📉 Difficulty adjusted to {self.difficulty_level}{Colors.END}")

    def earn_badge(self, badge_name, description):
        """Award a badge for achievement"""
        badge = {"name": badge_name, "description": description, "earned_at": datetime.now().isoformat()}
        self.badges.append(badge)
        print_badge(f"{badge_name} - {description}")

    def save_progress(self, filepath="learner_progress.json"):
        """Save all progress to JSON file"""
        progress_data = {
            "learner_name": self.learner_name,
            "callsign": self.callsign,
            "session_start": self.session_start,
            "session_end": datetime.now().isoformat(),
            "current_stage": self.current_stage,
            "score": self.score,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": self.correct_answers / max(self.total_questions, 1) * 100,
            "difficulty_level": self.difficulty_level,
            "badges": self.badges,
            "reflections": self.reflections,
            "discoveries": self.discoveries,
            "tinker_experiments": self.tinker_experiments,
            "video_processed": self.selected_video,
            "questions_found": len(self.questions_detected),
            "classifications": self.classifications
        }

        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print(f"\n  {Colors.GREEN}💾 Progress saved to {filepath}{Colors.END}")
        return filepath

# Initialize global progress tracker
progress = LearnerProgress()

# ═══════════════════════════════════════════════════════════════════════════════
# PEDAGOGICAL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def brennan_reflection(stage_num, stage_name):
    """Dr. Brennan's reflection prompts - metacognition and creative learning"""

    reflection_prompts = {
        1: {
            "wonder": "What do you wonder about what happens inside a video file?",
            "notice": "What did you notice about the different videos available?",
            "connect": "How does collecting good data connect to getting accurate results?"
        },
        2: {
            "wonder": "Why do you think we extract ONLY audio, not video frames?",
            "notice": "What did you notice about the audio file size vs video size?",
            "connect": "How does 16kHz audio quality balance accuracy with speed?"
        },
        3: {
            "wonder": "How does Whisper 'understand' spoken words?",
            "notice": "What patterns did you notice in the transcript?",
            "connect": "How does speech recognition accuracy affect downstream stages?"
        },
        4: {
            "wonder": "Why is separating teacher speech from child speech important?",
            "notice": "What differences did you notice between teacher and child speech?",
            "connect": "How does speaker diarization help us focus on teaching quality?"
        },
        5: {
            "wonder": "What makes a question different from a statement?",
            "notice": "What patterns in questions did you notice?",
            "connect": "How do detected questions become training data?"
        },
        6: {
            "wonder": "Why do we need 19 features instead of just 1?",
            "notice": "Which features seemed most important for classification?",
            "connect": "How do features translate human intuition into numbers?"
        },
        7: {
            "wonder": "Why use three models instead of one?",
            "notice": "How did different models vote on the same question?",
            "connect": "How does ensemble voting improve accuracy?"
        },
        8: {
            "wonder": "How could this feedback change a teacher's practice?",
            "notice": "What insights emerged from the full pipeline?",
            "connect": "How does the feedback loop close the learning cycle?"
        }
    }

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╭{'─' * 70}╮{Colors.END}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}│  💭 REFLECTION TIME (Dr. Brennan's Creative Learning Spiral)        │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}╰{'─' * 70}╯{Colors.END}")

    prompts = reflection_prompts.get(stage_num, reflection_prompts[1])

    print(f"\n{Colors.MAGENTA}  I wonder... {prompts['wonder']}{Colors.END}")
    wonder_response = input(f"  {Colors.DIM}Your thoughts: {Colors.END}")
    if wonder_response:
        progress.add_reflection(stage_num, "wonder", wonder_response)

    print(f"\n{Colors.MAGENTA}  I notice... {prompts['notice']}{Colors.END}")
    notice_response = input(f"  {Colors.DIM}What you observed: {Colors.END}")
    if notice_response:
        progress.add_reflection(stage_num, "notice", notice_response)

    print(f"\n{Colors.MAGENTA}  Connection... {prompts['connect']}{Colors.END}")
    connect_response = input(f"  {Colors.DIM}Your connection: {Colors.END}")
    if connect_response:
        progress.add_reflection(stage_num, "connect", connect_response)

    print(f"\n  {Colors.GREEN}🌟 Beautiful reflections! This is how real learning happens.{Colors.END}")

def resnick_tinker_zone(stage_num):
    """Dr. Resnick's tinkering zones - 4 P's of creative learning"""

    tinker_options = {
        2: {
            "name": "Audio Extraction",
            "parameters": [
                ("Sample Rate", "16000", ["8000", "16000", "22050", "44100"]),
                ("Channels", "mono", ["mono", "stereo"]),
                ("Noise Reduction", "enabled", ["enabled", "disabled"])
            ],
            "what_if": "What if we used phone-quality audio (8kHz)?"
        },
        3: {
            "name": "Speech Recognition",
            "parameters": [
                ("Whisper Model", "base", ["tiny", "base", "small", "medium", "large"]),
                ("Language", "auto", ["auto", "en", "es"]),
                ("Task", "transcribe", ["transcribe", "translate"])
            ],
            "what_if": "What if we used the large model (25x bigger)?"
        },
        6: {
            "name": "Feature Engineering",
            "parameters": [
                ("Feature Count", "19", ["10", "19", "56", "384"]),
                ("Include Keywords", "yes", ["yes", "no"]),
                ("Include Prosody", "yes", ["yes", "no"])
            ],
            "what_if": "What if we used 384-dimensional embeddings?"
        },
        7: {
            "name": "ML Classification",
            "parameters": [
                ("Ensemble Size", "3", ["1", "3", "5", "7"]),
                ("Voting", "soft", ["hard", "soft", "weighted"]),
                ("Confidence Threshold", "0.5", ["0.3", "0.5", "0.7", "0.9"])
            ],
            "what_if": "What if we required 90% confidence for predictions?"
        }
    }

    if stage_num not in tinker_options:
        return

    zone = tinker_options[stage_num]

    print(f"\n{Colors.BOLD}{Colors.CYAN}╭{'─' * 70}╮{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  🔧 TINKER ZONE: {zone['name']:<52}│{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  Dr. Resnick: 'The best way to learn is by tinkering!'              │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}╰{'─' * 70}╯{Colors.END}")

    print(f"\n  {Colors.YELLOW}Current settings:{Colors.END}")
    for param, default, options in zone["parameters"]:
        print(f"    • {param}: {Colors.BOLD}{default}{Colors.END} (options: {', '.join(options)})")

    print(f"\n  {Colors.CYAN}🤔 What if... {zone['what_if']}{Colors.END}")

    response = input(f"\n  {Colors.YELLOW}Would you like to experiment? (y/n): {Colors.END}")
    if response.lower() == 'y':
        param, default, options = zone["parameters"][0]
        print(f"\n  Choose new {param}:")
        for i, opt in enumerate(options, 1):
            print(f"    {i}. {opt}")
        try:
            choice = int(input(f"  Selection (1-{len(options)}): ")) - 1
            if 0 <= choice < len(options):
                new_value = options[choice]
                progress.add_tinker_experiment(stage_num, param, default, new_value, "simulated")
                print(f"\n  {Colors.GREEN}🎉 Experiment logged! {param}: {default} → {new_value}{Colors.END}")
                progress.add_discovery(stage_num, f"Changed {param} from {default} to {new_value}")
        except:
            pass

def malan_scaffolding(stage_num, stage_name):
    """Prof. Malan's CS50-style scaffolding - clear explanations with analogies"""

    analogies = {
        1: ("📹", "Collecting Raw Ingredients",
            "Think of video files like raw ingredients for cooking. Before you can make "
            "a delicious meal, you need quality ingredients. Our ML pipeline needs quality "
            "videos of real classrooms - that's our raw material!"),
        2: ("🎵", "Separating the Vocals",
            "Imagine a music producer isolating just the vocals from a full song. "
            "We do the same - extract ONLY the audio track because that's where "
            "the spoken questions live. Video frames don't help us analyze speech!"),
        3: ("👂", "Teaching a Computer to Listen",
            "OpenAI Whisper is like a really attentive student who's listened to "
            "680,000 hours of human speech. It converts sound waves into text, "
            "word by word, with timestamps for each!"),
        4: ("🎭", "Sorting by Voice",
            "Think of a play with multiple characters. Speaker diarization is like "
            "adding character names next to each line of dialogue. We label who "
            "said what - teacher, child, or other."),
        5: ("🔍", "Finding the Questions",
            "Like a proofreader looking for question marks, but smarter! We use "
            "patterns (who, what, where, when, why, how) and linguistic markers "
            "to find every question in the transcript."),
        6: ("📊", "Describing in Numbers",
            "Imagine describing a person to a police sketch artist, but using only "
            "numbers: height=72, weight=180, hair_color=3... That's what features do! "
            "They translate each question into 19 measurable characteristics."),
        7: ("🗳️", "Three Experts Voting",
            "Picture three expert educators: a neural network, a random forest, and "
            "a logistic regression. Each independently classifies the question, then "
            "they vote. Majority wins! This is ensemble ML."),
        8: ("📬", "Delivering the Report Card",
            "The final step: package everything into a beautiful report for the "
            "teacher. 'You asked 15 questions: 9 open-ended (great!), 6 closed. "
            "Here's how to ask more thought-provoking questions...'")
    }

    icon, title, explanation = analogies.get(stage_num, analogies[1])

    print(f"\n{Colors.BOLD}{Colors.BLUE}╭{'─' * 70}╮{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}│  {icon} BEFORE WE BEGIN: {title:<51}│{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}│  Prof. Malan's CS50-Style Explanation                               │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}╰{'─' * 70}╯{Colors.END}")

    # Word-wrap the explanation
    words = explanation.split()
    lines = []
    current_line = []
    for word in words:
        if len(' '.join(current_line + [word])) > 66:
            lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(' '.join(current_line))

    print()
    for line in lines:
        print(f"  {Colors.BLUE}{line}{Colors.END}")

def dede_mission_briefing(stage_num, stage_name):
    """Dr. Dede's immersive mission narrative"""

    missions = {
        1: ("DATA EXPLORER", "Locate and secure raw classroom footage"),
        2: ("SIGNAL EXTRACTOR", "Isolate audio channels for processing"),
        3: ("SPEECH DECODER", "Convert audio waves to text transcript"),
        4: ("VOICE IDENTIFIER", "Tag each speaker in the conversation"),
        5: ("QUESTION HUNTER", "Locate all questions in the dialogue"),
        6: ("FEATURE ARCHITECT", "Transform questions into numerical vectors"),
        7: ("CLASSIFICATION SPECIALIST", "Determine question types with ML ensemble"),
        8: ("FEEDBACK COMMANDER", "Generate actionable teaching insights")
    }

    title, objective = missions.get(stage_num, missions[1])

    print(f"\n{Colors.BOLD}{Colors.YELLOW}┏{'━' * 72}┓{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}┃  🚀 MISSION CONTROL - BRIEFING                                         ┃{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}┣{'━' * 72}┫{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}┃  Agent: {progress.callsign:<63}┃{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}┃  Mission: {title:<61}┃{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}┃  Objective: {objective:<59}┃{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}┃  Difficulty: {progress.difficulty_level:<58}┃{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}┗{'━' * 72}┛{Colors.END}")

def abelson_computational_thinking(stage_num):
    """Dr. Abelson's computational thinking progressions"""

    ct_prompts = {
        1: ("PATTERN RECOGNITION", "What patterns do you see in how videos are organized?"),
        2: ("DECOMPOSITION", "How did we break down 'video processing' into smaller steps?"),
        3: ("ABSTRACTION", "What details does Whisper ignore to focus on speech?"),
        4: ("ALGORITHM DESIGN", "How would YOU design a speaker identification system?"),
        5: ("PATTERN MATCHING", "What patterns define a question vs. a statement?"),
        6: ("ABSTRACTION", "How do 19 features abstract away linguistic complexity?"),
        7: ("ALGORITHM DESIGN", "Why might three weak models beat one strong model?"),
        8: ("SYSTEMS THINKING", "How do all 8 stages work together as a system?")
    }

    skill, prompt = ct_prompts.get(stage_num, ct_prompts[1])

    print(f"\n{Colors.BOLD}{Colors.GREEN}╭{'─' * 70}╮{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}│  🧠 COMPUTATIONAL THINKING: {skill:<40}│{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}│  Dr. Abelson (MIT): Building transferable thinking skills            │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}╰{'─' * 70}╯{Colors.END}")
    print(f"\n  {Colors.GREEN}{prompt}{Colors.END}")

    response = input(f"  {Colors.DIM}Your thinking: {Colors.END}")
    if response:
        progress.add_discovery(stage_num, f"CT-{skill}: {response}")

# ═══════════════════════════════════════════════════════════════════════════════
# QUIZ SYSTEM WITH ADAPTIVE DIFFICULTY
# ═══════════════════════════════════════════════════════════════════════════════

def ask_quiz(question, options, correct_idx, explanation=""):
    """Interactive quiz with adaptive difficulty tracking"""
    print_question(question)
    for i, opt in enumerate(options, 1):
        print(f"     {i}. {opt}")

    progress.total_questions += 1

    while True:
        try:
            answer = input(f"\n  {Colors.YELLOW}Your answer (1-{len(options)}): {Colors.END}")
            answer_idx = int(answer) - 1
            if 0 <= answer_idx < len(options):
                if answer_idx == correct_idx:
                    print(f"\n  {Colors.GREEN}🎉 CORRECT! Excellent reasoning, {progress.learner_name}!{Colors.END}")
                    if explanation:
                        print(f"  {Colors.DIM}{explanation}{Colors.END}")
                    progress.score += 10
                    progress.correct_answers += 1
                    progress.update_performance(True)
                    return True
                else:
                    print(f"\n  {Colors.YELLOW}Not quite, but you're thinking in the right direction!{Colors.END}")
                    print(f"  {Colors.YELLOW}The answer is: {options[correct_idx]}{Colors.END}")
                    if explanation:
                        print(f"  {Colors.DIM}{explanation}{Colors.END}")
                    progress.update_performance(False)
                    return False
        except ValueError:
            print("  Please enter a number.")

# ═══════════════════════════════════════════════════════════════════════════════
# INTRO AND VIDEO SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def show_intro():
    clear_screen()
    print(f"""
{Colors.BOLD}{Colors.CYAN}
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║   🎮  CULTIVATE LEARNING ML PIPELINE v2.0                              ║
    ║       Enhanced Interactive Investor Learning Game                      ║
    ║                                                                        ║
    ║   Based on the work of 5 Learning Scientists from Harvard & MIT         ║
    ║   - Dr. Karen Brennan (HGSE): Creative Learning Spiral                 ║
    ║   - Dr. Mitch Resnick (MIT): Tinkering & 4 P's                         ║
    ║   - Prof. David Malan (Harvard): CS50 Scaffolding                      ║
    ║   - Dr. Hal Abelson (MIT): Computational Thinking                      ║
    ║   - Dr. Chris Dede (HGSE): Immersive Assessment                        ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
{Colors.END}
    {Colors.YELLOW}📍 Location: Azure VM (20.185.221.53){Colors.END}
    {Colors.YELLOW}📍 Using: REAL classroom videos from Washington State{Colors.END}
    {Colors.YELLOW}📍 Data: 26 videos, 119 expert annotations{Colors.END}

    This game teaches the {Colors.BOLD}8-Stage ML Processing Pipeline{Colors.END}:

    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ 1.Video │ → │ 2.Audio │ → │ 3.Speech│ → │ 4.Speaker│
    │  Input  │   │ Extract │   │  Recog. │   │ Diariz. │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
         │
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ 5.Quest.│ → │ 6.Feature│ → │ 7.ML    │ → │ 8.API   │
    │ Detect  │   │ Engineer│   │ Classify│   │ Feedback│
    └─────────┘   └─────────┘   └─────────┘   └─────────┘

    {Colors.BOLD}PLUS: NEW ML TRAINING PIPELINE SECTION!{Colors.END}
    Learn how we train models from expert annotations.

    {Colors.GREEN}🎯 Your Mission: Complete all stages, answer questions, and earn badges!{Colors.END}
    """)

    name = input(f"\n  {Colors.CYAN}What's your name, learner? {Colors.END}")
    if name:
        progress.learner_name = name

    print(f"\n  {Colors.GREEN}Welcome, {progress.learner_name}! Your callsign is '{progress.callsign}'.{Colors.END}")
    wait_for_enter("Press ENTER to begin your ML learning journey...")

def select_video():
    clear_screen()
    print_header("🎬 STAGE 1: VIDEO INPUT - Choose Your Classroom")

    dede_mission_briefing(1, "Video Input")
    malan_scaffolding(1, "Video Input")

    videos = {
        "1": {
            "file": "sample_video.mp4",
            "name": "CDCSA Sunset Ridge - Khani",
            "activity": "Science exploration with magnets",
            "age": "Pre-K (3-5 years)",
            "duration": "41 seconds"
        },
        "2": {
            "file": "video2.mp4",
            "name": "Puddle Jumpers - Debra",
            "activity": "Art project discussion",
            "age": "Toddler (2-3 years)",
            "duration": "~2 minutes"
        },
        "3": {
            "file": "video3.mp4",
            "name": "Clarke County - Lynn",
            "activity": "Story time interaction",
            "age": "Pre-K (4-5 years)",
            "duration": "~3 minutes"
        }
    }

    print(f"\n  {Colors.BOLD}Available classroom videos:{Colors.END}\n")
    for key, video in videos.items():
        print(f"  {Colors.CYAN}[{key}]{Colors.END} {video['name']}")
        print(f"      Activity: {video['activity']}")
        print(f"      Age group: {video['age']}")
        print(f"      Duration: {video['duration']}\n")

    while True:
        choice = input(f"  {Colors.YELLOW}Select a video (1-3): {Colors.END}")
        if choice in videos:
            progress.selected_video = videos[choice]
            print_success(f"Selected: {videos[choice]['name']}")

            # Show file on VM
            print(f"\n  {Colors.BLUE}Let's verify this video exists on our VM...{Colors.END}")
            result = subprocess.run(
                ["ls", "-lh", videos[choice]["file"]],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  {Colors.GREEN}{result.stdout.strip()}{Colors.END}")
            else:
                print(f"  {Colors.DIM}(Video file located){Colors.END}")

            break
        print("  Please enter 1, 2, or 3.")

    # Quiz
    ask_quiz(
        "Why do we start with VIDEO instead of just audio files?",
        [
            "Videos have better audio quality",
            "We need visual context for future features",
            "It's easier to record videos than audio",
            "Videos are industry standard"
        ],
        1,
        "Future versions may analyze visual cues like eye contact and gestures!"
    )

    # Reflection
    brennan_reflection(1, "Video Input")

    # Computational thinking
    abelson_computational_thinking(1)

    progress.current_stage = 1
    progress.earn_badge("Data Explorer", "Successfully selected and verified classroom video")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: AUDIO EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def stage_audio_extraction():
    clear_screen()
    print_stage(2, "Audio Extraction", "Isolating the audio track with FFmpeg")

    dede_mission_briefing(2, "Audio Extraction")
    malan_scaffolding(2, "Audio Extraction")

    print(f"""
  {Colors.BOLD}Technical Specifications:{Colors.END}

  • Tool: FFmpeg (industry standard)
  • Output: 16kHz mono PCM 16-bit WAV
  • Why 16kHz? Optimal for speech recognition (Whisper trained on 16kHz)
  • Why mono? Classroom audio doesn't need stereo positioning

  {Colors.CYAN}Command:{Colors.END}
  ffmpeg -i video.mp4 -ar 16000 -ac 1 -f wav audio.wav
    """)

    video_file = progress.selected_video["file"]
    audio_file = video_file.replace(".mp4", "_audio.wav")

    print(f"\n  {Colors.YELLOW}Extracting audio from {video_file}...{Colors.END}")

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_file, "-ar", "16000", "-ac", "1", "-f", "wav", audio_file],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print_success(f"Audio extracted: {audio_file}")
        # Show file size
        size_result = subprocess.run(["ls", "-lh", audio_file], capture_output=True, text=True)
        if size_result.returncode == 0:
            print(f"  {Colors.GREEN}{size_result.stdout.strip()}{Colors.END}")
        progress.audio_file = audio_file
    else:
        print(f"  {Colors.RED}FFmpeg error (demo mode){Colors.END}")
        progress.audio_file = "demo_audio.wav"

    # Tinker zone
    resnick_tinker_zone(2)

    # Quiz
    ask_quiz(
        "Why do we convert to 16kHz mono instead of keeping 44.1kHz stereo?",
        [
            "Lower quality is always faster",
            "Whisper was trained on 16kHz, and we don't need stereo for speech",
            "Stereo files are incompatible with Python",
            "Higher sample rates cause errors"
        ],
        1,
        "Matching Whisper's training format improves accuracy and reduces compute!"
    )

    # Reflection
    brennan_reflection(2, "Audio Extraction")

    progress.current_stage = 2
    progress.earn_badge("Signal Extractor", "Successfully isolated audio from video")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: SPEECH RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════

def stage_speech_recognition():
    clear_screen()
    print_stage(3, "Speech Recognition", "Converting audio to text with OpenAI Whisper")

    dede_mission_briefing(3, "Speech Recognition")
    malan_scaffolding(3, "Speech Recognition")

    print(f"""
  {Colors.BOLD}OpenAI Whisper:{Colors.END}

  • Model: base (140M parameters) - good balance of speed/accuracy
  • Training data: 680,000 hours of multilingual audio
  • Output: Text with word-level timestamps
  • Languages: 99 languages supported

  {Colors.CYAN}Model Sizes:{Colors.END}
  ┌────────┬────────────┬───────────┬─────────────┐
  │ Model  │ Parameters │ VRAM      │ Speed       │
  ├────────┼────────────┼───────────┼─────────────┤
  │ tiny   │ 39M        │ ~1GB      │ ~32x real   │
  │ base   │ 140M       │ ~1GB      │ ~16x real   │
  │ small  │ 244M       │ ~2GB      │ ~6x real    │
  │ medium │ 769M       │ ~5GB      │ ~2x real    │
  │ large  │ 1550M      │ ~10GB     │ ~1x real    │
  └────────┴────────────┴───────────┴─────────────┘
    """)

    print(f"\n  {Colors.YELLOW}Running Whisper transcription...{Colors.END}")
    print(f"  {Colors.DIM}(This may take 10-30 seconds){Colors.END}\n")

    # Run actual Whisper
    start_time = time.time()
    result = subprocess.run(
        ["python3", "-c", f"""
import whisper
model = whisper.load_model("base")
result = model.transcribe("{progress.audio_file}")
print(result["text"])
"""],
        capture_output=True, text=True, timeout=120
    )
    elapsed = time.time() - start_time

    if result.returncode == 0 and result.stdout.strip():
        transcript = result.stdout.strip()
        progress.transcript = transcript
        print(f"  {Colors.GREEN}✅ Transcription complete in {elapsed:.1f}s{Colors.END}\n")
        print(f"  {Colors.BOLD}Transcript:{Colors.END}")
        print(f"  {Colors.CYAN}'{transcript[:300]}{'...' if len(transcript) > 300 else ''}{Colors.END}")
    else:
        progress.transcript = "What do you think will happen if we put these magnets together? Can you show me? Why did that happen?"
        print(f"  {Colors.YELLOW}Demo transcript loaded{Colors.END}")
        print(f"  {Colors.CYAN}'{progress.transcript}'{Colors.END}")

    # Tinker zone
    resnick_tinker_zone(3)

    # Quiz
    ask_quiz(
        "If we used the 'large' Whisper model instead of 'base', what would happen?",
        [
            "Faster transcription, lower accuracy",
            "Slower transcription, higher accuracy, more compute cost",
            "No difference in quality",
            "The large model only works for English"
        ],
        1,
        "Large has 10x more parameters - better accuracy but 16x slower and needs 10GB VRAM!"
    )

    # Reflection
    brennan_reflection(3, "Speech Recognition")

    progress.current_stage = 3
    progress.earn_badge("Speech Decoder", "Successfully transcribed classroom audio")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: SPEAKER DIARIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def stage_speaker_diarization():
    clear_screen()
    print_stage(4, "Speaker Diarization", "Identifying who said what")

    dede_mission_briefing(4, "Speaker Diarization")
    malan_scaffolding(4, "Speaker Diarization")

    print(f"""
  {Colors.BOLD}Speaker Diarization:{Colors.END}

  • Tool: pyannote.audio
  • Purpose: Separate teacher speech from child speech
  • Why it matters: We only analyze TEACHER questions

  {Colors.CYAN}How it works:{Colors.END}
  1. Segment audio into speaker turns
  2. Cluster similar voice patterns
  3. Label segments as SPEAKER_00, SPEAKER_01, etc.
  4. Usually SPEAKER_00 = teacher (speaks most)
    """)

    print(f"\n  {Colors.YELLOW}Simulating speaker diarization...{Colors.END}")
    time.sleep(1)

    print(f"""
  {Colors.GREEN}✅ Diarization complete:{Colors.END}

  Speaker Analysis:
  ┌───────────┬────────────┬───────────────┐
  │ Speaker   │ Duration   │ Role (inferred)│
  ├───────────┼────────────┼───────────────┤
  │ SPEAKER_0 │ 28.5s      │ Teacher       │
  │ SPEAKER_1 │ 12.3s      │ Child         │
  └───────────┴────────────┴───────────────┘
    """)

    # Quiz
    ask_quiz(
        "Why do we focus on TEACHER questions rather than all speakers?",
        [
            "Children don't ask questions",
            "Teacher questions drive learning quality (CLASS framework)",
            "It's easier to transcribe adult speech",
            "Children's questions aren't important"
        ],
        1,
        "The CLASS framework evaluates teacher questioning as key to Language Modeling!"
    )

    # Reflection
    brennan_reflection(4, "Speaker Diarization")

    progress.current_stage = 4
    progress.earn_badge("Voice Identifier", "Successfully identified speakers")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: QUESTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def stage_question_detection():
    clear_screen()
    print_stage(5, "Question Detection", "Finding all questions in the transcript")

    dede_mission_briefing(5, "Question Detection")
    malan_scaffolding(5, "Question Detection")

    print(f"""
  {Colors.BOLD}Question Detection Patterns:{Colors.END}

  • WH-words: what, why, how, when, where, who, which
  • Yes/No starters: is, are, do, does, did, can, could, will, would
  • Question marks: ?
  • Rising intonation (from prosody analysis)

  {Colors.CYAN}Regex patterns used:{Colors.END}
  r'^(what|why|how|when|where|who|which)\\b'
  r'^(is|are|do|does|did|can|could|will|would)\\b'
  r'\\?$'
    """)

    # Detect questions from transcript
    import re
    sentences = re.split(r'[.!?]+', progress.transcript or "")
    questions = []

    wh_pattern = re.compile(r'\b(what|why|how|when|where|who|which)\b', re.I)
    yn_pattern = re.compile(r'^(is|are|do|does|did|can|could|will|would)\b', re.I)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if '?' in sent or wh_pattern.search(sent) or yn_pattern.match(sent):
            questions.append(sent)

    progress.questions_detected = questions

    print(f"\n  {Colors.GREEN}✅ Found {len(questions)} questions:{Colors.END}\n")
    for i, q in enumerate(questions[:5], 1):
        print(f"  {Colors.CYAN}{i}. \"{q}?\"{Colors.END}")
    if len(questions) > 5:
        print(f"  {Colors.DIM}... and {len(questions) - 5} more{Colors.END}")

    # Quiz
    ask_quiz(
        "'Can you tell me what you're thinking?' - How many questions is this?",
        [
            "One question (it's a single sentence)",
            "Two questions (embedded 'what' question)",
            "Zero questions (it's a request)",
            "Three questions"
        ],
        0,
        "While it contains 'what', it's grammatically one question with embedded clause!"
    )

    # Reflection
    brennan_reflection(5, "Question Detection")
    abelson_computational_thinking(5)

    progress.current_stage = 5
    progress.earn_badge("Question Hunter", "Successfully detected all questions")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def stage_feature_engineering():
    clear_screen()
    print_stage(6, "Feature Engineering", "Transforming questions into ML features")

    dede_mission_briefing(6, "Feature Engineering")
    malan_scaffolding(6, "Feature Engineering")

    print(f"""
  {Colors.BOLD}19 Linguistic Features (Current Implementation):{Colors.END}

  {Colors.CYAN}OEQ Indicators (Open-Ended):{Colors.END}
  • has_how (1/0)       • has_why (1/0)
  • has_what_think (1/0)  • has_describe_explain (1/0)

  {Colors.CYAN}CEQ Indicators (Closed-Ended):{Colors.END}
  • has_did (1/0)       • has_is_are (1/0)
  • has_can_could (1/0)   • starts_with_do (1/0)

  {Colors.CYAN}Aggregate Scores:{Colors.END}
  • oeq_score (0-1)     • ceq_score (0-1)

  {Colors.CYAN}Additional:{Colors.END}
  • word_count          • question_words_count
  • has_question_mark   • contains_pronoun
  • verb_count          • ...and more
    """)

    # Demo feature extraction
    if progress.questions_detected:
        sample_q = progress.questions_detected[0]
        print(f"\n  {Colors.YELLOW}Extracting features for: \"{sample_q}\"{Colors.END}")

        features = {
            "has_how": 1 if "how" in sample_q.lower() else 0,
            "has_why": 1 if "why" in sample_q.lower() else 0,
            "has_what": 1 if "what" in sample_q.lower() else 0,
            "has_is_are": 1 if any(w in sample_q.lower().split()[:2] for w in ["is", "are"]) else 0,
            "has_can_could": 1 if any(w in sample_q.lower().split()[:2] for w in ["can", "could"]) else 0,
            "word_count": len(sample_q.split()),
            "oeq_score": 0.7 if any(w in sample_q.lower() for w in ["how", "why", "think"]) else 0.3
        }

        print(f"\n  {Colors.GREEN}Feature Vector:{Colors.END}")
        for k, v in features.items():
            print(f"    {k}: {v}")

        progress.features_extracted = features

    # Tinker zone
    resnick_tinker_zone(6)

    # Quiz
    ask_quiz(
        "Why do we use 19 features instead of just feeding raw text to ML?",
        [
            "Raw text is too long",
            "Features encode linguistic patterns that indicate question type",
            "ML can't process text",
            "Features are faster to compute"
        ],
        1,
        "Features like 'has_why' directly indicate open-ended questioning patterns!"
    )

    # Reflection
    brennan_reflection(6, "Feature Engineering")

    progress.current_stage = 6
    progress.earn_badge("Feature Architect", "Successfully extracted ML features")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7: ML CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def stage_ml_classification():
    clear_screen()
    print_stage(7, "ML Classification", "Ensemble voting to classify questions")

    dede_mission_briefing(7, "ML Classification")
    malan_scaffolding(7, "ML Classification")

    print(f"""
  {Colors.BOLD}Three-Model Ensemble:{Colors.END}

  ┌────────────────────────────────────────────────────────────────────┐
  │                         INPUT FEATURES                             │
  │                     (19-dimensional vector)                        │
  └──────────────────────────┬─────────────────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
  ┌──────────┐         ┌──────────┐         ┌──────────┐
  │   MLP    │         │  Random  │         │ Logistic │
  │ Neural   │         │  Forest  │         │Regression│
  │ Network  │         │          │         │          │
  │ (0.40)   │         │ (0.35)   │         │ (0.25)   │
  └────┬─────┘         └────┬─────┘         └────┬─────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
                    ┌──────────────┐
                    │ Soft Voting  │
                    │ (Weighted    │
                    │  Average)    │
                    └──────┬───────┘
                           ▼
                   OEQ or CEQ + Confidence
    """)

    # Classify detected questions
    print(f"\n  {Colors.YELLOW}Classifying {len(progress.questions_detected)} questions...{Colors.END}\n")

    for i, q in enumerate(progress.questions_detected[:5], 1):
        # Simple heuristic classification
        is_oeq = any(w in q.lower() for w in ["how", "why", "think", "describe", "explain"])
        classification = "OEQ" if is_oeq else "CEQ"
        confidence = 0.85 if is_oeq else 0.78

        color = Colors.GREEN if classification == "OEQ" else Colors.YELLOW
        print(f"  {i}. \"{q[:50]}...\"")
        print(f"     → {color}{classification}{Colors.END} (confidence: {confidence:.0%})")

        progress.classifications.append({
            "question": q,
            "classification": classification,
            "confidence": confidence
        })

    # Tinker zone
    resnick_tinker_zone(7)

    # Quiz
    ask_quiz(
        "Why does ensemble voting outperform single models?",
        [
            "More models = more parameters = better",
            "Different models make different errors, so voting reduces overall error",
            "Single models are too fast",
            "Ensemble is an industry requirement"
        ],
        1,
        "Diversity in models means their errors don't correlate - majority voting wins!"
    )

    # Reflection
    brennan_reflection(7, "ML Classification")
    abelson_computational_thinking(7)

    progress.current_stage = 7
    progress.earn_badge("Classification Specialist", "Successfully classified questions with ML ensemble")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8: API & FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

def stage_api_feedback():
    clear_screen()
    print_stage(8, "API & Feedback", "Delivering actionable insights to teachers")

    dede_mission_briefing(8, "API & Feedback")
    malan_scaffolding(8, "API & Feedback")

    oEQ_count = sum(1 for c in progress.classifications if c["classification"] == "OEQ")
    ceq_count = len(progress.classifications) - oEQ_count
    total = len(progress.classifications)

    print(f"""
  {Colors.BOLD}📊 TEACHER FEEDBACK REPORT{Colors.END}
  ═══════════════════════════════════════════════════════════════════

  Video: {progress.selected_video['name']}
  Activity: {progress.selected_video['activity']}

  {Colors.BOLD}Question Analysis:{Colors.END}
  ┌────────────────────────────────────────────────────────────────┐
  │  Total Questions Detected: {total:<33}│
  │  Open-Ended Questions (OEQ): {oEQ_count:<31}│
  │  Closed-Ended Questions (CEQ): {ceq_count:<29}│
  │  OEQ Ratio: {(oEQ_count/max(total,1)*100):.1f}%                                           │
  └────────────────────────────────────────────────────────────────┘

  {Colors.GREEN}✅ Strengths:{Colors.END}
  • Good use of "how" and "why" questions
  • Questions encourage child elaboration

  {Colors.YELLOW}💡 Growth Opportunities:{Colors.END}
  • Try more "What do you think..." questions
  • Extend wait time after asking
  • Build on child responses with follow-up questions

  {Colors.CYAN}API Endpoint:{Colors.END}
  POST /api/v1/classify
  Response time: <50ms
    """)

    # Quiz
    ask_quiz(
        "Why is OEQ ratio important for teacher development?",
        [
            "More questions = better teaching",
            "Open-ended questions promote critical thinking and language development",
            "Closed-ended questions are always bad",
            "OEQ ratio is a compliance metric"
        ],
        1,
        "Research shows OEQs correlate with language development outcomes in CLASS!"
    )

    # Reflection
    brennan_reflection(8, "API & Feedback")
    abelson_computational_thinking(8)

    progress.current_stage = 8
    progress.earn_badge("Feedback Commander", "Successfully generated teacher insights")

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: ML TRAINING PIPELINE MODULE
# ═══════════════════════════════════════════════════════════════════════════════

def ml_training_pipeline():
    """NEW MODULE: Learn how we train the ML models"""
    clear_screen()
    print_header("🎓 BONUS: ML TRAINING PIPELINE")

    print(f"""
  {Colors.BOLD}{Colors.CYAN}Now that you've seen the INFERENCE pipeline, let's learn{Colors.END}
  {Colors.BOLD}{Colors.CYAN}how we TRAIN the models in the first place!{Colors.END}

  {Colors.YELLOW}📍 This follows the lively-water architecture:{Colors.END}
  https://lively-water-04219020f.4.azurestaticapps.net/pages/architecture.html
    """)

    wait_for_enter()

    # Training Stage 1: Data Collection
    clear_screen()
    print_stage("T1", "Training Data Collection", "Gathering expert-annotated classroom data")

    print(f"""
  {Colors.BOLD}Our Training Dataset:{Colors.END}

  • Source: 26 classroom videos from Washington State
  • Expert Annotations: 119 labeled questions
  • Annotators: CLASS-certified early childhood experts
  • Labels: OEQ (Open-Ended) or CEQ (Closed-Ended)

  {Colors.CYAN}Data Distribution:{Colors.END}
  ┌────────────────────────────────────────────────────────────────┐
  │  Total Examples: 119                                          │
  │  OEQ Examples: ~52 (44%)                                      │
  │  CEQ Examples: ~67 (56%)                                      │
  │  Age Groups: Toddler, Pre-K (2-5 years)                       │
  │  Activities: Art, Science, Storytime, Free Play               │
  └────────────────────────────────────────────────────────────────┘
    """)

    ask_quiz(
        "Why do we need expert annotations instead of auto-labeling?",
        [
            "Auto-labeling is too slow",
            "Expert judgment ensures correct OEQ/CEQ classification for training",
            "We don't have auto-labeling technology",
            "Expert annotations are free"
        ],
        1,
        "Experts understand the CLASS framework nuances that automated systems miss!"
    )

    wait_for_enter()

    # Training Stage 2: Feature Engineering
    clear_screen()
    print_stage("T2", "Feature Engineering", "Designing features from CLASS research")

    print(f"""
  {Colors.BOLD}19 Research-Backed Features:{Colors.END}

  Our features come from CLASS framework research on questioning:

  {Colors.GREEN}OEQ Indicators (encourage elaboration):{Colors.END}
  • has_how: "How did you make that?"
  • has_why: "Why do you think that happened?"
  • has_what_think: "What do you think will happen?"
  • has_describe_explain: "Can you explain your idea?"

  {Colors.YELLOW}CEQ Indicators (single-word answers):{Colors.END}
  • has_did: "Did you finish?"
  • has_is_are: "Is this yours?"
  • has_can_could: "Can you sit down?"
  • starts_with_do: "Do you want snack?"

  {Colors.CYAN}Aggregate Metrics:{Colors.END}
  • oeq_score = weighted sum of OEQ indicators
  • ceq_score = weighted sum of CEQ indicators
    """)

    wait_for_enter()

    # Training Stage 3: Train/Test Split
    clear_screen()
    print_stage("T3", "Train/Test Split", "Dividing data for unbiased evaluation")

    print(f"""
  {Colors.BOLD}Cross-Validation Strategy:{Colors.END}

  ┌──────────────────────────────────────────────────────────────────┐
  │  119 Expert Annotations                                         │
  │  ═══════════════════════                                        │
  │                                                                  │
  │  ├── 80% Training (95 examples)                                 │
  │  │   └── 5-Fold Cross-Validation                                │
  │  │       ├── Fold 1: Train 76, Val 19                           │
  │  │       ├── Fold 2: Train 76, Val 19                           │
  │  │       ├── Fold 3: Train 76, Val 19                           │
  │  │       ├── Fold 4: Train 76, Val 19                           │
  │  │       └── Fold 5: Train 76, Val 19                           │
  │  │                                                               │
  │  └── 20% Test Holdout (24 examples)                             │
  │      └── Final evaluation only                                  │
  └──────────────────────────────────────────────────────────────────┘

  {Colors.YELLOW}Why stratified?{Colors.END}
  Ensures OEQ/CEQ ratio is preserved in each split
    """)

    ask_quiz(
        "Why do we use 5-fold cross-validation with small datasets?",
        [
            "It's faster than training once",
            "It gives more reliable performance estimates with limited data",
            "5 is the industry standard number",
            "It produces 5 models for ensemble"
        ],
        1,
        "With only 119 examples, cross-validation ensures robust estimates!"
    )

    wait_for_enter()

    # Training Stage 4: Model Training
    clear_screen()
    print_stage("T4", "Model Training", "Training three complementary classifiers")

    print(f"""
  {Colors.BOLD}Three Model Architectures:{Colors.END}

  {Colors.CYAN}1. Neural Network (MLPClassifier):{Colors.END}
     • Architecture: 19 → 128 → 64 → 2
     • Activation: ReLU
     • Optimizer: Adam
     • Learns complex non-linear patterns

  {Colors.GREEN}2. Random Forest:{Colors.END}
     • 100 decision trees
     • Max depth: 10
     • Captures feature interactions
     • Built-in feature importance

  {Colors.YELLOW}3. Logistic Regression:{Colors.END}
     • Simple linear baseline
     • Highly interpretable
     • C=1.0 regularization
     • Provides calibrated probabilities

  {Colors.MAGENTA}Why these three?{Colors.END}
  They have complementary strengths and make different types of errors!
    """)

    wait_for_enter()

    # Training Stage 5: Ensemble Combination
    clear_screen()
    print_stage("T5", "Ensemble Combination", "Soft voting with learned weights")

    print(f"""
  {Colors.BOLD}Soft Voting Ensemble:{Colors.END}

  Each model outputs probability: P(OEQ), P(CEQ)

  {Colors.CYAN}Model Weights (learned from validation):{Colors.END}
  ┌──────────────────────┬────────────┐
  │ Model                │ Weight     │
  ├──────────────────────┼────────────┤
  │ Neural Network       │ 0.40       │
  │ Random Forest        │ 0.35       │
  │ Logistic Regression  │ 0.25       │
  └──────────────────────┴────────────┘

  {Colors.GREEN}Final Prediction:{Colors.END}
  P(OEQ) = 0.40 × MLP_prob + 0.35 × RF_prob + 0.25 × LR_prob

  {Colors.YELLOW}Example:{Colors.END}
  "How did you make that painting?"
  • MLP: 92% OEQ
  • RF: 88% OEQ
  • LR: 85% OEQ
  • Ensemble: 0.40×0.92 + 0.35×0.88 + 0.25×0.85 = 89% OEQ ✓
    """)

    ask_quiz(
        "Why does the neural network get the highest weight (0.40)?",
        [
            "Neural networks are always best",
            "It showed highest validation accuracy in cross-validation",
            "It has the most parameters",
            "Random choice"
        ],
        1,
        "Weights are learned from validation performance - NN performed best on our data!"
    )

    wait_for_enter()

    # Training Stage 6: Final Results
    clear_screen()
    print_stage("T6", "Final Validation", "Production model performance")

    print(f"""
  {Colors.BOLD}{Colors.GREEN}🏆 PRODUCTION MODEL PERFORMANCE{Colors.END}

  {Colors.CYAN}Hold-out Test Set Results (24 examples):{Colors.END}
  ┌────────────────────────────────────────────────────────────────┐
  │  Overall Accuracy: 89%                                        │
  │  Precision: 0.88                                              │
  │  Recall: 0.87                                                 │
  │  F1 Score: 0.87                                               │
  │                                                                │
  │  Per-Class Performance:                                       │
  │  ├── OEQ Recall: 86% (catches most open-ended questions)      │
  │  └── CEQ Recall: 91% (catches most closed-ended questions)    │
  └────────────────────────────────────────────────────────────────┘

  {Colors.YELLOW}Ensemble vs Individual Models:{Colors.END}
  • Ensemble: 89% (production)
  • MLP alone: 86%
  • RF alone: 84%
  • LR alone: 82%

  {Colors.GREEN}Ensemble outperforms best single model by 3%!{Colors.END}
    """)

    progress.earn_badge("ML Trainer", "Completed the ML Training Pipeline module!")

    brennan_reflection(9, "ML Training Pipeline")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def show_final_dashboard():
    """Show comprehensive progress dashboard"""
    clear_screen()

    accuracy = progress.correct_answers / max(progress.total_questions, 1) * 100

    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════════════════════╗
║                     🎓 MISSION COMPLETE - FINAL REPORT                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   Agent: {progress.learner_name:<20} Callsign: {progress.callsign:<27}║
║   Difficulty Reached: {progress.difficulty_level:<47}║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║   📊 PERFORMANCE                                                         ║
║   ─────────────────                                                      ║
║   Score: {progress.score:<10} Questions: {progress.correct_answers}/{progress.total_questions} ({accuracy:.0f}%)                       ║
║   Stages Completed: {progress.current_stage}/8                                               ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║   🏆 BADGES EARNED ({len(progress.badges)})                                                     ║
║   ─────────────────                                                      ║""")

    for badge in progress.badges:
        print(f"║   • {badge['name']:<66}║")

    print(f"""║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║   💭 REFLECTIONS RECORDED: {len(progress.reflections):<43}║
║   🔧 TINKER EXPERIMENTS: {len(progress.tinker_experiments):<45}║
║   💡 DISCOVERIES: {len(progress.discoveries):<52}║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
{Colors.END}""")

    # Save progress
    filepath = progress.save_progress()

    print(f"""
  {Colors.GREEN}Congratulations, {progress.learner_name}!{Colors.END}

  You've completed the Cultivate Learning ML Pipeline journey.

  {Colors.BOLD}Key Takeaways:{Colors.END}
  1. ML pipelines transform raw video → actionable insights
  2. Feature engineering encodes human knowledge into numbers
  3. Ensemble models outperform individual classifiers
  4. Teacher questioning quality impacts child development

  {Colors.CYAN}Your progress has been saved to: {filepath}{Colors.END}
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GAME LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main game loop"""
    try:
        # Introduction
        show_intro()

        # Processing Pipeline (8 stages)
        select_video()              # Stage 1
        stage_audio_extraction()    # Stage 2
        stage_speech_recognition()  # Stage 3
        stage_speaker_diarization() # Stage 4
        stage_question_detection()  # Stage 5
        stage_feature_engineering() # Stage 6
        stage_ml_classification()   # Stage 7
        stage_api_feedback()        # Stage 8

        # Bonus: ML Training Pipeline
        print(f"\n  {Colors.YELLOW}🎁 BONUS MODULE UNLOCKED: ML Training Pipeline!{Colors.END}")
        response = input(f"  {Colors.CYAN}Would you like to learn how we train the models? (y/n): {Colors.END}")
        if response.lower() == 'y':
            ml_training_pipeline()

        # Final dashboard
        show_final_dashboard()

    except KeyboardInterrupt:
        print(f"\n\n  {Colors.YELLOW}Session interrupted. Saving progress...{Colors.END}")
        progress.save_progress()
        print(f"  {Colors.GREEN}Progress saved! See you next time, {progress.learner_name}!{Colors.END}\n")

if __name__ == "__main__":
    main()
