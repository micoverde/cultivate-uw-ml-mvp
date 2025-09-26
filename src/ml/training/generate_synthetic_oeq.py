#!/usr/bin/env python3
"""
Generate Synthetic OEQ Examples to Balance Dataset
Implements Issue #118: Synthetic Data for ML Training

Warren - This creates educationally-valid OEQ examples to balance our dataset!
"""

import random
import json
from typing import List, Dict

class SyntheticOEQGenerator:
    """Generate synthetic Open-Ended Questions for training data augmentation"""

    def __init__(self):
        # Templates based on actual OEQ patterns from CSV
        self.oeq_templates = {
            'why_questions': [
                "Why do you think {subject} {action}?",
                "Why is {subject} {attribute}?",
                "Why did that happen when {scenario}?",
                "Can you explain why {observation}?"
            ],
            'how_questions': [
                "How do {subject1} and {subject2} look different?",
                "How would you {action}?",
                "How could we {goal}?",
                "How do you feel about {topic}?",
                "How might {subject} change if {condition}?"
            ],
            'what_if_questions': [
                "What would happen if {scenario}?",
                "What if we {action}?",
                "What might happen next?",
                "What could we do about {problem}?"
            ],
            'thinking_questions': [
                "What do you think about {topic}?",
                "What makes you think {observation}?",
                "Tell me about {subject}.",
                "Describe what you see when {scenario}.",
                "What else could we {action}?"
            ],
            'comparison_questions': [
                "What's different between {subject1} and {subject2}?",
                "How are {subject1} and {subject2} similar?",
                "What do you notice about {subject}?",
                "What patterns do you see?"
            ]
        }

        # Educational contexts from actual classroom scenarios
        self.contexts = {
            'subjects': ['the blocks', 'the shapes', 'the colors', 'the animals',
                        'the plants', 'the water', 'the sand', 'the books',
                        'your drawing', 'the puzzle', 'the tower', 'the story'],
            'actions': ['move', 'grow', 'change', 'fall down', 'stick together',
                       'make that sound', 'look that way', 'float', 'sink'],
            'attributes': ['different', 'the same', 'bigger', 'important',
                          'special', 'colorful', 'soft', 'hard'],
            'scenarios': ['we mix them together', 'we take it apart',
                         'we add water', 'we turn it upside down',
                         'we share with friends', 'we try again'],
            'goals': ['solve this problem', 'make it better', 'help our friend',
                     'build something new', 'clean up', 'work together'],
            'problems': ['this mess', 'the broken toy', 'someone being sad',
                        'not having enough', 'things not working'],
            'topics': ['sharing', 'helping', 'learning', 'playing together',
                      'being kind', 'trying new things', 'making mistakes'],
            'observations': ['it works that way', 'that happened', 'you think so',
                           'it looks like that', 'we see this pattern']
        }

    def generate_oeq(self, count: int = 36) -> List[Dict]:
        """
        Generate synthetic OEQ examples

        Args:
            count: Number of OEQ examples to generate (default 36 to balance dataset)

        Returns:
            List of synthetic OEQ examples with metadata
        """
        synthetic_oeqs = []

        for i in range(count):
            # Select a random template category
            category = random.choice(list(self.oeq_templates.keys()))
            template = random.choice(self.oeq_templates[category])

            # Fill in the template with random context
            question = self._fill_template(template)

            # Add appropriate metadata
            metadata = {
                'question_text': question,
                'label': 'OEQ',
                'synthetic': True,
                'template_category': category,
                'has_wait_time': random.random() > 0.3,  # 70% have wait time
                'is_yes_no': False,  # OEQs are not yes/no
                'full_description': f"Synthetic OEQ: {question} - Generated for training balance",
                'video': f"SYNTHETIC_OEQ_{i+1:03d}",
                'question_num': 1
            }

            synthetic_oeqs.append(metadata)

        return synthetic_oeqs

    def _fill_template(self, template: str) -> str:
        """Fill in a template with random educational context"""
        question = template

        # Replace placeholders with random selections
        replacements = {
            '{subject}': random.choice(self.contexts['subjects']),
            '{subject1}': random.choice(self.contexts['subjects']),
            '{subject2}': random.choice(self.contexts['subjects']),
            '{action}': random.choice(self.contexts['actions']),
            '{attribute}': random.choice(self.contexts['attributes']),
            '{scenario}': random.choice(self.contexts['scenarios']),
            '{goal}': random.choice(self.contexts['goals']),
            '{problem}': random.choice(self.contexts['problems']),
            '{topic}': random.choice(self.contexts['topics']),
            '{observation}': random.choice(self.contexts['observations']),
            '{condition}': random.choice(self.contexts['scenarios'])
        }

        for placeholder, value in replacements.items():
            if placeholder in question:
                question = question.replace(placeholder, value)

        # Ensure proper capitalization and punctuation
        question = question[0].upper() + question[1:]
        if not question.endswith('?'):
            question += '?'

        return question

    def augment_existing_oeqs(self, existing_oeqs: List[Dict]) -> List[Dict]:
        """
        Create variations of existing OEQ examples

        Args:
            existing_oeqs: List of existing OEQ questions from dataset

        Returns:
            List of augmented OEQ variations
        """
        augmented = []

        variation_patterns = [
            lambda q: q.replace('What', 'What else'),
            lambda q: q.replace('How', 'How else'),
            lambda q: 'Can you explain ' + q.lower(),
            lambda q: 'Tell me more about ' + q.lower().replace('?', '.'),
            lambda q: q.replace('?', ' and why?'),
            lambda q: 'What makes you think ' + q.lower()
        ]

        for oeq in existing_oeqs[:10]:  # Augment first 10 OEQs
            if oeq.get('question_text'):
                original = oeq['question_text']

                # Apply random variation
                variation = random.choice(variation_patterns)
                new_question = variation(original)

                augmented.append({
                    'question_text': new_question,
                    'label': 'OEQ',
                    'synthetic': True,
                    'augmented_from': original,
                    'has_wait_time': oeq.get('has_wait_time', True),
                    'is_yes_no': False,
                    'full_description': f"Augmented OEQ from: {original}",
                    'video': f"AUGMENTED_{oeq.get('video', 'unknown')[:20]}",
                    'question_num': oeq.get('question_num', 1)
                })

        return augmented

def main():
    """Generate synthetic OEQ examples and combine with existing data"""

    print("🔬 GENERATING SYNTHETIC OEQ EXAMPLES")
    print("="*70)
    print("Issue #118: Using synthetic data for ML training")
    print()

    # Load existing data
    with open('data/retraining/extracted_questions_for_retraining.json', 'r') as f:
        existing_data = json.load(f)

    # Count existing labels
    oeq_count = sum(1 for q in existing_data if q['label'] == 'OEQ')
    ceq_count = sum(1 for q in existing_data if q['label'] == 'CEQ')

    print(f"Current dataset:")
    print(f"  • OEQ: {oeq_count}")
    print(f"  • CEQ: {ceq_count}")
    print(f"  • Imbalance ratio: {ceq_count/oeq_count:.2f}:1")
    print()

    # Generate synthetic OEQs
    generator = SyntheticOEQGenerator()

    # We need 36 more OEQs to balance (74 CEQ - 38 OEQ = 36)
    needed_oeqs = ceq_count - oeq_count
    print(f"Generating {needed_oeqs} synthetic OEQ examples...")

    synthetic_oeqs = generator.generate_oeq(count=needed_oeqs)

    # Also augment some existing OEQs
    existing_oeqs = [q for q in existing_data if q['label'] == 'OEQ']
    augmented_oeqs = generator.augment_existing_oeqs(existing_oeqs)

    print(f"  • Generated: {len(synthetic_oeqs)} new OEQs")
    print(f"  • Augmented: {len(augmented_oeqs)} variations")
    print()

    # Show some examples
    print("Sample synthetic OEQs:")
    print("-"*70)
    for i, oeq in enumerate(synthetic_oeqs[:5], 1):
        print(f"{i}. \"{oeq['question_text']}\"")
        print(f"   Category: {oeq.get('template_category', 'unknown')}")

    print("\nSample augmented OEQs:")
    print("-"*70)
    for i, oeq in enumerate(augmented_oeqs[:3], 1):
        print(f"{i}. \"{oeq['question_text']}\"")
        print(f"   Original: \"{oeq.get('augmented_from', 'unknown')}\"")

    # Combine all data
    balanced_dataset = existing_data + synthetic_oeqs + augmented_oeqs[:10]

    # Recount
    final_oeq = sum(1 for q in balanced_dataset if q['label'] == 'OEQ')
    final_ceq = sum(1 for q in balanced_dataset if q['label'] == 'CEQ')

    print("\n" + "="*70)
    print("📊 BALANCED DATASET CREATED:")
    print(f"  • Total samples: {len(balanced_dataset)}")
    print(f"  • OEQ: {final_oeq}")
    print(f"  • CEQ: {final_ceq}")
    print(f"  • New ratio: {final_ceq/final_oeq:.2f}:1 (much better!)")

    # Save balanced dataset
    output_file = 'data/retraining/balanced_questions_for_retraining.json'
    with open(output_file, 'w') as f:
        json.dump(balanced_dataset, f, indent=2)

    print(f"\n💾 Balanced dataset saved to: {output_file}")
    print("✅ Ready for retraining with balanced OEQ/CEQ distribution!")

if __name__ == "__main__":
    main()