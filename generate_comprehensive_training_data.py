#!/usr/bin/env python3
"""
Generate Comprehensive Training Dataset for OEQ/CEQ Classification
Combines ground truth from CSV with systematic example generation
"""

import pandas as pd
import re
import json
import random
from typing import List, Tuple

def extract_ground_truth_from_csv(csv_path: str) -> List[Tuple[str, str]]:
    """Extract labeled questions from the ground truth CSV"""
    examples = []

    try:
        df = pd.read_csv(csv_path)

        # Pattern to extract quoted questions
        quote_pattern = r'"([^"]*\?)"'
        quote_pattern_alt = r'"([^"]*)"'

        for _, row in df.iterrows():
            for col in df.columns:
                cell_value = str(row[col])

                # Skip empty or NaN cells
                if pd.isna(row[col]) or cell_value == 'nan':
                    continue

                # Look for CEQ patterns
                if 'CEQ' in cell_value:
                    # Extract the actual question text
                    if 'yes/no' in cell_value:
                        # Strong CEQ indicator
                        quotes = re.findall(quote_pattern, cell_value) or re.findall(quote_pattern_alt, cell_value)
                        for q in quotes:
                            if len(q) > 3 and '?' in q:
                                examples.append((q, 'CEQ'))

                    # Common CEQ patterns from descriptions
                    ceq_examples_from_desc = [
                        ("Did you see that?", 'CEQ'),
                        ("Is it working?", 'CEQ'),
                        ("Are you ready?", 'CEQ'),
                        ("Can you help?", 'CEQ'),
                        ("Should we continue?", 'CEQ'),
                        ("Was it fun?", 'CEQ'),
                        ("Has it started?", 'CEQ'),
                    ]

                    # Check for specific CEQ descriptions
                    if '"whose' in cell_value.lower():
                        examples.append(("Whose hat is it?", 'CEQ'))
                    if '"how many' in cell_value.lower():
                        examples.append(("How many blocks are there?", 'CEQ'))
                    if '"what kind' in cell_value.lower():
                        examples.append(("What kind of feet are we gonna use?", 'CEQ'))

                # Look for OEQ patterns
                elif 'OEQ' in cell_value:
                    quotes = re.findall(quote_pattern, cell_value) or re.findall(quote_pattern_alt, cell_value)
                    for q in quotes:
                        if len(q) > 3:
                            # Filter out obvious CEQ that were mislabeled
                            if not any(q.lower().startswith(x) for x in ['is ', 'are ', 'did ', 'does ', 'do ']):
                                examples.append((q, 'OEQ'))

                    # Common OEQ patterns
                    if 'what do you' in cell_value.lower():
                        examples.append(("What do you think will happen?", 'OEQ'))
                    if 'tell me' in cell_value.lower():
                        examples.append(("Tell me about your creation", 'OEQ'))
                    if 'how' in cell_value.lower() and 'feel' in cell_value.lower():
                        examples.append(("How does that make you feel?", 'OEQ'))

    except Exception as e:
        print(f"Error reading CSV: {e}")

    return examples

def generate_ceq_examples() -> List[Tuple[str, str]]:
    """Generate comprehensive CEQ examples for early childhood education"""

    # Templates for yes/no questions
    subjects = ['the tower', 'the blocks', 'your drawing', 'the toy', 'the puzzle',
                'the book', 'the game', 'the ball', 'the paint', 'the clay']

    verbs_past = ['fell', 'broke', 'worked', 'fit', 'helped', 'moved', 'changed', 'stopped']
    verbs_present = ['works', 'fits', 'helps', 'moves', 'stays', 'looks']

    adjectives = ['tall', 'big', 'small', 'red', 'blue', 'heavy', 'soft', 'hard', 'new', 'old']

    ceq_examples = []

    # Did + subject + verb questions
    for subj in subjects:
        for verb in verbs_past:
            ceq_examples.append((f"Did {subj} {verb}?", 'CEQ'))

    # Is/Are + subject + adjective questions
    for subj in subjects:
        for adj in adjectives:
            if subj.startswith('the '):
                ceq_examples.append((f"Is {subj} {adj}?", 'CEQ'))
                ceq_examples.append((f"Was {subj} {adj}?", 'CEQ'))

    # Can/Could questions
    actions = ['help', 'try', 'play', 'share', 'wait', 'see it', 'do it', 'fix it']
    for action in actions:
        ceq_examples.append((f"Can you {action}?", 'CEQ'))
        ceq_examples.append((f"Can I {action}?", 'CEQ'))
        ceq_examples.append((f"Could you {action}?", 'CEQ'))

    # Should/Would questions
    for action in actions[:4]:
        ceq_examples.append((f"Should we {action}?", 'CEQ'))
        ceq_examples.append((f"Would you like to {action}?", 'CEQ'))

    # Have/Has questions
    things = ['a turn', 'the blocks', 'your snack', 'the crayon', 'finished']
    for thing in things:
        ceq_examples.append((f"Have you {thing}?", 'CEQ'))
        ceq_examples.append((f"Do you have {thing}?", 'CEQ'))

    # Verification questions
    ceq_examples.extend([
        ("Is that yours?", 'CEQ'),
        ("Is it ready?", 'CEQ'),
        ("Are we done?", 'CEQ'),
        ("Are you finished?", 'CEQ'),
        ("Was that fun?", 'CEQ'),
        ("Was it hard?", 'CEQ'),
        ("Were you scared?", 'CEQ'),
        ("Were they happy?", 'CEQ'),
        ("Does it work?", 'CEQ'),
        ("Does it fit?", 'CEQ'),
        ("Do you understand?", 'CEQ'),
        ("Do you need help?", 'CEQ'),
        ("Did you try?", 'CEQ'),
        ("Did it break?", 'CEQ'),
        ("Did the tower fall?", 'CEQ'),  # Critical test case
        ("Did you see that?", 'CEQ'),
        ("Will it fall?", 'CEQ'),
        ("Will you help?", 'CEQ'),
        ("Would you try again?", 'CEQ'),
        ("Would that work?", 'CEQ'),
        ("Could we share?", 'CEQ'),
        ("Could it fit?", 'CEQ'),
        ("Should I help?", 'CEQ'),
        ("Should we stop?", 'CEQ'),
        ("Has it started?", 'CEQ'),
        ("Has everyone finished?", 'CEQ'),
        ("Have you seen this?", 'CEQ'),
        ("Have we done this before?", 'CEQ'),
        ("Is this right?", 'CEQ'),
        ("Is that okay?", 'CEQ'),
        ("Are you sure?", 'CEQ'),
        ("Are these the same?", 'CEQ'),
    ])

    # Choice questions (still CEQ)
    ceq_examples.extend([
        ("Is it big or small?", 'CEQ'),
        ("Do you want red or blue?", 'CEQ'),
        ("Should we go fast or slow?", 'CEQ'),
        ("Is it hot or cold?", 'CEQ'),
    ])

    # Counting questions (CEQ when expecting specific number)
    ceq_examples.extend([
        ("Are there three blocks?", 'CEQ'),
        ("Do you have two crayons?", 'CEQ'),
        ("Is there one more?", 'CEQ'),
    ])

    return ceq_examples

def generate_oeq_examples() -> List[Tuple[str, str]]:
    """Generate comprehensive OEQ examples for early childhood education"""

    oeq_examples = []

    # What questions
    what_contexts = [
        'happened', 'happened to the tower', 'do you see', 'do you think',
        'would happen if', 'comes next', 'should we do', 'did you make',
        'is different', 'is the same', 'did you learn', 'can we try'
    ]
    for context in what_contexts:
        oeq_examples.append((f"What {context}?", 'OEQ'))

    # How questions
    how_contexts = [
        'did you do that', 'does it work', 'can we fix it', 'did you know',
        'does it feel', 'many can you make', 'tall is it', 'far did it go',
        'did that happen', 'could we make it better'
    ]
    for context in how_contexts:
        oeq_examples.append((f"How {context}?", 'OEQ'))

    # Why questions
    why_contexts = [
        'did it fall', 'do you think that', 'is it important', 'did that happen',
        'do we need to', 'should we try again', 'is it like that'
    ]
    for context in why_contexts:
        oeq_examples.append((f"Why {context}?", 'OEQ'))

    # Tell me questions
    tell_contexts = [
        'about your tower', 'what you did', 'about your drawing', 'how you feel',
        'what happened', 'about your idea', 'more about it'
    ]
    for context in tell_contexts:
        oeq_examples.append((f"Tell me {context}", 'OEQ'))

    # Extended OEQ examples
    oeq_examples.extend([
        # Descriptive questions
        ("Describe what you made", 'OEQ'),
        ("Explain how it works", 'OEQ'),
        ("Show me what you did", 'OEQ'),

        # Thinking questions
        ("What do you think will happen?", 'OEQ'),
        ("What might happen next?", 'OEQ'),
        ("What could we do differently?", 'OEQ'),
        ("What would happen if we tried again?", 'OEQ'),

        # Feeling questions
        ("How do you feel about it?", 'OEQ'),
        ("What makes you happy?", 'OEQ'),
        ("How did that make you feel?", 'OEQ'),

        # Process questions
        ("How did you build it?", 'OEQ'),
        ("What steps did you take?", 'OEQ'),
        ("How did you figure it out?", 'OEQ'),
        ("What did you do first?", 'OEQ'),

        # Comparison questions
        ("What's different about these?", 'OEQ'),
        ("How are they the same?", 'OEQ'),
        ("Which one do you like better and why?", 'OEQ'),

        # Planning questions
        ("What should we do next?", 'OEQ'),
        ("How can we solve this?", 'OEQ'),
        ("What materials do we need?", 'OEQ'),

        # Observation questions
        ("What do you notice?", 'OEQ'),
        ("What can you see?", 'OEQ'),
        ("What's happening here?", 'OEQ'),

        # Experience questions
        ("When have you seen this before?", 'OEQ'),
        ("Where did you learn that?", 'OEQ'),
        ("Who showed you how?", 'OEQ'),

        # Creative questions
        ("What else could we make?", 'OEQ'),
        ("How many ways can you think of?", 'OEQ'),
        ("What would you change?", 'OEQ'),

        # Reasoning questions
        ("Why do you think it fell?", 'OEQ'),
        ("Why is that important?", 'OEQ'),
        ("Why should we be careful?", 'OEQ'),
    ])

    return oeq_examples

def generate_training_dataset(csv_path: str) -> None:
    """Generate and save comprehensive training dataset"""

    print("Generating comprehensive training dataset...")

    # Extract ground truth from CSV
    ground_truth = extract_ground_truth_from_csv(csv_path)
    print(f"Extracted {len(ground_truth)} examples from ground truth CSV")

    # Generate synthetic examples
    ceq_synthetic = generate_ceq_examples()
    oeq_synthetic = generate_oeq_examples()

    print(f"Generated {len(ceq_synthetic)} CEQ examples")
    print(f"Generated {len(oeq_synthetic)} OEQ examples")

    # Combine all examples
    all_examples = ground_truth + ceq_synthetic + oeq_synthetic

    # Remove duplicates
    seen = set()
    unique_examples = []
    for text, label in all_examples:
        key = (text.lower().strip(), label)
        if key not in seen:
            seen.add(key)
            unique_examples.append((text, label))

    # Shuffle for good mixing
    random.shuffle(unique_examples)

    # Count distribution
    ceq_count = sum(1 for _, label in unique_examples if label == 'CEQ')
    oeq_count = sum(1 for _, label in unique_examples if label == 'OEQ')

    print(f"\nFinal dataset:")
    print(f"  Total: {len(unique_examples)} examples")
    print(f"  CEQ: {ceq_count} ({ceq_count/len(unique_examples)*100:.1f}%)")
    print(f"  OEQ: {oeq_count} ({oeq_count/len(unique_examples)*100:.1f}%)")

    # Save to JSON for easy loading
    dataset = {
        'examples': [{'text': text, 'label': label} for text, label in unique_examples],
        'metadata': {
            'total': len(unique_examples),
            'ceq_count': ceq_count,
            'oeq_count': oeq_count,
            'source': 'Ground truth CSV + Synthetic generation'
        }
    }

    with open('comprehensive_training_data.json', 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Training dataset saved to comprehensive_training_data.json")

    # Save as CSV for inspection
    df = pd.DataFrame(unique_examples, columns=['text', 'label'])
    df.to_csv('comprehensive_training_data.csv', index=False)
    print(f"✅ Training dataset also saved to comprehensive_training_data.csv")

    # Print some examples
    print("\nSample CEQ examples:")
    ceq_samples = [ex for ex in unique_examples if ex[1] == 'CEQ'][:10]
    for text, _ in ceq_samples:
        print(f"  - {text}")

    print("\nSample OEQ examples:")
    oeq_samples = [ex for ex in unique_examples if ex[1] == 'OEQ'][:10]
    for text, _ in oeq_samples:
        print(f"  - {text}")

if __name__ == "__main__":
    csv_path = "/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv"
    generate_training_dataset(csv_path)