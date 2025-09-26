#!/usr/bin/env python3
"""
ANALYZE CSV DATA FOR RETRAINING ML MODEL
Check if we have proper OEQ vs CEQ labels and question text

Warren - This analyzes what training data we have for better OEQ/CEQ classification!
"""

import pandas as pd
import re

def analyze_csv_for_retraining():
    """Analyze the CSV to see what data we have for retraining"""

    print("🔍 ANALYZING CSV DATA FOR MODEL RETRAINING")
    print("="*70)

    # Load the CSV
    csv_path = "/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv"
    df = pd.read_csv(csv_path)

    print(f"Total rows in CSV: {len(df)}")
    print()

    # Extract all questions with their labels
    questions_with_labels = []

    # The CSV has Q1-Q8 descriptions that contain OEQ/CEQ labels
    for idx, row in df.iterrows():
        video_title = row['Video Title']

        # Check each question column (Q1 through Q8)
        for q_num in range(1, 9):
            q_desc_col = f'Q{q_num} description'

            if q_desc_col in row and pd.notna(row[q_desc_col]) and row[q_desc_col] != 'na':
                description = str(row[q_desc_col])

                # Extract the label (OEQ or CEQ)
                label = None
                if 'OEQ' in description:
                    label = 'OEQ'
                elif 'CEQ' in description:
                    label = 'CEQ'

                if label:
                    # Try to extract the actual question text if it's in quotes
                    question_text = None
                    quoted = re.findall(r'"([^"]*)"', description)
                    if quoted:
                        question_text = quoted[0]

                    questions_with_labels.append({
                        'video': video_title,
                        'question_num': q_num,
                        'label': label,
                        'question_text': question_text,
                        'full_description': description,
                        'has_wait_time': 'pause' in description.lower(),
                        'is_yes_no': 'yes/no' in description.lower()
                    })

    print(f"📊 EXTRACTED DATA SUMMARY:")
    print(f"Total questions with labels: {len(questions_with_labels)}")

    # Count labels
    oeq_count = sum(1 for q in questions_with_labels if q['label'] == 'OEQ')
    ceq_count = sum(1 for q in questions_with_labels if q['label'] == 'CEQ')

    print(f"• OEQ (Open-Ended): {oeq_count}")
    print(f"• CEQ (Closed-Ended): {ceq_count}")
    print(f"• Ratio: {oeq_count/ceq_count:.2f}:1 OEQ to CEQ")
    print()

    # Check how many have actual question text
    with_text = sum(1 for q in questions_with_labels if q['question_text'])
    print(f"Questions with actual text in quotes: {with_text}/{len(questions_with_labels)}")
    print()

    # Sample some questions with text
    print("📝 SAMPLE QUESTIONS WITH TEXT:")
    print("-"*70)

    samples_shown = 0
    for q in questions_with_labels:
        if q['question_text'] and samples_shown < 10:
            print(f"\nLabel: {q['label']}")
            print(f"Text: \"{q['question_text']}\"")
            print(f"Is yes/no: {q['is_yes_no']}")
            print(f"Description: {q['full_description'][:100]}...")
            samples_shown += 1

    print("\n" + "="*70)
    print("🎯 KEY PATTERNS FOR RETRAINING:")
    print("-"*70)

    # Analyze OEQ patterns
    oeq_questions = [q for q in questions_with_labels if q['label'] == 'OEQ' and q['question_text']]
    ceq_questions = [q for q in questions_with_labels if q['label'] == 'CEQ' and q['question_text']]

    print("\nOEQ PATTERNS:")
    oeq_starters = {}
    for q in oeq_questions:
        first_word = q['question_text'].split()[0].lower() if q['question_text'] else 'unknown'
        oeq_starters[first_word] = oeq_starters.get(first_word, 0) + 1

    for word, count in sorted(oeq_starters.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • '{word}': {count} times")

    print("\nCEQ PATTERNS:")
    ceq_starters = {}
    for q in ceq_questions:
        first_word = q['question_text'].split()[0].lower() if q['question_text'] else 'unknown'
        ceq_starters[first_word] = ceq_starters.get(first_word, 0) + 1

    for word, count in sorted(ceq_starters.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • '{word}': {count} times")

    # Check yes/no correlation
    ceq_yes_no = sum(1 for q in ceq_questions if q['is_yes_no'])
    print(f"\nCEQ questions that are yes/no: {ceq_yes_no}/{len(ceq_questions)} ({ceq_yes_no/len(ceq_questions)*100:.0f}%)")

    print("\n" + "="*70)
    print("💡 RETRAINING RECOMMENDATIONS:")
    print("-"*70)

    print("""
1. ✅ WE HAVE GOOD TRAINING DATA:
   • {total} labeled questions
   • Clear OEQ vs CEQ labels
   • {with_text} questions have actual text

2. 📊 CLASS IMBALANCE:
   • OEQ: {oeq} ({oeq_pct:.0f}%)
   • CEQ: {ceq} ({ceq_pct:.0f}%)
   • Need to handle imbalance in training

3. 🎯 KEY FEATURES TO EXTRACT:
   • First word (what, how, why vs can, do, is)
   • Presence of "yes/no" in description
   • Question marks and length
   • Specific patterns like "how many" (CEQ despite "how")

4. 📝 ADDITIONAL FEATURES AVAILABLE:
   • Wait time (pause information)
   • Rhetorical vs real questions
   • Age group context

5. 🚀 NEXT STEPS:
   • Extract all question text where available
   • Use descriptions to infer patterns when text missing
   • Create balanced training set
   • Add rule-based features for "how", "why", "what"
   • Retrain with focus on OEQ/CEQ distinction
""".format(
        total=len(questions_with_labels),
        with_text=with_text,
        oeq=oeq_count,
        oeq_pct=oeq_count/len(questions_with_labels)*100,
        ceq=ceq_count,
        ceq_pct=ceq_count/len(questions_with_labels)*100
    ))

    # Save the extracted data for retraining
    import json
    output_file = "extracted_questions_for_retraining.json"
    with open(output_file, 'w') as f:
        json.dump(questions_with_labels, f, indent=2)

    print(f"\n💾 Extracted {len(questions_with_labels)} questions saved to: {output_file}")
    print("\n✅ We have sufficient data to retrain for better OEQ/CEQ classification!")

if __name__ == "__main__":
    analyze_csv_for_retraining()