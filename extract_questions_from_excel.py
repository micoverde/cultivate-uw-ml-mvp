#!/usr/bin/env python3
"""
Extract actual educator questions from VideosAskingQuestions Excel file.

The questions are embedded in quotation marks within the description columns.
This script extracts them and pairs them with their OEQ/CEQ labels.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #196 - Extract real question text for model training
"""

import pandas as pd
import re
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_quoted_text(text):
    """Extract text within quotation marks."""
    if pd.isna(text):
        return []

    text = str(text)

    # Match various quotation marks: "text", "text", 'text'
    patterns = [
        r'"([^"]+)"',  # Standard double quotes
        r'"([^"]+)"',  # Curly double quotes
        r"'([^']+)'",  # Single quotes
    ]

    questions = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        questions.extend(matches)

    return questions

def extract_label_from_description(text):
    """Extract OEQ/CEQ label from description text."""
    if pd.isna(text):
        return None

    text_upper = str(text).upper()
    if 'OEQ' in text_upper:
        return 'OEQ'
    elif 'CEQ' in text_upper:
        return 'CEQ'
    return None

def main():
    logger.info("=" * 80)
    logger.info("🔍 EXTRACTING QUESTIONS FROM EXCEL FILE")
    logger.info("=" * 80)

    # Load Excel file
    excel_path = "/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions (9-18).xlsx"
    logger.info(f"📂 Loading {excel_path}...")
    df = pd.read_excel(excel_path)
    logger.info(f"✅ Loaded {len(df)} videos")

    questions = []
    labels = []
    sources = []  # Track where each question came from

    # Process each row (video)
    for idx, row in df.iterrows():
        video_title = row['Video Title']

        # Check Q description columns (Q1-Q8)
        for i in range(1, 9):
            desc_col = f'Q{i} description'

            if desc_col in df.columns:
                description = row[desc_col]

                # Extract label
                label = extract_label_from_description(description)

                if label:
                    # Extract quoted questions
                    quoted_questions = extract_quoted_text(description)

                    for question in quoted_questions:
                        questions.append(question.strip())
                        labels.append(label)
                        sources.append(f"{video_title} - Q{i}")
                        logger.info(f"   Found: [{label}] \"{question.strip()}\" from {video_title}")

        # Also check the main Description column
        if 'Description' in df.columns:
            description = row['Description']
            quoted_questions = extract_quoted_text(description)

            for question in quoted_questions:
                # Try to infer label from context or mark as unknown
                # For now, we'll skip these unless we can determine the label
                logger.info(f"   Found in description: \"{question.strip()}\" (label unknown)")

    logger.info(f"\n✅ Extracted {len(questions)} labeled questions with actual text")
    logger.info(f"📊 Class distribution:")

    label_counts = pd.Series(labels).value_counts()
    for label, count in label_counts.items():
        percentage = count / len(labels) * 100
        logger.info(f"   - {label}: {count} ({percentage:.1f}%)")

    # Save to CSV
    output_df = pd.DataFrame({
        'text': questions,
        'label': labels,
        'source': sources
    })

    output_path = Path("real_questions_with_text.csv")
    output_df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Saved to: {output_path}")

    # Show some examples
    logger.info("\n📋 Sample extracted questions:")
    for i in range(min(10, len(questions))):
        logger.info(f"\n{i+1}. [{labels[i]}] \"{questions[i]}\"")
        logger.info(f"   Source: {sources[i]}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ EXTRACTION COMPLETE")
    logger.info("=" * 80)

    return output_df

if __name__ == "__main__":
    df = main()
