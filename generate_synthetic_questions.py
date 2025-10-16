#!/usr/bin/env python3
"""
Generate synthetic OEQ and CEQ training data based on linguistic patterns.

Combines with real educator questions to create a comprehensive training set.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #196 - Create large training dataset with real patterns
"""

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OEQ examples - encourage critical thinking, explanation
OEQ_QUESTIONS = [
    # How questions (encourage explanation)
    "How did that happen?",
    "How does this work?",
    "How do you think it works?",
    "How can we solve this?",
    "How would you describe it?",
    "How did you figure that out?",
    "How do you know?",
    "How could we make it better?",
    "How is this similar to that?",
    "How are they different?",

    # Why questions (encourage reasoning)
    "Why did it fall?",
    "Why do you think that happened?",
    "Why is it important?",
    "Why did you choose that?",
    "Why do we need this?",
    "Why does it work that way?",
    "Why is it called that?",
    "Why do you think so?",

    # What do you think (solicit perspective)
    "What do you think happened?",
    "What do you think will happen next?",
    "What do you think about this?",
    "What do you think it means?",
    "What do you think we should do?",
    "What did you think when you saw that?",

    # Describe/Explain/Tell me (request elaboration)
    "Can you describe what you saw?",
    "Can you explain how you did that?",
    "Tell me about what you made?",
    "Describe what happened?",
    "Explain what you're thinking?",
    "Tell me more about that?",
    "Can you show me how it works?",

    # What else/What if (encourage expansion)
    "What else could we try?",
    "What else do you notice?",
    "What if we did it differently?",
    "What would happen if we changed it?",
    "What else can you tell me?",

    # Open-ended what/where/when
    "What happened next?",
    "What did we learn?",
    "What do you need help with?",
    "Where do you think it goes?",
    "When do you think we should do it?",
]

# CEQ examples - yes/no, simple factual answers
CEQ_QUESTIONS = [
    # Did questions
    "Did you like it?",
    "Did you finish?",
    "Did you see that?",
    "Did it work?",
    "Did you try it?",
    "Did you have fun?",

    # Is/Are questions
    "Is this correct?",
    "Is it big?",
    "Is it red?",
    "Are you ready?",
    "Are they the same?",
    "Is that yours?",

    # Can/Could questions (ability/permission)
    "Can you do it?",
    "Can you see it?",
    "Could you help me?",
    "Can we start?",
    "Could you pass that?",

    # Do/Does questions
    "Do you want more?",
    "Do you like this?",
    "Does it fit?",
    "Do you need help?",
    "Does it work?",

    # What/Which (specific selection)
    "What color is it?",
    "What number is this?",
    "Which one do you want?",
    "What's your name?",
    "Which is bigger?",

    # How many (counting)
    "How many are there?",
    "How many do you have?",
    "How many do you see?",

    # Yes/no clarifications
    "Is it this one?",
    "Was it you?",
    "Were you there?",
    "Will you try?",
    "Would you like that?",
    "Should we go?",
]

def main():
    logger.info("=" * 80)
    logger.info("🎯 GENERATING COMPREHENSIVE TRAINING DATASET")
    logger.info("=" * 80)

    # Load real educator questions
    logger.info("\n📂 Loading real educator questions...")
    real_df = pd.read_csv("real_questions_with_text.csv")
    logger.info(f"✅ Loaded {len(real_df)} real educator questions")
    logger.info(f"   - OEQ: {len(real_df[real_df['label']=='OEQ'])}")
    logger.info(f"   - CEQ: {len(real_df[real_df['label']=='CEQ'])}")

    # Create synthetic dataset
    logger.info("\n🔧 Generating synthetic questions based on linguistic patterns...")
    synthetic_oeq = pd.DataFrame({
        'text': OEQ_QUESTIONS,
        'label': ['OEQ'] * len(OEQ_QUESTIONS),
        'source': ['synthetic'] * len(OEQ_QUESTIONS)
    })

    synthetic_ceq = pd.DataFrame({
        'text': CEQ_QUESTIONS,
        'label': ['CEQ'] * len(CEQ_QUESTIONS),
        'source': ['synthetic'] * len(CEQ_QUESTIONS)
    })

    logger.info(f"✅ Generated {len(OEQ_QUESTIONS)} synthetic OEQ questions")
    logger.info(f"✅ Generated {len(CEQ_QUESTIONS)} synthetic CEQ questions")

    # Combine all data
    combined_df = pd.concat([real_df, synthetic_oeq, synthetic_ceq], ignore_index=True)

    # Remove duplicates (keep real over synthetic)
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')

    logger.info(f"\n📊 Final dataset statistics:")
    logger.info(f"   Total: {len(combined_df)} questions")

    label_counts = combined_df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = count / len(combined_df) * 100
        logger.info(f"   - {label}: {count} ({percentage:.1f}%)")

    # Save combined dataset
    output_path = "combined_training_data.csv"
    combined_df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Saved combined dataset to: {output_path}")

    # Show distribution by source
    logger.info("\n📋 Distribution by source:")
    source_counts = combined_df['source'].value_counts()
    for source, count in source_counts.items():
        logger.info(f"   - {source}: {count}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ DATASET GENERATION COMPLETE")
    logger.info("=" * 80)

    return combined_df

if __name__ == "__main__":
    df = main()
