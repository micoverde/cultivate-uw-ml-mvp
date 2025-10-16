#!/usr/bin/env python3
"""
Train ML models using REAL expert-coded questions from classroom videos.

This script processes the VideosAskingQuestions CSV which contains:
- Real educator questions transcribed from classroom videos
- Expert coding of each question as OEQ (open-ended) or CEQ (closed-ended)
- Context about pause time, child responses, etc.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #196 - Train with real expert-coded data
"""

import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

# Import the shared feature extractor
from src.ml.features.question_features import QuestionFeatureExtractor
from src.ml.training.ensemble_trainer_v2 import EnhancedEnsembleTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_questions_from_csv(csv_path: str):
    """
    Extract all questions and their labels from the video data CSV.

    The CSV has columns like:
    - Question 1, Q1 description
    - Question 2, Q2 description
    ...

    Q descriptions contain labels like "OEQ", "CEQ", "CEQ (yes/no)", etc.
    """
    logger.info(f"📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    logger.info(f"✅ Loaded {len(df)} videos")
    logger.info(f"📊 Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns

    questions = []
    labels = []

    # Process each row (video)
    for idx, row in df.iterrows():
        # Check each question column (Q1-Q8)
        for i in range(1, 9):
            q_col = f'Question {i} ' if i <= 5 else f'Question {i}'
            desc_col = f'Q{i} description'

            if q_col in df.columns and desc_col in df.columns:
                question_text = row[q_col]
                description = row[desc_col]

                # Skip if no question
                if pd.isna(question_text) or pd.isna(description):
                    continue

                # Skip 'na' entries
                if str(question_text).lower() == 'na' or str(description).lower() == 'na':
                    continue

                # Extract label from description
                # Descriptions are like: "OEQ with a pause...", "CEQ (yes/no)...", etc.
                desc_upper = str(description).upper()
                if 'OEQ' in desc_upper:
                    label = 'OEQ'
                elif 'CEQ' in desc_upper:
                    label = 'CEQ'
                else:
                    continue  # Skip if we can't determine label

                questions.append(str(question_text).strip())
                labels.append(label)

    logger.info(f"✅ Extracted {len(questions)} questions with labels")
    logger.info(f"📊 Class distribution:")
    label_counts = pd.Series(labels).value_counts()
    for label, count in label_counts.items():
        logger.info(f"   - {label}: {count} ({count/len(labels)*100:.1%})")

    return questions, labels

def show_sample_questions(questions, labels, n=10):
    """Show sample questions to verify extraction."""
    logger.info(f"\n📋 Sample Questions:")
    for i in range(min(n, len(questions))):
        logger.info(f"\n{i+1}. {labels[i]}: \"{questions[i]}\"")

def main():
    logger.info("=" * 80)
    logger.info("🚀 TRAINING WITH REAL EXPERT-CODED VIDEO DATA")
    logger.info("=" * 80)

    # Initialize feature extractor
    extractor = QuestionFeatureExtractor()

    # Load real video data
    csv_path = "/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv"
    questions, labels = extract_questions_from_csv(csv_path)

    # Show samples
    show_sample_questions(questions, labels, n=10)

    # Extract features
    logger.info("\n🔬 Extracting features using QuestionFeatureExtractor...")
    X = extractor.extract_batch(questions)
    logger.info(f"✅ Features shape: {X.shape}")

    # Encode labels
    label_map = {'CEQ': 0, 'OEQ': 1}
    y = np.array([label_map[label] for label in labels])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"\n📊 Data split:")
    logger.info(f"   - Training: {len(X_train)} samples")
    logger.info(f"   - Testing: {len(X_test)} samples")

    # Train ensemble model
    logger.info("\n🎯 Training 7-model ensemble...")
    ensemble_trainer = EnhancedEnsembleTrainer(model_type='ensemble')
    ensemble_trainer.train(X_train, y_train, X_test, y_test)

    # Train classic model
    logger.info("\n🎯 Training classic single model...")
    classic_trainer = EnhancedEnsembleTrainer(model_type='classic')
    classic_trainer.train(X_train, y_train, X_test, y_test)

    # Save models
    logger.info("\n💾 Saving models...")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    ensemble_path = Path(f"models/ensemble_{timestamp}.pkl")
    classic_path = Path(f"models/classic_{timestamp}.pkl")

    joblib.dump(ensemble_trainer, ensemble_path)
    joblib.dump(classic_trainer, classic_path)

    # Create latest symlinks
    ensemble_latest = Path("models/ensemble_latest.pkl")
    classic_latest = Path("models/classic_latest.pkl")

    if ensemble_latest.exists():
        ensemble_latest.unlink()
    if classic_latest.exists():
        classic_latest.unlink()

    ensemble_latest.symlink_to(ensemble_path.name)
    classic_latest.symlink_to(classic_path.name)

    logger.info(f"✅ Ensemble saved to: {ensemble_path}")
    logger.info(f"✅ Classic saved to: {classic_path}")
    logger.info(f"✅ Created 'latest' symlinks")

    # Test with real-world questions
    logger.info("\n🧪 Testing with real educator questions:")
    test_cases = [
        "Why did it fall?",
        "How does that work?",
        "What do you think happened?",
        "Can you describe what you saw?",
        "Did you like it?",
        "Is this correct?",
        "Can you do it?",
        "Do you want more?",
        "What do you see?",
        "How many are there?",
    ]

    for text in test_cases:
        features = extractor.extract(text).reshape(1, -1)

        # Scale features for prediction
        features_scaled = ensemble_trainer.scaler.transform(features)

        # Ensemble prediction
        ensemble_pred = ensemble_trainer.ensemble.predict(features_scaled)[0]
        ensemble_proba = ensemble_trainer.ensemble.predict_proba(features_scaled)[0]

        # Classic prediction (use first model from classic trainer)
        # Classic trainer has only 2 models: random_forest and logistic
        classic_model = list(classic_trainer.models.values())[0]  # Use random forest
        classic_features_scaled = classic_trainer.scaler.transform(features)
        classic_pred = classic_model.predict(classic_features_scaled)[0]
        classic_proba = classic_model.predict_proba(classic_features_scaled)[0]

        ensemble_label = "OEQ" if ensemble_pred == 1 else "CEQ"
        classic_label = "OEQ" if classic_pred == 1 else "CEQ"

        # Get feature explanation
        explanation = extractor.explain_features(text)

        logger.info(f"\n📝 \"{text}\"")
        logger.info(f"   Ensemble: {ensemble_label} (conf: {max(ensemble_proba):.1%}) - OEQ: {ensemble_proba[1]:.1%}, CEQ: {ensemble_proba[0]:.1%}")
        logger.info(f"   Classic:  {classic_label} (conf: {max(classic_proba):.1%}) - OEQ: {classic_proba[1]:.1%}, CEQ: {classic_proba[0]:.1%}")
        logger.info(f"   Features: OEQ_score={explanation['oeq_indicators']['total_score']}, CEQ_score={explanation['ceq_indicators']['total_score']}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE - Using real expert-coded classroom data!")
    logger.info("=" * 80)
    logger.info(f"\n📊 Models trained on {len(questions)} real educator questions")
    logger.info(f"📹 From real classroom videos with expert coding")
    logger.info(f"🎯 Ready for deployment!")

if __name__ == "__main__":
    main()
