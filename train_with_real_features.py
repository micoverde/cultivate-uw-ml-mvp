#!/usr/bin/env python3
"""
Train ensemble models using REAL feature extraction from text.

This script uses the shared QuestionFeatureExtractor to ensure
feature consistency between training and inference.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #196 - Improved model training with consistent features
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import the shared feature extractor
from src.ml.features.question_features import QuestionFeatureExtractor
from src.ml.training.ensemble_trainer_v2 import EnhancedEnsembleTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_data(csv_path: str):
    """Load training data from CSV file."""
    logger.info(f"📂 Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)

    logger.info(f"✅ Loaded {len(df)} samples")
    logger.info(f"📊 Columns: {df.columns.tolist()}")

    # Check class distribution
    if 'label' in df.columns:
        class_counts = df['label'].value_counts()
        logger.info(f"📊 Class distribution:")
        for label, count in class_counts.items():
            logger.info(f"   - {label}: {count} ({count/len(df)*100:.1f}%)")

    return df

def extract_features_from_text(df: pd.DataFrame, extractor: QuestionFeatureExtractor):
    """Extract features from text column using the shared extractor."""
    logger.info("🔬 Extracting features from text using QuestionFeatureExtractor...")

    texts = df['text'].tolist()
    features = extractor.extract_batch(texts)

    logger.info(f"✅ Extracted {features.shape[1]} features from {features.shape[0]} samples")
    logger.info(f"📝 Features: {extractor.get_feature_names()}")

    # Show some examples
    logger.info("\n📋 Sample Feature Extraction:")
    for i in range(min(3, len(texts))):
        explanation = extractor.explain_features(texts[i])
        logger.info(f"\nText: '{texts[i]}'")
        logger.info(f"Label: {df.iloc[i]['label']}")
        logger.info(f"OEQ Score: {explanation['oeq_indicators']['total_score']}")
        logger.info(f"CEQ Score: {explanation['ceq_indicators']['total_score']}")
        logger.info(f"{explanation['interpretation']}")

    return features

def main():
    logger.info("=" * 80)
    logger.info("🚀 TRAINING WITH REAL FEATURE EXTRACTION")
    logger.info("=" * 80)

    # Initialize feature extractor
    extractor = QuestionFeatureExtractor()

    # Load training data
    csv_path = "combined_training_data.csv"
    df = load_training_data(csv_path)

    # Extract features from text
    X = extract_features_from_text(df, extractor)
    y = df['label'].values

    # Encode labels
    label_map = {'CEQ': 0, 'OEQ': 1}
    y_encoded = np.array([label_map[label] for label in y])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    logger.info(f"\n📊 Data split:")
    logger.info(f"   - Training: {len(X_train)} samples")
    logger.info(f"   - Testing: {len(X_test)} samples")

    # Train ensemble model
    logger.info("\n🎯 Training 7-model ensemble...")
    ensemble_trainer = EnhancedEnsembleTrainer(model_type='ensemble')
    ensemble_trainer.train(X_train, y_train, X_test, y_test)

    # Train classic model
    logger.info("\n🎯 Training classic model...")
    classic_trainer = EnhancedEnsembleTrainer(model_type='classic')
    classic_trainer.train(X_train, y_train, X_test, y_test)

    # Save models
    logger.info("\n💾 Saving models...")
    ensemble_path = Path("models/ensemble_latest.pkl")
    classic_path = Path("models/classic_latest.pkl")

    import joblib
    joblib.dump(ensemble_trainer, ensemble_path)
    joblib.dump(classic_trainer, classic_path)

    logger.info(f"✅ Ensemble saved to: {ensemble_path}")
    logger.info(f"✅ Classic saved to: {classic_path}")

    # Test the models with real questions
    logger.info("\n🧪 Testing models with real questions:")
    test_questions = [
        ("Why did it fall?", "OEQ"),
        ("How does that work?", "OEQ"),
        ("What do you think happened?", "OEQ"),
        ("Did you like it?", "CEQ"),
        ("Is this correct?", "CEQ"),
        ("Can you do it?", "CEQ"),
    ]

    for text, expected in test_questions:
        features = extractor.extract(text).reshape(1, -1)

        # Ensemble prediction
        ensemble_pred = ensemble_trainer.ensemble.predict(features)[0]
        ensemble_proba = ensemble_trainer.ensemble.predict_proba(features)[0]

        # Classic prediction (use first model from classic trainer - random forest)
        classic_model = list(classic_trainer.models.values())[0]
        classic_pred = classic_model.predict(features)[0]
        classic_proba = classic_model.predict_proba(features)[0]

        ensemble_label = "OEQ" if ensemble_pred == 1 else "CEQ"
        classic_label = "OEQ" if classic_pred == 1 else "CEQ"

        logger.info(f"\nText: '{text}'")
        logger.info(f"Expected: {expected}")
        logger.info(f"Ensemble: {ensemble_label} (OEQ: {ensemble_proba[1]:.1%}, CEQ: {ensemble_proba[0]:.1%})")
        logger.info(f"Classic:  {classic_label} (OEQ: {classic_proba[1]:.1%}, CEQ: {classic_proba[0]:.1%})")

        # Show feature analysis
        explanation = extractor.explain_features(text)
        logger.info(f"Features: OEQ_score={explanation['oeq_indicators']['total_score']}, "
                   f"CEQ_score={explanation['ceq_indicators']['total_score']}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE - Models use real feature extraction!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
