#!/usr/bin/env python3
"""
Train balanced models with class weights to fix OEQ bias.

CRITICAL FIX: Models are biased toward OEQ, classifying CEQ questions like
"Did you like it?" as OEQ. This script fixes the bias by:

1. Using class_weight='balanced' in all models
2. Using balanced_accuracy scoring
3. Testing on critical CEQ examples

Author: Claude (Partner-Level Microsoft SDE)
Issue: #196 - Fix massive OEQ bias
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# Import shared feature extractor
from src.ml.features.question_features import QuestionFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("🚀 TRAINING BALANCED MODELS TO FIX OEQ BIAS")
    logger.info("=" * 80)

    # Load data
    logger.info("\n📂 Loading training data...")
    df = pd.read_csv("combined_training_data.csv")

    # Extract features
    logger.info("🔬 Extracting features...")
    extractor = QuestionFeatureExtractor()
    X = extractor.extract_batch(df['text'].tolist())
    y = df['label'].map({'CEQ': 0, 'OEQ': 1}).values

    # Show class distribution
    logger.info(f"\n📊 Class distribution:")
    logger.info(f"   CEQ: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    logger.info(f"   OEQ: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"\n📊 Split: {len(X_train)} train, {len(X_test)} test")

    # ========================================================================
    # Train CLASSIC model with class_weight='balanced'
    # ========================================================================
    logger.info("\n🎯 Training CLASSIC model (Random Forest) with balanced weights...")

    rf_classic = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # ← FIX THE BIAS
        random_state=42,
        n_jobs=-1
    )

    rf_classic.fit(X_train, y_train)
    y_pred_classic = rf_classic.predict(X_test)

    logger.info("\n📈 CLASSIC Model Results:")
    logger.info(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_classic):.1%}")

    cm = confusion_matrix(y_test, y_pred_classic)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                Predicted CEQ  Predicted OEQ")
    logger.info(f"Actual CEQ      {cm[0,0]:>13}  {cm[0,1]:>13}")
    logger.info(f"Actual OEQ      {cm[1,0]:>13}  {cm[1,1]:>13}")

    logger.info(f"\n{classification_report(y_test, y_pred_classic, target_names=['CEQ', 'OEQ'])}")

    # ========================================================================
    # Train ENSEMBLE with class_weight='balanced' for each model
    # ========================================================================
    logger.info("\n🎯 Training ENSEMBLE (4 models) with balanced weights...")

    # Create models with balanced class weights
    models = []

    # 1. Neural Network (with class_weight through sample_weight)
    nn = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    # Calculate sample weights for neural network
    class_weights = np.where(y_train == 0, len(y_train) / (2 * np.sum(y_train == 0)),
                                          len(y_train) / (2 * np.sum(y_train == 1)))
    nn.fit(X_train_scaled, y_train, sample_weight=class_weights)
    nn_calibrated = CalibratedClassifierCV(nn, cv='prefit')
    nn_calibrated.fit(X_train_scaled, y_train)
    models.append(('neural_network', nn_calibrated))

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_calibrated = CalibratedClassifierCV(rf, cv='prefit')
    rf_calibrated.fit(X_train, y_train)
    models.append(('random_forest', rf_calibrated))

    # 3. SVM
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm.fit(X_train_scaled, y_train)
    svm_calibrated = CalibratedClassifierCV(svm, cv='prefit')
    svm_calibrated.fit(X_train_scaled, y_train)
    models.append(('svm', svm_calibrated))

    # 4. Logistic Regression
    lr = LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        l1_ratio=0.5,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    lr_calibrated = CalibratedClassifierCV(lr, cv='prefit')
    lr_calibrated.fit(X_train_scaled, y_train)
    models.append(('logistic', lr_calibrated))

    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train_scaled, y_train)

    y_pred_ensemble = ensemble.predict(X_test_scaled)

    logger.info("\n📈 ENSEMBLE Model Results:")
    logger.info(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_ensemble):.1%}")

    cm_ens = confusion_matrix(y_test, y_pred_ensemble)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                Predicted CEQ  Predicted OEQ")
    logger.info(f"Actual CEQ      {cm_ens[0,0]:>13}  {cm_ens[0,1]:>13}")
    logger.info(f"Actual OEQ      {cm_ens[1,0]:>13}  {cm_ens[1,1]:>13}")

    logger.info(f"\n{classification_report(y_test, y_pred_ensemble, target_names=['CEQ', 'OEQ'])}")

    # ========================================================================
    # Test on critical CEQ examples
    # ========================================================================
    logger.info("\n🧪 Testing on CRITICAL CEQ examples:")
    ceq_examples = [
        "Did you like it?",
        "Is this correct?",
        "Can you do it?",
        "Do you want more?",
        "Are you ready?",
    ]

    for text in ceq_examples:
        features = extractor.extract(text).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Classic prediction
        classic_pred = rf_classic.predict(features)[0]
        classic_proba = rf_classic.predict_proba(features)[0]

        # Ensemble prediction
        ensemble_pred = ensemble.predict(features_scaled)[0]
        ensemble_proba = ensemble.predict_proba(features_scaled)[0]

        classic_label = "OEQ" if classic_pred == 1 else "CEQ"
        ensemble_label = "OEQ" if ensemble_pred == 1 else "CEQ"

        logger.info(f"\nText: '{text}'")
        logger.info(f"Expected: CEQ")
        logger.info(f"Classic:  {classic_label} (CEQ: {classic_proba[0]:.1%}, OEQ: {classic_proba[1]:.1%}) {'✅' if classic_label == 'CEQ' else '❌ WRONG'}")
        logger.info(f"Ensemble: {ensemble_label} (CEQ: {ensemble_proba[0]:.1%}, OEQ: {ensemble_proba[1]:.1%}) {'✅' if ensemble_label == 'CEQ' else '❌ WRONG'}")

    # ========================================================================
    # Save models in compatible format
    # ========================================================================
    logger.info("\n💾 Saving models...")

    # Save classic model (wrap in trainer-like object)
    classic_trainer = type('obj', (object,), {
        'models': {'random_forest': rf_classic},
        'scaler': StandardScaler().fit(X_train),  # Identity scaler for classic (no scaling)
        'model_type': 'classic'
    })()

    # Save ensemble (wrap in trainer-like object)
    ensemble_trainer = type('obj', (object,), {
        'ensemble': ensemble,
        'models': dict(models),
        'scaler': scaler,
        'model_type': 'ensemble'
    })()

    joblib.dump(classic_trainer, "models/classic_latest.pkl")
    joblib.dump(ensemble_trainer, "models/ensemble_latest.pkl")

    logger.info("✅ Saved models/classic_latest.pkl")
    logger.info("✅ Saved models/ensemble_latest.pkl")

    logger.info("\n" + "=" * 80)
    logger.info("✅ BALANCED MODEL TRAINING COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
