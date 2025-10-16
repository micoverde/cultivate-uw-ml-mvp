#!/usr/bin/env python3
"""
Train 7-model ensemble with balanced class weights to fix OEQ bias.

CRITICAL FIX: Restore 7-model ensemble AND fix OEQ bias simultaneously.

7 Models:
1. Neural Network (MLP)
2. Random Forest
3. XGBoost
4. SVM
5. Logistic Regression
6. LightGBM
7. Gradient Boosting

ALL with class_weight='balanced' to fix the bias problem!

Author: Claude (Partner-Level Microsoft SDE)
Issue: #196 - Restore 7 models + Fix OEQ bias
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# Try to import optional models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available")

# Import shared feature extractor and model trainer
from src.ml.features.question_features import QuestionFeatureExtractor
from src.ml.model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("🚀 TRAINING 7-MODEL ENSEMBLE WITH BALANCED WEIGHTS")
    logger.info("=" * 80)

    # Load data
    logger.info("\n📂 Loading training data...")
    df = pd.read_csv("combined_training_data_v4.csv")

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
    # Train CLASSIC model
    # ========================================================================
    logger.info("\n🎯 Training CLASSIC model (Random Forest)...")

    rf_classic = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
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
    # Train 7-MODEL ENSEMBLE
    # ========================================================================
    logger.info("\n🎯 Training 7-MODEL ENSEMBLE with balanced weights...")

    models = []

    # Calculate sample weights for neural network
    class_weights = np.where(y_train == 0,
                            len(y_train) / (2 * np.sum(y_train == 0)),
                            len(y_train) / (2 * np.sum(y_train == 1)))

    # 1. Neural Network
    logger.info("  Training Neural Network...")
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
    nn.fit(X_train_scaled, y_train, sample_weight=class_weights)
    models.append(('neural_network', nn))

    # 2. Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models.append(('random_forest', rf))

    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        logger.info("  Training XGBoost...")
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        models.append(('xgboost', xgb_model))

    # 4. SVM
    logger.info("  Training SVM...")
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm.fit(X_train_scaled, y_train)
    models.append(('svm', svm))

    # 5. Logistic Regression
    logger.info("  Training Logistic Regression...")
    lr = LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        l1_ratio=0.5,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    models.append(('logistic', lr))

    # 6. LightGBM
    if LIGHTGBM_AVAILABLE:
        logger.info("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42
        )
        lgb_model.fit(X_train_scaled, y_train)
        models.append(('lightgbm', lgb_model))

    # 7. Gradient Boosting
    logger.info("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    # GradientBoosting doesn't have class_weight, use sample_weight
    gb.fit(X_train_scaled, y_train, sample_weight=class_weights)
    models.append(('gradient_boosting', gb))

    logger.info(f"\n✅ Created ensemble with {len(models)} models:")
    for name, _ in models:
        logger.info(f"   - {name}")

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
    # CRITICAL TESTING: CEQ Examples
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🧪 CRITICAL TESTING: CEQ Examples (MUST classify as CEQ!)")
    logger.info("=" * 80)

    ceq_examples = [
        "Did you like it?",
        "Is this correct?",
        "Can you do it?",
        "Do you want more?",
        "Are you ready?",
        "Did you finish?",
        "Is it big?",
    ]

    ceq_correct_classic = 0
    ceq_correct_ensemble = 0

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

        classic_correct = classic_label == "CEQ"
        ensemble_correct = ensemble_label == "CEQ"

        if classic_correct:
            ceq_correct_classic += 1
        if ensemble_correct:
            ceq_correct_ensemble += 1

        logger.info(f"\nText: '{text}'")
        logger.info(f"Expected: CEQ")
        logger.info(f"Classic:  {classic_label} (CEQ: {classic_proba[0]:.1%}, OEQ: {classic_proba[1]:.1%}) {'✅' if classic_correct else '❌ WRONG!'}")
        logger.info(f"Ensemble: {ensemble_label} (CEQ: {ensemble_proba[0]:.1%}, OEQ: {ensemble_proba[1]:.1%}) {'✅' if ensemble_correct else '❌ WRONG!'}")

    # ========================================================================
    # OEQ Examples (ensure no regression)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🧪 OEQ Examples (ensure we didn't break OEQ classification)")
    logger.info("=" * 80)

    oeq_examples = [
        "How did that happen?",
        "Why did it fall?",
        "What do you think happened?",
        "How can we make it better?",
        "Why do you think so?",
    ]

    oeq_correct_classic = 0
    oeq_correct_ensemble = 0

    for text in oeq_examples:
        features = extractor.extract(text).reshape(1, -1)
        features_scaled = scaler.transform(features)

        classic_pred = rf_classic.predict(features)[0]
        classic_proba = rf_classic.predict_proba(features)[0]

        ensemble_pred = ensemble.predict(features_scaled)[0]
        ensemble_proba = ensemble.predict_proba(features_scaled)[0]

        classic_label = "OEQ" if classic_pred == 1 else "CEQ"
        ensemble_label = "OEQ" if ensemble_pred == 1 else "CEQ"

        classic_correct = classic_label == "OEQ"
        ensemble_correct = ensemble_label == "OEQ"

        if classic_correct:
            oeq_correct_classic += 1
        if ensemble_correct:
            oeq_correct_ensemble += 1

        logger.info(f"\nText: '{text}'")
        logger.info(f"Expected: OEQ")
        logger.info(f"Classic:  {classic_label} (CEQ: {classic_proba[0]:.1%}, OEQ: {classic_proba[1]:.1%}) {'✅' if classic_correct else '❌ WRONG!'}")
        logger.info(f"Ensemble: {ensemble_label} (CEQ: {ensemble_proba[0]:.1%}, OEQ: {ensemble_proba[1]:.1%}) {'✅' if ensemble_correct else '❌ WRONG!'}")

    # ========================================================================
    # Summary Scores
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL SCORES")
    logger.info("=" * 80)
    logger.info(f"\nCEQ Test Accuracy:")
    logger.info(f"  Classic:  {ceq_correct_classic}/{len(ceq_examples)} ({ceq_correct_classic/len(ceq_examples)*100:.0f}%)")
    logger.info(f"  Ensemble: {ceq_correct_ensemble}/{len(ceq_examples)} ({ceq_correct_ensemble/len(ceq_examples)*100:.0f}%)")

    logger.info(f"\nOEQ Test Accuracy:")
    logger.info(f"  Classic:  {oeq_correct_classic}/{len(oeq_examples)} ({oeq_correct_classic/len(oeq_examples)*100:.0f}%)")
    logger.info(f"  Ensemble: {oeq_correct_ensemble}/{len(oeq_examples)} ({oeq_correct_ensemble/len(oeq_examples)*100:.0f}%)")

    # ========================================================================
    # Save models
    # ========================================================================
    logger.info("\n💾 Saving models...")

    # Save classic model
    classic_trainer = ModelTrainer(
        models={'random_forest': rf_classic},
        scaler=StandardScaler().fit(X_train),  # Identity scaler for classic
        model_type='classic'
    )

    # Save ensemble
    ensemble_trainer = ModelTrainer(
        models=dict(models),
        scaler=scaler,
        model_type='ensemble',
        ensemble=ensemble
    )

    joblib.dump(classic_trainer, "models/classic_latest.pkl")
    joblib.dump(ensemble_trainer, "models/ensemble_latest.pkl")

    logger.info("✅ Saved models/classic_latest.pkl")
    logger.info("✅ Saved models/ensemble_latest.pkl")

    logger.info("\n" + "=" * 80)
    logger.info(f"✅ {len(models)}-MODEL ENSEMBLE TRAINING COMPLETE WITH BALANCED WEIGHTS!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
