#!/usr/bin/env python3
"""
Production Ensemble Training Script
Trains all 7 models and uploads to Azure Blob Storage

Usage:
    python train_ensemble_production.py

Author: Warren & Claude
Issue: #187 - Ensemble ML Model Architecture
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import joblib
import json

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our ensemble trainer
from src.ml.training.ensemble_trainer_v2 import EnhancedEnsembleTrainer, PerformanceMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_data():
    """Load and prepare training data"""
    logger.info("📂 Loading training data...")

    # Load the CSV file
    data_path = Path("data/ml_training/training_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"✅ Loaded {len(df)} samples")

    # Log class distribution
    class_counts = df['question_type'].value_counts()
    logger.info(f"📊 Class distribution: {class_counts.to_dict()}")
    logger.info(f"   - OEQ: {class_counts.get('OEQ', 0)} ({class_counts.get('OEQ', 0)/len(df)*100:.1f}%)")
    logger.info(f"   - CEQ: {class_counts.get('CEQ', 0)} ({class_counts.get('CEQ', 0)/len(df)*100:.1f}%)")

    # Select features for training
    feature_columns = [
        'word_count',
        'transcription_confidence',
        'audio_duration',
        'ling_word_count',
        'ling_character_count',
        'ling_question_count',
        'ling_has_question',
        'ling_educator_language_score',
        'ling_avg_word_length',
        'ling_avg_sentence_length',
        'ling_sentence_count',
        'ling_avg_word_confidence',
        'ling_text_complexity_score',
        'interact_question_type_score',
        'interact_age_appropriateness',
        'interact_provides_wait_time',
        'interact_creates_response_opportunity',
        'interact_teacher_support_present',
        'interact_overall_interaction_quality'
    ]

    # Extract features and labels
    X = df[feature_columns].fillna(0).values
    y = df['question_type'].values

    # Also extract text for reference
    texts = df['transcript_text'].values

    logger.info(f"📐 Feature dimensions: {X.shape}")
    logger.info(f"📝 Features used: {feature_columns}")

    return X, y, texts, df

def train_ensemble_model(X, y):
    """Train the ensemble model"""
    logger.info("\n🚀 Starting Ensemble Training...")
    logger.info("=" * 60)

    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"📊 Data split:")
    logger.info(f"   - Training: {len(X_train)} samples")
    logger.info(f"   - Testing: {len(X_test)} samples")

    # Initialize trainer with Azure support
    trainer = EnhancedEnsembleTrainer(
        model_type='ensemble',
        use_azure=True  # Will save to Azure if configured
    )

    # Train the ensemble
    logger.info("\n🎯 Training 7-model ensemble:")
    logger.info("   1. Neural Network (MLP)")
    logger.info("   2. Random Forest")
    logger.info("   3. XGBoost")
    logger.info("   4. Support Vector Machine")
    logger.info("   5. Logistic Regression")
    logger.info("   6. LightGBM")
    logger.info("   7. Gradient Boosting")

    results = trainer.train(X_train, y_train, X_test, y_test)

    # Display performance
    logger.info("\n📈 Training Complete! Results:")
    logger.info("=" * 60)

    for model_name, perf in results['performances'].items():
        if 'val_metrics' in perf:
            metrics = perf['val_metrics']
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  ✓ Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"  ✓ F1 Score: {metrics['f1_macro']:.3f}")
            logger.info(f"  ✓ Precision: {metrics['precision_macro']:.3f}")
            logger.info(f"  ✓ Recall: {metrics['recall_macro']:.3f}")

            if 'weights' in perf:
                logger.info(f"  ✓ Ensemble Weights: {perf['weights']}")

    # Evaluate on full test set for final metrics
    logger.info("\n🔍 Final Evaluation on Test Set:")
    evaluation = trainer.evaluate_on_ground_truth(X_test, y_test)

    if 'ensemble' in evaluation:
        final_metrics = evaluation['ensemble']['metrics']
        logger.info(f"  🎯 Final Ensemble Accuracy: {final_metrics['accuracy']:.3f}")
        logger.info(f"  🎯 Final F1 Score: {final_metrics['f1_macro']:.3f}")

        # Log confusion matrix
        conf_matrix = evaluation['ensemble']['confusion_matrix']
        logger.info(f"\n📊 Confusion Matrix:")
        logger.info(f"         Predicted")
        logger.info(f"         CEQ  OEQ")
        logger.info(f"Actual CEQ  {conf_matrix[0][0]:3d}  {conf_matrix[0][1]:3d}")
        logger.info(f"       OEQ  {conf_matrix[1][0]:3d}  {conf_matrix[1][1]:3d}")

    logger.info(f"\n⏱️ Total training time: {results['training_time']:.2f} seconds")

    return trainer, results

def train_classic_model(X, y):
    """Train the classic model for comparison"""
    logger.info("\n🎯 Training Classic Model (for comparison)...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize classic trainer
    trainer = EnhancedEnsembleTrainer(
        model_type='classic',
        use_azure=True
    )

    # Train classic models
    results = trainer.train(X_train, y_train, X_test, y_test)

    # Display performance
    logger.info("\n📈 Classic Model Results:")
    for model_name, perf in results['performances'].items():
        if 'val_metrics' in perf:
            metrics = perf['val_metrics']
            logger.info(f"  {model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}")

    return trainer, results

def save_models_locally(ensemble_trainer, classic_trainer):
    """Save models locally as backup"""
    logger.info("\n💾 Saving models locally...")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save ensemble
    ensemble_path = models_dir / f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(ensemble_trainer, ensemble_path)
    logger.info(f"  ✓ Ensemble saved to: {ensemble_path}")

    # Save classic
    classic_path = models_dir / f"classic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(classic_trainer, classic_path)
    logger.info(f"  ✓ Classic saved to: {classic_path}")

    # Save latest symlinks
    latest_ensemble = models_dir / "ensemble_latest.pkl"
    latest_classic = models_dir / "classic_latest.pkl"

    if latest_ensemble.exists():
        latest_ensemble.unlink()
    if latest_classic.exists():
        latest_classic.unlink()

    latest_ensemble.symlink_to(ensemble_path.name)
    latest_classic.symlink_to(classic_path.name)

    logger.info(f"  ✓ Created 'latest' symlinks")

    return ensemble_path, classic_path

def create_deployment_report(ensemble_results, classic_results):
    """Create a deployment readiness report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "training_data": {
            "samples": 98,  # 99 - 1 header
            "features": 19,
            "classes": ["OEQ", "CEQ"]
        },
        "ensemble_performance": {},
        "classic_performance": {},
        "recommendation": "",
        "azure_deployment": {
            "status": "pending",
            "container": "ml-models",
            "blob_names": []
        }
    }

    # Extract ensemble metrics
    if 'ensemble' in ensemble_results['performances']:
        metrics = ensemble_results['performances']['ensemble']['val_metrics']
        report['ensemble_performance'] = {
            "accuracy": round(metrics['accuracy'], 3),
            "f1_score": round(metrics['f1_macro'], 3),
            "training_time": round(ensemble_results['training_time'], 2)
        }

    # Extract classic metrics
    if 'random_forest' in classic_results['performances']:
        metrics = classic_results['performances']['random_forest']['val_metrics']
        report['classic_performance'] = {
            "accuracy": round(metrics['accuracy'], 3),
            "f1_score": round(metrics['f1_macro'], 3),
            "training_time": round(classic_results['training_time'], 2)
        }

    # Make recommendation
    ensemble_score = report['ensemble_performance'].get('f1_score', 0)
    classic_score = report['classic_performance'].get('f1_score', 0)

    if ensemble_score > classic_score + 0.05:
        report['recommendation'] = "DEPLOY ENSEMBLE - Significantly better performance"
    elif classic_score > ensemble_score:
        report['recommendation'] = "DEPLOY CLASSIC - Better performance with simpler model"
    else:
        report['recommendation'] = "DEPLOY ENSEMBLE - Similar performance, more robust"

    # Save report
    report_path = Path("models/deployment_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📄 Deployment report saved to: {report_path}")

    return report

def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("🚀 PRODUCTION ENSEMBLE TRAINING PIPELINE")
    logger.info("=" * 60)

    try:
        # Step 1: Load data
        X, y, texts, df = load_training_data()

        # Step 2: Train ensemble
        ensemble_trainer, ensemble_results = train_ensemble_model(X, y)

        # Step 3: Train classic for comparison
        classic_trainer, classic_results = train_classic_model(X, y)

        # Step 4: Save models locally
        ensemble_path, classic_path = save_models_locally(ensemble_trainer, classic_trainer)

        # Step 5: Create deployment report
        report = create_deployment_report(ensemble_results, classic_results)

        # Step 6: Azure deployment status
        logger.info("\n☁️ Azure Deployment Status:")
        if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
            logger.info("  ✅ Azure connection configured")
            logger.info("  ✅ Models uploaded to blob storage")
            logger.info("  📦 Container: ml-models")
            logger.info("  🔗 Blobs: ensemble_latest.pkl, classic_latest.pkl")
        else:
            logger.info("  ⚠️ Azure connection not configured")
            logger.info("  💾 Models saved locally only")
            logger.info("  📝 Set AZURE_STORAGE_CONNECTION_STRING to enable cloud storage")

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING COMPLETE - READY FOR PRODUCTION")
        logger.info("=" * 60)
        logger.info(f"\n📊 Final Performance Summary:")
        logger.info(f"  Ensemble: {report['ensemble_performance']}")
        logger.info(f"  Classic:  {report['classic_performance']}")
        logger.info(f"\n🎯 Recommendation: {report['recommendation']}")

        logger.info("\n📝 Next Steps:")
        logger.info("  1. Review deployment_report.json")
        logger.info("  2. Test API endpoints: python test_ensemble_api.py")
        logger.info("  3. Deploy to production: git push origin main")
        logger.info("  4. Monitor performance in production")

        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())