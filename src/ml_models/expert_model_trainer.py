"""
Expert-validated ML model trainer for educator interaction analysis.

This module trains the InteractionAnalyzer model using expert-annotated
data from Cultivate Learning researchers, focusing on:
- Open-ended vs closed-ended question classification
- Wait time behavior detection
- Scaffolding technique recognition
- CLASS framework scoring
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, List, Tuple
import logging
from transformers import AutoTokenizer
from .interaction_analyzer import InteractionAnalyzer, TrainingPipeline

logger = logging.getLogger(__name__)

class ExpertAnnotatedDataset(Dataset):
    """Dataset class for expert-annotated educator interaction data."""

    def __init__(self,
                 training_data: List[Dict],
                 tokenizer,
                 max_length: int = 512):
        """Initialize dataset with expert annotations."""
        self.data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Prepare training samples
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict]:
        """Convert expert annotations to training samples."""
        samples = []

        for example in self.data:
            analysis = example['analysis']
            questions = analysis['questions']

            # Create samples for each question
            for question in questions:
                # Convert question type to numeric label
                if question['question_type'] == 'open_ended':
                    question_label = 0
                elif question['question_type'] == 'closed_ended':
                    question_label = 1
                else:
                    question_label = 2  # unknown

                # Extract wait time score
                wait_time_score = 1.0 if question['wait_time']['waits_for_response'] else 0.0

                # Use question description as text input
                text = question['description']

                # Overall quality score from expert analysis
                overall_quality = question['quality_score']

                # Generate CLASS scores (normalized 0-1)
                class_scores = self._extract_class_scores(example['class_framework'])

                samples.append({
                    'text': text,
                    'question_label': question_label,
                    'wait_time_score': wait_time_score,
                    'quality_score': overall_quality,
                    'class_scores': class_scores,
                    'expert_validated': True
                })

        logger.info(f"Generated {len(samples)} training samples from expert annotations")
        return samples

    def _extract_class_scores(self, class_framework: Dict) -> List[float]:
        """Extract CLASS scores and normalize to 0-1 scale."""
        scores = []

        # Extract all CLASS indicators (normalized from 1-7 scale to 0-1)
        for domain in ['language_modeling', 'quality_feedback', 'concept_development']:
            domain_scores = class_framework.get(domain, {})
            for indicator, score in domain_scores.items():
                normalized_score = (score - 1) / 6  # Convert 1-7 to 0-1
                scores.append(max(0, min(1, normalized_score)))  # Clamp to 0-1

        # Ensure we have exactly 12 scores (4 per domain)
        while len(scores) < 12:
            scores.append(0.0)

        return scores[:12]  # Take first 12 scores

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize text
        encoding = self.tokenizer(
            sample['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'question_labels': torch.tensor(sample['question_label'], dtype=torch.long),
            'depth_scores': torch.tensor(sample['wait_time_score'], dtype=torch.float),
            'class_scores': torch.tensor(sample['class_scores'], dtype=torch.float),
            'quality_scores': torch.tensor(sample['quality_score'], dtype=torch.float)
        }


class ExpertModelTrainer:
    """Trainer for ML model using expert-validated annotations."""

    def __init__(self,
                 data_path: str = "data/training_examples/expert_training_dataset.json",
                 model_save_path: str = "models/expert_trained_model.pt"):
        """Initialize trainer with expert data."""
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load expert annotations
        self.training_data = self._load_expert_data()

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = InteractionAnalyzer()

        # Training hyperparameters optimized for expert data
        self.batch_size = 8  # Small batch size for limited data
        self.learning_rate = 2e-5
        self.num_epochs = 10  # More epochs for small dataset
        self.validation_split = 0.2

    def _load_expert_data(self) -> List[Dict]:
        """Load expert-annotated training data."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} expert-annotated examples")
            return data
        except Exception as e:
            logger.error(f"Error loading expert data from {self.data_path}: {e}")
            return []

    def prepare_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation data loaders."""
        # Split data into train/validation (no stratification due to limited data)
        train_data, val_data = train_test_split(
            self.training_data,
            test_size=self.validation_split,
            random_state=42,
            shuffle=True
        )

        # Create datasets
        train_dataset = ExpertAnnotatedDataset(train_data, self.tokenizer)
        val_dataset = ExpertAnnotatedDataset(val_data, self.tokenizer)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        return train_loader, val_loader

    def train_model(self) -> Dict:
        """Train the model on expert-annotated data."""
        logger.info("Starting expert-validated model training...")

        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders()

        # Initialize training pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = TrainingPipeline(
            model=self.model,
            learning_rate=self.learning_rate,
            device=device
        )

        # Training history
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracy': [],
            'epoch_metrics': []
        }

        best_val_accuracy = 0.0

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            epoch_train_losses = []
            pipeline.model.train()

            for batch_idx, batch in enumerate(train_loader):
                losses = pipeline.train_step(batch)
                epoch_train_losses.append(losses['total_loss'])

                if batch_idx % 5 == 0:  # Log every 5 batches
                    logger.info(f"Batch {batch_idx}: Loss = {losses['total_loss']:.4f}")

            avg_train_loss = np.mean(epoch_train_losses)
            training_history['train_losses'].append(avg_train_loss)

            # Validation phase
            val_metrics = pipeline.evaluate(val_loader)
            training_history['val_losses'].append(val_metrics['total'])

            # Calculate validation accuracy
            val_accuracy = self._calculate_validation_accuracy(val_loader, pipeline.model)
            training_history['val_accuracy'].append(val_accuracy)

            # Log epoch results
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_metrics['total'],
                'val_accuracy': val_accuracy,
                'val_question_loss': val_metrics['question'],
                'val_depth_loss': val_metrics['depth'],
                'val_class_loss': val_metrics['class']
            }
            training_history['epoch_metrics'].append(epoch_metrics)

            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_metrics['total']:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_model(pipeline.model, epoch + 1, val_accuracy)
                logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")

        # Final evaluation
        final_metrics = self._final_evaluation(val_loader, pipeline.model)
        training_history['final_metrics'] = final_metrics

        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")

        return training_history

    def _calculate_validation_accuracy(self, val_loader: DataLoader, model: nn.Module) -> float:
        """Calculate validation accuracy for question classification."""
        model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                text_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }

                outputs = model(text_inputs)
                question_logits = outputs['question_logits']
                predicted_labels = torch.argmax(question_logits, dim=1)

                correct_predictions += (predicted_labels == batch['question_labels']).sum().item()
                total_predictions += batch['question_labels'].size(0)

        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

    def _final_evaluation(self, val_loader: DataLoader, model: nn.Module) -> Dict:
        """Perform comprehensive final evaluation."""
        model.eval()

        all_predictions = []
        all_labels = []
        all_quality_predictions = []
        all_quality_targets = []

        with torch.no_grad():
            for batch in val_loader:
                text_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }

                outputs = model(text_inputs)

                # Question classification
                question_logits = outputs['question_logits']
                predicted_labels = torch.argmax(question_logits, dim=1)

                all_predictions.extend(predicted_labels.cpu().numpy())
                all_labels.extend(batch['question_labels'].cpu().numpy())

                # Quality prediction
                quality_predictions = outputs['overall_quality']
                all_quality_predictions.extend(quality_predictions.cpu().numpy())
                all_quality_targets.extend(batch['quality_scores'].cpu().numpy())

        # Classification metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        classification_report_dict = classification_report(
            all_labels,
            all_predictions,
            target_names=['open_ended', 'closed_ended', 'unknown'],
            output_dict=True
        )

        # Quality prediction metrics
        quality_mse = np.mean((np.array(all_quality_predictions) - np.array(all_quality_targets)) ** 2)
        quality_correlation = np.corrcoef(all_quality_predictions, all_quality_targets)[0, 1]

        return {
            'question_classification_accuracy': accuracy,
            'classification_report': classification_report_dict,
            'quality_prediction_mse': quality_mse,
            'quality_correlation': quality_correlation,
            'total_validation_samples': len(all_labels)
        }

    def _save_model(self, model: nn.Module, epoch: int, accuracy: float):
        """Save trained model with metadata."""
        model_state = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
            'training_data_source': 'expert_annotations',
            'model_architecture': 'InteractionAnalyzer',
            'tokenizer_name': 'bert-base-uncased',
            'training_samples': len(self.training_data),
            'created_at': '2025-09-23'
        }

        torch.save(model_state, self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Train model on expert annotations
    trainer = ExpertModelTrainer()

    print("🚀 Starting ML model training on expert-validated data...")
    print("📚 Using 25 expert-annotated educator-child interaction examples")
    print("🎯 Training for open-ended question detection and quality assessment")

    # Train the model
    training_history = trainer.train_model()

    if training_history:
        final_metrics = training_history['final_metrics']
        print(f"\n✅ Training completed!")
        print(f"🎯 Final question classification accuracy: {final_metrics['question_classification_accuracy']:.3f}")
        print(f"📊 Quality prediction correlation: {final_metrics['quality_correlation']:.3f}")
        print(f"🔍 Total validation samples: {final_metrics['total_validation_samples']}")
        print(f"💾 Model saved to: models/expert_trained_model.pt")
        print("\n🎓 Model ready for demo deployment!")
    else:
        print("❌ Training failed. Check data and dependencies.")