"""
Core ML model for analyzing educator-child interactions.

Combines text and video analysis to provide comprehensive assessment
of interaction quality based on early childhood education research.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class InteractionAnalyzer(nn.Module):
    """Multi-modal ML model for educator-child interaction analysis."""

    def __init__(self,
                 text_model_name: str = "bert-base-uncased",
                 hidden_dim: int = 768,
                 num_quality_indicators: int = 12,
                 fusion_dim: int = 256):
        """Initialize the interaction analyzer model.

        Args:
            text_model_name: Pre-trained text model name
            hidden_dim: Hidden dimension size
            num_quality_indicators: Number of CLASS framework indicators
            fusion_dim: Fusion layer dimension
        """
        super().__init__()

        # Text processing components
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Question detection head
        self.question_classifier = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, 3)  # open-ended, closed-ended, not-question
        )

        # Conversation depth analyzer
        self.depth_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, 1)  # depth score 0-1
        )

        # CLASS framework scorer
        self.class_scorer = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, num_quality_indicators)  # 12 CLASS indicators
        )

        # Multi-modal fusion (for future video integration)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, hidden_dim)
        )

        # Overall quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, 1)  # overall quality 0-1
        )

    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            text_inputs: Tokenized text inputs

        Returns:
            Dictionary of model outputs
        """
        # Encode text
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_dim]

        # Generate predictions
        question_logits = self.question_classifier(text_features)
        depth_score = torch.sigmoid(self.depth_analyzer(text_features))
        class_scores = torch.sigmoid(self.class_scorer(text_features))
        overall_quality = torch.sigmoid(self.quality_scorer(text_features))

        return {
            'question_logits': question_logits,
            'depth_score': depth_score,
            'class_scores': class_scores,
            'overall_quality': overall_quality,
            'text_features': text_features
        }

    def predict_interaction(self, transcript: str) -> Dict:
        """Predict quality indicators for a conversation transcript.

        Args:
            transcript: Conversation transcript text

        Returns:
            Dictionary of quality predictions
        """
        self.eval()
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                transcript,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            # Get predictions
            outputs = self.forward(inputs)

            # Process outputs
            question_probs = torch.softmax(outputs['question_logits'], dim=-1)
            question_type = torch.argmax(question_probs, dim=-1).item()
            question_types = ['open_ended', 'closed_ended', 'not_question']

            class_scores = outputs['class_scores'].squeeze().cpu().numpy()
            class_indicators = [
                'concept_analysis', 'concept_creating', 'concept_integration', 'concept_connections',
                'feedback_scaffolding', 'feedback_encouraging', 'feedback_specific', 'feedback_back_forth',
                'language_conversations', 'language_open_questions', 'language_repetition', 'language_advanced'
            ]

            results = {
                'question_type': question_types[question_type],
                'question_confidence': question_probs.max().item(),
                'depth_score': outputs['depth_score'].item(),
                'overall_quality': outputs['overall_quality'].item(),
                'class_scores': dict(zip(class_indicators, class_scores.tolist()))
            }

            return results


class TrainingPipeline:
    """Training pipeline for the interaction analyzer model."""

    def __init__(self,
                 model: InteractionAnalyzer,
                 learning_rate: float = 2e-5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize training pipeline.

        Args:
            model: InteractionAnalyzer model
            learning_rate: Learning rate for optimizer
            device: Training device
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Loss functions
        self.question_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Perform one training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary of losses
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        text_inputs = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device)
        }
        outputs = self.model(text_inputs)

        # Calculate losses
        question_loss = self.question_loss(
            outputs['question_logits'],
            batch['question_labels'].to(self.device)
        )

        depth_loss = self.regression_loss(
            outputs['depth_score'].squeeze(),
            batch['depth_scores'].to(self.device)
        )

        class_loss = self.regression_loss(
            outputs['class_scores'],
            batch['class_scores'].to(self.device)
        )

        quality_loss = self.regression_loss(
            outputs['overall_quality'].squeeze(),
            batch['quality_scores'].to(self.device)
        )

        # Combined loss
        total_loss = question_loss + depth_loss + class_loss + quality_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'question_loss': question_loss.item(),
            'depth_loss': depth_loss.item(),
            'class_loss': class_loss.item(),
            'quality_loss': quality_loss.item()
        }

    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_losses = {'total': 0, 'question': 0, 'depth': 0, 'class': 0, 'quality': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                text_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                outputs = self.model(text_inputs)

                # Calculate losses
                question_loss = self.question_loss(
                    outputs['question_logits'],
                    batch['question_labels'].to(self.device)
                )
                depth_loss = self.regression_loss(
                    outputs['depth_score'].squeeze(),
                    batch['depth_scores'].to(self.device)
                )
                class_loss = self.regression_loss(
                    outputs['class_scores'],
                    batch['class_scores'].to(self.device)
                )
                quality_loss = self.regression_loss(
                    outputs['overall_quality'].squeeze(),
                    batch['quality_scores'].to(self.device)
                )
                total_loss = question_loss + depth_loss + class_loss + quality_loss

                total_losses['total'] += total_loss.item()
                total_losses['question'] += question_loss.item()
                total_losses['depth'] += depth_loss.item()
                total_losses['class'] += class_loss.item()
                total_losses['quality'] += quality_loss.item()
                num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses


class FeedbackGenerator:
    """Generates research-based feedback from model predictions."""

    def __init__(self):
        """Initialize feedback generator with research-based templates."""
        self.feedback_templates = {
            'high_quality': [
                "Excellent use of open-ended questions that promote critical thinking!",
                "Your conversation shows great depth and follows the child's interests.",
                "Strong back-and-forth interaction that supports language development."
            ],
            'moderate_quality': [
                "Good interaction with opportunities for enhancement.",
                "Consider adding more 'how' and 'why' questions to extend thinking.",
                "Your responsiveness to the child is developing well."
            ],
            'needs_improvement': [
                "Focus on asking more open-ended questions to promote thinking.",
                "Try to create more back-and-forth conversation opportunities.",
                "Build on the child's responses with follow-up questions."
            ]
        }

        self.specific_suggestions = {
            'open_ended_questions': [
                "Replace 'Is this red?' with 'What do you notice about this color?'",
                "Instead of 'Do you like this?', try 'What interests you about this?'",
                "Change 'Can you count them?' to 'How might we figure out how many there are?'"
            ],
            'conversation_depth': [
                "Use phrases like 'Tell me more about...' to extend conversations",
                "Build on children's ideas with 'That reminds me of...'",
                "Ask follow-up questions that connect to their experiences"
            ],
            'responsiveness': [
                "Follow the child's lead when they show interest in something",
                "Use wait time to give children space to think and respond",
                "Acknowledge and build on children's contributions"
            ]
        }

    def generate_feedback(self, predictions: Dict, transcript: str) -> Dict:
        """Generate personalized feedback based on model predictions.

        Args:
            predictions: Model predictions
            transcript: Original transcript

        Returns:
            Dictionary of feedback and recommendations
        """
        overall_quality = predictions['overall_quality']
        depth_score = predictions['depth_score']
        class_scores = predictions['class_scores']

        # Determine overall quality level
        if overall_quality > 0.7:
            quality_level = 'high_quality'
        elif overall_quality > 0.4:
            quality_level = 'moderate_quality'
        else:
            quality_level = 'needs_improvement'

        feedback = {
            'overall_message': np.random.choice(self.feedback_templates[quality_level]),
            'strengths': [],
            'improvements': [],
            'specific_suggestions': [],
            'research_basis': []
        }

        # Analyze specific areas
        self._analyze_question_quality(feedback, predictions)
        self._analyze_conversation_depth(feedback, predictions)
        self._analyze_class_indicators(feedback, predictions)

        # Add research citations
        feedback['research_basis'] = [
            "Hart & Risley (1995): Quality of adult-child conversations predicts language development",
            "Pianta et al. (2008): CLASS framework indicators for effective teaching",
            "Rowe (2012): Open-ended questions support vocabulary growth"
        ]

        return feedback

    def _analyze_question_quality(self, feedback: Dict, predictions: Dict):
        """Analyze question quality and add feedback."""
        if predictions['question_type'] == 'open_ended':
            feedback['strengths'].append("Uses open-ended questions effectively")
        else:
            feedback['improvements'].append("Increase use of open-ended questions")
            feedback['specific_suggestions'].extend(
                np.random.choice(self.specific_suggestions['open_ended_questions'], 2)
            )

    def _analyze_conversation_depth(self, feedback: Dict, predictions: Dict):
        """Analyze conversation depth and add feedback."""
        if predictions['depth_score'] > 0.6:
            feedback['strengths'].append("Creates conversations with good depth")
        else:
            feedback['improvements'].append("Build more depth in conversations")
            feedback['specific_suggestions'].extend(
                np.random.choice(self.specific_suggestions['conversation_depth'], 2)
            )

    def _analyze_class_indicators(self, feedback: Dict, predictions: Dict):
        """Analyze CLASS indicators and add feedback."""
        class_scores = predictions['class_scores']

        # Find strongest and weakest areas
        strong_areas = [k for k, v in class_scores.items() if v > 0.7]
        weak_areas = [k for k, v in class_scores.items() if v < 0.4]

        if strong_areas:
            feedback['strengths'].append(f"Strong performance in: {', '.join(strong_areas[:2])}")

        if weak_areas:
            feedback['improvements'].append(f"Focus areas: {', '.join(weak_areas[:2])}")


if __name__ == "__main__":
    # Example usage
    model = InteractionAnalyzer()
    feedback_gen = FeedbackGenerator()

    sample_transcript = """
    Teacher: What are you building?
    Child: A castle!
    Teacher: Why did you choose to make a castle?
    Child: Because I like princesses.
    Teacher: How do you think the princess feels in her castle?
    """

    predictions = model.predict_interaction(sample_transcript)
    feedback = feedback_gen.generate_feedback(predictions, sample_transcript)

    print("Predictions:", predictions)
    print("Feedback:", feedback)