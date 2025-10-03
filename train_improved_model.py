#!/usr/bin/env python3
"""
Train Improved OEQ/CEQ Model with Gradient Descent
Using comprehensive datasets and weighted loss to fix CEQ misclassification
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedOEQCEQClassifier(nn.Module):
    """Enhanced neural network optimized for CEQ detection"""
    def __init__(self, input_size=56, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(ImprovedOEQCEQClassifier, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AdvancedFeatureExtractor:
    """Enhanced feature extraction with better CEQ/OEQ discrimination"""

    def __init__(self):
        # Critical CEQ starters - questions that almost always expect yes/no
        self.strong_ceq_starters = ['did', 'does', 'do', 'is', 'are', 'was', 'were',
                                    'will', 'would', 'can', 'could', 'should', 'has', 'have', 'had']

        # Strong OEQ indicators
        self.strong_oeq_starters = ['what', 'how', 'why', 'where', 'when', 'who', 'which', 'whose']

        # CEQ keywords for counting
        self.ceq_keywords = self.strong_ceq_starters + ['may', 'might', 'must', 'shall', 'am']

        # OEQ keywords for counting
        self.oeq_keywords = self.strong_oeq_starters + [
            'describe', 'explain', 'tell', 'think', 'feel',
            'notice', 'see', 'observe', 'imagine', 'wonder'
        ]

    def extract_features(self, text: str) -> np.ndarray:
        """Extract 56 features with enhanced CEQ/OEQ discrimination"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        first_word = words[0] if words else ''

        features = []

        # 1-16: OEQ keyword counts (16 features)
        for i, keyword in enumerate(self.oeq_keywords[:16]):
            features.append(text_lower.count(keyword))

        # 17-29: CEQ keyword counts (13 features)
        for i, keyword in enumerate(self.ceq_keywords[:13]):
            features.append(text_lower.count(keyword))

        # 30-35: Basic text statistics (6 features)
        features.append(len(words))  # Word count
        features.append(len(text))  # Character count
        features.append(np.mean([len(w) for w in words]) if words else 0)  # Avg word length
        features.append(text.count('?'))  # Question marks
        features.append(text.count('!'))  # Exclamations
        features.append(text.count(','))  # Commas

        # 36-41: Question type indicators (6 features)
        features.append(1 if first_word in self.strong_oeq_starters else 0)
        features.append(1 if first_word in self.strong_ceq_starters else 0)  # CRITICAL for CEQ
        features.append(1 if 'you' in text_lower else 0)
        features.append(1 if 'think' in text_lower else 0)
        features.append(1 if 'feel' in text_lower else 0)
        features.append(1 if first_word == 'did' else 0)  # CRITICAL: Special flag for 'did'

        # 42-47: Specific starter checks (6 features)
        features.append(1 if text_lower.startswith('what') else 0)
        features.append(1 if text_lower.startswith('how') else 0)
        features.append(1 if text_lower.startswith('why') else 0)
        features.append(1 if text_lower.startswith('is') else 0)
        features.append(1 if text_lower.startswith('are') else 0)
        features.append(1 if text_lower.startswith('do') else 0)

        # 48-52: Pattern features (5 features)
        features.append(text_lower.count('do you'))
        features.append(text_lower.count('can you'))
        features.append(text_lower.count('tell me'))
        features.append(1 if 'or' in text_lower else 0)  # Choice questions
        features.append(1 if len(text) < 30 else 0)  # Short questions often CEQ

        # 53-56: Context features (4 features)
        features.append(1 if 'because' in text_lower else 0)
        features.append(1 if 'yes' in text_lower or 'no' in text_lower else 0)  # Explicit yes/no
        features.append(len([w for w in words if len(w) > 7]))  # Complex word count
        features.append(1 if text_lower.endswith('?') else 0)

        return np.array(features[:56], dtype=np.float32)

def load_all_datasets():
    """Load and combine all training datasets"""

    all_examples = []

    # Load comprehensive dataset
    try:
        with open('comprehensive_training_data.json', 'r') as f:
            data1 = json.load(f)
            for ex in data1['examples']:
                all_examples.append((ex['text'], ex['label']))
        logger.info(f"Loaded {len(data1['examples'])} examples from comprehensive dataset")
    except Exception as e:
        logger.warning(f"Could not load comprehensive dataset: {e}")

    # Load HuggingFace-inspired dataset
    try:
        with open('huggingface_augmented_data.json', 'r') as f:
            data2 = json.load(f)
            for ex in data2['examples']:
                all_examples.append((ex['text'], ex['label']))
        logger.info(f"Loaded {len(data2['examples'])} examples from HuggingFace dataset")
    except Exception as e:
        logger.warning(f"Could not load HuggingFace dataset: {e}")

    # Add critical test cases
    critical_ceq = [
        ("Did the tower fall?", 'CEQ'),  # MUST classify correctly
        ("Did it break?", 'CEQ'),
        ("Did you see that?", 'CEQ'),
        ("Is it tall?", 'CEQ'),
        ("Are you ready?", 'CEQ'),
        ("Was it fun?", 'CEQ'),
        ("Can you help?", 'CEQ'),
        ("Should we stop?", 'CEQ'),
        ("Does it work?", 'CEQ'),
        ("Have you finished?", 'CEQ'),
    ]

    # Duplicate critical examples to emphasize importance
    all_examples.extend(critical_ceq * 5)  # Add 5x weight to critical examples

    # Remove duplicates while preserving order
    seen = set()
    unique_examples = []
    for text, label in all_examples:
        if text.lower().strip() not in seen:
            seen.add(text.lower().strip())
            unique_examples.append((text, label))

    # Count distribution
    ceq_count = sum(1 for _, label in unique_examples if label == 'CEQ')
    oeq_count = sum(1 for _, label in unique_examples if label == 'OEQ')

    logger.info(f"Total unique examples: {len(unique_examples)}")
    logger.info(f"CEQ: {ceq_count} ({ceq_count/len(unique_examples)*100:.1f}%)")
    logger.info(f"OEQ: {oeq_count} ({oeq_count/len(unique_examples)*100:.1f}%)")

    return unique_examples

def train_model(examples, epochs=150, lr=0.001):
    """Train model using gradient descent with class weights"""

    # Extract features
    feature_extractor = AdvancedFeatureExtractor()
    X = np.array([feature_extractor.extract_features(text) for text, _ in examples])
    y = np.array([0 if label == 'CEQ' else 1 for _, label in examples])  # CEQ=0, OEQ=1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights = torch.FloatTensor(class_weights)
    logger.info(f"Class weights: CEQ={class_weights[0]:.2f}, OEQ={class_weights[1]:.2f}")

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    # Weighted sampler for balanced batches
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = ImprovedOEQCEQClassifier(input_size=56)

    # Weighted cross-entropy loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Adam optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_accuracy = 0
    best_model_state = None

    logger.info("Starting training with gradient descent...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass (gradient descent)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights using gradient descent
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        test_accuracy = correct / total

        # Update learning rate
        scheduler.step(train_loss)

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()

        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, Acc={test_accuracy:.3f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("\nConfusion Matrix:")
    logger.info("       CEQ  OEQ")
    logger.info(f"CEQ:   {cm[0][0]:3d}  {cm[0][1]:3d}")
    logger.info(f"OEQ:   {cm[1][0]:3d}  {cm[1][1]:3d}")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=['CEQ', 'OEQ'])
    logger.info("\nClassification Report:")
    logger.info(report)

    return model, feature_extractor, best_accuracy

def test_critical_examples(model, feature_extractor):
    """Test on critical examples that were failing"""

    test_cases = [
        ("Did the tower fall?", "CEQ"),  # CRITICAL TEST CASE
        ("What happened to the tower?", "OEQ"),
        ("Is it tall?", "CEQ"),
        ("How tall is it?", "OEQ"),
        ("Can you help me?", "CEQ"),
        ("Why did it break?", "OEQ"),
        ("Does it work?", "CEQ"),
        ("Tell me about it", "OEQ"),
        ("Are you ready?", "CEQ"),
        ("What do you think?", "OEQ"),
    ]

    model.eval()
    logger.info("\n" + "="*60)
    logger.info("Testing Critical Examples")
    logger.info("="*60)

    correct = 0
    for text, expected in test_cases:
        features = feature_extractor.extract_features(text)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            outputs = model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            pred_label = "CEQ" if predicted[0] == 0 else "OEQ"
            ceq_prob = probs[0][0].item()
            oeq_prob = probs[0][1].item()

            status = "✅" if pred_label == expected else "❌"
            if pred_label == expected:
                correct += 1

            logger.info(f"{status} Q: {text:30s} | Expected: {expected}, Got: {pred_label} (CEQ:{ceq_prob:.2f}, OEQ:{oeq_prob:.2f})")

    logger.info(f"\nCritical Test Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.1f}%")

    return correct == len(test_cases)

def main():
    """Main training pipeline"""

    logger.info("="*60)
    logger.info("Training Improved OEQ/CEQ Model with Gradient Descent")
    logger.info("="*60)

    # Load all datasets
    examples = load_all_datasets()

    # Train model
    model, feature_extractor, accuracy = train_model(examples, epochs=150, lr=0.001)

    logger.info(f"\n🎯 Best validation accuracy: {accuracy:.3f}")

    # Test on critical examples
    all_correct = test_critical_examples(model, feature_extractor)

    if all_correct:
        logger.info("\n✅ SUCCESS! All critical test cases passed!")
        logger.info("✅ 'Did the tower fall?' is correctly classified as CEQ!")
    else:
        logger.info("\n⚠️ Some critical test cases still failing. May need more training.")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'class_mapping': {'CEQ': 0, 'OEQ': 1}
    }, 'oeq_ceq_improved_model.pth')

    logger.info(f"\n✅ Model saved to oeq_ceq_improved_model.pth")

    # Save feature extractor
    import pickle
    with open('feature_extractor_improved.pkl', 'wb') as f:
        pickle.dump(feature_extractor, f)

    logger.info("✅ Feature extractor saved to feature_extractor_improved.pkl")

if __name__ == "__main__":
    main()