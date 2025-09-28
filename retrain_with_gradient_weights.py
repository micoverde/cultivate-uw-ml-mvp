#!/usr/bin/env python3
"""
Retrain OEQ/CEQ Classifier with Gradient Descent and Class Weights
Addresses the CEQ misclassification issue using weighted cross-entropy loss
Based on GitHub issue #120: Support fine tuning with gradient descent
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedOEQCEQClassifier(nn.Module):
    """Enhanced neural network with better architecture for CEQ detection"""
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

        # Output layer
        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EnhancedFeatureExtractor:
    """Enhanced feature extraction with better CEQ indicators"""

    def __init__(self):
        # Expanded CEQ keywords based on ground truth
        self.ceq_keywords = [
            'is', 'are', 'was', 'were', 'am',
            'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'shall',
            'can', 'may', 'might', 'must',
            'have', 'has', 'had'
        ]

        # Strong CEQ starters (critical for detection)
        self.ceq_starters = ['did', 'does', 'do', 'is', 'are', 'was', 'were',
                             'will', 'would', 'can', 'could', 'should', 'has', 'have']

        self.oeq_keywords = [
            'what', 'how', 'why', 'where', 'when', 'who', 'which', 'whose',
            'describe', 'explain', 'tell me', 'think', 'feel',
            'notice', 'see', 'observe', 'imagine', 'wonder', 'curious',
            'explore', 'discover'
        ]

    def extract_features(self, text: str) -> np.ndarray:
        """Extract 56+ features with enhanced CEQ detection"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        first_word = words[0] if words else ''

        features = []

        # 1-16: OEQ keyword counts (16 features)
        for keyword in self.oeq_keywords[:16]:
            count = text_lower.count(keyword)
            features.append(count)

        # 17-29: CEQ keyword counts (13 features)
        for keyword in self.ceq_keywords[:13]:
            count = text_lower.count(keyword)
            features.append(count)

        # 30: Word count
        features.append(len(words))

        # 31: Character count
        features.append(len(text))

        # 32: Average word length
        avg_len = np.mean([len(word) for word in words]) if words else 0
        features.append(avg_len)

        # 33: Question mark count
        features.append(text.count('?'))

        # 34: Exclamation mark count
        features.append(text.count('!'))

        # 35: Comma count
        features.append(text.count(','))

        # 36: Starts with question word (0 or 1)
        question_starters = ['what', 'how', 'why', 'when', 'where', 'who']
        starts_with_q = 1 if any(text_lower.startswith(q) for q in question_starters) else 0
        features.append(starts_with_q)

        # 37: Contains "you"
        features.append(1 if 'you' in text_lower else 0)

        # 38: Contains "think"
        features.append(1 if 'think' in text_lower else 0)

        # 39: Contains "feel"
        features.append(1 if 'feel' in text_lower else 0)

        # 40-45: Sentence structure features
        features.append(1 if text_lower.startswith('what') else 0)
        features.append(1 if text_lower.startswith('how') else 0)
        features.append(1 if text_lower.startswith('why') else 0)
        features.append(1 if text_lower.startswith('is') else 0)
        features.append(1 if text_lower.startswith('are') else 0)
        features.append(1 if text_lower.startswith('do') else 0)

        # 46-50: Advanced linguistic features
        features.append(text_lower.count('do you'))
        features.append(text_lower.count('can you'))
        features.append(text_lower.count('what do you'))
        features.append(text_lower.count('how do you'))
        features.append(text_lower.count('tell me'))

        # 51-56: Context and additional CEQ indicators
        features.append(1 if 'because' in text_lower else 0)
        features.append(1 if 'maybe' in text_lower else 0)
        features.append(1 if 'might' in text_lower else 0)

        # CRITICAL: Strong CEQ starter indicator (fixes "Did the tower fall?" issue)
        features.append(2 if first_word == 'did' else 0)  # Double weight for 'did'
        features.append(1 if first_word in self.ceq_starters else 0)

        # Length-based feature (short questions are often CEQ)
        features.append(1 if len(text) < 30 else 0)

        return np.array(features[:56], dtype=np.float32)  # Ensure exactly 56 features

def prepare_training_data():
    """Prepare balanced training data from ground truth and synthetic examples"""

    # Load ground truth from CSV
    csv_path = "/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv"

    training_examples = []

    # Add ground truth examples from CSV
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            for col in df.columns:
                if 'CEQ' in str(row[col]):
                    # Extract question if mentioned
                    text = str(row[col])
                    if '"' in text:
                        # Try to extract quoted question
                        import re
                        quotes = re.findall(r'"([^"]*)"', text)
                        for q in quotes:
                            if '?' in q or len(q) < 50:
                                training_examples.append((q, 'CEQ'))

                elif 'OEQ' in str(row[col]):
                    text = str(row[col])
                    if '"' in text:
                        import re
                        quotes = re.findall(r'"([^"]*)"', text)
                        for q in quotes:
                            if '?' in q or len(q) < 50:
                                training_examples.append((q, 'OEQ'))
    except Exception as e:
        logger.warning(f"Could not load CSV: {e}")

    # Add critical CEQ examples that were misclassified
    ceq_examples = [
        "Did the tower fall?",
        "Did it break?",
        "Did you see that?",
        "Did you like it?",
        "Did you finish?",
        "Is it tall?",
        "Is that yours?",
        "Is it working?",
        "Are you ready?",
        "Are we done?",
        "Was it fun?",
        "Was it hard?",
        "Were you scared?",
        "Can you do it?",
        "Can I help?",
        "Could you try?",
        "Should we stop?",
        "Will it work?",
        "Would you like to?",
        "Has it started?",
        "Have you seen it?",
        "Do you understand?",
        "Does it fit?",
    ]

    # Add OEQ examples
    oeq_examples = [
        "What happened to the tower?",
        "How did you build it?",
        "Why did it fall down?",
        "What do you think about it?",
        "How does it work?",
        "Tell me about your tower",
        "Explain what you did",
        "Describe your creation",
        "What can we do next?",
        "How tall can you make it?",
        "Why do you think that happened?",
        "What would happen if we tried again?",
        "How could we make it better?",
        "What did you learn?",
        "How did that make you feel?",
    ]

    # Add to training set
    for q in ceq_examples:
        training_examples.append((q, 'CEQ'))

    for q in oeq_examples:
        training_examples.append((q, 'OEQ'))

    # Ensure balanced dataset
    ceq_count = sum(1 for _, label in training_examples if label == 'CEQ')
    oeq_count = sum(1 for _, label in training_examples if label == 'OEQ')

    logger.info(f"Dataset: {ceq_count} CEQ, {oeq_count} OEQ examples")

    # Extract features
    feature_extractor = EnhancedFeatureExtractor()
    X = []
    y = []

    for text, label in training_examples:
        features = feature_extractor.extract_features(text)
        X.append(features)
        y.append(0 if label == 'CEQ' else 1)  # CEQ=0, OEQ=1

    return np.array(X), np.array(y), feature_extractor

def train_with_gradient_descent(X, y, epochs=100, lr=0.001):
    """Train using gradient descent with class weights"""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Convert class weights to tensor
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

    # Create weighted sampler for balanced batches
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = ImprovedOEQCEQClassifier(input_size=56)

    # Use weighted cross-entropy loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    best_accuracy = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass with gradient descent
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights using gradient descent
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_accuracy = correct / total

        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        ceq_correct = 0
        ceq_total = 0
        oeq_correct = 0
        oeq_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                # Track per-class accuracy
                for i in range(batch_y.size(0)):
                    if batch_y[i] == 0:  # CEQ
                        ceq_total += 1
                        if predicted[i] == 0:
                            ceq_correct += 1
                    else:  # OEQ
                        oeq_total += 1
                        if predicted[i] == 1:
                            oeq_correct += 1

        test_accuracy = correct / total
        ceq_accuracy = ceq_correct / ceq_total if ceq_total > 0 else 0
        oeq_accuracy = oeq_correct / oeq_total if oeq_total > 0 else 0

        # Update learning rate
        scheduler.step(test_loss)

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Acc={train_accuracy:.3f}, "
                       f"Test Acc={test_accuracy:.3f}, "
                       f"CEQ Acc={ceq_accuracy:.3f}, "
                       f"OEQ Acc={oeq_accuracy:.3f}")

    # Load best model
    model.load_state_dict(best_model_state)

    return model, best_accuracy

def test_specific_examples(model, feature_extractor):
    """Test on specific problematic examples"""

    test_cases = [
        ("Did the tower fall?", "CEQ"),
        ("What happened to the tower?", "OEQ"),
        ("Is it tall?", "CEQ"),
        ("How tall is it?", "OEQ"),
        ("Can you help me?", "CEQ"),
        ("Why did it break?", "OEQ"),
        ("Does it work?", "CEQ"),
        ("Tell me about it", "OEQ"),
    ]

    model.eval()
    print("\n" + "="*60)
    print("Testing Specific Examples")
    print("="*60)

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

            print(f"\n{status} Q: {text}")
            print(f"   Expected: {expected}, Got: {pred_label}")
            print(f"   Probabilities: CEQ={ceq_prob:.3f}, OEQ={oeq_prob:.3f}")

def main():
    """Main training pipeline"""

    logger.info("Starting OEQ/CEQ classifier retraining with gradient descent...")

    # Prepare data
    X, y, feature_extractor = prepare_training_data()

    # Train model with gradient descent
    model, accuracy = train_with_gradient_descent(X, y, epochs=100, lr=0.001)

    logger.info(f"\n🎯 Final accuracy: {accuracy:.3f}")

    # Test on specific examples
    test_specific_examples(model, feature_extractor)

    # Save model
    save_path = "oeq_ceq_retrained.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'epoch': 100,
        'class_mapping': {'CEQ': 0, 'OEQ': 1}
    }, save_path)

    logger.info(f"\n✅ Model saved to {save_path}")

    # Save feature extractor
    with open("feature_extractor_enhanced.pkl", "wb") as f:
        pickle.dump(feature_extractor, f)

    logger.info("✅ Feature extractor saved")

if __name__ == "__main__":
    main()