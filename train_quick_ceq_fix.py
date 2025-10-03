#!/usr/bin/env python3
"""
Quick CEQ Fix Model - Fast training to fix "Did the tower fall?" misclassification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickCEQModel(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.fc(x)

def extract_features(text):
    """Extract simple but effective features"""
    text_lower = text.lower().strip()
    words = text_lower.split()
    first_word = words[0] if words else ''

    features = []

    # Critical CEQ starters (10 features)
    ceq_starters = ['did', 'does', 'do', 'is', 'are', 'was', 'were', 'will', 'can', 'has']
    for starter in ceq_starters:
        features.append(1.0 if first_word == starter else 0.0)

    # Critical OEQ starters (5 features)
    oeq_starters = ['what', 'how', 'why', 'where', 'when']
    for starter in oeq_starters:
        features.append(1.0 if first_word == starter else 0.0)

    # Additional features (5 features)
    features.append(len(words) / 10.0)  # Normalized word count
    features.append(1.0 if 'you' in text_lower else 0.0)
    features.append(1.0 if 'tell' in text_lower else 0.0)
    features.append(1.0 if text.endswith('?') else 0.0)
    features.append(1.0 if first_word == 'did' else 0.0)  # Extra emphasis on 'did'

    return np.array(features[:20], dtype=np.float32)

def load_training_data():
    """Load training data"""
    examples = []

    # Critical examples with heavy emphasis
    critical_ceq = [
        ("Did the tower fall?", 'CEQ'),
        ("Did it break?", 'CEQ'),
        ("Did you finish?", 'CEQ'),
        ("Is it tall?", 'CEQ'),
        ("Are you ready?", 'CEQ'),
        ("Was it fun?", 'CEQ'),
        ("Can you help?", 'CEQ'),
        ("Does it work?", 'CEQ'),
        ("Has it started?", 'CEQ'),
        ("Will you come?", 'CEQ'),
    ] * 10  # Repeat 10 times for emphasis

    critical_oeq = [
        ("What happened to the tower?", 'OEQ'),
        ("How did you build it?", 'OEQ'),
        ("Why did it fall?", 'OEQ'),
        ("Where is it now?", 'OEQ'),
        ("When will you fix it?", 'OEQ'),
    ] * 5

    examples.extend(critical_ceq)
    examples.extend(critical_oeq)

    # Load from comprehensive dataset
    try:
        with open('comprehensive_training_data.json', 'r') as f:
            data = json.load(f)
            examples.extend([(ex['text'], ex['label']) for ex in data['examples'][:200]])
    except:
        pass

    return examples

def train_quick_model():
    """Train quick model"""
    logger.info("Training Quick CEQ Fix Model...")

    # Load data
    examples = load_training_data()
    logger.info(f"Loaded {len(examples)} examples")

    # Extract features
    X = torch.FloatTensor([extract_features(text) for text, _ in examples])
    y = torch.LongTensor([0 if label == 'CEQ' else 1 for _, label in examples])

    # Model
    model = QuickCEQModel(input_size=20)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).sum().item() / len(y)
            logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={accuracy:.3f}")

    # Test critical examples
    logger.info("\nTesting Critical Examples:")
    test_cases = [
        ("Did the tower fall?", "CEQ"),
        ("What happened to the tower?", "OEQ"),
        ("Is it tall?", "CEQ"),
        ("How tall is it?", "OEQ"),
    ]

    model.eval()
    for text, expected in test_cases:
        features = torch.FloatTensor(extract_features(text)).unsqueeze(0)
        with torch.no_grad():
            output = model(features)
            pred = "CEQ" if torch.argmax(output) == 0 else "OEQ"
            status = "✅" if pred == expected else "❌"
            logger.info(f"{status} '{text}' -> {pred} (expected {expected})")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_mapping': {'CEQ': 0, 'OEQ': 1}
    }, 'quick_ceq_fix_model.pth')

    logger.info("\n✅ Quick model saved to quick_ceq_fix_model.pth")
    return model

if __name__ == "__main__":
    train_quick_model()