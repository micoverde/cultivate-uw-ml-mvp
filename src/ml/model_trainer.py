"""
Model trainer wrapper for saving/loading ML models.

This module provides a simple wrapper class for saving and loading
trained ML models in a format compatible with the API.
"""

class ModelTrainer:
    """Simple wrapper for model saving compatibility."""
    def __init__(self, models, scaler, model_type, ensemble=None):
        self.models = models
        self.scaler = scaler
        self.model_type = model_type
        self.ensemble = ensemble
