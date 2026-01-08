"""
Hybrid Feature Extractor - Combining Rule-Based and Semantic Features

This module implements a hybrid approach that combines:
- 19 hand-crafted linguistic features (domain knowledge)
- 384 semantic embedding features (transformer-based)
- Total: 403 features for enhanced OEQ/CEQ classification

Author: Principal SDE 1 (Cultivate ML Platform Team)
Issue: #299 - USER STORY 3.1: Hybrid Feature Extractor Implementation
Feature: #298 - Demo 4 Auto Feature Extractor Comparison
"""

import numpy as np
from typing import List, Literal, Optional, Dict, Any
from functools import lru_cache
import logging
import time

from .question_features import QuestionFeatureExtractor

logger = logging.getLogger(__name__)

# Lazy load sentence transformers to avoid startup overhead
_transformer_model = None
_transformer_load_time = None


def _get_transformer():
    """Lazy load the sentence transformer model."""
    global _transformer_model, _transformer_load_time

    if _transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            start_time = time.time()
            logger.info("Loading sentence transformer model: all-MiniLM-L6-v2")
            _transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            _transformer_load_time = time.time() - start_time
            logger.info(f"Transformer loaded in {_transformer_load_time:.2f}s")

        except ImportError as e:
            logger.error(f"sentence-transformers not installed: {e}")
            raise ImportError(
                "sentence-transformers is required for semantic features. "
                "Install with: pip install sentence-transformers"
            )

    return _transformer_model


class HybridFeatureExtractor:
    """
    Hybrid feature extractor combining rule-based and transformer features.

    Modes:
        - 'rule_only': 19-dimensional hand-crafted features
        - 'semantic_only': 384-dimensional transformer embeddings
        - 'hybrid': 403-dimensional concatenated features (default)

    Thread-safe, lazy-loads transformer on first use.

    Example:
        >>> extractor = HybridFeatureExtractor(mode='hybrid')
        >>> features = extractor.extract("What happened to your tower?")
        >>> print(features.shape)  # (403,)
    """

    # Feature dimensions
    RULE_DIM = 19
    SEMANTIC_DIM = 384
    HYBRID_DIM = RULE_DIM + SEMANTIC_DIM  # 403

    def __init__(
        self,
        mode: Literal['rule_only', 'semantic_only', 'hybrid'] = 'hybrid',
        device: Optional[str] = None,
        cache_embeddings: bool = True,
        normalize_semantic: bool = True
    ):
        """
        Initialize the hybrid feature extractor.

        Args:
            mode: Extraction mode - 'rule_only', 'semantic_only', or 'hybrid'
            device: Device for transformer ('cpu', 'cuda', or None for auto)
            cache_embeddings: Whether to cache semantic embeddings (LRU cache)
            normalize_semantic: Whether to L2-normalize semantic embeddings
        """
        self.mode = mode
        self.device = device
        self.cache_embeddings = cache_embeddings
        self.normalize_semantic = normalize_semantic

        # Initialize rule-based extractor
        self.rule_extractor = QuestionFeatureExtractor()

        # Lazy load transformer (only when needed)
        self._transformer = None
        self._warm = False

        # Feature name cache
        self._feature_names = None

        logger.info(f"HybridFeatureExtractor initialized with mode='{mode}'")

    def _ensure_transformer(self):
        """Ensure transformer is loaded (lazy loading)."""
        if self._transformer is None and self.mode in ('semantic_only', 'hybrid'):
            self._transformer = _get_transformer()
            if self.device:
                self._transformer = self._transformer.to(self.device)
            self._warm = True

    def extract(self, text: str) -> np.ndarray:
        """
        Extract features from a single text string.

        Args:
            text: The question text to analyze

        Returns:
            numpy array of shape (feature_dim,) where feature_dim depends on mode:
                - 'rule_only': 19
                - 'semantic_only': 384
                - 'hybrid': 403
        """
        if not text or not text.strip():
            return self._empty_features()

        start_time = time.time()

        if self.mode == 'rule_only':
            features = self.rule_extractor.extract(text)

        elif self.mode == 'semantic_only':
            features = self._extract_semantic(text)

        else:  # hybrid
            rule_features = self.rule_extractor.extract(text)
            semantic_features = self._extract_semantic(text)
            features = np.concatenate([rule_features, semantic_features])

        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Extracted {len(features)} features in {latency_ms:.1f}ms")

        return features

    def _extract_semantic(self, text: str) -> np.ndarray:
        """Extract semantic embedding from text."""
        self._ensure_transformer()

        if self.cache_embeddings:
            embedding = self._cached_encode(text)
        else:
            embedding = self._transformer.encode(text, convert_to_numpy=True)

        if self.normalize_semantic:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.astype(np.float32)

    @lru_cache(maxsize=1000)
    def _cached_encode(self, text: str) -> np.ndarray:
        """Cached encoding with LRU cache."""
        return self._transformer.encode(text, convert_to_numpy=True)

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Batch extraction with optimal batching for transformer.

        Args:
            texts: List of question texts to analyze
            batch_size: Batch size for transformer encoding
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_texts, feature_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.get_feature_dim())

        start_time = time.time()
        n_texts = len(texts)

        if self.mode == 'rule_only':
            features = self.rule_extractor.extract_batch(texts)

        elif self.mode == 'semantic_only':
            features = self._batch_semantic(texts, batch_size, show_progress)

        else:  # hybrid
            rule_features = self.rule_extractor.extract_batch(texts)
            semantic_features = self._batch_semantic(texts, batch_size, show_progress)
            features = np.hstack([rule_features, semantic_features])

        latency_ms = (time.time() - start_time) * 1000
        per_text_ms = latency_ms / n_texts if n_texts > 0 else 0
        logger.info(
            f"Batch extracted {n_texts} texts in {latency_ms:.1f}ms "
            f"({per_text_ms:.1f}ms/text)"
        )

        return features

    def _batch_semantic(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool
    ) -> np.ndarray:
        """Batch semantic extraction."""
        self._ensure_transformer()

        embeddings = self._transformer.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        if self.normalize_semantic:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def get_feature_dim(self) -> int:
        """Get the dimension of the feature vector for current mode."""
        if self.mode == 'rule_only':
            return self.RULE_DIM
        elif self.mode == 'semantic_only':
            return self.SEMANTIC_DIM
        else:
            return self.HYBRID_DIM

    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names.

        Returns:
            List of feature names matching the feature vector dimensions
        """
        if self._feature_names is None:
            names = []

            if self.mode in ('rule_only', 'hybrid'):
                names.extend(self.rule_extractor.get_feature_names())

            if self.mode in ('semantic_only', 'hybrid'):
                names.extend([f'semantic_{i}' for i in range(self.SEMANTIC_DIM)])

            self._feature_names = names

        return self._feature_names.copy()

    def get_mode(self) -> str:
        """Return current extraction mode."""
        return self.mode

    def set_mode(self, mode: Literal['rule_only', 'semantic_only', 'hybrid']):
        """
        Change extraction mode.

        Args:
            mode: New extraction mode
        """
        self.mode = mode
        self._feature_names = None  # Reset cached names
        logger.info(f"Mode changed to '{mode}'")

    def warm_up(self) -> float:
        """
        Pre-load transformer model.

        Returns:
            Warm-up time in seconds
        """
        if self._warm:
            return 0.0

        start_time = time.time()
        self._ensure_transformer()

        # Run a dummy encoding to fully warm up
        _ = self._transformer.encode("warm up", convert_to_numpy=True)

        warm_up_time = time.time() - start_time
        logger.info(f"Warm-up completed in {warm_up_time:.2f}s")
        return warm_up_time

    def _empty_features(self) -> np.ndarray:
        """Return zero features for empty input."""
        return np.zeros(self.get_feature_dim(), dtype=np.float32)

    def explain_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features with explanations for interpretability.

        Args:
            text: The question text to analyze

        Returns:
            Dictionary with features, explanations, and insights
        """
        features = self.extract(text)

        explanation = {
            'text': text,
            'mode': self.mode,
            'feature_dim': len(features),
            'features': {}
        }

        if self.mode in ('rule_only', 'hybrid'):
            rule_features = features[:self.RULE_DIM]
            rule_explanation = self.rule_extractor.explain_features(text)
            explanation['rule_based'] = {
                'features': rule_explanation['features'],
                'oeq_indicators': rule_explanation['oeq_indicators'],
                'ceq_indicators': rule_explanation['ceq_indicators'],
                'interpretation': rule_explanation['interpretation']
            }

        if self.mode in ('semantic_only', 'hybrid'):
            semantic_start = 0 if self.mode == 'semantic_only' else self.RULE_DIM
            semantic_features = features[semantic_start:semantic_start + self.SEMANTIC_DIM]

            # Get similar training examples (if available)
            explanation['semantic'] = {
                'embedding_norm': float(np.linalg.norm(semantic_features)),
                'embedding_mean': float(np.mean(semantic_features)),
                'embedding_std': float(np.std(semantic_features)),
                'top_dimensions': self._get_top_dimensions(semantic_features, k=5)
            }

        return explanation

    def _get_top_dimensions(self, embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Get top k dimensions by absolute magnitude."""
        abs_values = np.abs(embedding)
        top_indices = np.argsort(abs_values)[-k:][::-1]

        return [
            {'dim': int(i), 'value': float(embedding[i])}
            for i in top_indices
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics and configuration."""
        stats = {
            'mode': self.mode,
            'feature_dim': self.get_feature_dim(),
            'rule_dim': self.RULE_DIM,
            'semantic_dim': self.SEMANTIC_DIM,
            'transformer_loaded': self._transformer is not None,
            'cache_enabled': self.cache_embeddings,
            'normalize_semantic': self.normalize_semantic
        }

        if self._transformer is not None and hasattr(self._cached_encode, 'cache_info'):
            cache_info = self._cached_encode.cache_info()
            stats['cache_hits'] = cache_info.hits
            stats['cache_misses'] = cache_info.misses
            stats['cache_size'] = cache_info.currsize

        return stats


# Singleton instances for easy import
_hybrid_extractor = None
_rule_extractor = None
_semantic_extractor = None


def get_hybrid_extractor() -> HybridFeatureExtractor:
    """Get singleton hybrid extractor instance."""
    global _hybrid_extractor
    if _hybrid_extractor is None:
        _hybrid_extractor = HybridFeatureExtractor(mode='hybrid')
    return _hybrid_extractor


def get_rule_extractor() -> HybridFeatureExtractor:
    """Get singleton rule-only extractor instance."""
    global _rule_extractor
    if _rule_extractor is None:
        _rule_extractor = HybridFeatureExtractor(mode='rule_only')
    return _rule_extractor


def get_semantic_extractor() -> HybridFeatureExtractor:
    """Get singleton semantic-only extractor instance."""
    global _semantic_extractor
    if _semantic_extractor is None:
        _semantic_extractor = HybridFeatureExtractor(mode='semantic_only')
    return _semantic_extractor


if __name__ == "__main__":
    # Test the hybrid feature extractor
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("HYBRID FEATURE EXTRACTOR TEST")
    print("=" * 70)

    test_questions = [
        ("What do you think happened?", "OEQ"),
        ("Did you like it?", "CEQ"),
        ("How could you make it stronger?", "OEQ"),
        ("Is this correct?", "CEQ"),
        ("Why did it fall?", "OEQ"),
    ]

    # Test all three modes
    for mode in ['rule_only', 'semantic_only', 'hybrid']:
        print(f"\n--- Mode: {mode} ---")
        extractor = HybridFeatureExtractor(mode=mode)

        # Warm up for semantic modes
        if mode != 'rule_only':
            warm_up_time = extractor.warm_up()
            print(f"Warm-up time: {warm_up_time:.2f}s")

        for text, expected in test_questions:
            start = time.time()
            features = extractor.extract(text)
            latency_ms = (time.time() - start) * 1000

            print(f"\nText: '{text}'")
            print(f"  Expected: {expected}")
            print(f"  Features: {features.shape} ({len(features)} dims)")
            print(f"  Latency: {latency_ms:.1f}ms")

            if mode in ('rule_only', 'hybrid'):
                # Show rule-based indicators
                oeq_score = features[15] if mode == 'rule_only' else features[15]
                ceq_score = features[16] if mode == 'rule_only' else features[16]
                print(f"  OEQ Score: {oeq_score:.1f}, CEQ Score: {ceq_score:.1f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
