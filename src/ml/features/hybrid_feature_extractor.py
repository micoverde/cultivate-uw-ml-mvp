"""
Hybrid Feature Extractor for Demo 4: Feature Extraction Comparison

Supports three extraction modes:
1. rule_only: 19 hand-crafted linguistic features (fast, interpretable)
2. semantic_only: 384-dimensional sentence embeddings (deep learning)
3. hybrid: Combined 403 features (19 rule + 384 semantic)

Author: Claude (Partner-Level Microsoft SDE)
Feature: #298 - Demo 4 Auto Feature Extractor Comparison
"""

import time
import numpy as np
from typing import List, Dict, Any, Literal, Optional
from functools import lru_cache

from .question_features import QuestionFeatureExtractor


class HybridFeatureExtractor:
    """
    Feature extraction with three modes for Demo 4 comparison.

    Modes:
        - rule_only: 19 hand-crafted features (2ms latency)
        - semantic_only: 384 sentence transformer features (~40ms latency)
        - hybrid: 403 combined features (~45ms latency)
    """

    RULE_DIM = 19
    SEMANTIC_DIM = 384
    HYBRID_DIM = RULE_DIM + SEMANTIC_DIM  # 403

    def __init__(
        self,
        mode: Literal['rule_only', 'semantic_only', 'hybrid'] = 'hybrid',
        cache_embeddings: bool = True,
        normalize_semantic: bool = True
    ):
        """
        Initialize the hybrid feature extractor.

        Args:
            mode: Extraction mode ('rule_only', 'semantic_only', 'hybrid')
            cache_embeddings: Whether to cache semantic embeddings
            normalize_semantic: Whether to L2-normalize semantic embeddings
        """
        self.mode = mode
        self.cache_embeddings = cache_embeddings
        self.normalize_semantic = normalize_semantic

        # Rule-based extractor (always available)
        self.rule_extractor = QuestionFeatureExtractor()

        # Lazy-loaded semantic model
        self._semantic_model = None
        self._model_loaded = False

    @property
    def output_dim(self) -> int:
        """Get output feature dimension based on mode."""
        if self.mode == 'rule_only':
            return self.RULE_DIM
        elif self.mode == 'semantic_only':
            return self.SEMANTIC_DIM
        else:
            return self.HYBRID_DIM

    def _load_semantic_model(self):
        """Lazy load the sentence transformer model."""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._model_loaded = True
        except ImportError:
            raise ImportError(
                "sentence-transformers required for semantic mode. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer: {e}")

    def extract_rule_features(self, text: str) -> np.ndarray:
        """Extract 19 rule-based features."""
        return self.rule_extractor.extract(text)

    def extract_semantic_features(self, text: str) -> np.ndarray:
        """Extract 384-dimensional semantic embedding."""
        self._load_semantic_model()

        embedding = self._semantic_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_semantic
        )
        return embedding.astype(np.float32)

    @lru_cache(maxsize=1000)
    def _cached_semantic(self, text: str) -> tuple:
        """Cached semantic embedding (returns tuple for hashability)."""
        return tuple(self.extract_semantic_features(text))

    def extract(self, text: str) -> np.ndarray:
        """
        Extract features based on current mode.

        Args:
            text: Question text to analyze

        Returns:
            Feature vector of shape (output_dim,)
        """
        if self.mode == 'rule_only':
            return self.extract_rule_features(text)

        elif self.mode == 'semantic_only':
            if self.cache_embeddings:
                return np.array(self._cached_semantic(text), dtype=np.float32)
            return self.extract_semantic_features(text)

        else:  # hybrid
            rule_features = self.extract_rule_features(text)

            if self.cache_embeddings:
                semantic_features = np.array(
                    self._cached_semantic(text), dtype=np.float32
                )
            else:
                semantic_features = self.extract_semantic_features(text)

            return np.concatenate([rule_features, semantic_features])

    def extract_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract features for multiple texts.

        Args:
            texts: List of question texts
            batch_size: Batch size for semantic model

        Returns:
            Feature matrix of shape (n_samples, output_dim)
        """
        if self.mode == 'rule_only':
            return self.rule_extractor.extract_batch(texts)

        elif self.mode == 'semantic_only':
            self._load_semantic_model()
            embeddings = self._semantic_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_semantic
            )
            return embeddings.astype(np.float32)

        else:  # hybrid
            rule_features = self.rule_extractor.extract_batch(texts)

            self._load_semantic_model()
            semantic_features = self._semantic_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_semantic
            )

            return np.concatenate(
                [rule_features, semantic_features.astype(np.float32)],
                axis=1
            )

    def extract_with_timing(self, text: str) -> Dict[str, Any]:
        """
        Extract features with timing information for Demo 4.

        Returns dict with features and latency.
        """
        start = time.perf_counter()
        features = self.extract(text)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            'features': features,
            'latency_ms': latency_ms,
            'mode': self.mode,
            'dim': len(features)
        }

    def compare_all_modes(self, text: str) -> Dict[str, Any]:
        """
        Compare all three extraction modes for Demo 4 UI.

        Returns comparison data for rule-based, semantic, and hybrid.
        """
        results = {}

        # Rule-based extraction
        start = time.perf_counter()
        rule_features = self.extract_rule_features(text)
        rule_latency = (time.perf_counter() - start) * 1000

        results['rule_based'] = {
            'features': rule_features.tolist(),
            'feature_count': len(rule_features),
            'latency_ms': round(rule_latency, 2),
            'feature_names': self.rule_extractor.get_feature_names()
        }

        # Semantic extraction (only if model available)
        try:
            start = time.perf_counter()
            semantic_features = self.extract_semantic_features(text)
            semantic_latency = (time.perf_counter() - start) * 1000

            results['semantic'] = {
                'features': semantic_features.tolist()[:10],  # First 10 for display
                'feature_count': len(semantic_features),
                'latency_ms': round(semantic_latency, 2),
                'model': 'all-MiniLM-L6-v2'
            }

            # Hybrid (combine both)
            hybrid_latency = rule_latency + semantic_latency
            results['hybrid'] = {
                'feature_count': len(rule_features) + len(semantic_features),
                'latency_ms': round(hybrid_latency, 2),
                'composition': f'{len(rule_features)} rule + {len(semantic_features)} semantic'
            }

        except (ImportError, RuntimeError) as e:
            # Semantic model not available
            results['semantic'] = {
                'error': str(e),
                'feature_count': 384,
                'latency_ms': None,
                'available': False
            }
            results['hybrid'] = {
                'error': 'Semantic model not available',
                'feature_count': 403,
                'latency_ms': None,
                'available': False
            }

        return results

    def warm_up(self) -> float:
        """
        Warm up the semantic model (load into memory).

        Returns warm-up time in seconds.
        """
        if self.mode == 'rule_only':
            return 0.0

        start = time.perf_counter()
        self._load_semantic_model()
        # Run a dummy encoding to fully warm up
        _ = self._semantic_model.encode("warm up", convert_to_numpy=True)
        return time.perf_counter() - start

    def get_feature_explanation(self, text: str) -> Dict[str, Any]:
        """Get detailed feature explanation for a question."""
        return self.rule_extractor.explain_features(text)


# Factory function for creating extractors with specific modes
def create_extractor(mode: str = 'hybrid') -> HybridFeatureExtractor:
    """Create a HybridFeatureExtractor with the specified mode."""
    return HybridFeatureExtractor(mode=mode)


if __name__ == "__main__":
    # Test the hybrid feature extractor
    print("=" * 60)
    print("HYBRID FEATURE EXTRACTOR TEST")
    print("=" * 60)

    test_questions = [
        "What do you think happened to your tower?",
        "Did your tower fall down?",
        "How could you make it stronger?",
        "Is this correct?",
    ]

    # Test rule-only mode
    print("\n--- Rule-Only Mode (19 features) ---")
    extractor = HybridFeatureExtractor(mode='rule_only')
    for q in test_questions:
        result = extractor.extract_with_timing(q)
        print(f"Q: {q}")
        print(f"   Dim: {result['dim']}, Latency: {result['latency_ms']:.2f}ms")

    # Test semantic mode (if available)
    print("\n--- Semantic Mode (384 features) ---")
    try:
        extractor = HybridFeatureExtractor(mode='semantic_only')
        extractor.warm_up()
        for q in test_questions:
            result = extractor.extract_with_timing(q)
            print(f"Q: {q}")
            print(f"   Dim: {result['dim']}, Latency: {result['latency_ms']:.2f}ms")
    except ImportError as e:
        print(f"Semantic mode not available: {e}")

    # Test hybrid mode
    print("\n--- Hybrid Mode (403 features) ---")
    try:
        extractor = HybridFeatureExtractor(mode='hybrid')
        for q in test_questions:
            result = extractor.extract_with_timing(q)
            print(f"Q: {q}")
            print(f"   Dim: {result['dim']}, Latency: {result['latency_ms']:.2f}ms")
    except ImportError as e:
        print(f"Hybrid mode not available: {e}")

    # Compare all modes
    print("\n--- Mode Comparison ---")
    extractor = HybridFeatureExtractor(mode='hybrid')
    comparison = extractor.compare_all_modes(test_questions[0])
    print(f"Question: {test_questions[0]}")
    for mode, data in comparison.items():
        print(f"  {mode}: {data.get('feature_count')} features, {data.get('latency_ms')}ms")
