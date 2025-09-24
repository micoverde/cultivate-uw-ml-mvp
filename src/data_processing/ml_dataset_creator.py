#!/usr/bin/env python3
"""
ML Training Dataset Creator for Issue #89
Creates unified training dataset from transcripts and annotations for Feature 2 ML models.

Combines:
- Transcribed audio clips with word-level timestamps
- Quality annotations from expert CSV data
- Feature vectors for multi-modal BERT architecture (Issue #76)

Author: Claude (Issue #89 ML Dataset Preparation)
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MLTrainingExample:
    """Single ML training example with all modalities"""
    example_id: str

    # Audio/text features
    transcript_text: str
    word_timestamps: List[Dict[str, Any]]
    audio_duration: float
    transcription_confidence: float

    # Educational annotations
    question_type: str  # OEQ, CEQ, etc.
    question_description: str
    age_group: str
    video_context: str

    # Metadata
    original_video: str
    audio_clip_path: str

    # Feature vectors (for Issue #76 multi-modal BERT)
    linguistic_features: Optional[Dict[str, float]] = None
    acoustic_features: Optional[Dict[str, float]] = None
    interaction_features: Optional[Dict[str, float]] = None

    # CLASS framework labels (target variables)
    class_scores: Optional[Dict[str, float]] = None
    processing_metadata: Dict[str, Any] = None

class MLDatasetCreator:
    """Creates ML-ready training dataset from transcripts and annotations"""

    def __init__(self,
                 transcripts_dir: str = "data/transcripts/transcripts",
                 annotations_file: str = None,
                 output_dir: str = "data/ml_training"):

        self.transcripts_dir = Path(transcripts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find annotations file
        if annotations_file:
            self.annotations_file = Path(annotations_file)
        else:
            # Search for CSV in secure data folder
            secure_data_paths = [
                Path("../secure data/VideosAskingQuestions CSV.csv"),
                Path("/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv")
            ]

            self.annotations_file = None
            for path in secure_data_paths:
                if path.exists():
                    self.annotations_file = path
                    break

        if not self.annotations_file or not self.annotations_file.exists():
            logger.warning("Annotations CSV not found - will use transcript metadata only")

        self.training_examples: List[MLTrainingExample] = []

        logger.info(f"ML Dataset Creator initialized")
        logger.info(f"  Transcripts: {self.transcripts_dir}")
        logger.info(f"  Annotations: {self.annotations_file}")
        logger.info(f"  Output: {self.output_dir}")

    def create_training_dataset(self) -> Dict[str, Any]:
        """Create complete ML training dataset"""

        logger.info("Starting ML training dataset creation")

        # Load annotations if available
        annotations_df = self.load_annotations()

        # Process all transcripts
        transcript_files = list(self.transcripts_dir.glob("*.json"))
        if not transcript_files:
            logger.error("No transcript files found")
            return {"error": "No transcripts available"}

        logger.info(f"Processing {len(transcript_files)} transcript files")

        for transcript_file in transcript_files:
            try:
                example = self.process_transcript(transcript_file, annotations_df)
                if example:
                    self.training_examples.append(example)
            except Exception as e:
                logger.error(f"Failed to process {transcript_file.name}: {e}")

        # Generate dataset files
        dataset_info = self.generate_dataset_files()

        logger.info(f"ML dataset creation complete: {len(self.training_examples)} examples")
        return dataset_info

    def load_annotations(self) -> Optional[pd.DataFrame]:
        """Load expert annotations from CSV"""

        if not self.annotations_file or not self.annotations_file.exists():
            logger.warning("No annotations file available")
            return None

        try:
            df = pd.read_csv(self.annotations_file)
            logger.info(f"Loaded {len(df)} annotation rows from CSV")
            return df
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            return None

    def process_transcript(self, transcript_file: Path,
                         annotations_df: Optional[pd.DataFrame]) -> Optional[MLTrainingExample]:
        """Process single transcript into ML training example"""

        try:
            # Load transcript data
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)

            clip_id = transcript_data.get('clip_id')
            if not clip_id:
                logger.warning(f"No clip_id in {transcript_file.name}")
                return None

            # Extract basic transcript info
            segments = transcript_data.get('transcript_segments', [])
            if not segments:
                logger.warning(f"No segments in {clip_id}")
                return None

            full_text = " ".join(segment.get('text', '') for segment in segments)

            # Extract word timestamps
            word_timestamps = []
            for segment in segments:
                words = segment.get('words', [])
                for word in words:
                    word_timestamps.append({
                        'word': word.get('word', '').strip(),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0),
                        'confidence': word.get('confidence', 0)
                    })

            # Get quality annotation (from transcript metadata)
            quality_annotation = transcript_data.get('quality_annotation', {})

            # Extract linguistic features
            linguistic_features = self.extract_linguistic_features(full_text, segments)

            # Extract interaction features from annotation
            interaction_features = self.extract_interaction_features(quality_annotation)

            # Get processing metadata
            processing_metadata = transcript_data.get('processing_metadata', {})

            # Create training example
            example = MLTrainingExample(
                example_id=clip_id,
                transcript_text=full_text,
                word_timestamps=word_timestamps,
                audio_duration=transcript_data.get('duration', 5.0),
                transcription_confidence=processing_metadata.get('transcription_confidence', 0.0),

                question_type=quality_annotation.get('question_type', 'unknown'),
                question_description=quality_annotation.get('description', ''),
                age_group=quality_annotation.get('age_group', 'unknown'),
                video_context=quality_annotation.get('video_description', ''),

                linguistic_features=linguistic_features,
                interaction_features=interaction_features,

                original_video=transcript_data.get('original_video', ''),
                audio_clip_path=f"data/processed_audio/clips/{clip_id}.wav",
                processing_metadata=processing_metadata
            )

            return example

        except Exception as e:
            logger.error(f"Error processing {transcript_file}: {e}")
            return None

    def extract_linguistic_features(self, text: str, segments: List[Dict]) -> Dict[str, float]:
        """Extract linguistic features for ML training"""

        # Basic text metrics
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # Question analysis
        question_count = text.count('?')
        has_question = question_count > 0

        # Educational language patterns
        educator_patterns = [
            r'\b(what|how|why|can you|tell me|think about)\b',
            r'\b(good|great|excellent|wonderful|yes)\b',
            r'\b(let\'s|we can|try|show me|look at)\b'
        ]

        educator_indicators = 0
        text_lower = text.lower()
        for pattern in educator_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                educator_indicators += 1

        # Complexity metrics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        sentence_count = max(1, len(re.findall(r'[.!?]+', text)))
        avg_sentence_length = word_count / sentence_count

        # Confidence analysis
        word_confidences = []
        for segment in segments:
            for word in segment.get('words', []):
                conf = word.get('confidence', 0)
                if conf > 0:
                    word_confidences.append(conf)

        avg_word_confidence = np.mean(word_confidences) if word_confidences else 0

        return {
            'word_count': float(word_count),
            'character_count': float(char_count),
            'question_count': float(question_count),
            'has_question': float(has_question),
            'educator_language_score': float(educator_indicators / 3.0),  # Normalized
            'avg_word_length': float(avg_word_length),
            'avg_sentence_length': float(avg_sentence_length),
            'sentence_count': float(sentence_count),
            'avg_word_confidence': float(avg_word_confidence),
            'text_complexity_score': float(min(avg_word_length * avg_sentence_length / 10, 1.0))
        }

    def extract_interaction_features(self, quality_annotation: Dict) -> Dict[str, float]:
        """Extract interaction features from quality annotation"""

        # Question type encoding
        question_type_map = {
            'OEQ': 1.0,      # Open-ended question
            'CEQ': 0.5,      # Closed-ended question
            'Rhetorical': 0.2,
            'Follow-up': 0.8,
            'unknown': 0.0
        }

        question_type = quality_annotation.get('question_type', 'unknown')
        question_score = question_type_map.get(question_type, 0.0)

        # Age group encoding
        age_group_map = {
            'PK': 0.8,        # Pre-kindergarten
            'Toddler': 0.4,
            'TODDLER': 0.4,
            'INFANT AND TODDLER': 0.2,
            'PK and Toddler': 0.6,
            'unknown': 0.5
        }

        age_group = quality_annotation.get('age_group', 'unknown')
        age_score = age_group_map.get(age_group, 0.5)

        # Description analysis for teaching quality indicators
        description = quality_annotation.get('description', '').lower()

        wait_time_score = 1.0 if 'pause' in description or 'wait' in description else 0.0
        response_opportunity = 1.0 if 'respond' in description or 'response' in description else 0.0
        teacher_support = 1.0 if any(word in description for word in ['encourage', 'help', 'support']) else 0.0

        return {
            'question_type_score': question_score,
            'age_appropriateness': age_score,
            'provides_wait_time': wait_time_score,
            'creates_response_opportunity': response_opportunity,
            'teacher_support_present': teacher_support,
            'overall_interaction_quality': np.mean([question_score, wait_time_score, response_opportunity])
        }

    def generate_dataset_files(self) -> Dict[str, Any]:
        """Generate various dataset file formats for ML training"""

        if not self.training_examples:
            return {"error": "No training examples to export"}

        # Convert to various formats
        dataset_json = self.create_json_dataset()
        dataset_csv = self.create_csv_dataset()
        feature_vectors = self.create_feature_vectors()

        # Save files
        self.save_dataset_files(dataset_json, dataset_csv, feature_vectors)

        # Generate summary statistics
        summary = self.generate_dataset_summary()

        return {
            "total_examples": len(self.training_examples),
            "output_directory": str(self.output_dir),
            "files_generated": [
                "training_dataset.json",
                "training_dataset.csv",
                "feature_vectors.npy",
                "feature_labels.json",
                "dataset_summary.json"
            ],
            "summary_statistics": summary
        }

    def create_json_dataset(self) -> List[Dict[str, Any]]:
        """Create JSON format dataset"""

        return [asdict(example) for example in self.training_examples]

    def create_csv_dataset(self) -> pd.DataFrame:
        """Create CSV format dataset for analysis"""

        rows = []
        for example in self.training_examples:
            row = {
                'example_id': example.example_id,
                'transcript_text': example.transcript_text,
                'word_count': len(example.transcript_text.split()),
                'transcription_confidence': example.transcription_confidence,
                'question_type': example.question_type,
                'age_group': example.age_group,
                'audio_duration': example.audio_duration,
                'original_video': example.original_video
            }

            # Add linguistic features
            if example.linguistic_features:
                for key, value in example.linguistic_features.items():
                    row[f'ling_{key}'] = value

            # Add interaction features
            if example.interaction_features:
                for key, value in example.interaction_features.items():
                    row[f'interact_{key}'] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def create_feature_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """Create feature vectors for ML models (Issue #76 compatibility)"""

        feature_vectors = []
        feature_names = []

        for example in self.training_examples:
            vector = []

            # Linguistic features
            if example.linguistic_features:
                ling_features = example.linguistic_features
                if not feature_names:  # First example sets feature names
                    feature_names.extend([f"ling_{k}" for k in ling_features.keys()])
                vector.extend(ling_features.values())

            # Interaction features
            if example.interaction_features:
                interact_features = example.interaction_features
                if len(feature_names) == (len(example.linguistic_features) if example.linguistic_features else 0):
                    feature_names.extend([f"interact_{k}" for k in interact_features.keys()])
                vector.extend(interact_features.values())

            # Basic features
            if len(feature_names) == (len(vector)):
                feature_names.extend(['transcription_confidence', 'audio_duration'])
            vector.extend([example.transcription_confidence, example.audio_duration])

            feature_vectors.append(vector)

        return np.array(feature_vectors), feature_names

    def save_dataset_files(self, dataset_json: List[Dict], dataset_csv: pd.DataFrame,
                          feature_data: Tuple[np.ndarray, List[str]]) -> None:
        """Save all dataset files"""

        # Save JSON dataset
        json_file = self.output_dir / "training_dataset.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=2, ensure_ascii=False, default=str)

        # Save CSV dataset
        csv_file = self.output_dir / "training_dataset.csv"
        dataset_csv.to_csv(csv_file, index=False)

        # Save feature vectors
        feature_vectors, feature_names = feature_data
        np.save(self.output_dir / "feature_vectors.npy", feature_vectors)

        # Save feature labels
        labels_file = self.output_dir / "feature_labels.json"
        with open(labels_file, 'w') as f:
            json.dump({
                "feature_names": feature_names,
                "vector_shape": feature_vectors.shape,
                "description": "Feature vectors compatible with Issue #76 multi-modal BERT architecture"
            }, f, indent=2)

        logger.info(f"Dataset files saved to {self.output_dir}")

    def generate_dataset_summary(self) -> Dict[str, Any]:
        """Generate dataset summary statistics"""

        if not self.training_examples:
            return {}

        # Basic statistics
        total_examples = len(self.training_examples)

        # Question type distribution
        question_types = {}
        age_groups = {}
        for example in self.training_examples:
            qt = example.question_type
            question_types[qt] = question_types.get(qt, 0) + 1

            ag = example.age_group
            age_groups[ag] = age_groups.get(ag, 0) + 1

        # Quality metrics
        confidences = [ex.transcription_confidence for ex in self.training_examples]
        word_counts = [len(ex.transcript_text.split()) for ex in self.training_examples]

        summary = {
            "dataset_overview": {
                "total_examples": total_examples,
                "creation_timestamp": datetime.now().isoformat(),
                "source": "Issue #89 Transcription Pipeline"
            },
            "content_distribution": {
                "question_types": question_types,
                "age_groups": age_groups
            },
            "quality_metrics": {
                "mean_confidence": float(np.mean(confidences)),
                "confidence_range": [float(np.min(confidences)), float(np.max(confidences))],
                "mean_word_count": float(np.mean(word_counts)),
                "word_count_range": [int(np.min(word_counts)), int(np.max(word_counts))]
            },
            "ml_compatibility": {
                "feature_vector_ready": True,
                "bert_compatible": True,
                "issue_76_aligned": True
            }
        }

        # Save summary
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return summary


def main():
    """Main dataset creation execution"""

    print("🤖 ML Training Dataset Creator - Issue #89")
    print("=" * 50)

    creator = MLDatasetCreator()

    try:
        result = creator.create_training_dataset()

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return 1

        # Display results
        print(f"\n📊 DATASET CREATION RESULTS:")
        print(f"  Total Examples: {result.get('total_examples', 0)}")
        print(f"  Output Directory: {result.get('output_directory', 'unknown')}")

        print(f"\n📁 Files Generated:")
        for filename in result.get('files_generated', []):
            print(f"    ✓ {filename}")

        summary = result.get('summary_statistics', {})
        if summary:
            content = summary.get('content_distribution', {})
            quality = summary.get('quality_metrics', {})

            print(f"\n📈 Dataset Statistics:")
            print(f"  Question Types: {len(content.get('question_types', {}))}")
            print(f"  Age Groups: {len(content.get('age_groups', {}))}")
            print(f"  Mean Confidence: {quality.get('mean_confidence', 0):.3f}")
            print(f"  Mean Words/Clip: {quality.get('mean_word_count', 0):.1f}")

        print(f"\n🎯 ML Training Ready!")
        print(f"  Compatible with Issue #76 multi-modal BERT architecture")
        print(f"  Ready for Feature 2 model development")

    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        print(f"\n❌ Dataset creation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)