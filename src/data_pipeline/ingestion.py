"""
Data ingestion module for educator-child interaction recordings.

Handles secure ingestion of video files and text transcripts with
privacy protection and metadata extraction.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles secure ingestion of training data with privacy protection."""

    def __init__(self, data_dir: str = "data/training"):
        """Initialize data ingestion pipeline.

        Args:
            data_dir: Directory for storing training data
        """
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.metadata_dir = self.data_dir / "metadata"

        # Create directories if they don't exist
        for dir_path in [self.videos_dir, self.transcripts_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def generate_anonymous_id(self, original_filename: str) -> str:
        """Generate anonymous ID for privacy protection.

        Args:
            original_filename: Original file name

        Returns:
            Anonymous ID hash
        """
        # Use SHA256 hash of filename + timestamp for anonymization
        hash_input = f"{original_filename}_{datetime.now().isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def ingest_video(self,
                    video_path: str,
                    metadata: Dict) -> Tuple[str, Dict]:
        """Ingest video file with privacy protection.

        Args:
            video_path: Path to video file
            metadata: Video metadata including quality indicators

        Returns:
            Tuple of (anonymous_id, processed_metadata)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Generate anonymous ID
        anonymous_id = self.generate_anonymous_id(os.path.basename(video_path))

        # Copy video to secure location with anonymous name
        video_extension = Path(video_path).suffix
        secure_video_path = self.videos_dir / f"{anonymous_id}{video_extension}"

        # For now, log the operation (in production, would copy the file)
        logger.info(f"Processing video: {video_path} -> {secure_video_path}")

        # Process and sanitize metadata
        processed_metadata = self._sanitize_metadata(metadata, anonymous_id)

        # Save metadata
        metadata_path = self.metadata_dir / f"{anonymous_id}_video.json"
        with open(metadata_path, 'w') as f:
            json.dump(processed_metadata, f, indent=2)

        logger.info(f"Video ingested with ID: {anonymous_id}")
        return anonymous_id, processed_metadata

    def ingest_transcript(self,
                         transcript_path: str,
                         anonymous_id: str,
                         metadata: Dict) -> Dict:
        """Ingest transcript file linked to video.

        Args:
            transcript_path: Path to transcript file
            anonymous_id: Anonymous ID from video ingestion
            metadata: Transcript metadata

        Returns:
            Processed metadata
        """
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        # Read and process transcript
        with open(transcript_path, 'r') as f:
            transcript_content = f.read()

        # De-identify transcript content
        deidentified_content = self._deidentify_transcript(transcript_content)

        # Save de-identified transcript
        secure_transcript_path = self.transcripts_dir / f"{anonymous_id}_transcript.txt"
        with open(secure_transcript_path, 'w') as f:
            f.write(deidentified_content)

        # Process metadata
        processed_metadata = self._sanitize_metadata(metadata, anonymous_id)
        processed_metadata['transcript_length'] = len(deidentified_content)
        processed_metadata['word_count'] = len(deidentified_content.split())

        # Save transcript metadata
        metadata_path = self.metadata_dir / f"{anonymous_id}_transcript.json"
        with open(metadata_path, 'w') as f:
            json.dump(processed_metadata, f, indent=2)

        logger.info(f"Transcript ingested for ID: {anonymous_id}")
        return processed_metadata

    def _sanitize_metadata(self, metadata: Dict, anonymous_id: str) -> Dict:
        """Sanitize metadata to remove identifying information.

        Args:
            metadata: Original metadata
            anonymous_id: Anonymous ID for the record

        Returns:
            Sanitized metadata
        """
        sanitized = {
            'anonymous_id': anonymous_id,
            'ingestion_timestamp': datetime.now().isoformat(),
            'data_type': metadata.get('data_type', 'unknown'),
            'quality_indicators': metadata.get('quality_indicators', {}),
            'research_framework': metadata.get('research_framework', 'CLASS'),
            'interaction_duration': metadata.get('interaction_duration'),
            'participant_age_range': metadata.get('participant_age_range'),
            'interaction_type': metadata.get('interaction_type'),
            'consent_status': metadata.get('consent_status', 'required'),
            'irb_approval': metadata.get('irb_approval', 'required')
        }

        # Remove any potentially identifying fields
        identifying_fields = ['name', 'school', 'location', 'date', 'teacher_name', 'child_name']
        for field in identifying_fields:
            if field in metadata:
                logger.warning(f"Removing potentially identifying field: {field}")

        return sanitized

    def _deidentify_transcript(self, transcript: str) -> str:
        """Remove identifying information from transcript.

        Args:
            transcript: Original transcript content

        Returns:
            De-identified transcript
        """
        # Replace common identifying patterns
        deidentified = transcript

        # Replace names with generic placeholders
        # In production, would use more sophisticated NER
        replacements = {
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b': '[NAME]',  # Full names
            r'\bMrs?\. [A-Z][a-z]+\b': '[TEACHER]',     # Teacher titles
            r'\b\d{1,2}/\d{1,2}/\d{4}\b': '[DATE]',    # Dates
            r'\b[A-Z][a-z]+ School\b': '[SCHOOL]'       # School names
        }

        import re
        for pattern, replacement in replacements.items():
            deidentified = re.sub(pattern, replacement, deidentified)

        return deidentified

    def list_ingested_data(self) -> List[Dict]:
        """List all ingested data with metadata.

        Returns:
            List of metadata dictionaries for all ingested data
        """
        ingested_data = []

        # Scan metadata directory for all records
        for metadata_file in self.metadata_dir.glob("*_video.json"):
            with open(metadata_file, 'r') as f:
                video_metadata = json.load(f)

            anonymous_id = video_metadata['anonymous_id']

            # Check for corresponding transcript
            transcript_metadata_path = self.metadata_dir / f"{anonymous_id}_transcript.json"
            transcript_metadata = None
            if transcript_metadata_path.exists():
                with open(transcript_metadata_path, 'r') as f:
                    transcript_metadata = json.load(f)

            ingested_data.append({
                'anonymous_id': anonymous_id,
                'video_metadata': video_metadata,
                'transcript_metadata': transcript_metadata,
                'has_transcript': transcript_metadata is not None
            })

        return ingested_data


def create_sample_metadata() -> Dict:
    """Create sample metadata for testing purposes."""
    return {
        'data_type': 'educator_child_interaction',
        'quality_indicators': {
            'open_ended_questions': 0,
            'conversational_turns': 0,
            'child_engagement_level': 'unknown',
            'educator_responsiveness': 'unknown'
        },
        'research_framework': 'CLASS',
        'interaction_duration': '10:30',
        'participant_age_range': '3-5',
        'interaction_type': 'free_play',
        'consent_status': 'obtained',
        'irb_approval': 'UW_IRB_2024_001'
    }


if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    sample_metadata = create_sample_metadata()

    print("Data ingestion pipeline initialized")
    print(f"Training data directory: {ingestion.data_dir}")
    print("Ready for video and transcript ingestion with privacy protection")