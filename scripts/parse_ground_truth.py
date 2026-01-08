#!/usr/bin/env python3
"""
Ground Truth Data Pipeline - USER STORY 1.2
Parse Cultivate Learning expert annotations CSV into structured JSON

This script:
1. Parses VideosAskingQuestions CSV with expert annotations
2. Extracts OEQ/CEQ classifications from descriptions
3. Links to existing Whisper transcripts
4. Generates video_catalog.json for Demo 3 frontend

Author: Claude (Partner-Level Microsoft SDE)
Issue: #290 - USER STORY 1.2: Ground Truth Data Pipeline
"""

import csv
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


def parse_timestamp(ts: str) -> float:
    """Convert timestamp string (M:SS or MM:SS) to seconds"""
    if not ts or ts.strip().lower() == 'na':
        return -1

    ts = ts.strip()
    # Handle range format "0:00-0:45" - take start time
    if '-' in ts:
        ts = ts.split('-')[0].strip()

    parts = ts.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        try:
            return int(minutes) * 60 + float(seconds)
        except ValueError:
            return -1
    return -1


def extract_question_type(description: str) -> str:
    """Extract OEQ or CEQ from description text"""
    if not description:
        return 'UNKNOWN'

    desc_upper = description.upper()

    # Check for OEQ indicators
    if 'OEQ' in desc_upper or 'OPEN' in desc_upper:
        return 'OEQ'

    # Check for CEQ indicators
    if 'CEQ' in desc_upper or 'CLOSED' in desc_upper or 'YES/NO' in desc_upper:
        return 'CEQ'

    return 'UNKNOWN'


def extract_behavioral_notes(description: str) -> Dict[str, Any]:
    """Extract behavioral annotations from description"""
    notes = {
        'pauses_for_response': False,
        'rhetorical': False,
        'child_responds': False,
        'teacher_answers_own': False,
        'restates_question': False
    }

    if not description:
        return notes

    desc_lower = description.lower()

    # Detect pause behavior
    if 'pause' in desc_lower and 'no pause' not in desc_lower and "doesn't pause" not in desc_lower:
        notes['pauses_for_response'] = True

    # Detect rhetorical questions
    if 'rhetorical' in desc_lower:
        notes['rhetorical'] = True

    # Detect child response
    if 'child' in desc_lower and ('respond' in desc_lower or 'answer' in desc_lower):
        notes['child_responds'] = True

    # Detect teacher self-answering
    if 'answers' in desc_lower and ('own' in desc_lower or 'herself' in desc_lower or 'himself' in desc_lower):
        notes['teacher_answers_own'] = True

    # Detect question restatement
    if 'restate' in desc_lower or 'rephrase' in desc_lower or 'repeat' in desc_lower:
        notes['restates_question'] = True

    return notes


def find_transcripts_for_asset(asset_id: str, transcripts_dir: Path) -> List[Dict]:
    """Find all transcript files for a given asset ID"""
    transcripts = []

    if not transcripts_dir.exists():
        return transcripts

    pattern = f"{asset_id}_q*_transcript.json"

    for transcript_file in transcripts_dir.glob(pattern):
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                transcripts.append({
                    'filename': transcript_file.name,
                    'question_id': transcript_file.stem.split('_')[1],  # e.g., 'q1'
                    'transcript_text': ' '.join(
                        seg.get('text', '') for seg in data.get('transcript_segments', [])
                    ).strip(),
                    'confidence': data.get('processing_metadata', {}).get('transcription_confidence', 0),
                    'word_count': data.get('processing_metadata', {}).get('total_words', 0)
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {transcript_file}: {e}")

    return sorted(transcripts, key=lambda x: x['question_id'])


def get_video_file_info(filename: str, secure_data_dir: Path) -> Dict[str, Any]:
    """Get file information for a video"""
    # Try different extensions and case variations
    extensions = ['.mp4', '.MP4', '.mov', '.MOV']

    for ext in extensions:
        # Try exact filename
        video_path = secure_data_dir / filename
        if video_path.exists():
            stat = video_path.stat()
            return {
                'exists': True,
                'local_path': str(video_path),
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }

        # Try with different extension
        base = Path(filename).stem
        video_path = secure_data_dir / f"{base}{ext}"
        if video_path.exists():
            stat = video_path.stat()
            return {
                'exists': True,
                'local_path': str(video_path),
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }

    return {'exists': False, 'local_path': None, 'file_size_mb': 0}


def generate_display_name(filename: str) -> str:
    """Generate a human-readable display name from filename"""
    # Remove extension
    name = Path(filename).stem

    # Remove asset numbers at end (e.g., "004", "013")
    name = re.sub(r'\s*\d{3}$', '', name)

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def parse_ground_truth_csv(csv_path: str) -> Dict[str, Any]:
    """
    Parse the Cultivate Learning expert annotations CSV

    Returns a dictionary of videos keyed by asset ID
    """
    videos = {}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            video_title = row.get('Video Title', '').strip()
            asset_id = row.get('Asset #', '').strip()

            if not asset_id or not video_title:
                continue

            # Parse age group
            age_group = row.get('Age group', 'UNKNOWN').strip().upper()
            if 'TODDLER' in age_group:
                age_group = 'TODDLER'
            elif 'INFANT' in age_group:
                age_group = 'INFANT'
            elif 'PK' in age_group or 'PRE' in age_group:
                age_group = 'PK'

            video = {
                'id': asset_id,
                'filename': video_title,
                'display_name': generate_display_name(video_title),
                'age_group': age_group,
                'timestamp_range': row.get('Timestamp', '').strip(),
                'description': row.get('Description', '').strip(),
                'questions': []
            }

            # Parse questions Q1-Q8
            for i in range(1, 9):
                # CSV has inconsistent column naming
                timestamp_col = f'Question {i} '
                desc_col = f'Q{i} description'

                timestamp = row.get(timestamp_col, '').strip()
                description = row.get(desc_col, '').strip()

                if timestamp and timestamp.lower() != 'na':
                    question = {
                        'id': f'q{i}',
                        'timestamp': timestamp,
                        'timestamp_seconds': parse_timestamp(timestamp),
                        'ground_truth_type': extract_question_type(description),
                        'description': description,
                        'behavioral_notes': extract_behavioral_notes(description),
                        'has_transcript': False,
                        'transcript_text': None,
                        'ml_prediction': None
                    }
                    video['questions'].append(question)

            videos[asset_id] = video

    return videos


def build_video_catalog(
    csv_path: str,
    transcripts_dir: str,
    secure_data_dir: str,
    output_path: str
) -> Dict[str, Any]:
    """
    Build comprehensive video catalog for Demo 3

    Combines:
    - Expert annotations from CSV
    - Existing Whisper transcripts
    - Video file metadata
    """
    print(f"Parsing ground truth from: {csv_path}")
    videos = parse_ground_truth_csv(csv_path)
    print(f"Found {len(videos)} videos with annotations")

    transcripts_path = Path(transcripts_dir)
    secure_path = Path(secure_data_dir)

    catalog = {
        'videos': [],
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'source_csv': csv_path,
            'transcripts_dir': transcripts_dir,
            'secure_data_dir': secure_data_dir
        },
        'statistics': {
            'total_videos': 0,
            'total_questions': 0,
            'questions_by_type': {'OEQ': 0, 'CEQ': 0, 'UNKNOWN': 0},
            'videos_by_age_group': {'PK': 0, 'TODDLER': 0, 'INFANT': 0, 'UNKNOWN': 0},
            'videos_with_transcripts': 0,
            'questions_with_transcripts': 0
        }
    }

    for asset_id, video in videos.items():
        # Get video file info
        file_info = get_video_file_info(video['filename'], secure_path)
        video['file_info'] = file_info

        # Find and link transcripts
        transcripts = find_transcripts_for_asset(asset_id, transcripts_path)
        transcript_map = {t['question_id']: t for t in transcripts}

        has_any_transcript = False
        for question in video['questions']:
            q_id = question['id']
            if q_id in transcript_map:
                question['has_transcript'] = True
                question['transcript_text'] = transcript_map[q_id]['transcript_text']
                question['transcript_confidence'] = transcript_map[q_id]['confidence']
                has_any_transcript = True
                catalog['statistics']['questions_with_transcripts'] += 1

        video['has_transcripts'] = has_any_transcript
        video['transcript_count'] = len(transcripts)

        # Update statistics
        catalog['statistics']['total_questions'] += len(video['questions'])

        for q in video['questions']:
            q_type = q['ground_truth_type']
            if q_type in catalog['statistics']['questions_by_type']:
                catalog['statistics']['questions_by_type'][q_type] += 1

        age = video['age_group']
        if age in catalog['statistics']['videos_by_age_group']:
            catalog['statistics']['videos_by_age_group'][age] += 1
        else:
            catalog['statistics']['videos_by_age_group']['UNKNOWN'] += 1

        if has_any_transcript:
            catalog['statistics']['videos_with_transcripts'] += 1

        catalog['videos'].append(video)

    catalog['statistics']['total_videos'] = len(catalog['videos'])

    # Sort videos by display name
    catalog['videos'].sort(key=lambda v: v['display_name'])

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nCatalog generated: {output_path}")
    print(f"Statistics:")
    print(f"  Total videos: {catalog['statistics']['total_videos']}")
    print(f"  Total questions: {catalog['statistics']['total_questions']}")
    print(f"  Questions by type: {catalog['statistics']['questions_by_type']}")
    print(f"  Videos by age group: {catalog['statistics']['videos_by_age_group']}")
    print(f"  Videos with transcripts: {catalog['statistics']['videos_with_transcripts']}")
    print(f"  Questions with transcripts: {catalog['statistics']['questions_with_transcripts']}")

    return catalog


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Parse Cultivate Learning ground truth CSV')
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV file with expert annotations (required)'
    )
    parser.add_argument(
        '--transcripts',
        default='./data/transcripts/transcripts',
        help='Path to directory with Whisper transcripts'
    )
    parser.add_argument(
        '--secure-data',
        default='.',
        help='Path to secure data directory with video files'
    )
    parser.add_argument(
        '--output',
        default='./unified-demos/demo3/data/video_catalog.json',
        help='Output path for video catalog JSON'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build catalog
    catalog = build_video_catalog(
        csv_path=args.csv,
        transcripts_dir=args.transcripts,
        secure_data_dir=args.secure_data,
        output_path=args.output
    )

    print(f"\nDone! Video catalog saved to: {args.output}")


if __name__ == '__main__':
    main()
