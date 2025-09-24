#!/usr/bin/env python3
"""
Enhanced Video-CSV Alignment Test with Better Filename Matching

Addresses special character encoding and filename variations found in initial test.
"""

import os
import pandas as pd
from pathlib import Path
import html

def enhanced_find_video_file(video_title: str, video_dir: Path) -> Path:
    """Enhanced video file matching with comprehensive variations"""

    # Direct match
    direct_path = video_dir / video_title
    if direct_path.exists():
        return direct_path

    # Handle HTML entities and special characters
    cleaned_title = html.unescape(video_title)  # Convert &#39; to '
    cleaned_title = cleaned_title.strip()       # Remove trailing spaces

    # Generate all possible variations
    variations = [
        video_title,
        cleaned_title,
        video_title.replace(' ', '_'),
        cleaned_title.replace(' ', '_'),
        video_title.rstrip(),
        cleaned_title.rstrip()
    ]

    # Add extension variations
    extended_variations = []
    for var in variations:
        extended_variations.extend([
            var,
            var.replace('.mp4', '.MP4'),
            var.replace('.mp4', '.mov'),
            var.replace('.mp4', '.MOV')
        ])

    # Test all variations
    for variation in extended_variations:
        file_path = video_dir / variation
        if file_path.exists():
            print(f"✅ Found match: '{video_title}' → '{file_path.name}'")
            return file_path

    # Fuzzy matching - check if base name exists in any video file
    base_name = video_title.split('.')[0].lower().strip()

    for video_file in video_dir.glob("*.*"):
        if video_file.suffix.lower() in ['.mp4', '.mov']:
            file_base = video_file.stem.lower()

            # Check various similarity conditions
            if (base_name in file_base or
                file_base in base_name or
                base_name.replace(' ', '') in file_base.replace(' ', '') or
                base_name.replace('_', ' ') in file_base.replace('_', ' ')):

                print(f"🔍 Fuzzy match: '{video_title}' → '{video_file.name}'")
                return video_file

    print(f"❌ No match found for: '{video_title}'")
    return None

def test_specific_missing_files():
    """Test the specific files that were missing"""

    video_dir = Path("/home/warrenjo/src/tmp2/secure data")
    missing_files = [
        "Book about Flowers SOM SUBS .mp4",
        "Being Aware of a Toddler's Needs"
    ]

    print("🔍 Testing specific missing files:")
    print("="*50)

    for missing_file in missing_files:
        print(f"\nTesting: '{missing_file}'")
        found_file = enhanced_find_video_file(missing_file, video_dir)

        if found_file:
            print(f"  ✅ Found: {found_file}")
        else:
            print(f"  ❌ Still missing")

            # Show what files exist with similar names
            base = missing_file.split('.')[0].lower()
            similar_files = []
            for f in video_dir.glob("*.*"):
                if any(word in f.name.lower() for word in base.split()[:2]):  # First 2 words
                    similar_files.append(f.name)

            if similar_files:
                print(f"  🔍 Similar files found: {similar_files}")

def main():
    """Enhanced alignment test"""

    # Test specific missing files first
    test_specific_missing_files()

    # Quick recount of total questions across all CSV rows
    print("\n" + "="*60)
    print("QUESTION COUNT VERIFICATION")
    print("="*60)

    csv_file = Path("/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv")
    df = pd.read_csv(csv_file)

    total_questions = 0
    for idx, row in df.iterrows():
        video_questions = 0
        for q_num in range(1, 9):
            question_col = f'Question {q_num} '  # Note trailing space for some
            if question_col not in row:
                question_col = f'Question {q_num}'  # Try without space

            if question_col in row and pd.notna(row[question_col]) and row[question_col] != 'na':
                video_questions += 1
                total_questions += 1

        print(f"📹 {row['Video Title'][:40]:40} - {video_questions} questions")

    print(f"\n📊 Total questions in CSV: {total_questions}")
    print(f"📊 Target questions needed: 119")
    print(f"📊 Difference: {total_questions - 119}")

if __name__ == "__main__":
    main()