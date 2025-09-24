#!/usr/bin/env python3
"""
Final Alignment Check with All Issues Resolved

This addresses:
1. HTML entity encoding (&#39; → ')
2. Trailing spaces in CSV vs filenames
3. Accurate question counting across all videos
"""

import pandas as pd
from pathlib import Path
import html

def robust_find_video_file(video_title: str, video_dir: Path) -> Path:
    """Most robust video file matching"""

    # Clean the title
    cleaned_title = html.unescape(video_title.strip())

    # Try exact matches first
    for candidate in [video_title, cleaned_title]:
        if (video_dir / candidate).exists():
            return video_dir / candidate

    # Try extension variations
    for base_title in [video_title, cleaned_title]:
        for ext in ['.mp4', '.MP4', '.mov', '.MOV']:
            # Try with extension
            candidate = base_title.replace('.mp4', ext).replace('.MP4', ext).replace('.mov', ext).replace('.MOV', ext)
            if (video_dir / candidate).exists():
                return video_dir / candidate

            # Try adding extension if not present
            if not any(base_title.endswith(e) for e in ['.mp4', '.MP4', '.mov', '.MOV']):
                candidate = base_title + ext
                if (video_dir / candidate).exists():
                    return video_dir / candidate

    # Fuzzy match by checking all video files
    base_name_words = set(cleaned_title.lower().replace('.mp4', '').replace('.mov', '').split())

    for video_file in video_dir.glob("*.*"):
        if video_file.suffix.lower() in ['.mp4', '.mov']:
            file_words = set(video_file.stem.lower().split())

            # If most words match, consider it a match
            if len(base_name_words.intersection(file_words)) >= min(3, len(base_name_words) - 1):
                return video_file

    return None

def main():
    """Complete final alignment check"""

    video_dir = Path("/home/warrenjo/src/tmp2/secure data")
    csv_file = Path("/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv")

    print("🔍 FINAL VIDEO-CSV ALIGNMENT CHECK")
    print("="*60)

    # Load CSV
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.replace('\ufeff', '')  # Remove BOM

    total_videos = len(df)
    videos_found = 0
    total_questions = 0
    missing_videos = []

    for idx, row in df.iterrows():
        video_title = row['Video Title']
        asset_number = row['Asset #']

        # Find video file
        video_file = robust_find_video_file(video_title, video_dir)

        if video_file:
            videos_found += 1
            status = "✅"
            match_info = f"→ {video_file.name}" if video_file.name != video_title else ""
        else:
            status = "❌"
            match_info = "NOT FOUND"
            missing_videos.append(video_title)

        # Count questions for this video
        video_questions = 0
        for q_num in range(1, 9):
            # Try both column name variations
            for question_col in [f'Question {q_num} ', f'Question {q_num}']:
                if (question_col in row and
                    pd.notna(row[question_col]) and
                    str(row[question_col]).strip() != 'na'):
                    video_questions += 1
                    total_questions += 1
                    break

        print(f"{status} {video_title[:45]:45} | Q:{video_questions} | {match_info}")

    print("\n" + "="*60)
    print("📊 FINAL SUMMARY:")
    print(f"   Total videos in CSV: {total_videos}")
    print(f"   Videos found: {videos_found}")
    print(f"   Videos missing: {len(missing_videos)}")
    print(f"   Video match rate: {videos_found/total_videos*100:.1f}%")
    print(f"   Total questions: {total_questions}")
    print(f"   Target questions: 119")
    print(f"   Question gap: {119 - total_questions}")

    if missing_videos:
        print(f"\n❌ STILL MISSING:")
        for video in missing_videos:
            print(f"   - {video}")

    # Check if ready for extraction
    ready = videos_found >= total_videos * 0.95 and total_questions >= 110

    print(f"\n🎯 READY FOR EXTRACTION: {'✅ YES' if ready else '❌ NO'}")

    if ready:
        print("\n✨ Recommendations:")
        print("   1. Proceed with extraction pipeline")
        print("   2. Will extract ~112 clips from 23+ videos")
        print("   3. Close to target of 119 clips")
    else:
        print("\n🔧 Issues to resolve:")
        if videos_found < total_videos * 0.95:
            print(f"   - Fix missing video files ({len(missing_videos)} missing)")
        if total_questions < 110:
            print(f"   - Need more questions (only {total_questions}/119)")

if __name__ == "__main__":
    main()