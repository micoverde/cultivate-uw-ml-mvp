#!/usr/bin/env python3
"""
Test Video-CSV Alignment Script for STORY 1.1 (Issue #88)

Validates that video files exist and timestamps align correctly with CSV annotations
before running the full extraction pipeline.

Author: Claude (Issue #88 Testing)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoCSVAlignmentTester:
    """Test alignment between video files and CSV annotations"""

    def __init__(self,
                 video_dir: str = "/home/warrenjo/src/tmp2/secure data",
                 csv_file: str = "/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv"):

        self.video_dir = Path(video_dir)
        self.csv_file = Path(csv_file)
        self.alignment_results = {
            'total_videos_in_csv': 0,
            'videos_found': 0,
            'videos_missing': [],
            'total_questions': 0,
            'valid_timestamps': 0,
            'invalid_timestamps': [],
            'video_matches': {},
            'duration_mismatches': []
        }

    def parse_timestamp(self, timestamp_str: str) -> Optional[float]:
        """Parse timestamp string to seconds"""
        try:
            if pd.isna(timestamp_str) or timestamp_str == 'na':
                return None

            parts = str(timestamp_str).strip().split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            else:
                return None
        except (ValueError, AttributeError):
            return None

    def find_video_file(self, video_title: str) -> Optional[Path]:
        """Find video file matching the title with comprehensive matching"""

        # Direct match first
        direct_path = self.video_dir / video_title
        if direct_path.exists():
            return direct_path

        # Handle filename variations
        video_title_clean = video_title.replace('&#39;', "'")

        # Try common variations
        variations = [
            video_title,
            video_title_clean,
            video_title.replace('.mp4', '.MP4'),
            video_title.replace('.mp4', '.mov'),
            video_title.replace('.mp4', '.MOV'),
            video_title_clean.replace('.mp4', '.MP4'),
            video_title_clean.replace('.mp4', '.mov'),
            video_title_clean.replace('.mp4', '.MOV')
        ]

        for variation in variations:
            file_path = self.video_dir / variation
            if file_path.exists():
                return file_path

        # Fuzzy matching for similar names
        base_name = video_title.split('.')[0].lower()
        for video_file in self.video_dir.glob("*.*"):
            if video_file.suffix.lower() in ['.mp4', '.mov']:
                if base_name in video_file.name.lower():
                    return video_file

        return None

    def get_video_duration(self, video_path: Path) -> Optional[float]:
        """Get video duration using moviepy (only if available)"""
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(str(video_path)) as video:
                return video.duration
        except ImportError:
            logger.warning("moviepy not available - cannot check video duration")
            return None
        except Exception as e:
            logger.error(f"Error getting duration for {video_path.name}: {e}")
            return None

    def test_single_video(self, row: pd.Series) -> Dict:
        """Test alignment for a single video"""
        video_title = row['Video Title']
        asset_number = str(row['Asset #'])

        logger.info(f"Testing: {video_title} (Asset: {asset_number})")

        result = {
            'video_title': video_title,
            'asset_number': asset_number,
            'video_found': False,
            'video_path': None,
            'video_duration': None,
            'questions': [],
            'timestamp_range': None,
            'alignment_issues': []
        }

        # Find video file
        video_path = self.find_video_file(video_title)
        if video_path:
            result['video_found'] = True
            result['video_path'] = str(video_path)
            result['video_duration'] = self.get_video_duration(video_path)
            self.alignment_results['videos_found'] += 1
        else:
            result['alignment_issues'].append("Video file not found")
            self.alignment_results['videos_missing'].append(video_title)

        # Parse timestamps and questions
        questions_data = []
        timestamps = []

        for q_num in range(1, 9):
            question_col = f'Question {q_num} '  # Note trailing space
            desc_col = f'Q{q_num} description'

            if question_col in row and pd.notna(row[question_col]) and row[question_col] != 'na':
                timestamp_str = row[question_col]
                timestamp_seconds = self.parse_timestamp(timestamp_str)
                description = row.get(desc_col, '')

                question_data = {
                    'question_number': q_num,
                    'timestamp_str': timestamp_str,
                    'timestamp_seconds': timestamp_seconds,
                    'description': description,
                    'valid': timestamp_seconds is not None
                }

                questions_data.append(question_data)
                self.alignment_results['total_questions'] += 1

                if timestamp_seconds is not None:
                    timestamps.append(timestamp_seconds)
                    self.alignment_results['valid_timestamps'] += 1
                else:
                    self.alignment_results['invalid_timestamps'].append(f"{video_title} Q{q_num}: {timestamp_str}")

        result['questions'] = questions_data

        # Check timestamp range vs video duration
        if timestamps and result['video_duration']:
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)
            result['timestamp_range'] = (min_timestamp, max_timestamp)

            # Check if timestamps exceed video duration
            if max_timestamp > result['video_duration']:
                issue = f"Timestamp {max_timestamp}s exceeds video duration {result['video_duration']:.1f}s"
                result['alignment_issues'].append(issue)
                self.alignment_results['duration_mismatches'].append(f"{video_title}: {issue}")

        return result

    def load_and_validate_csv(self) -> pd.DataFrame:
        """Load and validate CSV structure"""
        try:
            logger.info(f"Loading CSV: {self.csv_file}")

            # Load with proper encoding
            df = pd.read_csv(self.csv_file, encoding='utf-8')

            # Clean BOM characters
            df.columns = df.columns.str.replace('\ufeff', '')

            logger.info(f"CSV loaded: {len(df)} rows")
            logger.info(f"Columns: {list(df.columns)}")

            # Validate required columns
            required_columns = ['Video Title', 'Asset #']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Check for question columns
            question_columns = [col for col in df.columns if col.startswith('Question ')]
            logger.info(f"Found question columns: {question_columns}")

            return df

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def generate_test_report(self, results: List[Dict]):
        """Generate comprehensive alignment test report"""

        report = {
            'summary': self.alignment_results.copy(),
            'detailed_results': results,
            'recommendations': []
        }

        # Calculate success rates
        total_videos = len(results)
        videos_found = sum(1 for r in results if r['video_found'])
        total_questions = sum(len(r['questions']) for r in results)
        valid_questions = sum(1 for r in results for q in r['questions'] if q['valid'])

        report['summary'].update({
            'video_match_rate': (videos_found / total_videos * 100) if total_videos > 0 else 0,
            'timestamp_validity_rate': (valid_questions / total_questions * 100) if total_questions > 0 else 0,
            'total_alignment_issues': sum(len(r['alignment_issues']) for r in results)
        })

        # Generate recommendations
        if self.alignment_results['videos_missing']:
            report['recommendations'].append(
                f"Missing video files: {len(self.alignment_results['videos_missing'])} videos not found"
            )

        if self.alignment_results['invalid_timestamps']:
            report['recommendations'].append(
                f"Invalid timestamps: {len(self.alignment_results['invalid_timestamps'])} timestamps could not be parsed"
            )

        if self.alignment_results['duration_mismatches']:
            report['recommendations'].append(
                f"Duration mismatches: {len(self.alignment_results['duration_mismatches'])} timestamps exceed video duration"
            )

        if report['summary']['video_match_rate'] == 100 and report['summary']['timestamp_validity_rate'] > 95:
            report['recommendations'].append("✅ Alignment looks good - ready for full extraction pipeline")

        return report

    def print_detailed_results(self, results: List[Dict]):
        """Print detailed test results to console"""

        print("\n" + "="*80)
        print("VIDEO-CSV ALIGNMENT TEST RESULTS")
        print("="*80)

        for result in results[:5]:  # Show first 5 for brevity
            print(f"\n📹 {result['video_title']}")
            print(f"   Asset: {result['asset_number']}")
            print(f"   Found: {'✅' if result['video_found'] else '❌'}")

            if result['video_found']:
                print(f"   Path: {result['video_path']}")
                if result['video_duration']:
                    print(f"   Duration: {result['video_duration']:.1f}s")

            if result['questions']:
                print(f"   Questions: {len(result['questions'])}")
                for q in result['questions']:
                    status = "✅" if q['valid'] else "❌"
                    print(f"     Q{q['question_number']}: {q['timestamp_str']} ({q['timestamp_seconds']}s) {status}")

            if result['alignment_issues']:
                print(f"   Issues: {', '.join(result['alignment_issues'])}")

        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more videos")

        # Summary statistics
        print(f"\n📊 SUMMARY:")
        print(f"   Videos in CSV: {len(results)}")
        print(f"   Videos found: {self.alignment_results['videos_found']}")
        print(f"   Videos missing: {len(self.alignment_results['videos_missing'])}")
        print(f"   Total questions: {self.alignment_results['total_questions']}")
        print(f"   Valid timestamps: {self.alignment_results['valid_timestamps']}")
        print(f"   Invalid timestamps: {len(self.alignment_results['invalid_timestamps'])}")

        if self.alignment_results['videos_missing']:
            print(f"\n❌ MISSING VIDEOS:")
            for video in self.alignment_results['videos_missing']:
                print(f"   - {video}")

        if self.alignment_results['invalid_timestamps']:
            print(f"\n❌ INVALID TIMESTAMPS:")
            for timestamp in self.alignment_results['invalid_timestamps'][:10]:  # Show first 10
                print(f"   - {timestamp}")

        if self.alignment_results['duration_mismatches']:
            print(f"\n⚠️  DURATION MISMATCHES:")
            for mismatch in self.alignment_results['duration_mismatches']:
                print(f"   - {mismatch}")

    def run_alignment_test(self) -> bool:
        """Run complete alignment test"""
        try:
            logger.info("Starting video-CSV alignment test")

            # Load CSV
            df = self.load_and_validate_csv()
            self.alignment_results['total_videos_in_csv'] = len(df)

            # Test each video
            results = []
            for idx, row in df.iterrows():
                try:
                    result = self.test_single_video(row)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to test row {idx}: {e}")

            # Generate and display results
            self.print_detailed_results(results)
            report = self.generate_test_report(results)

            # Save detailed report
            import json
            report_path = Path("/home/warrenjo/src/tmp2/cultivate-uw-ml-mvp/data/alignment_test_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(f"\n📄 Detailed report saved: {report_path}")

            # Determine if ready for extraction
            video_match_rate = report['summary']['video_match_rate']
            timestamp_validity_rate = report['summary']['timestamp_validity_rate']

            success = video_match_rate >= 90 and timestamp_validity_rate >= 95

            print(f"\n🎯 READINESS ASSESSMENT:")
            print(f"   Video match rate: {video_match_rate:.1f}%")
            print(f"   Timestamp validity rate: {timestamp_validity_rate:.1f}%")
            print(f"   Ready for extraction: {'✅ YES' if success else '❌ NO'}")

            return success

        except Exception as e:
            logger.error(f"Alignment test failed: {e}")
            return False

def main():
    """Main test runner"""
    print("🧪 Testing video-CSV alignment for Issue #88...")

    tester = VideoCSVAlignmentTester()
    success = tester.run_alignment_test()

    if success:
        print("\n✅ Alignment test passed - ready to proceed with extraction!")
    else:
        print("\n❌ Alignment issues detected - review results before extraction")
        sys.exit(1)

if __name__ == "__main__":
    main()