#!/usr/bin/env python3
"""
Validation Script for Issue #89 Transcription Setup

Tests that all required dependencies are installed and the pipeline
can be initialized without running full transcription.

Author: Claude (Issue #89 Setup Validation)
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""

    print("🔍 Checking transcription dependencies...")
    print("=" * 50)

    dependencies = {
        'Core Libraries': [
            ('numpy', 'np'),
            ('pandas', 'pd'),
            ('librosa', 'librosa'),
            ('soundfile', 'sf')
        ],
        'ML Libraries': [
            ('torch', 'torch'),
            ('torchaudio', 'torchaudio'),
            ('whisper', 'whisper'),
        ],
        'Speaker Diarization': [
            ('pyannote.audio', 'pyannote.audio'),
            ('speechbrain', 'speechbrain')
        ]
    }

    all_available = True

    for category, libs in dependencies.items():
        print(f"\n📦 {category}:")

        for lib_name, import_name in libs:
            try:
                __import__(import_name)
                print(f"  ✅ {lib_name}")
            except ImportError as e:
                print(f"  ❌ {lib_name} - {e}")
                all_available = False

    return all_available

def check_environment():
    """Check environment configuration"""

    print(f"\n🌍 Environment Configuration:")
    print("=" * 50)

    # Check Hugging Face token
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print(f"  ✅ HUGGINGFACE_HUB_TOKEN is set (length: {len(hf_token)})")
    else:
        print("  ⚠️  HUGGINGFACE_HUB_TOKEN not set - speaker diarization may require authentication")

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"     GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ℹ️  CUDA not available - using CPU (slower but functional)")
    except ImportError:
        print("  ❌ Cannot check CUDA - torch not available")

def validate_pipeline_setup():
    """Test basic pipeline initialization"""

    print(f"\n🧪 Testing Pipeline Setup:")
    print("=" * 50)

    try:
        # Import pipeline components
        sys.path.insert(0, str(Path(__file__).parent))
        from transcription_pipeline import TranscriptionPipeline

        print("  ✅ TranscriptionPipeline import successful")

        # Test pipeline initialization
        pipeline = TranscriptionPipeline(
            clips_dir="test_clips",
            metadata_file="test_metadata.json",
            output_dir="test_output",
            whisper_model="base"  # Smallest model for testing
        )

        print("  ✅ Pipeline initialization successful")

        # Test data structures
        from transcription_pipeline import (
            TranscriptWord,
            SpeakerSegment,
            TranscriptSegment,
            ClipTranscript
        )

        # Create test instances
        test_word = TranscriptWord("hello", 0.0, 0.5, 0.95, "educator")
        test_speaker = SpeakerSegment("educator_01", 0.0, 2.0, 0.85, "educator")

        print("  ✅ Data structures working correctly")

        # Test JSON encoder
        from transcription_pipeline import TranscriptionEncoder
        import json
        import numpy as np

        test_data = {
            'numpy_int': np.int32(42),
            'numpy_float': np.float32(3.14),
            'numpy_bool': np.bool_(True),
            'numpy_array': np.array([1, 2, 3])
        }

        json_str = json.dumps(test_data, cls=TranscriptionEncoder)
        parsed = json.loads(json_str)

        print("  ✅ JSON serialization working correctly")

        return True

    except Exception as e:
        print(f"  ❌ Pipeline setup failed: {e}")
        return False

def check_input_data():
    """Check if Issue #88 output data is available"""

    print(f"\n📁 Checking Issue #88 Input Data:")
    print("=" * 50)

    # Expected paths from Issue #88
    clips_dir = Path("data/processed_audio/clips")
    metadata_file = Path("data/processed_audio/metadata/clip_manifest.json")

    if clips_dir.exists():
        clip_count = len(list(clips_dir.glob("*.wav")))
        print(f"  ✅ Audio clips directory found: {clip_count} clips")
    else:
        print(f"  ❌ Audio clips directory not found: {clips_dir}")
        print("     Run Issue #88 video-audio extraction first")

    if metadata_file.exists():
        print(f"  ✅ Clip metadata file found: {metadata_file}")

        # Check metadata content
        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            clips = metadata.get('clips', [])
            print(f"     Metadata contains {len(clips)} clip records")

            if clips:
                sample_clip = clips[0]
                required_fields = ['clip_filename', 'asset_number', 'question_number']
                missing_fields = [f for f in required_fields if f not in sample_clip]

                if not missing_fields:
                    print("     ✅ Metadata format is correct")
                else:
                    print(f"     ⚠️  Missing fields in metadata: {missing_fields}")

        except Exception as e:
            print(f"     ❌ Error reading metadata: {e}")
    else:
        print(f"  ❌ Clip metadata file not found: {metadata_file}")
        print("     Run Issue #88 video-audio extraction first")

def main():
    """Main validation function"""

    print("🎤 Issue #89: Transcription Pipeline Setup Validation")
    print("=" * 70)

    # Run all validation checks
    deps_ok = check_dependencies()
    check_environment()
    pipeline_ok = validate_pipeline_setup()
    check_input_data()

    # Final summary
    print(f"\n📊 Validation Summary:")
    print("=" * 50)

    if deps_ok and pipeline_ok:
        print("✅ All core components are ready!")
        print("📋 Next steps:")
        print("   1. Ensure HUGGINGFACE_HUB_TOKEN is set for speaker diarization")
        print("   2. Run Issue #88 if audio clips are not available")
        print("   3. Execute: python src/data_processing/transcription_pipeline.py")
    else:
        print("❌ Setup issues detected:")
        if not deps_ok:
            print("   - Install missing dependencies: pip install -r requirements-transcription.txt")
        if not pipeline_ok:
            print("   - Fix pipeline setup issues shown above")

    print(f"\n💡 For GPU acceleration:")
    print("   - Install CUDA-compatible PyTorch")
    print("   - Ensure sufficient GPU memory (4GB+ recommended)")

    return deps_ok and pipeline_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)