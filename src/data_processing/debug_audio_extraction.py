#!/usr/bin/env python3
"""
Debug script to identify audio extraction issues
"""

import sys
import traceback
from pathlib import Path

try:
    from moviepy.editor import VideoFileClip
    import librosa
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_single_video():
    """Test audio extraction on a single video"""

    video_dir = Path("/home/warrenjo/src/tmp2/secure data")
    test_video = video_dir / "Structure Activity Video_SUBS.mp4"

    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        return False

    print(f"Testing: {test_video.name}")

    try:
        print("Loading video...")
        with VideoFileClip(str(test_video)) as video:
            print(f"Video duration: {video.duration:.2f} seconds")

            if video.audio is None:
                print("ERROR: No audio track found")
                return False

            print("Audio track found")
            print(f"Audio duration: {video.audio.duration:.2f} seconds")

            # Try different extraction methods
            print("\n--- Method 1: to_soundarray with default fps ---")
            try:
                audio_array = video.audio.to_soundarray()
                print(f"Audio array shape: {audio_array.shape}")
                print(f"Audio array dtype: {audio_array.dtype}")
                print("SUCCESS: to_soundarray() worked")
            except Exception as e:
                print(f"FAILED: to_soundarray() - {e}")
                traceback.print_exc()

            print("\n--- Method 2: to_soundarray with specified fps ---")
            try:
                audio_array = video.audio.to_soundarray(fps=16000)
                print(f"Audio array shape: {audio_array.shape}")
                print(f"Audio array dtype: {audio_array.dtype}")
                print("SUCCESS: to_soundarray(fps=16000) worked")

                # Test mono conversion
                if len(audio_array.shape) > 1:
                    print(f"Converting stereo to mono: {audio_array.shape} -> ", end="")
                    if audio_array.shape[1] > 1:
                        mono_array = np.mean(audio_array, axis=1)
                    else:
                        mono_array = audio_array.flatten()
                    print(f"{mono_array.shape}")
                    print("SUCCESS: Mono conversion worked")

            except Exception as e:
                print(f"FAILED: to_soundarray(fps=16000) - {e}")
                traceback.print_exc()

            print("\n--- Method 3: Using librosa directly ---")
            try:
                # Write temporary audio file and load with librosa
                temp_audio = "/tmp/test_audio.wav"
                video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                audio_data, sr = librosa.load(temp_audio, sr=16000)
                print(f"Librosa audio shape: {audio_data.shape}")
                print(f"Librosa sample rate: {sr}")
                print("SUCCESS: Librosa method worked")
            except Exception as e:
                print(f"FAILED: Librosa method - {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"FAILED: Video loading - {e}")
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    print("🧪 Debug Audio Extraction")
    print("=" * 40)
    test_single_video()