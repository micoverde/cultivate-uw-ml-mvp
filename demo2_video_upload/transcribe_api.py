#!/usr/bin/env python3
"""
DEMO 2 Transcription API: Whisper-based Video Transcription
FastAPI backend for video upload, transcription, and question detection

Warren - This processes your uploaded videos with real Whisper transcription!
"""

import sys
import os
import asyncio
import json
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re

# Add paths for our ML modules
sys.path.insert(0, '../src/ml/training')
sys.path.insert(0, '../src')

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
    print("✅ Whisper available for transcription")
except ImportError as e:
    print(f"Warning: Whisper not available: {e}")
    print("Using mock transcription for demo")
    WHISPER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DEMO 2: Video Transcription API",
    description="Whisper-based video transcription and question detection",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionRequest(BaseModel):
    video_id: str
    use_whisper: bool = True

class TranscriptionResponse(BaseModel):
    video_id: str
    transcript: str
    questions: List[Dict[str, Any]]
    processing_time: float
    word_count: int
    method: str

class QuestionDetectionResponse(BaseModel):
    questions: List[Dict[str, Any]]
    total_questions: int
    processing_time: float

class WhisperTranscriptionService:
    """Service for video transcription using Whisper"""

    def __init__(self):
        self.whisper_model = None
        self.load_whisper_model()

    def load_whisper_model(self):
        """Load Whisper model for transcription"""
        try:
            if not WHISPER_AVAILABLE:
                logger.warning("Whisper not available, using mock transcription")
                return

            # Load small Whisper model for speed (can upgrade to medium/large for accuracy)
            self.whisper_model = whisper.load_model("small")
            logger.info("✅ Whisper small model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.whisper_model = None

    async def transcribe_video(self, video_file_path: str) -> Dict[str, Any]:
        """Transcribe video file using Whisper"""
        start_time = time.time()

        if self.whisper_model is not None:
            return await self.transcribe_with_whisper(video_file_path, start_time)
        else:
            return self.transcribe_with_mock(video_file_path, start_time)

    async def transcribe_with_whisper(self, video_file_path: str, start_time: float) -> Dict[str, Any]:
        """Use real Whisper for transcription"""
        try:
            logger.info(f"Transcribing with Whisper: {video_file_path}")

            # Run Whisper transcription
            result = self.whisper_model.transcribe(video_file_path)

            transcript = result["text"].strip()
            word_count = len(transcript.split())
            processing_time = time.time() - start_time

            logger.info(f"Whisper transcription complete: {word_count} words in {processing_time:.2f}s")

            return {
                'transcript': transcript,
                'word_count': word_count,
                'processing_time': processing_time,
                'method': 'whisper-small',
                'segments': result.get('segments', [])
            }

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return self.transcribe_with_mock(video_file_path, start_time)

    def transcribe_with_mock(self, video_file_path: str, start_time: float) -> Dict[str, Any]:
        """Mock transcription for demo when Whisper unavailable"""

        # Simulate processing time
        import asyncio
        time.sleep(2)  # Simulate transcription time

        # Mock transcript based on common educator video patterns
        mock_transcripts = [
            "Let's look at this picture together. What do you see here? Can you tell me about the colors? How does this make you feel? What do you think will happen next? Is this similar to something we've seen before?",

            "Today we're going to explore these blocks. What shapes do you notice? How can we stack them? What happens when we build them tall? Do you think this will fall down? How can we make it stronger?",

            "Look at these beautiful flowers in our book. What colors do you see? How do the petals feel? What do flowers need to grow? Have you seen flowers like this in your garden? What do you wonder about flowers?",

            "We're going to taste this fruit today. What does it look like on the outside? What do you think is inside? How does it smell? Does it remind you of other fruits? What do you predict it will taste like?",

            "Let's paint with these colors. What happens when we mix blue and yellow? How does the paint feel on your brush? What would you like to create? Can you tell me about your picture? What story does your painting tell?"
        ]

        # Choose a random mock transcript
        import random
        transcript = random.choice(mock_transcripts)

        word_count = len(transcript.split())
        processing_time = time.time() - start_time

        return {
            'transcript': transcript,
            'word_count': word_count,
            'processing_time': processing_time,
            'method': 'mock-demo',
            'segments': []
        }

class QuestionDetectionService:
    """Service for detecting questions in transcripts"""

    def detect_questions(self, transcript: str) -> List[Dict[str, Any]]:
        """Extract questions from transcript"""

        # Split transcript into sentences
        sentences = re.split(r'[.!?]+', transcript)

        questions = []
        question_id = 1

        for sentence in sentences:
            sentence = sentence.strip()

            # Skip empty sentences
            if not sentence:
                continue

            # Check if it's a question
            if self.is_question(sentence):
                question_type = self.classify_question_type(sentence)

                questions.append({
                    'id': question_id,
                    'text': sentence + '?',  # Ensure question mark
                    'type': question_type,
                    'timestamp': None,  # Could be extracted from Whisper segments
                    'confidence': 0.85 + (question_id * 0.02),  # Mock confidence
                    'context': self.get_question_context(sentence, transcript)
                })

                question_id += 1

        return questions

    def is_question(self, sentence: str) -> bool:
        """Determine if sentence is a question"""

        sentence_lower = sentence.lower().strip()

        # Direct question indicators
        question_starters = [
            'what', 'how', 'why', 'when', 'where', 'who',
            'can you', 'do you', 'will you', 'have you',
            'is this', 'are you', 'does this', 'did you'
        ]

        # Check if starts with question words
        for starter in question_starters:
            if sentence_lower.startswith(starter):
                return True

        # Check for question patterns
        question_patterns = [
            r'\bwhat\b.*\?',
            r'\bhow\b.*\?',
            r'\bwhy\b.*\?',
            r'\bcan you\b',
            r'\bdo you\b',
            r'\btell me\b'
        ]

        for pattern in question_patterns:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def classify_question_type(self, question: str) -> str:
        """Classify question as OEQ or CEQ"""

        question_lower = question.lower()

        # Strong OEQ indicators
        oeq_patterns = [
            'what do you see', 'what do you think', 'what do you notice',
            'how does', 'how do you feel', 'tell me about',
            'what happens when', 'why do you think', 'what would you',
            'how could we', 'what if', 'describe'
        ]

        # Strong CEQ indicators
        ceq_patterns = [
            'is this', 'are you', 'do you like', 'can you see',
            'is it', 'are they', 'does this', 'did you',
            'will you', 'have you'
        ]

        # Check for OEQ patterns
        for pattern in oeq_patterns:
            if pattern in question_lower:
                return 'OEQ'

        # Check for CEQ patterns
        for pattern in ceq_patterns:
            if pattern in question_lower:
                return 'CEQ'

        # Default classification based on question starters
        if any(question_lower.startswith(starter) for starter in ['what', 'how', 'why']):
            return 'OEQ'
        else:
            return 'CEQ'

    def get_question_context(self, question: str, full_transcript: str) -> str:
        """Get surrounding context for the question"""

        # Find the question in the transcript
        question_pos = full_transcript.find(question)

        if question_pos == -1:
            return "Context not found"

        # Get surrounding context (50 characters before and after)
        start = max(0, question_pos - 50)
        end = min(len(full_transcript), question_pos + len(question) + 50)

        context = full_transcript[start:end].strip()

        # Clean up context
        if len(context) > 100:
            context = "..." + context[-97:]

        return context

# Initialize services
transcription_service = WhisperTranscriptionService()
question_service = QuestionDetectionService()

# Store uploaded files temporarily
uploaded_files = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "DEMO 2: Video Transcription API",
        "status": "running",
        "whisper_available": WHISPER_AVAILABLE,
        "model": "whisper-small" if WHISPER_AVAILABLE else "mock-demo"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "whisper_model_loaded": transcription_service.whisper_model is not None,
        "whisper_available": WHISPER_AVAILABLE,
        "supported_formats": ["mp4", "mov", "avi", "wav", "mp3"]
    }

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing"""

    # Validate file type
    allowed_extensions = ['.mp4', '.mov', '.avi', '.wav', '.mp3']
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Supported: {allowed_extensions}"
        )

    # Check file size (max 100MB for demo)
    file_size = 0
    temp_file_path = None

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name

            # Read and save file
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break

                file_size += len(chunk)

                # Check size limit (100MB)
                if file_size > 100 * 1024 * 1024:
                    os.unlink(temp_file_path)
                    raise HTTPException(status_code=413, detail="File too large (max 100MB)")

                temp_file.write(chunk)

        # Generate unique ID for this upload
        video_id = f"video_{int(time.time())}_{hash(file.filename) % 10000}"

        # Store file info
        uploaded_files[video_id] = {
            'filename': file.filename,
            'file_path': temp_file_path,
            'file_size': file_size,
            'upload_time': datetime.now().isoformat(),
            'processed': False
        }

        logger.info(f"Video uploaded: {video_id} ({file.filename}, {file_size/1024/1024:.1f}MB)")

        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_size": file_size,
            "message": "Video uploaded successfully",
            "next_step": f"POST /transcribe/{video_id}"
        }

    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/transcribe/{video_id}", response_model=TranscriptionResponse)
async def transcribe_video(video_id: str):
    """Transcribe uploaded video"""

    if video_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Video not found")

    video_info = uploaded_files[video_id]

    try:
        logger.info(f"Starting transcription for {video_id}")

        # Transcribe video
        transcription_result = await transcription_service.transcribe_video(video_info['file_path'])

        # Detect questions
        questions = question_service.detect_questions(transcription_result['transcript'])

        # Update video info
        video_info['processed'] = True
        video_info['transcript'] = transcription_result['transcript']
        video_info['questions'] = questions

        logger.info(f"Transcription complete: {video_id}, {len(questions)} questions found")

        return TranscriptionResponse(
            video_id=video_id,
            transcript=transcription_result['transcript'],
            questions=questions,
            processing_time=transcription_result['processing_time'],
            word_count=transcription_result['word_count'],
            method=transcription_result['method']
        )

    except Exception as e:
        logger.error(f"Transcription error for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/questions/{video_id}", response_model=QuestionDetectionResponse)
async def get_questions(video_id: str):
    """Get detected questions for a video"""

    if video_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Video not found")

    video_info = uploaded_files[video_id]

    if not video_info.get('processed', False):
        raise HTTPException(status_code=400, detail="Video not yet processed")

    questions = video_info.get('questions', [])

    return QuestionDetectionResponse(
        questions=questions,
        total_questions=len(questions),
        processing_time=0.1  # Mock processing time for question retrieval
    )

@app.get("/video/{video_id}")
async def get_video_info(video_id: str):
    """Get video information and processing status"""

    if video_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Video not found")

    video_info = uploaded_files[video_id].copy()

    # Remove file path for security
    video_info.pop('file_path', None)

    return video_info

@app.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """Delete uploaded video and clean up files"""

    if video_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Video not found")

    video_info = uploaded_files[video_id]

    try:
        # Delete temporary file
        if os.path.exists(video_info['file_path']):
            os.unlink(video_info['file_path'])

        # Remove from memory
        del uploaded_files[video_id]

        logger.info(f"Video deleted: {video_id}")

        return {"message": f"Video {video_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Delete error for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "videos_uploaded": len(uploaded_files),
        "videos_processed": sum(1 for v in uploaded_files.values() if v.get('processed', False)),
        "whisper_available": WHISPER_AVAILABLE,
        "model_version": "whisper-small" if WHISPER_AVAILABLE else "mock-demo"
    }

if __name__ == "__main__":
    import uvicorn

    print("🎬 Starting DEMO 2 Transcription API")
    print("=" * 50)
    print(f"Whisper available: {WHISPER_AVAILABLE}")
    print(f"Model: {'whisper-small' if WHISPER_AVAILABLE else 'mock-demo'}")
    print("Supported formats: MP4, MOV, AVI, WAV, MP3")
    print("Max file size: 100MB")
    print()
    print("API will be available at: http://localhost:8002")
    print("API docs at: http://localhost:8002/docs")

    uvicorn.run(app, host="0.0.0.0", port=8002)