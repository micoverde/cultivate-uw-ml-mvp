#!/usr/bin/env python3
"""
FastAPI Main Application for Cultivate Learning ML MVP

Provides API endpoints for educator transcript analysis and ML predictions.
Implements real-time analysis pipeline for stakeholder demo.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #46 - Story 2.1: Submit educator transcripts for analysis
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
import json
from typing import Dict, List
from datetime import datetime

# Import routers
from .endpoints.transcript_analysis import router as transcript_router
from .endpoints.educator_response_analysis import (
    educator_response_service,
    EducatorResponseRequest,
    EducatorResponseAnalysisResult
)
from .endpoints.admin import router as admin_router
from .endpoints.video_analysis import router as video_router
from .endpoints.model_management import router as model_router

# Import security middleware
from .security.middleware import SecurityMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting Cultivate Learning ML MVP API")
    logger.info("Loading ML models and initializing services...")

    # TODO: Initialize ML models here
    # await load_ml_models()

    yield

    # Shutdown
    logger.info("🛑 Shutting down API")

# Create FastAPI application
app = FastAPI(
    title="Cultivate Learning ML MVP",
    description="AI-powered educator interaction analysis API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Configure CORS for both development and production
import os

# Production origins for Azure Static Web Apps and custom domains
production_origins = [
    "https://cultivate-ml-demo.azurewebsites.net",
    "https://app.cultivate-learning.com",
    "https://demo.cultivate-learning.com",
    "https://zealous-dune-*.westus2.2.azurestaticapps.net",
    "https://zealous-mushroom-*.westus2.2.azurestaticapps.net",
    "https://*.azurestaticapps.net",
    "https://cultivate-frontend-prod.azurestaticapps.net"
]

# Development origins
development_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:3001"  # Vite alternate port
]

# Combine origins based on environment
allowed_origins = development_origins
if os.getenv("ENVIRONMENT") == "production":
    allowed_origins = production_origins + development_origins
else:
    # For development/staging, allow all localhost and azurestaticapps
    allowed_origins = development_origins + production_origins

# Add security middleware first (before CORS)
app.add_middleware(SecurityMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

# Include routers
app.include_router(transcript_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(video_router, prefix="/api/v1")
app.include_router(model_router)  # Already has /api/v1/models prefix

# Educator Response Analysis Endpoints (PIVOT for MVP Sprint 1)
@app.post("/api/analyze/educator-response", response_model=dict)
async def submit_educator_response(request: EducatorResponseRequest):
    """
    Submit educator response for AI coaching analysis.

    PIVOT: Core endpoint for MVP Sprint 1 demo script requirements.
    Users type responses to scenarios and receive structured coaching feedback.
    """
    try:
        analysis_id = await educator_response_service.analyze_educator_response(request)
        return {"analysis_id": analysis_id, "status": "submitted"}
    except Exception as e:
        logger.error(f"Failed to submit educator response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/educator-response/status/{analysis_id}")
async def get_educator_response_status(analysis_id: str):
    """Get status of educator response analysis."""
    try:
        status = educator_response_service.get_analysis_status(analysis_id)
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/educator-response/results/{analysis_id}", response_model=EducatorResponseAnalysisResult)
async def get_educator_response_results(analysis_id: str):
    """Get completed educator response analysis results."""
    try:
        results = educator_response_service.get_analysis_results(analysis_id)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cultivate Learning ML MVP API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }

# WebSocket connection manager for real-time features (Milestone #106)
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[websocket] = {
            "session_id": session_id,
            "connected_at": datetime.now()
        }
        # Security: Sanitize session_id for logging to prevent log injection
        safe_session_id = session_id.replace('\n', '').replace('\r', '').replace('\t', ' ')[:50] if session_id else 'unknown'
        logger.info(f"WebSocket connected: {safe_session_id}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            session_data = self.connection_data.pop(websocket, {})
            logger.info(f"WebSocket disconnected: {session_data.get('session_id')}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/realtime/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time analysis updates"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types for real-time demo
            if message.get("type") == "transcript_chunk":
                # Send back real-time processing update
                response = {
                    "type": "processing_update",
                    "session_id": session_id,
                    "status": "analyzing",
                    "chunk_processed": message.get("chunk", ""),
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_personal_message(json.dumps(response), websocket)

            elif message.get("type") == "demo_event":
                # Handle demo interaction events
                response = {
                    "type": "demo_response",
                    "session_id": session_id,
                    "event": message.get("event"),
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_personal_message(json.dumps(response), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/video/{video_id}")
async def video_websocket_endpoint(websocket: WebSocket, video_id: str):
    """WebSocket endpoint for real-time video processing updates"""
    await manager.connect(websocket, video_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle video processing status requests
            if message.get("type") == "status_request":
                # Import video processing status from video_analysis module
                from .endpoints.video_analysis import video_processing_status

                if video_id in video_processing_status:
                    status_data = video_processing_status[video_id]
                    response = {
                        "type": "video_status_update",
                        "video_id": video_id,
                        "status": status_data["status"],
                        "progress": status_data["progress_percentage"],
                        "message": status_data["message"],
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    response = {
                        "type": "video_status_error",
                        "video_id": video_id,
                        "error": "Video not found",
                        "timestamp": datetime.now().isoformat()
                    }

                await manager.send_personal_message(json.dumps(response), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "cultivate-ml-api",
        "version": "1.0.0",
        "features": {
            "websockets": True,
            "real_time_analysis": True,
            "enhanced_demo": True
        },
        "active_connections": len(manager.active_connections)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )