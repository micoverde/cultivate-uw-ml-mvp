#!/usr/bin/env python3
"""
FastAPI Main Application for Cultivate Learning ML MVP

Provides API endpoints for educator transcript analysis and ML predictions.
Implements real-time analysis pipeline for stakeholder demo.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #46 - Story 2.1: Submit educator transcripts for analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

# Import routers
from .endpoints.transcript_analysis import router as transcript_router

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
    "https://zealous-dune-*.westus2.2.azurestaticapps.net",
    "https://zealous-mushroom-*.westus2.2.azurestaticapps.net",
    "https://*.azurestaticapps.net"
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(transcript_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cultivate Learning ML MVP API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "cultivate-ml-api",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )