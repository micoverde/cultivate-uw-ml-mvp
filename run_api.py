#!/usr/bin/env python3
"""
FastAPI application entry point for Cultivate Learning ML API
"""
import uvicorn

if __name__ == "__main__":
    # Run the FastAPI application
    # The src.api.main module contains the FastAPI app instance
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
