#!/usr/bin/env python3
"""
Simple FastAPI runner for local testing and development
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)