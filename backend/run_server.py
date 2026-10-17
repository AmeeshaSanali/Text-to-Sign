#!/usr/bin/env python3
"""Convenient entry point to start the FastAPI translation server."""

import os
import sys
import uvicorn

# Add directory containing 'src' to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
