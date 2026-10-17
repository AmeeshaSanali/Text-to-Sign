#!/usr/bin/env python3
"""Convenient entry point to run the interactive Text-to-Gloss CLI."""

import os
import sys

# Add directory containing 'src' to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.cli import run_cli

if __name__ == "__main__":
    run_cli()
