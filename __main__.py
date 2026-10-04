"""Enables `python -m csvchat`."""
import os
import sys

try:
    from .csvchat import main
except ImportError:  # running the file directly: python csvchat/__main__.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from csvchat import main

sys.exit(main())
