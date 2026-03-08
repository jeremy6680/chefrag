# conftest.py
import sys
from pathlib import Path

# Add app/ to sys.path so tests can import agent, indexer, auth directly
sys.path.insert(0, str(Path(__file__).parent / "app"))