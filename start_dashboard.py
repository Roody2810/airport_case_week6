"""
Start het dashboard zonder iets in de terminal te typen:
open dit bestand in VS Code en klik op 'Run Python File' (het driehoekje rechtsboven).
Het dashboard opent daarna in je browser op http://localhost:8501
"""
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=Path(__file__).parent)
