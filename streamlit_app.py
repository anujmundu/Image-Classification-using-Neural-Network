# streamlit_app.py
"""Streamlit Community Cloud Root Entrypoint.

This script launches the 11-Architecture Neural Network Vision Benchmark Suite
directly from the repository root for one-click Streamlit Community Cloud deployment.
"""

import sys
import runpy
from pathlib import Path

# Add repository root to system path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Target the main Streamlit application
APP_PATH = REPO_ROOT / "demo" / "streamlit_demo.py"

if __name__ == "__main__":
    runpy.run_path(str(APP_PATH), run_name="__main__")
