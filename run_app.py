import subprocess
import sys
import os

BASE = os.path.dirname(sys.executable)
streamlit = os.path.join(BASE, "Scripts", "streamlit.exe")

subprocess.run([
    streamlit, "run", "app.py",
    "--server.headless=true",
    "--server.port=8501"
])
