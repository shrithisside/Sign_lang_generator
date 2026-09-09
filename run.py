#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import webbrowser
import threading

def open_browser():
    time.sleep(1.5)
    print("\n🌐 Opening website in your default browser: http://127.0.0.1:8000/")
    webbrowser.open("http://127.0.0.1:8000/")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # Use virtual environment python if present
    venv_python = os.path.join(base_dir, 'venv', 'bin', 'python')
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    threading.Thread(target=open_browser, daemon=True).start()

    print("=" * 58)
    print("        🤟 Voice2Gesture AI — Live Server Runner          ")
    print("=" * 58)
    print("Starting Django server on http://127.0.0.1:8000/")
    print("Press Ctrl+C to stop the server.\n")

    cmd = [python_bin, 'manage.py', 'runserver', '0.0.0.0:8000']
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")
