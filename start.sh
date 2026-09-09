#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

echo "========================================================"
echo "         🤟 Voice2Gesture AI — Server Launcher          "
echo "========================================================"

# Detect local IP
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")

echo "Starting server on:"
echo "  👉 Local:   http://127.0.0.1:8000/"
echo "  👉 Network: http://${LOCAL_IP}:8000/ (for phones/tablets)"
echo "--------------------------------------------------------"

# Auto-open browser after a short delay
(sleep 1.5 && open "http://127.0.0.1:8000/") &

# Run Django development server
if [ -f "./venv/bin/python" ]; then
    ./venv/bin/python manage.py runserver 0.0.0.0:8000
else
    python3 manage.py runserver 0.0.0.0:8000
fi
