#!/usr/bin/env bash
# GoChiDUBB Studio — Server Launcher
# ====================================
# Uses the server manager for clean process lifecycle.
# Run: ./start.sh
# Or:  python tools/gochidubb_serverctl.py start
set -euo pipefail
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Load .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Start Ollama if not already running
if ! curl -s http://localhost:1234/v1/api/tags &>/dev/null; then
    if command -v ollama &>/dev/null; then
        echo "Starting Ollama..."
        ollama serve &>/dev/null &
        sleep 2
    fi
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  GoChiDUBB Studio                    ║"
echo "║  http://localhost:8910               ║"
echo "║                                      ║"
echo "║  Manage with:                        ║"
echo "║    python tools/gochidubb_serverctl.py status   ║"
echo "║    python tools/gochidubb_serverctl.py stop     ║"
echo "║    python tools/gochidubb_serverctl.py logs -f  ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Use the server manager to start in foreground
exec python tools/gochidubb_serverctl.py foreground