#!/bin/bash
# NBA 3PT Dashboard — launcher
# First run: creates a local .venv and installs dependencies automatically.
# Every run after that: just launches the dashboard.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV" ]; then
  echo "Setting up virtual environment..."
  python3 -m venv "$VENV"
fi

# Install / upgrade dependencies
echo "Checking dependencies..."
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$DIR/requirements.txt"

# Launch the dashboard
echo "Starting dashboard..."
"$VENV/bin/streamlit" run "$DIR/app.py"
