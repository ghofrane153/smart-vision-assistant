#!/usr/bin/env bash
# Smart Vision Assistant - setup for Linux / macOS
# Run from the repository root: bash setup-linux.sh
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Creating virtual environment .venv ..."
    python3 -m venv .venv
fi

echo "Installing packages (the first time can take several minutes) ..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ "$(uname)" = "Linux" ] && ! command -v espeak-ng >/dev/null && ! command -v espeak >/dev/null; then
    echo "NOTE: text-to-speech needs espeak-ng on Linux: sudo apt install espeak-ng"
fi

echo
echo "Setup finished. Checking the environment ..."
.venv/bin/python scripts/00_check_environment.py
