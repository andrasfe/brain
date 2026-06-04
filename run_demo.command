#!/bin/bash
# Double-clickable launcher for a live brain run (executes on macOS).
cd "$HOME/brain" || exit 1
echo "── installing deps (first run only) ─────────────────────────"
python3 -m pip install -q httpx python-dotenv PyYAML 2>/dev/null \
  || python3 -m pip install -q --user httpx python-dotenv PyYAML 2>/dev/null \
  || python3 -m pip install -q --break-system-packages httpx python-dotenv PyYAML
echo "── running the brain ────────────────────────────────────────"
python3 run.py --yes "Write a Python script primes.py that prints the first 15 prime numbers, run it, and report the exact output."
echo
echo "── done. press Return to close ──────────────────────────────"
read _
