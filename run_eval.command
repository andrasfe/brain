#!/bin/bash
# Double-clickable launcher for the MAP-paper comparison eval (runs on macOS).
cd "$HOME/brain" || exit 1
echo "── ensuring deps ────────────────────────────────────────────"
python3 -m pip install -q httpx python-dotenv PyYAML 2>/dev/null \
  || python3 -m pip install -q --user httpx python-dotenv PyYAML 2>/dev/null \
  || python3 -m pip install -q --break-system-packages httpx python-dotenv PyYAML
echo "── running eval: brain (multi-agent) vs qwen-alone ──────────"
echo "   (Tower of Hanoi + graph traversal, 10 instances each)"
python3 -m eval.run_eval --n 1 --disks 3
echo
echo "── done. results in eval/results/ — press Return to close ───"
read _
