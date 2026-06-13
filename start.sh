#!/usr/bin/env bash
# start.sh — bring up the whole brain: daemon + web UI, and check Joplin.
#
# Launches the always-on brain daemon (with the webhook input adapter so the
# web UI's "add task" works) and the dashboard, then reports the Joplin Data
# API / Joplin Server status. Logs go to daemon.log / webui.log. Ctrl-C here
# stops the foreground tail; use `./start.sh stop` to kill the services.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
WEBHOOK_PORT="${WEBHOOK_PORT:-8765}"
UI_PORT="${UI_PORT:-8800}"
JOPLIN_API="${JOPLIN_API:-http://localhost:41184}"

stop() {
  echo "stopping brain services…"
  pkill -f "brain.daemon" 2>/dev/null || true
  pkill -f "brain.webui"  2>/dev/null || true
  echo "stopped."
}
[ "${1:-}" = "stop" ] && { stop; exit 0; }

# don't double-launch
pkill -f "brain.daemon" 2>/dev/null || true
pkill -f "brain.webui"  2>/dev/null || true
sleep 1

echo "🧠 starting brain daemon (webhook :$WEBHOOK_PORT)…"
nohup "$PY" -m brain.daemon --no-stdin --tick-seconds 2 --idle-rate 0 \
  --webhook-port "$WEBHOOK_PORT" >> daemon.log 2>&1 &
sleep 3

echo "🖥  starting web UI (http://127.0.0.1:$UI_PORT)…"
nohup "$PY" -m brain.webui --port "$UI_PORT" >> webui.log 2>&1 &
sleep 2

echo
echo "── status ───────────────────────────────────"
echo " daemon : $(pgrep -f brain.daemon >/dev/null && echo up || echo DOWN)"
echo " web UI : http://127.0.0.1:$UI_PORT"

# Joplin Data API (a Joplin client must expose it: desktop Web Clipper, or the
# headless terminal app). Notes published here sync to your Joplin Server.
if curl -s --max-time 3 "$JOPLIN_API/ping" 2>/dev/null | grep -q JoplinClipperServer; then
  echo " joplin : Data API up at $JOPLIN_API  ✓ (set joplin.enabled + JOPLIN_TOKEN to publish)"
else
  echo " joplin : Data API NOT reachable at $JOPLIN_API"
  echo "          → enable it: Joplin desktop → Settings → Web Clipper → Enable,"
  echo "            copy the token to JOPLIN_TOKEN; or run the headless terminal app."
  echo "          → for cross-device sync, stand up Joplin Server:"
  echo "            docker compose -f deploy/joplin-server.compose.yml up -d"
fi
echo "─────────────────────────────────────────────"
echo "tailing daemon.log (Ctrl-C to detach; services keep running)…"
echo
exec tail -n 5 -f daemon.log
