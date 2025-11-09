!/usr/bin/env bash
#
# Simple wrapper to start/stop the Aim experiment-tracking UI.
# -----------------------------------------------------------------
# Usage:
#   ./aim_ui.sh up   [--repo /some/dir] [--port 43800]
#   ./aim_ui.sh down [--repo /some/dir] [--port 43800]
#
# A PID file is kept next to the repo:  <repo>/.aim_server.pid
# A log  file is kept next to the repo: <repo>/.aim_server.log
# -----------------------------------------------------------------

set -euo pipefail

# ----------------------------- defaults --------------------------
# AIM_REPO="/home/yb107/cvpr2025/DukeDiffSeg"
# AIM_REPO="/home/yb107/cvpr2025/aim_repo/dukediffseg"
AIM_REPO="/home/yb107/cvpr2025/aim_repo/dukediffseg6_0"
AIM_PORT=43800
ACTION=""
# -----------------------------------------------------------------

# ------------- tiny CLI parser (only --repo / --port) ------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    up|down) ACTION="$1"; shift ;;
    --repo)  AIM_REPO="$2"; shift 2 ;;
    --port)  AIM_PORT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 {up|down} [--repo DIR] [--port PORT]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$ACTION" ]]; then
  echo "Error: you must specify 'up' or 'down'"; exit 1
fi
# -----------------------------------------------------------------

# PIDFILE="$AIM_REPO/aim/.aim_server.pid"
# LOGFILE="$AIM_REPO/aim/.aim_server.log"
PIDFILE="$AIM_REPO/.aim_server.pid"
LOGFILE="$AIM_REPO/.aim_server.log"

mkdir -p "$AIM_REPO"

# ----------------------------- helpers ---------------------------
is_running() {
  # Return 0 (true) if a PID in $PIDFILE is alive
  [[ -f "$PIDFILE" ]] && ps -p "$(cat "$PIDFILE")" > /dev/null 2>&1
}

kill_pidfile() {
  if is_running; then
    PGID=$(ps -o pgid= -p "$(cat "$PIDFILE")" | tr -d ' ')
    echo "Stopping Aim UI (PGID $PGID)..."
    kill -9 "-$PGID" 2>/dev/null || true
    echo "Stopped."
  fi
  rm -f "$PIDFILE"
}
# -----------------------------------------------------------------

case "$ACTION" in
  up)
      # Is the port already taken?
      if lsof -i :"$AIM_PORT" >/dev/null 2>&1; then
        echo "⚠️  Aim UI (or something else) already listening on port $AIM_PORT."
        exit 0
      fi

      # Clean stale pidfile if needed
      if [[ -f "$PIDFILE" ]] && ! is_running; then
        rm -f "$PIDFILE"
      fi

      echo "Starting Aim UI on port $AIM_PORT ..."
      # Start in its own process-group so we can kill the whole tree later
      cd $AIM_REPO
      # setsid nohup pipenv run aim up --repo "aim/" --port "$AIM_PORT" \
      setsid nohup pipenv run aim up --repo "." --port "$AIM_PORT" \
            > "$LOGFILE" 2>&1 &
      echo $! > "$PIDFILE"

      echo "✅ Aim UI running:"
      echo "   URL   : http://localhost:$AIM_PORT"
      echo "   Log   : $LOGFILE"
      echo "   PID   : $(cat "$PIDFILE")"
      echo "To stop : $0 down --repo \"$AIM_REPO\" --port $AIM_PORT"
      ;;
  down)
      if [[ ! -f "$PIDFILE" ]]; then
        echo "No PID file at $PIDFILE — nothing to stop."
        exit 0
      fi
      kill_pidfile
      ;;
esac
