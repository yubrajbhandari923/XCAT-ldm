#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <SUFFIX>"
  echo "Examples:"
  echo "  $(basename "$0") _ldm"
  echo "  $(basename "$0") ldm   # same as _ldm"
}

# Require exactly one arg or show help
if [[ $# -ne 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi

SUFFIX="$1"
# Normalize: ensure suffix begins with "_"
if [[ "${SUFFIX}" != _* ]]; then
  SUFFIX="_${SUFFIX}"
fi

PIDFILE="/home/yb107/logs/train${SUFFIX}.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "❌ PID file not found: $PIDFILE"
    exit 1
fi

PID=$(cat $PIDFILE)
echo "Killing training process group with PGID = $PID ..."
kill -9 -$PID
rm -f "$PIDFILE"
echo "✅ Training process and subprocesses killed."