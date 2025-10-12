#!/bin/bash

PIDFILE="/home/yb107/logs/train_.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "❌ PID file not found: $PIDFILE"
    exit 1
fi

PID=$(cat $PIDFILE)
echo "Killing training process group with PGID = $PID ..."
kill -9 -$PID
rm -f "$PIDFILE"
echo "✅ Training process and subprocesses killed."