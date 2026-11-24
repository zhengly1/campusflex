#!/usr/bin/env bash
# Terminal demo: run on the server; no browser or local port forwarding required.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python
if [ "${1:-}" = '--help' ] || [ "${1:-}" = '-h' ]; then
  "$PY" -m campusflex fault-demo --help
  exit 0
fi
BROKER=''
cleanup() {
  if [ -n "$BROKER" ]; then
    kill "$BROKER" 2>/dev/null || true
    wait "$BROKER" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if ! (exec 3<>/dev/tcp/127.0.0.1/1883) 2>/dev/null; then
  MOSQ="${MOSQUITTO:-$HOME/mosquitto-env/sbin/mosquitto}"
  CONF="${MOSQUITTO_CONF:-$HOME/mosquitto.conf}"
  [ -x "$MOSQ" ] && [ -f "$CONF" ] || { echo 'Mosquitto is unavailable; see README.' >&2; exit 1; }
  mkdir -p demo_out
  "$MOSQ" -c "$CONF" >> demo_out/terminal_mosquitto.log 2>&1 &
  BROKER=$!
  for _ in {1..30}; do
    (exec 3<>/dev/tcp/127.0.0.1/1883) 2>/dev/null && break
    kill -0 "$BROKER" 2>/dev/null || { echo 'Mosquitto startup failed' >&2; exit 1; }
    sleep .1
  done
fi
"$PY" -u -m campusflex fault-demo "$@"
