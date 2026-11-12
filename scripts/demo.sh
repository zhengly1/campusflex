#!/usr/bin/env bash
# End-to-end demo; writes only to demo_out/ (results/ stays the evidence of record).
#
#   scripts/demo.sh [--port 8501] [--no-dashboard]
#
#   1. IoT  – one evening of the planner over MQTT (falls back to the in-process bus, and says
#             so, when no broker is reachable)
#   2. LLM  – the live E1 C/D episode replayed from the recorded deepseek-flash replies (no API
#             call); with DEEPSEEK_API_KEY set, also one live call per role (llm-check, ~1 cent)
#   3. Smart devices – per-device dispatch, learned usage profiles, sensor fault detection
#   4. Dashboard – Streamlit on 127.0.0.1:PORT: the replayed LLM episode (Decision trace page)
#                  and the smart device results of step 3 (Smart devices page)
#
# Environment (see README): .venv with paho-mqtt and streamlit; Mosquitto at ~/mosquitto-env
# with ~/mosquitto.conf, started here when nothing listens on 1883 and stopped on exit.
set -uo pipefail
{
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python
PORT=8501 DASH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --no-dashboard) DASH=0; shift ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
OUT=demo_out
mkdir -p "$OUT"
MOSQ="${MOSQUITTO:-$HOME/mosquitto-env/sbin/mosquitto}"
CONF="${MOSQUITTO_CONF:-$HOME/mosquitto.conf}"
BROKER=""
if ! (exec 3<>/dev/tcp/127.0.0.1/1883) 2>/dev/null; then
  if [ -x "$MOSQ" ] && [ -f "$CONF" ]; then
    "$MOSQ" -c "$CONF" > "$OUT/mosquitto.log" 2>&1 &
    BROKER=$!
    sleep 1
    echo "started Mosquitto (pid $BROKER)"
  else
    echo "no MQTT broker on 1883 and none at $MOSQ: step 1 will use the in-process bus"
  fi
fi
cleanup() { [ -n "$BROKER" ] && kill "$BROKER" 2>/dev/null && wait "$BROKER" 2>/dev/null; BROKER=""; }   # wait: port 1883 is free on return
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM HUP PIPE
step() { echo; echo "== $1"; shift; "$@" || echo "!! step failed (exit $?)"; }

step "1/4 IoT: one evening over MQTT (fault scenario, seed 10, planner arm B)" \
  "$PY" -m campusflex run --scenario fault --seed 10 --policy arm_B --device mqtt --save "$OUT/episode_mqtt.json"
step "2/4 LLM: replay of the live E1 C/D episode (recorded deepseek-flash replies, no API call)" \
  "$PY" -m campusflex.dashboard.replay_llm_demo --out "$OUT/episode_llm.json"
if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  step "2b live LLM: one call per role" \
    "$PY" -m campusflex llm-check --repeats 1 --no-projection --out "$OUT/llm_check.json"
fi
smart() {          # full JSON in $OUT/smart.log; one line per arm here
  "$PY" -m campusflex smart --out-dir "$OUT/smart" > "$OUT/smart.log" || return
  "$PY" - "$OUT/smart/summary.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"{'arm':20} {'kWh':>6} {'thermal %':>9} {'CO2>1000 %':>10} {'manual adj.':>11}")
for a, x in s["eval"].items():
    d = x["discomfort_pct"]
    print(f"{a:20} {x['kwh']:6.1f} {d['thermal']:9.1f} {d['co2']:10.1f} {x['adjustments_per_evening']:11.1f}")
ad = s["anomaly_detection"]
print(f"sensor faults detected {ad['detected']}/{ad['faults']}, false alarms {ad['false_alarms']} "
      f"(simulated estimates; details on the dashboard's Smart devices page)")
PYEOF
}
step "3/4 Smart devices: IoT dispatch, learned profiles, sensor control" smart
[ -f "$OUT/smart/personas.md" ] && cat "$OUT/smart/personas.md"
if [ "$DASH" = 1 ]; then
  echo
  echo "== 4/4 Dashboard: http://127.0.0.1:$PORT  (from a laptop: ssh -L $PORT:127.0.0.1:$PORT $(whoami)@$(hostname))"
  echo "   Decision trace: $OUT/episode_llm.json (switch to $OUT/episode_mqtt.json in the sidebar)"
  echo "   Smart devices:  $OUT/smart; Ctrl-C stops"
  "$PY" -m streamlit run campusflex/dashboard/app.py --server.headless true --server.address 127.0.0.1 \
    --server.port "$PORT" --browser.gatherUsageStats false -- --episode "$OUT/episode_llm.json" \
    --smart-dir "$OUT/smart"
fi
exit 0
}
