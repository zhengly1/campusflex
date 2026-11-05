#!/usr/bin/env bash
# Ledger items 5-8 with the live LLM, in README order. Stops at the first step that exits
# non-zero or records a not_run / aborted status (missing key, account error such as HTTP 402).
#
#   export DEEPSEEK_API_KEY=...      # read from the environment only; this script never stores it
#   scripts/run_live.sh [--reduced] [--from STEP] [--prices IN HIT OUT]
#
# Steps: llm_check e1_CD e5_live t1 e2 e3 e4 e6_review e6_eval tables ledger.
# The freeze gate is NOT re-run: the live runs must use results/frozen_config.json as it is
# (llm-check warns if the prompts or model changed since the freeze). --from resumes after a
# fix (e.g. a top-up); the resumed step runs again from its start (E1 C/D, E3, E5 live and
# llm-check are uncached by design, so their calls are made and billed again).
# --prices passes USD per 1M tokens (input miss, input cache hit, output) to llm-check.
# Logs: results/live_logs/<step>.log and progress.txt.
set -uo pipefail
{   # the whole script is read before it runs, so editing the file mid-run cannot derail it
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
REDUCED=() FROM="" PRICES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --reduced) REDUCED=(--reduced); shift ;;
    --from) FROM="${2:?--from needs a step name}"; shift 2 ;;
    --prices) PRICES=(--price-in "${2:?}" --price-cache-hit "${3:?}" --price-out "${4:?}"); shift 4 ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

STEPS=(llm_check e1_CD e5_live t1 e2 e3 e4 e6_review e6_eval tables ledger)
if [ -n "$FROM" ] && [[ " ${STEPS[*]} " != *" $FROM "* ]]; then
  echo "--from must be one of: ${STEPS[*]}" >&2; exit 2
fi

cd "$ROOT" || exit 1
case "${CAMPUSFLEX_PROVIDER:-deepseek}" in
  deepseek)  KEYVAR=DEEPSEEK_API_KEY ;;
  anthropic) KEYVAR=ANTHROPIC_API_KEY ;;
  *) echo "CAMPUSFLEX_PROVIDER must be deepseek or anthropic, got ${CAMPUSFLEX_PROVIDER}" >&2; exit 2 ;;
esac
if [ -z "${!KEYVAR:-}" ]; then
  echo "$KEYVAR is not set (export it in this shell; do not write it into a file in the repo)" >&2
  exit 2
fi

LOGS="results/live_logs"
mkdir -p "$LOGS"

cmd_for() {
  case "$1" in
    llm_check) echo "$PY -m campusflex llm-check ${PRICES[*]:-}" ;;
    e1_CD)     echo "$PY -m campusflex e1 --arms CD ${REDUCED[*]:-}" ;;
    e5_live)   echo "$PY -m campusflex e5 --live" ;;
    t1)        echo "$PY -m campusflex t1" ;;
    e2)        echo "$PY -m campusflex e2 --llm" ;;
    e3)        echo "$PY -m campusflex e3 --llm" ;;
    e4)        echo "$PY -m campusflex e4 --llm" ;;
    e6_review) echo "$PY -m campusflex e6 review --kind llm" ;;
    e6_eval)   echo "$PY -m campusflex e6 evaluate" ;;
    tables)    echo "$PY -m campusflex tables" ;;
    ledger)    echo "$PY -m campusflex ledger" ;;
  esac
}

files_for() {
  case "$1" in
    llm_check) echo results/llm_check.json ;;
    e1_CD)     echo results/e1_CD.json ;;
    e5_live)   echo results/e5_live.json ;;
    t1)        echo results/trace_T1.json ;;
    e2)        echo results/e2.json ;;
    e3)        echo results/e3.json ;;
    e4)        echo results/e4.json ;;
    e6_review) echo results/l5_history.json ;;
    e6_eval)   echo results/e6.json ;;
  esac
}

# Non-zero (and the reasons) when a result file records a run that did not happen.
check_files() {
  [ $# -eq 0 ] && return 0
  "$PY" - "$@" <<'EOF'
import json, sys
bad = []
def walk(o, f, path):
    if isinstance(o, dict):
        st = o.get("status")
        if st in ("not_run", "aborted") or (path == "" and f.endswith("llm_check.json") and st != "ready"):
            why = o.get("reason") or "; ".join(map(str, o.get("warnings") or []))
            bad.append(f"{f}{path or '/'}: {st} ({str(why)[:400]})")
        for k, v in o.items():
            if k not in ("records", "texts", "rows", "episodes", "history", "steps", "decisions"):
                walk(v, f, f"{path}/{k}")
for f in sys.argv[1:]:
    try:
        walk(json.load(open(f)), f, "")
    except (OSError, ValueError) as e:
        bad.append(f"{f}: unreadable ({e})")
print("\n".join(bad))
sys.exit(1 if bad else 0)
EOF
}

started=0
[ -z "$FROM" ] && started=1
for s in "${STEPS[@]}"; do
  [ "$s" = "$FROM" ] && started=1
  [ "$started" = 1 ] || continue
  c="$(cmd_for "$s")"
  t0=$(date +%s)
  echo "[$(date '+%F %T')] $s: $c"
  # shellcheck disable=SC2086
  $c > "$LOGS/$s.log" 2>&1
  rc=$?
  echo "$s rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOGS/progress.txt"
  # shellcheck disable=SC2046
  why="$(check_files $(files_for "$s"))"; chk=$?
  # a command that writes nothing when it cannot run prints its not_run status instead
  if [ $chk -eq 0 ] && grep -qE '"status": "(not_run|aborted)"| not run: ' "$LOGS/$s.log"; then
    why="$LOGS/$s.log: $(grep -m1 -E '"reason"| not run: ' "$LOGS/$s.log")"; chk=1
  fi
  if [ $rc -ne 0 ] || [ $chk -ne 0 ]; then
    echo "stopped at $s (exit $rc). See $LOGS/$s.log" >&2
    [ -n "$why" ] && echo "$why" >&2
    echo "fix the cause, then resume with: scripts/run_live.sh --from $s" >&2
    exit 1
  fi
done
echo "all steps done; see results/evidence_ledger.md"
exit 0
}
