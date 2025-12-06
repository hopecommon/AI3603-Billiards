#!/usr/bin/env bash
set -euo pipefail

CONCURRENCY="${CONCURRENCY:-5}"
GAMES="${GAMES:-4}"
SEED_BASE="${SEED_BASE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/../evaluate.py"
RUN_LOG_DIR="$SCRIPT_DIR/../parallel_runs"
mkdir -p "$RUN_LOG_DIR"

declare -a pids=()
declare -a run_logs=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup EXIT INT TERM
for i in $(seq 1 "$CONCURRENCY"); do
  seed=$((SEED_BASE + i))
  export PYTHONHASHSEED=$seed
  run_log="$RUN_LOG_DIR/run-${i}-$(date +%Y%m%d-%H%M%S).log"
  echo "[parallel] start job $i -> $run_log"
  (
    cd "$SCRIPT_DIR/.." &&
    PYTHONUNBUFFERED=1 ./.venv/bin/python "$EVAL_SCRIPT" --games "$GAMES"
  ) >"$run_log" 2>&1 &
  pids+=($!)
  run_logs+=("$run_log")
done

fail=false
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=true
  fi
done

if [[ ${#run_logs[@]} -eq 0 ]]; then
  echo "[parallel] warning: 没有检测到 run log。"
  exit 0
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python - <<'PY' "$REPO_ROOT" "${run_logs[@]}"
import sys
from collections import Counter, defaultdict
repo_root = sys.argv[1]
logs = sys.argv[2:]
sys.path.insert(0, repo_root)
from eval.analyze_log import parse_log, build_summary  # type: ignore

agg_games = 0
agg_hits = []
agg_wins = Counter()
agg_shots = Counter()
agg_events = defaultdict(Counter)

for log in logs:
    stats = parse_log(log)
    summary = build_summary(stats, log)
    agg_games += summary['total_games']
    agg_hits.extend(summary['hit_counts'])
    agg_wins.update(summary['winners'])
    agg_shots.update(summary['shots_by_agent'])
    for event, counts in summary['events'].items():
        for agent, val in counts.items():
            agg_events[event][agent] += val

print(f"[parallel] 新日志数量: {len(logs)}")
print(f"[parallel] 总对局数: {agg_games}")
if agg_games:
    avg_hits = sum(agg_hits) / agg_games
    print(f"[parallel] 平均击球数: {avg_hits:.2f} (最少 {min(agg_hits)}, 最多 {max(agg_hits)})")
if agg_wins:
    print("[parallel] 胜场统计:")
    for agent, cnt in agg_wins.items():
        print(f"  - {agent}: {cnt}")
if agg_shots:
    print("[parallel] 出杆统计:")
    for agent, cnt in agg_shots.items():
        print(f"  - {agent}: {cnt}")
if agg_events:
    print("[parallel] 事件统计:")
    for event, counts in agg_events.items():
        agents = " ".join(f"{agent}:{cnt}" for agent, cnt in counts.items())
        print(f"  * {event}: {agents}")
PY "$REPO_ROOT" "${new_logs[@]}"

if $fail; then
  exit 1
fi
