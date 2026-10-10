#!/bin/bash
# Replay one session of the corpus in the revision-3 configuration.
#   bash evaluation/replay_rev3/run_session.sh <label> <session id> <off|on>
# Refuses to start if the care plan has been altered (see the
# care-plan contamination note in revision-plan.md).
cd "$(dirname "$0")/../.."
label=$1; session=$2; cache=$3
lines=$(wc -l < mock_data/789.txt)
if [ "$lines" -ne 40 ]; then
  echo "ABORT: mock_data/789.txt has $lines lines, expected 40"; exit 1
fi
flag=false; [ "$cache" = "on" ] && flag=true
out=evaluation/replay_rev3/cache_$cache/$label
CANONICALIZATION_CACHE=$flag ENFORCEMENT_MODE=declared PYTHONIOENCODING=utf-8 \
  .venv/Scripts/python.exe -m evaluation.replay_runner --session "$session" \
  --out "$out.json" --review-csv "$out.csv" > "$out.log" 2>&1
echo "exit $?"
.venv/Scripts/python.exe - "$out.json" <<'PY'
import json, sys, collections
t = json.load(open(sys.argv[1], encoding="utf8"))["transcripts"][0]
E = [json.loads(l) for l in open(t["structured_log"], encoding="utf8")]
c = collections.Counter(e.get("decision_type") for e in E)
print(f"{t['transcript']}: {t['outcome']}, error={t.get('error')}, "
      f"turns {t['turns_replayed']}/{t['turns_in_transcript']}, "
      f"{t['wall_seconds']} s, canonicalizations {t['canonicalization_count']} "
      f"(unknown {t['canonicalization_unknown']}), "
      f"goal planner {t.get('goal_planner_calls')}, "
      f"overrides {c.get('hard_enforcement', 0)} "
      f"(model-chosen wait/recovery {len(t['hard_enforcements'])}), "
      f"mock_data restored {t.get('mock_data_restored')}")
print("  log:", dict(c))
PY
echo "care plan lines after: $(wc -l < mock_data/789.txt)"
