#!/bin/bash
# Phase 4: replay one session with the meal-dependency care plan.
#   bash evaluation/replay_rev3/run_phase4.sh <label> <session id> <layer|nolayer>
# Memory off and declared enforcement, as in the reference corpus; the
# nolayer arm disables the validator (VALIDATOR_DISABLED=true).
cd "$(dirname "$0")/../.."
label=$1; session=$2; arm=$3
lines=$(wc -l < mock_data/789.txt)
if [ "$lines" -ne 40 ]; then
  echo "ABORT: mock_data/789.txt has $lines lines, expected 40"; exit 1
fi
disabled=false; [ "$arm" = "nolayer" ] && disabled=true
out=evaluation/replay_rev3/phase4/$arm/$label
CANONICALIZATION_CACHE=false ENFORCEMENT_MODE=declared VALIDATOR_DISABLED=$disabled \
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m evaluation.replay_runner \
  --session "$session" --care-plan-variant mock_data/variants/789_meal_dependencies.txt \
  --out "$out.json" --review-csv "$out.csv" > "$out.log" 2>&1
echo "exit $?"
.venv/Scripts/python.exe - "$out.json" <<'PY'
import json, sys, collections
t = json.load(open(sys.argv[1], encoding="utf8"))["transcripts"][0]
E = [json.loads(l) for l in open(t["structured_log"], encoding="utf8")]
c = collections.Counter(e.get("decision_type") for e in E)
print(f"{t['transcript']}: {t['outcome']}, error={t.get('error')}, "
      f"turns {t['turns_replayed']}/{t['turns_in_transcript']}, "
      f"{t['wall_seconds']} s, variant={t.get('care_plan_variant')}, "
      f"overrides {c.get('hard_enforcement', 0)}, "
      f"mock_data restored {t.get('mock_data_restored')}")
maps = collections.Counter((e["activity_name"], e["canonical_action"])
                           for e in E if e.get("decision_type") == "canonicalization")
print("  mappings:", dict(maps))
print("  log:", dict(c))
PY
echo "care plan lines after: $(wc -l < mock_data/789.txt)"
