#!/bin/bash
# Five sequential runs of the structured, temperature-0 verifier baseline for
# one model, each annotated with the serving provider when on OpenRouter.
#   bash evaluation/baseline_rev3/run_model.sh <provider> <model> <tag> [free]
# With "free" the JSON schema is not imposed (temperature 0 still applies).
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
prov=$1; model=$2; tag=$3
flag="--structured"; [ "$4" = "free" ] && flag=""
for i in 1 2 3 4 5; do
  out=evaluation/baseline_rev3/${tag}_run$i.json
  $PY -m evaluation.llm_verifier_baseline --provider $prov --model $model \
    --temperature 0 $flag --out $out > ${out%.json}.log 2>&1
  grep -E 'accuracy:|Parse errors' ${out%.json}.log
  if [ "$prov" = "openrouter" ]; then
    sleep 5; $PY -m evaluation.openrouter_generation_info $out
  fi
done
echo "DONE $tag"
