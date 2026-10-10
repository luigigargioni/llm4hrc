#!/bin/bash
# Repeats the LLM-verifier baseline 4 more times per model (run 1 = existing files).
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run_model() { prov=$1; model=$2; tag=$3
  for i in 2 3 4 5; do
    $PY -m evaluation.llm_verifier_baseline --provider $prov --model $model \
      --out evaluation/baseline_reps/${tag}_run$i.json > evaluation/baseline_reps/${tag}_run$i.log 2>&1
  done; }
run_model openai gpt-5.4-mini gpt54mini &
run_model openrouter openai/gpt-oss-120b gptoss120b &
run_model openrouter qwen/qwen3-30b-a3b-instruct-2507 qwen3-30b &
run_model openrouter mistralai/mistral-small-3.2-24b-instruct mistral-small &
run_model openrouter meta-llama/llama-3.3-70b-instruct llama33-70b &
wait; echo ALL DONE
