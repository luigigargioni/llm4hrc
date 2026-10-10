"""Benchmark candidate models on the graph's most expensive single call.

A replayed session spends most of its wall clock inside task_progress: the
node regenerates the whole task on every cycle, and the answer is a large
structured object, so its latency and output cost dominate. Comparing
models on that exact call — the node's own prompts, task, and tool schema —
says more than a generic benchmark, and costs one request per model.

Usage::

    python -m evaluation.node_model_probe \
        --models openai/gpt-oss-120b,openai/gpt-oss-20b
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    ),
    override=True,
)

import graph.graph_init  # noqa: F401,E402 - breaks the node import cycle
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402

import nodes.task_progress_node as task_progress  # noqa: E402
from utils.models import define_model  # noqa: E402


def load_recorded_task() -> dict:
    """Rebuild a realistic task from the most recent replay log."""
    logs = glob.glob("log/*.txt")
    if not logs:
        raise SystemExit("nessun log disponibile: esegui prima un replay")
    latest = max(logs, key=os.path.getmtime)
    lines = open(latest, encoding="utf-8", errors="replace").read().splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "next_steps_node - scheduled_activities":
            continue
        for candidate in lines[index + 1:index + 6]:
            if candidate.strip().startswith("["):
                activities = json.loads(candidate.strip())
                return {
                    "patient_name": "Anna",
                    "patient_surname": "Test",
                    "patient_information": "Care-home patient.",
                    "activities": activities,
                }
    raise SystemExit("nessun task trovato nel log piu recente")


def probe(model_id: str, task: dict, with_tool: bool) -> dict:
    """Issue the task_progress call against one model and time it."""
    system = PromptTemplate.from_template(
        task_progress.INITIAL_PROMPT
    ).invoke({}).to_string()
    human = PromptTemplate.from_template(
        task_progress.NEW_REQUEST_PROMPT
    ).invoke({
        "task": task,
        "situation_perception": "Patient is speaking",
        "patient_request": "I would like to read a book now",
        "vocal_answer": "Of course, Anna.",
        "robot_action": "SPEAKING",
    }).to_string()

    template = task_progress.response_template if with_tool else None
    model = define_model(template, provider="openrouter", model=model_id)

    started = time.perf_counter()
    try:
        response = model.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
    except Exception as exc:  # noqa: BLE001 - a rejection is a result
        return {
            "model": model_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:160],
            "seconds": round(time.perf_counter() - started, 1),
        }
    elapsed = time.perf_counter() - started

    usage = response.usage_metadata or {}
    details = usage.get("output_token_details") or {}
    payload = response.to_json()["kwargs"]
    tool_calls = payload.get("tool_calls") or []

    activities = None
    if tool_calls:
        args = tool_calls[0].get("args") or {}
        updated = args.get("updated_task")
        if isinstance(updated, dict):
            activities = len(updated.get("activities") or [])

    return {
        "model": model_id,
        "ok": True,
        "seconds": round(elapsed, 1),
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "reasoning": int(details.get("reasoning") or 0),
        "tool_call": bool(tool_calls),
        "activities_returned": activities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True,
                        help="comma-separated OpenRouter model ids")
    parser.add_argument("--no-tool", action="store_true",
                        help="probe without binding the response schema")
    parser.add_argument("--out", default="evaluation/node_model_probe.json")
    args = parser.parse_args()

    task = load_recorded_task()
    print(f"task di prova: {len(task['activities'])} attivita\n")

    header = (f"{'modello':42} {'ok':>3} {'sec':>6} {'in':>7} {'out':>7} "
              f"{'reas':>6} {'tool':>5} {'att':>4}")
    print(header)
    results = []
    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        result = probe(model_id, task, with_tool=not args.no_tool)
        results.append(result)
        if result["ok"]:
            print(f"{result['model']:42} {'si':>3} {result['seconds']:6} "
                  f"{result['input']:7} {result['output']:7} "
                  f"{result['reasoning']:6} "
                  f"{str(result['tool_call']):>5} "
                  f"{str(result['activities_returned']):>4}")
        else:
            print(f"{result['model']:42} {'no':>3} {result['seconds']:6}   "
                  f"{result['error']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"task_activities": len(task["activities"]),
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nrisultati: {args.out}")


if __name__ == "__main__":
    main()
