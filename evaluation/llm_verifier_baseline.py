"""LLM-as-verifier baseline for comparison with the symbolic layer.

For each scenario in the scenario suite, this module asks an LLM to
verify the executability of each activity given the world state and
the canonical action's preconditions. The LLM's output is compared
against the same ground truth used by the symbolic layer evaluation.

This provides a baseline answering: "How well does an LLM perform the
validation step that our deterministic Plan Validator does?"

Metrics collected:
  - accuracy: fraction of activities correctly classified
  - per-status precision/recall
  - latency per scenario
  - token usage (prompt + completion)

Output: llm_verifier_baseline_results.json
"""

import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path, override=True)

from evaluation.scenarios import SCENARIOS
from planning.canonical_actions import CANONICAL_ACTIONS

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from utils.models import define_model


VERIFIER_SYSTEM_PROMPT = """\
You are a plan validator for a care-home robot coaching system.

Given:
- A world state (a set of boolean facts, where True means the fact holds).
- A list of activities, each with a canonical action, a criticality flag,
  a scheduled time (or no time), and a done flag.
- The canonical action dictionary with preconditions and effects.

Your task: for EACH activity, determine its executability status as one of:
  - EXECUTABLE: all preconditions satisfied, time window OK (or no time)
  - BLOCKED_CRITICAL: critical activity with unsatisfied preconditions
  - BLOCKED_NON_CRITICAL: non-critical activity with unsatisfied preconditions,
    or unknown/missing canonical action, or expired time window
  - DEFERRED: blocked ONLY by time window (activity is too early, not expired)

Rules:
1. If an activity is done=True, it is EXECUTABLE (skip validation).
2. If the canonical action is "unknown" or not in the dictionary,
   the activity is BLOCKED_NON_CRITICAL.
3. Check each precondition: if any is False in the world state, the activity
   is blocked. If the activity is critical -> BLOCKED_CRITICAL, else
   BLOCKED_NON_CRITICAL.
4. If preconditions are all satisfied but the activity has a scheduled time
   and the current time is BEFORE the window (scheduled - 30min), it is DEFERRED.
5. If preconditions are all satisfied but the current time is AFTER the window
   (scheduled + 30min), it is BLOCKED_NON_CRITICAL.
6. If preconditions are all satisfied and the time is within the window
   (or no time is set), it is EXECUTABLE.

Return ONLY valid JSON in this format:
{
  "activities": [
    {
      "activity_name": "<name>",
      "status": "EXECUTABLE|BLOCKED_CRITICAL|BLOCKED_NON_CRITICAL|DEFERRED",
      "reasons": ["<reason1>", ...]
    }
  ]
}
"""

VERIFIER_USER_TEMPLATE = """\
Current time: {current_time}

World state (only True facts shown):
{world_state_true}

Canonical action dictionary (name: preconditions -> effects):
{action_dict}

Activities to validate:
{activities_json}
"""


# The answer format as a JSON schema. With --structured the model is asked to
# follow it through the provider's structured-output support, which removes
# the free-prose answers that made one model's row measure format compliance
# rather than validation.
VERDICT_SCHEMA = {
    "name": "plan_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "activities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "activity_name": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": [
                                "EXECUTABLE",
                                "BLOCKED_CRITICAL",
                                "BLOCKED_NON_CRITICAL",
                                "DEFERRED",
                            ],
                        },
                        "reasons": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["activity_name", "status", "reasons"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["activities"],
        "additionalProperties": False,
    },
}


def _structured(model):
    """Bind the verdict schema as the required response format.

    On OpenRouter, require_parameters routes the request only to providers
    that honour response_format; without it a provider that ignores the
    schema would silently answer in free text.
    """
    kwargs = {
        "response_format": {"type": "json_schema", "json_schema": VERDICT_SCHEMA}
    }
    if os.getenv("LLM_PROVIDER") == "openrouter":
        extra_body = dict(getattr(model, "extra_body", None) or {})
        # require_parameters needs a provider supporting every parameter,
        # and models without reasoning have none that accepts `reasoning`.
        # Only gpt-oss is run with a reasoning effort.
        if "gpt-oss" not in _model_id():
            extra_body.pop("reasoning", None)
        provider = dict(extra_body.get("provider") or {})
        provider["require_parameters"] = True
        extra_body["provider"] = provider
        kwargs["extra_body"] = extra_body
    return model.bind(**kwargs)


def _format_world_state_true(world_state: dict) -> str:
    """Format world state showing only True facts."""
    true_facts = sorted([k for k, v in world_state.items() if v])
    return "\n".join(f"  - {f}: True" for f in true_facts) if true_facts else "  (none)"


def _format_action_dict() -> str:
    """Format the canonical action dictionary compactly."""
    lines = []
    for name, defn in sorted(CANONICAL_ACTIONS.items()):
        pre = ", ".join(defn.get("preconditions", [])) or "(none)"
        eff = ", ".join(defn.get("effects", [])) or "(none)"
        lines.append(f"  {name}: pre=[{pre}] -> eff=[{eff}]")
    return "\n".join(lines)


def _format_activities(activities: list) -> str:
    """Format activities for the LLM prompt."""
    lines = []
    for a in activities:
        lines.append(json.dumps({
            "activity_name": a["activity_name"],
            "canonical_action": a.get("canonical_action", "MISSING"),
            "critical": a.get("critical", False),
            "activity_time": a.get("activity_time"),
            "without_time": a.get("without_time", True),
            "done": a.get("done", False),
        }))
    return "\n".join(lines)


def _parse_llm_response(raw_content: str) -> dict | None:
    """Parse the LLM JSON response, stripping code fences."""
    content = raw_content.strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()
    if content.startswith("```"):
        content = content.removeprefix("```").strip()
    if content.endswith("```"):
        content = content.removesuffix("```").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _extract_llm_verdict(parsed: dict, activities: list) -> dict:
    """Extract observed outcomes from the LLM response, aligned to activities."""
    llm_acts = {a["activity_name"]: a["status"] for a in parsed.get("activities", [])}

    observed = {
        "executable": [],
        "blocked_critical": [],
        "blocked_non_critical": [],
        "deferred": [],
    }
    for a in activities:
        name = a["activity_name"]
        status = llm_acts.get(name, "BLOCKED_NON_CRITICAL")
        # Normalize status names
        status_upper = status.upper().replace(" ", "_")
        if status_upper == "EXECUTABLE":
            observed["executable"].append(name)
        elif status_upper == "BLOCKED_CRITICAL":
            observed["blocked_critical"].append(name)
        elif status_upper == "BLOCKED_NON_CRITICAL":
            observed["blocked_non_critical"].append(name)
        elif status_upper == "DEFERRED":
            observed["deferred"].append(name)
        else:
            # Unknown status -> default to blocked non-critical
            observed["blocked_non_critical"].append(name)
    return observed


def _compare_with_expected(observed: dict, expected: dict) -> dict:
    """Compare observed vs expected, return per-activity match details."""
    all_match = True
    details = {}
    for key in ["executable", "blocked_critical", "blocked_non_critical", "deferred"]:
        obs_set = set(observed.get(key, []))
        exp_set = set(expected.get(key, []))
        match = obs_set == exp_set
        details[key] = {
            "match": match,
            "expected": sorted(exp_set),
            "observed": sorted(obs_set),
            "missing": sorted(exp_set - obs_set),
            "extra": sorted(obs_set - exp_set),
        }
        if not match:
            all_match = False
    return {"all_pass": all_match, "details": details}


def _count_activity_accuracy(observed: dict, expected: dict, activities: list) -> dict:
    """Per-activity accuracy: how many activities are in the correct bucket."""
    correct = 0
    total = len(activities)
    for a in activities:
        name = a["activity_name"]
        # Find which bucket expected puts it in
        exp_bucket = None
        for key in ["executable", "blocked_critical", "blocked_non_critical", "deferred"]:
            if name in expected.get(key, []):
                exp_bucket = key
                break
        # Find which bucket observed puts it in
        obs_bucket = None
        for key in ["executable", "blocked_critical", "blocked_non_critical", "deferred"]:
            if name in observed.get(key, []):
                obs_bucket = key
                break
        if exp_bucket == obs_bucket:
            correct += 1
    return {"correct": correct, "total": total, "accuracy": correct / total if total > 0 else 0}


def _model_id() -> str:
    """The model actually in use, whichever provider is configured."""
    provider = os.getenv("LLM_PROVIDER", "")
    key = ("OPENROUTER_MODEL" if provider == "openrouter"
           else "GROQ_MODEL" if provider == "groq"
           else "OLLAMA_MODEL" if provider == "ollama"
           else "OPENAI_MODEL")
    return os.getenv(key, "unknown")


def run_llm_verifier_baseline(
    output_file: str = "evaluation/llm_verifier_baseline_results.json",
    temperature: float | None = None,
    structured: bool = False,
    only: set | None = None,
) -> dict:
    """Run the LLM verifier baseline on all scenarios."""
    model = define_model(temperature=temperature)
    if structured:
        model = _structured(model)
    scenarios = [s for s in SCENARIOS if not only or s["id"] in only]

    results = []
    total_correct = 0
    total_activities = 0
    total_scenarios_pass = 0
    total_latency = 0.0

    action_dict_str = _format_action_dict()

    for scenario in scenarios:
        sid = scenario["id"]
        desc = scenario["description"]
        expected = scenario["expected"]
        activities = scenario["activities"]

        user_prompt = PromptTemplate.from_template(VERIFIER_USER_TEMPLATE).invoke({
            "current_time": scenario["fake_datetime"],
            "world_state_true": _format_world_state_true(scenario["world_state"]),
            "action_dict": action_dict_str,
            "activities_json": _format_activities(activities),
        })

        messages = [
            SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt.to_string()),
        ]

        start_time = time.time()
        token_usage = {}
        raw_content = ""
        generation_id = None
        try:
            response = model.invoke(messages)
            latency = time.time() - start_time
            total_latency += latency

            # Kept so the serving provider can be looked up afterwards:
            # OpenRouter routes each request to one of many providers.
            generation_id = (response.response_metadata or {}).get("id")
            raw_content = response.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            # Count token usage if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage = {
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                    "total_tokens": response.usage_metadata.get("total_tokens", 0),
                }

            parsed = _parse_llm_response(raw_content)

            if parsed is None:
                # JSON parse failure -> all blocked non-critical (fail-closed)
                observed = {
                    "executable": [],
                    "blocked_critical": [],
                    "blocked_non_critical": [a["activity_name"] for a in activities],
                    "deferred": [],
                }
                parse_error = True
            else:
                observed = _extract_llm_verdict(parsed, activities)
                parse_error = False

        except Exception as e:
            latency = time.time() - start_time
            total_latency += latency
            observed = {
                "executable": [],
                "blocked_critical": [],
                "blocked_non_critical": [a["activity_name"] for a in activities],
                "deferred": [],
            }
            parse_error = True
            raw_content = f"ERROR: {e}"

        comparison = _compare_with_expected(observed, expected)
        accuracy = _count_activity_accuracy(observed, expected, activities)

        total_correct += accuracy["correct"]
        total_activities += accuracy["total"]
        if comparison["all_pass"]:
            total_scenarios_pass += 1

        results.append({
            "id": sid,
            "description": desc,
            "pass": comparison["all_pass"],
            "accuracy": accuracy["accuracy"],
            "latency_seconds": round(latency, 3),
            "parse_error": parse_error,
            "observed": observed,
            "expected": expected,
            "comparison": comparison["details"],
            "token_usage": token_usage,
            "generation_id": generation_id,
            "raw_llm_output": raw_content[:2000] if not parse_error else raw_content[:500],
        })

        status = "PASS" if comparison["all_pass"] else "FAIL"
        print(f"  [{status}] {sid}: {desc} (accuracy={accuracy['accuracy']:.0%}, {latency:.2f}s)")

    total = len(scenarios)
    overall_accuracy = total_correct / total_activities if total_activities > 0 else 0
    avg_latency = total_latency / total if total > 0 else 0

    # Aggregate token usage
    total_prompt_tokens = sum(r["token_usage"].get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r["token_usage"].get("completion_tokens", 0) for r in results)
    total_tokens = sum(r["token_usage"].get("total_tokens", 0) for r in results)

    summary = {
        "total_scenarios": total,
        "scenarios_passed": total_scenarios_pass,
        "scenario_pass_rate": round(total_scenarios_pass / total, 4) if total > 0 else 0,
        "total_activities": total_activities,
        "activities_correctly_classified": total_correct,
        "activity_accuracy": round(overall_accuracy, 4),
        "avg_latency_seconds": round(avg_latency, 3),
        "total_latency_seconds": round(total_latency, 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "model": _model_id(),
        "provider": os.getenv("LLM_PROVIDER", "unknown"),
        "temperature": temperature,
        "structured_output": structured,
        "reasoning_effort": os.getenv("OPENROUTER_REASONING_EFFORT"),
        "parse_errors": sum(1 for r in results if r["parse_error"]),
    }

    output = {
        "summary": summary,
        "scenarios": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-verifier baseline")
    parser.add_argument("--provider", default=None,
                        help="override LLM_PROVIDER for this run")
    parser.add_argument("--model", default=None,
                        help="override the model id for this run")
    parser.add_argument("--out", default=None, help="results file")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature (default: global 0.3)")
    parser.add_argument("--structured", action="store_true",
                        help="require the JSON schema as response format")
    parser.add_argument("--only", default=None,
                        help="comma-separated scenario ids")
    args = parser.parse_args()

    # load_dotenv ran at import with override=True, so command-line
    # environment variables were already replaced by the .env values.
    # Applying the overrides here makes the choice explicit and final.
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        provider = os.environ.get("LLM_PROVIDER", "")
        key = ("OPENROUTER_MODEL" if provider == "openrouter"
               else "GROQ_MODEL" if provider == "groq"
               else "OLLAMA_MODEL" if provider == "ollama"
               else "OPENAI_MODEL")
        os.environ[key] = args.model

    output_file = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "llm_verifier_baseline_results.json",
    )

    print("=" * 60)
    print("  LLM-AS-VERIFIER BASELINE")
    print("  Comparison with symbolic layer on 20 scenarios")
    print("=" * 60)
    print()

    results = run_llm_verifier_baseline(
        output_file,
        temperature=args.temperature,
        structured=args.structured,
        only={x.strip() for x in args.only.split(",")} if args.only else None,
    )

    s = results["summary"]
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Scenarios passed:       {s['scenarios_passed']}/{s['total_scenarios']} ({s['scenario_pass_rate']:.1%})")
    print(f"  Activity accuracy:      {s['activities_correctly_classified']}/{s['total_activities']} ({s['activity_accuracy']:.1%})")
    print(f"  Avg latency:            {s['avg_latency_seconds']:.3f}s")
    print(f"  Total tokens:           {s['total_tokens']}")
    print(f"  Parse errors:           {s['parse_errors']}")
    print(f"  Model:                  {s['model']}")
    print(f"  Results saved to:       {output_file}")
    print("=" * 60)
