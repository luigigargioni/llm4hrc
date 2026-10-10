"""Scenario runner for the symbolic layer evaluation suite.

Executes each scenario through the symbolic pipeline:
  world_state_updater -> plan_validator

In ablation mode, plan_validator is skipped (all activities marked executable),
simulating the system without the symbolic layer. A second ablation keeps the
base Event Planner's time-window check, which the base architecture already
applied, so that only what it could not refuse is credited to the validator.

Compares observed vs. expected outcomes and produces aggregate metrics.
"""

import json
import os
import sys
from datetime import datetime
from copy import deepcopy
from typing import Any

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.scenarios import SCENARIOS
from nodes.plan_validator_node import plan_validator_node
from nodes.world_state_updater_node import world_state_updater_node
from deterministic.activity_planner import (
    ExpertScheduler,
    TOLERANCE_START_TIME_ACTIVITY,
)


def _build_state(scenario: dict) -> dict:
    """Build a minimal GraphState-compatible dict from a scenario."""
    fake_dt = datetime.strptime(scenario["fake_datetime"], "%H:%M")
    today = datetime.today()
    fake_datetime = datetime.combine(today, fake_dt.time())

    return {
        "task": None,
        "progress_checking_output": None,
        "conversation_history": None,
        "updated_task": {
            "patient_name": "Test",
            "patient_surname": "Patient",
            "patient_information": "Test patient",
            "datetime": scenario["fake_datetime"],
            "activities": deepcopy(scenario["activities"]),
        },
        "robot_action": None,
        "vocal_answer": None,
        "patient_request": scenario.get("patient_request", ""),
        "situation_perception": "",
        "robot_behaviour": "proactive",
        "doctor_specs": None,
        "vector_store": None,
        "end": False,
        "fake_datetime": fake_datetime,
        "patient_id": None,
        "doctor_specifications": None,
        "scheduled_activities": None,
        "next_activities": None,
        "progress_checking_similarity_search": None,
        "world_state": deepcopy(scenario["world_state"]),
    }


def _run_symbolic(state: dict) -> dict:
    """Run the symbolic pipeline: world_state_updater -> plan_validator."""
    state = world_state_updater_node(state)
    state = plan_validator_node(state)
    return state


def _run_ablation(state: dict) -> dict:
    """Run without plan_validator: all activities marked executable."""
    state = world_state_updater_node(state)
    for activity in state["updated_task"]["activities"]:
        activity["executable"] = True
        activity["not_executable_reasons"] = []
        activity["arbitrated_status"] = "EXECUTABLE"
        activity["recovery_actions"] = []
    return state


def _run_event_planner_only(state: dict) -> dict:
    """Run without plan_validator, keeping the base Event Planner's timing.

    The base architecture already refuses a timed activity outside its
    tolerance window: the Event Planner only offers activities whose time
    falls within it. This ablation reproduces that check, with the Event
    Planner's own time parser, so that the comparison credits the
    validator only with what the base architecture could not refuse.
    Activities without a time are self-schedulable and always admitted.
    """
    state = world_state_updater_node(state)
    scheduler = ExpertScheduler()
    now = state["fake_datetime"]
    for activity in state["updated_task"]["activities"]:
        status = "EXECUTABLE"
        time_text = activity.get("activity_time")
        if not activity.get("done") and time_text and not activity.get(
                "without_time"):
            try:
                scheduled = scheduler._parse_time(time_text)
            except (ValueError, TypeError):
                scheduled = None
            if scheduled is None:
                status = "BLOCKED_NON_CRITICAL"
            else:
                scheduled = datetime.combine(now.date(), scheduled.time())
                if now < scheduled - TOLERANCE_START_TIME_ACTIVITY:
                    status = "DEFERRED"
                elif now > scheduled + TOLERANCE_START_TIME_ACTIVITY:
                    status = "BLOCKED_NON_CRITICAL"
        activity["executable"] = status == "EXECUTABLE"
        activity["not_executable_reasons"] = []
        activity["arbitrated_status"] = status
        activity["recovery_actions"] = []
    return state


def _extract_observed(state: dict) -> dict:
    """Extract observed outcomes from state after running the pipeline."""
    activities = state["updated_task"]["activities"]
    return {
        "executable": [a["activity_name"] for a in activities if a.get("executable") is True],
        "blocked_critical": [
            a["activity_name"] for a in activities
            if a.get("arbitrated_status") == "BLOCKED_CRITICAL"
        ],
        "blocked_non_critical": [
            a["activity_name"] for a in activities
            if a.get("arbitrated_status") == "BLOCKED_NON_CRITICAL"
        ],
        "deferred": [
            a["activity_name"] for a in activities
            if a.get("arbitrated_status") == "DEFERRED"
        ],
        "recovery_actions": {
            a["activity_name"]: [r["canonical_action"] for r in a.get("recovery_actions", [])]
            for a in activities
            if a.get("recovery_actions")
        },
    }


def _compare(observed: dict, expected: dict) -> dict:
    """Compare observed vs. expected. Returns match details."""
    checks = {}
    all_pass = True

    for key in ["executable", "blocked_critical", "blocked_non_critical", "deferred"]:
        obs_set = set(observed.get(key, []))
        exp_set = set(expected.get(key, []))
        match = obs_set == exp_set
        checks[key] = {
            "match": match,
            "expected": sorted(exp_set),
            "observed": sorted(obs_set),
            "missing": sorted(exp_set - obs_set),
            "extra": sorted(obs_set - exp_set),
        }
        if not match:
            all_pass = False

    # Check recovery actions (subset check: expected actions must be present)
    exp_recovery = expected.get("recovery_actions", {})
    obs_recovery = observed.get("recovery_actions", {})
    for act_name, exp_actions in exp_recovery.items():
        obs_actions = set(obs_recovery.get(act_name, []))
        exp_set = set(exp_actions)
        match = exp_set.issubset(obs_actions)
        checks[f"recovery_{act_name}"] = {
            "match": match,
            "expected": sorted(exp_set),
            "observed": sorted(obs_actions),
        }
        if not match:
            all_pass = False

    # Check that no unexpected recovery actions are present
    for act_name in obs_recovery:
        if act_name not in exp_recovery:
            checks[f"recovery_{act_name}"] = {
                "match": False,
                "expected": [],
                "observed": sorted(obs_recovery[act_name]),
                "note": "unexpected recovery actions",
            }
            all_pass = False

    return {"all_pass": all_pass, "checks": checks}


def run_evaluation(output_file: str = "evaluation/evaluation_results.json") -> dict:
    """Run all scenarios and produce evaluation results."""
    results = []
    pass_count = 0
    total_violations_intercepted = 0
    total_violations_missed_ablation = 0
    total_violations_missed_event_planner = 0
    total_violations_missed_symbolic = 0
    total_recovery_suggestions = 0

    for scenario in SCENARIOS:
        sid = scenario["id"]
        desc = scenario["description"]
        expected = scenario["expected"]

        # --- Symbolic run ---
        state_sym = _build_state(scenario)
        state_sym = _run_symbolic(state_sym)
        observed_sym = _extract_observed(state_sym)
        comparison = _compare(observed_sym, expected)

        if comparison["all_pass"]:
            pass_count += 1

        # Count violations intercepted by symbolic layer
        blocked_sym = (
            len(observed_sym["blocked_critical"])
            + len(observed_sym["blocked_non_critical"])
            + len(observed_sym["deferred"])
        )
        total_violations_intercepted += blocked_sym

        # Count recovery suggestions
        for acts in observed_sym["recovery_actions"].values():
            total_recovery_suggestions += len(acts)

        # --- Ablation run ---
        state_abl = _build_state(scenario)
        state_abl = _run_ablation(state_abl)
        observed_abl = _extract_observed(state_abl)

        # In ablation, activities that should be blocked are missed
        should_be_blocked = (
            set(expected.get("blocked_critical", []))
            | set(expected.get("blocked_non_critical", []))
            | set(expected.get("deferred", []))
        )
        # In ablation all are executable, so all blocked ones are missed
        total_violations_missed_ablation += len(should_be_blocked)
        # With the Event Planner's timing only, what it already refuses
        state_ep = _build_state(scenario)
        state_ep = _run_event_planner_only(state_ep)
        observed_ep = _extract_observed(state_ep)
        obs_blocked_ep = (
            set(observed_ep["blocked_critical"])
            | set(observed_ep["blocked_non_critical"])
            | set(observed_ep["deferred"])
        )
        missed_ep = should_be_blocked - obs_blocked_ep
        total_violations_missed_event_planner += len(missed_ep)
        # In symbolic, check if any were missed (shouldn't be if test passes)
        obs_blocked_sym = (
            set(observed_sym["blocked_critical"])
            | set(observed_sym["blocked_non_critical"])
            | set(observed_sym["deferred"])
        )
        total_violations_missed_symbolic += len(should_be_blocked - obs_blocked_sym)

        results.append({
            "id": sid,
            "description": desc,
            "pass": comparison["all_pass"],
            "expected": expected,
            "observed_symbolic": observed_sym,
            "observed_ablation": observed_abl,
            "observed_event_planner_only": observed_ep,
            "comparison": comparison["checks"],
            "violations_intercepted_symbolic": blocked_sym,
            "violations_missed_ablation": len(should_be_blocked),
            "violations_missed_event_planner_only": sorted(missed_ep),
            "violations_missed_symbolic": len(should_be_blocked - obs_blocked_sym),
        })

        status = "PASS" if comparison["all_pass"] else "FAIL"
        print(f"  [{status}] {sid}: {desc}")

    # Aggregate metrics
    total = len(SCENARIOS)
    success_rate = pass_count / total if total > 0 else 0
    ablation_error_reduction = 0
    if total_violations_missed_ablation > 0:
        ablation_error_reduction = (
            (total_violations_missed_ablation - total_violations_missed_symbolic)
            / total_violations_missed_ablation
        )

    summary = {
        "total_scenarios": total,
        "passed": pass_count,
        "failed": total - pass_count,
        "success_rate": round(success_rate, 4),
        "violations_intercepted_symbolic": total_violations_intercepted,
        "violations_missed_ablation": total_violations_missed_ablation,
        "violations_missed_event_planner_only": (
            total_violations_missed_event_planner
        ),
        "violations_missed_symbolic": total_violations_missed_symbolic,
        "recovery_suggestions": total_recovery_suggestions,
        "ablation_error_reduction": round(ablation_error_reduction, 4),
    }

    output = {
        "summary": summary,
        "scenarios": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return output
