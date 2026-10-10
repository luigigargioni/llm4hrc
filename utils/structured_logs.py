"""Structured logging for the symbolic planning layer.

Produces machine-readable JSON logs of every decision made by the
symbolic layer (action canonicalizer, plan validator, goal planner,
world state updater). These logs serve both debugging and evaluation
purposes (explainability, ablation studies, failure mode analysis).

Each log entry is a JSON object with:
  - timestamp: ISO format
  - node: which node produced the entry
  - decision_type: category of decision
  - ...fields specific to the decision type

All entries are appended to a dedicated structured log file, separate
from the human-readable log file managed by utils/logs.py.
"""

import json
from datetime import datetime
from pathlib import Path

STRUCTURED_LOG_DIR = "log/structured"
_structured_log_filename: str | None = None


def _ensure_log_file() -> str:
    """Lazily create the structured log file for this run."""
    global _structured_log_filename
    if _structured_log_filename is None:
        Path(STRUCTURED_LOG_DIR).mkdir(parents=True, exist_ok=True)
        _structured_log_filename = (
            f"{STRUCTURED_LOG_DIR}/"
            f"{datetime.now().strftime('%d%m%Y_%H%M%S')}_structured.jsonl"
        )
        # Write empty file
        with open(_structured_log_filename, "w", encoding="utf-8") as f:
            pass
    return _structured_log_filename


def log_structured(entry: dict) -> None:
    """
    Append a structured log entry as a JSON line (JSONL format).

    The entry dict should contain at minimum:
      - node: str (e.g. "plan_validator", "goal_planner")
      - decision_type: str (e.g. "precondition_check", "goal_decomposition")
    A timestamp is added automatically.
    """
    entry["timestamp"] = datetime.now().isoformat()
    filename = _ensure_log_file()
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ------------------------------------------------------------------
# Convenience functions for each symbolic node
# ------------------------------------------------------------------

def log_canonicalizer_decision(
    activity_name: str,
    raw_llm_output: str,
    canonical_action: str,
    is_known: bool,
    source: str = "model",
) -> None:
    """Log a single canonicalization decision.

    `source` separates a fresh model answer from one reused from the
    session cache, so the two can be counted apart when a run is analysed.
    """
    log_structured({
        "node": "action_canonicalizer",
        "decision_type": "canonicalization",
        "activity_name": activity_name,
        "raw_llm_output": raw_llm_output,
        "canonical_action": canonical_action,
        "is_known": is_known,
        "source": source,
        "outcome": "mapped" if is_known else "unknown_fail_closed",
    })


def log_precondition_check(
    activity_name: str,
    canonical_action: str,
    preconditions_checked: list[dict],
    all_satisfied: bool,
    missing: list[str],
) -> None:
    """Log precondition verification for one activity."""
    log_structured({
        "node": "plan_validator",
        "decision_type": "precondition_check",
        "activity_name": activity_name,
        "canonical_action": canonical_action,
        "preconditions": preconditions_checked,
        "all_satisfied": all_satisfied,
        "missing_preconditions": missing,
    })


def log_temporal_check(
    activity_name: str,
    activity_time: str | None,
    without_time: bool,
    within_window: bool,
    earliest: str | None,
    latest: str | None,
    current_time: str | None,
) -> None:
    """Log temporal window check for one activity."""
    log_structured({
        "node": "plan_validator",
        "decision_type": "temporal_check",
        "activity_name": activity_name,
        "activity_time": activity_time,
        "without_time": without_time,
        "within_window": within_window,
        "earliest": earliest,
        "latest": latest,
        "current_time": current_time,
    })


def log_arbitration(
    activity_name: str,
    executable: bool,
    not_executable_reasons: list[str],
    arbitrated_status: str,
    is_critical: bool,
    priority: int,
    recovery_actions: list[dict],
) -> None:
    """Log the final arbitration decision for one activity."""
    log_structured({
        "node": "plan_validator",
        "decision_type": "arbitration",
        "activity_name": activity_name,
        "executable": executable,
        "not_executable_reasons": not_executable_reasons,
        "arbitrated_status": arbitrated_status,
        "is_critical": is_critical,
        "priority": priority,
        "recovery_actions": recovery_actions,
    })


def log_goal_decomposition(
    goal_predicate: str | None,
    decomposition_success: bool,
    action_chain: list[dict] | None,
    fallback_used: bool,
    num_activities_generated: int,
) -> None:
    """Log the goal decomposition decision in the goal planner."""
    log_structured({
        "node": "goal_planner",
        "decision_type": "goal_decomposition",
        "goal_predicate": goal_predicate,
        "decomposition_success": decomposition_success,
        "action_chain": [
            a.get("canonical_action", str(a)) for a in (action_chain or [])
        ],
        "fallback_used": fallback_used,
        "num_activities_generated": num_activities_generated,
    })


def log_goal_predicate_extraction(
    patient_request: str,
    goal_predicate: str | None,
    generate_new_activities: bool,
) -> None:
    """Log the LLM goal predicate extraction."""
    log_structured({
        "node": "goal_planner",
        "decision_type": "goal_predicate_extraction",
        "patient_request": patient_request,
        "goal_predicate": goal_predicate,
        "generate_new_activities": generate_new_activities,
    })


def log_world_state_update(
    activity_name: str,
    canonical_action: str,
    effects_applied: list[str],
    world_state_before: dict,
    world_state_after: dict,
) -> None:
    """Log a world state change caused by a completed activity."""
    # Only log the changed keys to keep entries compact
    changed = {
        k: {"before": world_state_before.get(k), "after": world_state_after.get(k)}
        for k in effects_applied
    }
    log_structured({
        "node": "world_state_updater",
        "decision_type": "world_state_update",
        "activity_name": activity_name,
        "canonical_action": canonical_action,
        "effects_applied": effects_applied,
        "changed_facts": changed,
    })


def log_hard_enforcement(
    violated_activities: list[str],
    robot_action_before: str,
    robot_action_after: str,
    arbitrated_statuses: list[str],
    recovery_suggested: bool,
    vocal_answer_before: str = "",
    vocal_answer_after: str = "",
    declared_activity: str | None = None,
    match_mode: str | None = None,
    enforcement_mode: str = "declared",
) -> None:
    """Log the deterministic hard enforcement override in next_steps."""
    log_structured({
        "node": "next_steps",
        "decision_type": "hard_enforcement",
        "violated_activities": violated_activities,
        "robot_action_before": robot_action_before,
        "robot_action_after": robot_action_after,
        "arbitrated_statuses": arbitrated_statuses,
        "recovery_suggested": recovery_suggested,
        # Kept so an override can be classified afterwards: an answer that
        # already refused the activity means the substring check fired on a
        # correct refusal, not on a proposal.
        "vocal_answer_before": vocal_answer_before,
        "vocal_answer_after": vocal_answer_after,
        # What the reply declared it was proposing, and which rule fired:
        # "declared" for the declaration, "fallback_name_match" when the
        # declaration was missing, "name_match" in legacy mode.
        "declared_activity": declared_activity,
        "match_mode": match_mode,
        "enforcement_mode": enforcement_mode,
    })
