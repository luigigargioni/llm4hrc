"""Test that structured logging produces valid JSONL entries.

This test calls the structured log functions directly and verifies
that the output file contains valid JSON lines with the expected fields.
"""

import json
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.structured_logs import (  # noqa: E402
    log_structured,
    log_canonicalizer_decision,
    log_precondition_check,
    log_temporal_check,
    log_arbitration,
    log_goal_decomposition,
    log_goal_predicate_extraction,
    log_world_state_update,
    log_hard_enforcement,
    _structured_log_filename,
)


def get_latest_structured_log():
    """Get the most recent structured log file."""
    files = glob.glob("log/structured/*_structured.jsonl")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def test_basic_structured_log():
    """Test that log_structured produces valid JSONL."""
    log_structured({
        "node": "test",
        "decision_type": "test_entry",
        "message": "hello",
    })

    logfile = get_latest_structured_log()
    assert logfile is not None, "No structured log file found"

    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    last_entry = json.loads(lines[-1])
    assert last_entry["node"] == "test"
    assert last_entry["decision_type"] == "test_entry"
    assert "timestamp" in last_entry
    print("  Passed!")


def test_canonicalizer_log():
    """Test canonicalizer decision log."""
    log_canonicalizer_decision(
        activity_name="Take your pills",
        raw_llm_output="take_medication",
        canonical_action="take_medication",
        is_known=True,
    )
    log_canonicalizer_decision(
        activity_name="Do some yoga",
        raw_llm_output="yoga_meditation",
        canonical_action="unknown",
        is_known=False,
    )

    logfile = get_latest_structured_log()
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = [json.loads(l) for l in lines]
    canon_entries = [e for e in entries if e.get("decision_type") == "canonicalization"]
    assert len(canon_entries) >= 2

    known = [e for e in canon_entries if e["is_known"]]
    unknown = [e for e in canon_entries if not e["is_known"]]
    assert len(known) >= 1
    assert len(unknown) >= 1
    assert known[-1]["canonical_action"] == "take_medication"
    assert unknown[-1]["canonical_action"] == "unknown"
    assert unknown[-1]["outcome"] == "unknown_fail_closed"
    print("  Passed!")


def test_precondition_check_log():
    """Test precondition check log."""
    log_precondition_check(
        activity_name="Full Rehabilitation Session",
        canonical_action="full_rehabilitation_session",
        preconditions_checked=[
            {"precondition": "stretching_completed", "satisfied": False},
            {"precondition": "patient_fed", "satisfied": False},
        ],
        all_satisfied=False,
        missing=["stretching_completed", "patient_fed"],
    )

    logfile = get_latest_structured_log()
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = [json.loads(l) for l in lines]
    precond_entries = [e for e in entries if e.get("decision_type") == "precondition_check"]
    assert len(precond_entries) >= 1
    assert precond_entries[-1]["all_satisfied"] is False
    assert len(precond_entries[-1]["missing_preconditions"]) == 2
    print("  Passed!")


def test_arbitration_log():
    """Test arbitration log."""
    log_arbitration(
        activity_name="Full Rehabilitation Session",
        executable=False,
        not_executable_reasons=["missing_precondition: stretching_completed"],
        arbitrated_status="BLOCKED_CRITICAL",
        is_critical=True,
        priority=1,
        recovery_actions=[
            {"canonical_action": "stretching_session", "description": "Stretching", "satisfies": ["stretching_completed"]},
        ],
    )

    logfile = get_latest_structured_log()
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = [json.loads(l) for l in lines]
    arb_entries = [e for e in entries if e.get("decision_type") == "arbitration"]
    assert len(arb_entries) >= 1
    assert arb_entries[-1]["arbitrated_status"] == "BLOCKED_CRITICAL"
    assert arb_entries[-1]["is_critical"] is True
    assert len(arb_entries[-1]["recovery_actions"]) == 1
    print("  Passed!")


def test_goal_decomposition_log():
    """Test goal decomposition log."""
    log_goal_decomposition(
        goal_predicate="rehabilitation_completed",
        decomposition_success=True,
        action_chain=[
            {"canonical_action": "have_lunch"},
            {"canonical_action": "stretching_session"},
            {"canonical_action": "full_rehabilitation_session"},
        ],
        fallback_used=False,
        num_activities_generated=3,
    )
    log_goal_decomposition(
        goal_predicate=None,
        decomposition_success=False,
        action_chain=None,
        fallback_used=True,
        num_activities_generated=2,
    )

    logfile = get_latest_structured_log()
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = [json.loads(l) for l in lines]
    decomp_entries = [e for e in entries if e.get("decision_type") == "goal_decomposition"]
    assert len(decomp_entries) >= 2

    success = [e for e in decomp_entries if e["decomposition_success"]]
    fallback = [e for e in decomp_entries if e["fallback_used"]]
    assert len(success) >= 1
    assert len(fallback) >= 1
    assert success[-1]["num_activities_generated"] == 3
    print("  Passed!")


def test_world_state_update_log():
    """Test world state update log."""
    ws_before = {"patient_fed": False, "after_meal": False}
    ws_after = {"patient_fed": True, "after_meal": True}

    log_world_state_update(
        activity_name="Have lunch",
        canonical_action="have_lunch",
        effects_applied=["patient_fed", "after_meal"],
        world_state_before=ws_before,
        world_state_after=ws_after,
    )

    logfile = get_latest_structured_log()
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = [json.loads(l) for l in lines]
    ws_entries = [e for e in entries if e.get("decision_type") == "world_state_update"]
    assert len(ws_entries) >= 1
    assert ws_entries[-1]["changed_facts"]["patient_fed"]["before"] is False
    assert ws_entries[-1]["changed_facts"]["patient_fed"]["after"] is True
    print("  Passed!")


def test_hard_enforcement_log():
    """Test hard enforcement log."""
    log_hard_enforcement(
        violated_activities=["full rehabilitation session"],
        robot_action_before="SPEAKING",
        robot_action_after="SUGGEST_RECOVERY_ACTION",
        arbitrated_statuses=["BLOCKED_CRITICAL"],
        recovery_suggested=True,
        declared_activity="Full Rehabilitation Session",
        match_mode="declared",
    )

    logfile = get_latest_structured_log()
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = [json.loads(l) for l in lines]
    he_entries = [e for e in entries if e.get("decision_type") == "hard_enforcement"]
    assert len(he_entries) >= 1
    assert he_entries[-1]["robot_action_before"] == "SPEAKING"
    assert he_entries[-1]["robot_action_after"] == "SUGGEST_RECOVERY_ACTION"
    assert he_entries[-1]["recovery_suggested"] is True
    # Which rule fired, so declared-intent and name-match overrides can be
    # told apart when the corpus is analysed.
    assert he_entries[-1]["match_mode"] == "declared"
    assert he_entries[-1]["declared_activity"] == "Full Rehabilitation Session"
    assert he_entries[-1]["enforcement_mode"] == "declared"
    print("  Passed!")


def test_all_entries_valid_json():
    """Verify every line in the structured log is valid JSON."""
    logfile = get_latest_structured_log()
    assert logfile is not None

    with open(logfile, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Line {i+1} is not valid JSON: {e}")

    print("  Passed!")


if __name__ == "__main__":
    print("\n=== Test: basic structured log ===")
    test_basic_structured_log()

    print("\n=== Test: canonicalizer log ===")
    test_canonicalizer_log()

    print("\n=== Test: precondition check log ===")
    test_precondition_check_log()

    print("\n=== Test: arbitration log ===")
    test_arbitration_log()

    print("\n=== Test: goal decomposition log ===")
    test_goal_decomposition_log()

    print("\n=== Test: world state update log ===")
    test_world_state_update_log()

    print("\n=== Test: hard enforcement log ===")
    test_hard_enforcement_log()

    print("\n=== Test: all entries valid JSON ===")
    test_all_entries_valid_json()

    print("\n=== ALL STRUCTURED LOG TESTS PASSED! ===")
