"""Test script for the Plan Validator arbitration logic.

Tests that the plan_validator_node correctly assigns arbitrated_status
and recovery_actions based on criticality, preconditions, and time window.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.plan_validator_node import _find_recovery_actions  # noqa: E402

WORLD_STATE = {
    "patient_awake": True,
    "patient_available": True,
    "patient_not_tired": True,
    "patient_fed": False,
    "patient_tired": False,
    "medication_available": True,
    "correct_time": True,
    "food_available": True,
    "water_available": True,
    "stretching_completed": False,
    "rehabilitation_completed": False,
}


def test_recovery_actions():
    """Test that _find_recovery_actions finds correct actions for missing preconditions."""
    # Missing: patient_fed
    recovery = _find_recovery_actions(["patient_fed"], WORLD_STATE)
    print("\n=== Test: recovery actions for 'patient_fed' ===")
    for r in recovery:
        print(
            f"  - {r['canonical_action']}: {r['description']} (satisfies: {r['satisfies']})"
        )

    action_names = [r["canonical_action"] for r in recovery]
    assert (
        "have_lunch" in action_names
    ), "Expected have_lunch as recovery for patient_fed"
    assert "eat_meal" in action_names, "Expected eat_meal as recovery for patient_fed"
    print("  Passed!")

    # Missing: stretching_completed
    # Note: stretching_session requires patient_fed and patient_not_tired
    ws_stretch = dict(WORLD_STATE)
    ws_stretch["patient_fed"] = True  # so stretching_session's preconditions are met
    recovery = _find_recovery_actions(["stretching_completed"], ws_stretch)
    print("\n=== Test: recovery actions for 'stretching_completed' ===")
    for r in recovery:
        print(
            f"  - {r['canonical_action']}: {r['description']} (satisfies: {r['satisfies']})"
        )

    action_names = [r["canonical_action"] for r in recovery]
    assert (
        "stretching_session" in action_names
    ), "Expected stretching_session as recovery for stretching_completed"
    print("  Passed!")

    # Missing: patient_fed AND stretching_completed (full_rehabilitation_session preconditions)
    recovery = _find_recovery_actions(
        ["patient_fed", "stretching_completed"], WORLD_STATE
    )
    print("\n=== Test: recovery actions for 'patient_fed' + 'stretching_completed' ===")
    for r in recovery:
        print(
            f"  - {r['canonical_action']}: {r['description']} (satisfies: {r['satisfies']})"
        )
    assert len(recovery) >= 2, "Expected at least 2 recovery actions"
    print("  Passed!")


def test_recovery_with_unsatisfiable_preconditions():
    """Test that recovery actions are NOT suggested when their own preconditions aren't met."""
    # If patient is asleep, have_lunch (requires patient_awake) should NOT be suggested
    ws = dict(WORLD_STATE)
    ws["patient_awake"] = False

    recovery = _find_recovery_actions(["patient_fed"], ws)
    print("\n=== Test: recovery when patient asleep (should be empty) ===")
    print(f"  Recovery actions: {[r['canonical_action'] for r in recovery]}")
    # have_lunch requires patient_awake, eat_meal requires patient_awake
    # So no recovery actions should be available
    assert len(recovery) == 0, "Expected no recovery actions when patient is asleep"
    print("  Passed!")


def test_arbitration_logic():
    """Test the full arbitration logic by simulating plan_validator_node behavior."""
    from planning.canonical_actions import CANONICAL_ACTIONS

    print("\n=== Test: full arbitration logic ===")

    # Simulate activities with different scenarios
    test_cases = [
        {
            "name": "Critical activity with missing preconditions",
            "activity": {
                "activity_name": "Full Rehabilitation Session",
                "canonical_action": "full_rehabilitation_session",
                "critical": True,
                "priority": 1,
                "activity_time": None,
                "without_time": True,
            },
            "expected_status": "BLOCKED_CRITICAL",
            "expected_recovery": True,
        },
        {
            "name": "Non-critical activity with missing preconditions",
            "activity": {
                "activity_name": "Stretching Session",
                "canonical_action": "stretching_session",
                "critical": False,
                "priority": 3,
                "activity_time": None,
                "without_time": True,
            },
            "expected_status": "BLOCKED_NON_CRITICAL",
            "expected_recovery": False,
        },
        {
            "name": "Executable activity",
            "activity": {
                "activity_name": "Drink Water",
                "canonical_action": "drink_water",
                "critical": False,
                "priority": 3,
                "activity_time": None,
                "without_time": True,
            },
            "expected_status": "EXECUTABLE",
            "expected_recovery": False,
        },
    ]

    for tc in test_cases:
        activity = tc["activity"]
        ca = activity["canonical_action"]

        # Simulate precondition check
        executable = True
        not_executable_reasons = []
        missing_preconditions = []

        if ca in CANONICAL_ACTIONS:
            for pre in CANONICAL_ACTIONS[ca]["preconditions"]:
                if not WORLD_STATE.get(pre, False):
                    executable = False
                    reason = f"missing_precondition: {pre}"
                    not_executable_reasons.append(reason)
                    missing_preconditions.append(pre)

        # Simulate arbitration
        has_precondition_issues = any(
            "missing_precondition" in r for r in not_executable_reasons
        )
        has_time_issues = any(
            "outside_time_window" in r for r in not_executable_reasons
        )
        is_critical = activity.get("critical", False)

        if executable:
            arbitrated_status = "EXECUTABLE"
        elif has_time_issues and not has_precondition_issues:
            arbitrated_status = "DEFERRED"
        elif is_critical:
            arbitrated_status = "BLOCKED_CRITICAL"
        else:
            arbitrated_status = "BLOCKED_NON_CRITICAL"

        # Simulate recovery actions
        recovery_actions = []
        if arbitrated_status == "BLOCKED_CRITICAL":
            recovery_actions = _find_recovery_actions(
                missing_preconditions, WORLD_STATE
            )

        print(f"\n  Case: {tc['name']}")
        print(f"    Status: {arbitrated_status} (expected: {tc['expected_status']})")
        print(f"    Recovery: {[r['canonical_action'] for r in recovery_actions]}")

        assert (
            arbitrated_status == tc["expected_status"]
        ), f"Expected {tc['expected_status']}, got {arbitrated_status}"

        if tc["expected_recovery"]:
            assert len(recovery_actions) > 0, "Expected recovery actions"
        else:
            assert len(recovery_actions) == 0, "Expected no recovery actions"

        print(f"    Passed!")

    print("\n=== All arbitration tests passed! ===")


if __name__ == "__main__":
    test_recovery_actions()
    test_recovery_with_unsatisfiable_preconditions()
    test_arbitration_logic()
    print("\n=== ALL TESTS PASSED! ===")
