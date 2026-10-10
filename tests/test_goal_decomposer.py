"""Quick test script for the GoalDecomposer backward chaining integration."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deterministic.goal_decomposer import GoalDecomposer  # noqa: E402

# Minimal world state matching INITIAL_GRAPH_STATE["world_state"]
# (duplicated here to avoid langchain_core dependency in test environment)
WORLD_STATE = {
    "patient_awake": True, "patient_sleeping": False, "patient_available": True,
    "patient_not_tired": True, "patient_tired": False, "patient_rested": False,
    "patient_in_room": True, "patient_fed": False, "patient_hydrated": True,
    "patient_in_pain": False, "patient_stressed": False, "patient_lonely": True,
    "patient_bored": True, "patient_motivated": True, "patient_relaxed": False,
    "patient_happier": False, "patient_sad": False, "patient_emotionally_stable": True,
    "patient_focused": True, "patient_stabilized": False, "patient_reassured": False,
    "patient_socialized": False, "patient_prepared_for_day": False,
    "medication_available": True, "medication_prepared": False, "correct_time": True,
    "medication_taken": False, "after_meal": True, "infection_treated": False,
    "meal_completed": True, "blood_pressure_checked": False, "blood_glucose_checked": False,
    "vitals_checked": False, "patient_monitored": False,
    "phone_available": True, "relative_contacted": False, "nurse_available": True,
    "internet_available": True, "device_available": True,
    "rehabilitation_needed": True, "rehabilitation_completed": False,
    "stretching_completed": False, "mobility_checked": False, "mobility_improved": False,
    "muscles_relaxed": False,
    "book_available": True, "reading_completed": False, "cognitive_training_completed": False,
    "quiet_environment": True, "tv_on": False, "lights_on": True,
    "food_available": True, "water_available": True, "good_weather": True,
    "music_device_available": True,
    "breakfast_completed": True, "lunch_completed": False, "dinner_completed": False,
    "morning_routine_completed": True, "evening_routine_completed": False,
}


def test_rehabilitation_chain():
    """Test that 'rehabilitation_completed' decomposes into a valid action chain."""
    gd = GoalDecomposer()

    # Use initial world state but block the direct path (physical_rehabilitation)
    # by setting patient_not_tired = False. Since rest no longer requires the
    # patient to be tired, the chain is rest -> physical_rehabilitation.
    ws = dict(WORLD_STATE)
    ws["patient_not_tired"] = False
    ws["patient_fed"] = False
    ws["stretching_completed"] = False

    chain = gd.decompose_goal("rehabilitation_completed", ws)

    print("\n=== Test: rehabilitation_completed (blocked direct path) ===")
    print(f"Chain length: {len(chain)}")
    for i, action in enumerate(chain):
        print(f"  {i+1}. {action['canonical_action']} - {action['description']}")
        print(f"     preconditions: {action['preconditions']}")
        print(f"     effects: {action['effects']}")

    assert len(chain) > 0, "Expected non-empty action chain"

    # Verify the last action achieves the goal
    assert "rehabilitation_completed" in chain[-1]["effects"], \
        f"Expected last action to achieve rehabilitation_completed, " \
        f"got {chain[-1]['canonical_action']} with effects {chain[-1]['effects']}"

    # Verify all preconditions of each action are either in the world state
    # or produced by an earlier action in the chain
    achieved = set(ws.keys())  # facts already true in world state
    for i, action in enumerate(chain):
        for precond in action["preconditions"]:
            assert precond in achieved, \
                f"Action {i+1} ({action['canonical_action']}) requires " \
                f"'{precond}' but it's not achieved by world state or " \
                f"earlier actions. Achieved so far: {achieved}"
        for effect in action["effects"]:
            achieved.add(effect)

    print("\nAll assertions passed!")

    # Test activities conversion
    activities = gd.actions_to_activities(chain)
    print(f"\nConverted to {len(activities)} activities:")
    for i, act in enumerate(activities):
        print(f"  {i+1}. {act['activity_name']} (ca: {act['canonical_action']})")

    return chain


def test_already_satisfied_goal():
    """Test that an already-satisfied goal returns an empty chain."""
    gd = GoalDecomposer()
    ws = dict(WORLD_STATE)
    ws["medication_taken"] = True

    chain = gd.decompose_goal("medication_taken", ws)
    print(f"\n=== Test: already satisfied goal ===")
    print(f"Chain length: {len(chain)} (expected 0)")
    assert len(chain) == 0, "Expected empty chain for already-satisfied goal"
    print("Passed!")


def test_unprovable_goal():
    """Test that an unprovable goal returns an empty chain."""
    gd = GoalDecomposer()
    ws = dict(WORLD_STATE)

    chain = gd.decompose_goal("nonexistent_goal_predicate", ws)
    print(f"\n=== Test: unprovable goal ===")
    print(f"Chain length: {len(chain)} (expected 0)")
    assert len(chain) == 0, "Expected empty chain for unprovable goal"
    print("Passed!")


if __name__ == "__main__":
    test_rehabilitation_chain()
    test_already_satisfied_goal()
    test_unprovable_goal()
    print("\n=== All tests passed! ===")
