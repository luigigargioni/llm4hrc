"""Tests for the revision-3 changes to the symbolic layer.

- condition facts lapse (World State Updater), and effects are applied once;
- arbitration edge cases (duplicates, critical unknown, expired + missing);
- dated activity times (a postponed activity moved to another day);
- plain-language rendering of blocked activities for the Next Steps prompt.
"""

import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "offline-test")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph.graph_init  # noqa: F401,E402  - breaks the node import cycle
from copy import deepcopy  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from nodes.world_state_updater_node import (  # noqa: E402
    _patient_report,
    world_state_updater_node,
)
from nodes.plan_validator_node import plan_validator_node  # noqa: E402
from nodes.next_steps_node import (  # noqa: E402
    _activity_view,
    _plain_reason,
    _to_patient,
)
from utils.interfaces import INITIAL_GRAPH_STATE  # noqa: E402

DAY = datetime(2026, 9, 23)


def _activity(name, action, done=False, critical=False, time=None):
    return {
        "activity_name": name,
        "canonical_action": action,
        "critical": critical,
        "priority": 3,
        "activity_time": time,
        "without_time": time is None,
        "activity_duration": 30,
        "done": done,
        "skipped": False,
        "activity_details": "",
        "execution_notes": None,
    }


def _state(activities, at, world_state=None, meta=None):
    state = deepcopy(INITIAL_GRAPH_STATE)
    state["updated_task"] = {"activities": activities}
    state["fake_datetime"] = at
    state["situation_perception"] = "Patient is speaking"
    if world_state is not None:
        state["world_state"] = world_state
    if meta is not None:
        state["world_state_meta"] = meta
    return state


def _by_name(state, name):
    return next(
        a for a in state["updated_task"]["activities"]
        if a["activity_name"] == name
    )


def test_tiredness_lapses():
    """A walk makes the patient tired for two hours, then not tired."""
    walk = _activity("Walk in the park", "walk_in_park", done=True)
    exercises = _activity("Physical exercises", "exercise")

    state = _state([walk, exercises], DAY.replace(hour=10))
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_tired"] is True
    state = plan_validator_node(state)
    assert _by_name(state, "Physical exercises")["executable"] is False

    # One hour later the patient is still tired.
    state["fake_datetime"] = DAY.replace(hour=11)
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_tired"] is True

    # Two hours after the walk the tiredness has lapsed. The walk is still
    # listed as done, but its effects are not re-applied.
    state["fake_datetime"] = DAY.replace(hour=12)
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_tired"] is False
    assert state["world_state"]["patient_not_tired"] is True
    state = plan_validator_node(state)
    assert _by_name(state, "Physical exercises")["executable"] is True
    print("  Passed!")


def test_meal_facts_lapse():
    """patient_fed lapses after four hours, after_meal after two."""
    meal = _activity("Eat breakfast", "eat_meal", done=True)
    state = _state([meal], DAY.replace(hour=8))
    assert state["world_state"]["after_meal"] is False
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_fed"] is True
    assert state["world_state"]["after_meal"] is True

    state["fake_datetime"] = DAY.replace(hour=10, minute=30)
    state = world_state_updater_node(state)
    assert state["world_state"]["after_meal"] is False
    assert state["world_state"]["patient_fed"] is True

    state["fake_datetime"] = DAY.replace(hour=12, minute=30)
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_fed"] is False
    print("  Passed!")


def test_second_activity_renews_fact():
    """A second completed activity sets the fact again and restarts it."""
    walk = _activity("Walk in the park", "walk_in_park", done=True,
                     time="10:00")
    state = _state([walk], DAY.replace(hour=10))
    state = world_state_updater_node(state)

    exercise = _activity("Physical exercises", "exercise", done=True,
                         time="11:30")
    state["updated_task"]["activities"].append(exercise)
    state["fake_datetime"] = DAY.replace(hour=11, minute=30)
    state = world_state_updater_node(state)

    # 12:30 is more than two hours after the walk but not after the exercises.
    state["fake_datetime"] = DAY.replace(hour=12, minute=30)
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_tired"] is True
    state["fake_datetime"] = DAY.replace(hour=13, minute=30)
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_tired"] is False
    print("  Passed!")


def test_patient_report():
    """What the patient says about their tiredness is an observation."""
    corpus = [  # the six statements in the user-study transcripts
        "ok thanks. I'm tired, I go to sleep a little",
        "Hi Robert, I'm a little bit tired. What do you suggest?",
        "i'm sorry my friend but i'm tired today, can i go for a walk "
        "tomorrow?",
        "i'm quite tired, i prefer to relax and listen some music instead",
        "i decide to do my excercise but i'm still a little bit tired, so "
        "can you suggest some light excercise",
        "Siamo tornati... un po' stanca, ma è andata bene.",
    ]
    for text in corpus:
        assert _patient_report(text) == "patient_tired", text
    for text in ("I'm not tired at all", "I don't feel tired",
                 "non sono stanca", "I am no longer tired"):
        assert _patient_report(text) == "patient_not_tired", text
    for text in ("I'm tired of waiting", "sono stanca di aspettare",
                 "Let's go for a walk", None):
        assert _patient_report(text) is None, text

    # In the updater the report overrides the fact inferred from the walk:
    # an hour later the patient says she is not tired and may exercise.
    walk = _activity("Walk in the park", "walk_in_park", done=True)
    exercises = _activity("Physical exercises", "exercise")
    state = _state([walk, exercises], DAY.replace(hour=10))
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_tired"] is True
    state["fake_datetime"] = DAY.replace(hour=11)
    state["patient_request"] = "I'm not tired anymore, let's exercise"
    state = world_state_updater_node(state)
    assert state["world_state"]["patient_not_tired"] is True
    assert state["world_state"]["patient_tired"] is False
    state = plan_validator_node(state)
    assert _by_name(state, "Physical exercises")["executable"] is True

    # And a patient who says she is tired is tired, with no activity done.
    state = _state([exercises], DAY.replace(hour=16))
    state["patient_request"] = "I'm still a little bit tired"
    state = world_state_updater_node(state)
    state = plan_validator_node(state)
    assert _by_name(state, "Physical exercises")["executable"] is False
    print("  Passed!")


def test_rest_needs_no_tiredness():
    """A patient who is not tired may still rest."""
    rest = _activity("Rest", "rest")
    state = _state([rest], DAY.replace(hour=10))
    state = world_state_updater_node(state)
    state = plan_validator_node(state)
    assert _by_name(state, "Rest")["arbitrated_status"] == "EXECUTABLE"
    print("  Passed!")


def test_duplicate_critical_is_non_critical():
    first = _activity("Take Ibuprofen", "take_medication", critical=True)
    second = _activity("Take Ibuprofen", "take_medication", critical=True)
    state = _state([first, second], DAY.replace(hour=10))
    state = plan_validator_node(state)
    activities = state["updated_task"]["activities"]
    assert activities[0]["arbitrated_status"] == "EXECUTABLE"
    assert activities[1]["arbitrated_status"] == "BLOCKED_NON_CRITICAL"
    assert activities[1]["recovery_actions"] == []
    print("  Passed!")


def test_expired_critical_gets_no_recovery():
    """Missing precondition and expired window: no recovery is proposed."""
    antibiotic = _activity("Take Antibiotic", "take_antibiotic",
                           critical=True, time="08:00")
    state = _state([antibiotic], DAY.replace(hour=10))
    state = plan_validator_node(state)
    activity = _by_name(state, "Take Antibiotic")
    assert activity["arbitrated_status"] == "BLOCKED_CRITICAL"
    assert activity["recovery_actions"] == []

    # Inside its window the same activity gets its recovery (a meal).
    antibiotic = _activity("Take Antibiotic", "take_antibiotic",
                           critical=True, time="10:00")
    state = _state([antibiotic], DAY.replace(hour=10))
    state = plan_validator_node(state)
    recovery = _by_name(state, "Take Antibiotic")["recovery_actions"]
    assert "eat_meal" in [r["canonical_action"] for r in recovery]
    print("  Passed!")


def test_critical_unknown_stays_non_critical():
    activity = _activity("Garden therapy", "unknown", critical=True)
    state = _state([activity], DAY.replace(hour=10))
    state = plan_validator_node(state)
    result = _by_name(state, "Garden therapy")
    assert result["arbitrated_status"] == "BLOCKED_NON_CRITICAL"
    print("  Passed!")


def test_dated_times():
    """A walk moved to tomorrow is deferred; one left yesterday expired."""
    tomorrow = (DAY + timedelta(days=1)).strftime("%d-%m-%Y")
    yesterday = (DAY - timedelta(days=1)).strftime("%d-%m-%Y")
    today = DAY.strftime("%d-%m-%Y")
    walks = [
        _activity("Walk tomorrow", "rest", time=f"{tomorrow} 10:00"),
        _activity("Walk yesterday", "rest", time=f"{yesterday} 10:00"),
        _activity("Walk today", "rest", time=f"{today} 10:00"),
    ]
    state = _state(walks, DAY.replace(hour=10))
    state = plan_validator_node(state)
    assert _by_name(state, "Walk tomorrow")["arbitrated_status"] == "DEFERRED"
    assert (_by_name(state, "Walk yesterday")["arbitrated_status"]
            == "BLOCKED_NON_CRITICAL")
    assert _by_name(state, "Walk today")["arbitrated_status"] == "EXECUTABLE"
    print("  Passed!")


def test_plain_language_view():
    """The prompt view carries no identifiers the model could read out."""
    blocked = _activity("Take Antibiotic", "take_antibiotic", critical=True)
    blocked["not_executable_reasons"] = [
        "missing_precondition: after_meal",
        "outside_time_window_early:13:00",
    ]
    blocked["arbitrated_status"] = "BLOCKED_CRITICAL"
    blocked["recovery_actions"] = [
        {"canonical_action": "eat_meal", "description": "Consume a meal"},
    ]
    view = _activity_view(blocked, blocked=True)
    rendered = str({k: v for k, v in view.items() if k != "activity_name"})
    assert "_" not in rendered.replace("why_not_now", "").replace(
        "what_would_make_it_possible", "")
    assert "precondition" not in rendered
    assert "BLOCKED" not in rendered
    assert view["what_would_make_it_possible"] == ["Consume a meal"]
    assert _plain_reason("missing precondition: patient_fed") == (
        "the patient has not eaten yet")
    # Spoken to the patient, the same reasons are in the second person.
    assert _to_patient(_plain_reason("missing_precondition: mobility_checked")
                       ) == "your mobility has not been checked yet"
    assert _to_patient(_plain_reason("missing_precondition: patient_awake")
                       ) == "you are asleep"
    print("  Passed!")


def test_goal_planner_leaves_care_plan_alone():
    """Emergent activities join the session's task, not the care plan file.

    The Goal Planner used to append its fallback activities to
    mock_data/<patient>.txt, which later sessions read back as the care plan.
    """
    import json
    import nodes.goal_planner_node as goal_planner

    class _Reply:
        content = json.dumps({
            "generate_new_activities": True,
            "goal_predicate": None,
            "new_activities": [_activity("Prepare shopping list", None,
                                         time="07:20")],
        })

    class _Model:
        def invoke(self, messages):
            return _Reply()

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mock_data")
    patient_files = [name for name in os.listdir(data_dir)
                     if os.path.isfile(os.path.join(data_dir, name))]
    before = {name: open(os.path.join(data_dir, name), "rb").read()
              for name in patient_files}

    original = goal_planner.define_model
    goal_planner.define_model = lambda *args, **kwargs: _Model()
    try:
        state = _state([_activity("Take Ibuprofen", "take_medication")],
                       DAY.replace(hour=7))
        state["task"] = {"activities": [_activity("Take Ibuprofen",
                                                  "take_medication")]}
        state["patient_id"] = "789"
        state["patient_request"] = "Remind me to prepare a shopping list"
        state = goal_planner.goal_planner_node(state)
    finally:
        goal_planner.define_model = original

    names = [a["activity_name"] for a in state["task"]["activities"]]
    assert "Prepare shopping list" in names
    after = {name: open(os.path.join(data_dir, name), "rb").read()
             for name in os.listdir(data_dir)
             if os.path.isfile(os.path.join(data_dir, name))}
    assert after == before, "the Goal Planner wrote to mock_data"
    print("  Passed!")


def test_task_progress_rejects_truncated_task():
    """A regenerated task missing done/skipped is not accepted.

    In the replay the model once returned two of five activities, one with
    neither flag and its details ending in "(truncated)"; the Robot Effector
    then failed on activity["done"].
    """
    import json
    import nodes.task_progress_node as task_progress
    from edges.robot_effector_conditional_edges import (
        robot_effector_conditional_edges,
    )

    current = [_activity(n, a) for n, a in (
        ("Take Ibuprofen", "take_medication"), ("Walk in the park", None),
        ("Read a book", None))]
    truncated = {"activities": [
        _activity("Take Ibuprofen", "take_medication", done=True),
        {"activity_name": "Walk in the park", "activity_time": "10:00",
         "activity_details": "30-minute... (truncated)"},
    ]}

    class _Result:
        def __init__(self, task):
            self.task = task

        def to_json(self):
            return {"kwargs": {"content": json.dumps(
                {"updated_task": self.task})}}

    class _Model:
        def __init__(self, task):
            self.task = task

        def invoke(self, messages):
            return _Result(self.task)

    original = task_progress.define_model
    try:
        renamed = [dict(a) for a in current]
        renamed[1]["activity_name"] = "Walk outside for mobility"
        cases = (
            (truncated, 3, "Walk in the park"),       # missing flags
            ({"activities": current[:1]}, 3, "Walk in the park"),  # dropped
            ({"activities": renamed}, 3, "Walk outside for mobility"),
        )
        for reply, expected, second in cases:
            task_progress.define_model = (
                lambda *args, reply=reply, **kwargs: _Model(reply))
            state = _state(current, DAY.replace(hour=10))
            state["task"] = {"activities": [dict(a) for a in current]}
            state = task_progress.task_progress_node(state)
            activities = state["updated_task"]["activities"]
            assert len(activities) == expected
            assert activities[1]["activity_name"] == second
    finally:
        task_progress.define_model = original

    # The edge itself no longer fails on a missing flag.
    state = _state([], DAY.replace(hour=10))
    state["task"] = {"activities": [{"activity_name": "Walk in the park"}]}
    assert robot_effector_conditional_edges(state) == "robot_perception_node"
    print("  Passed!")


def test_recovery_message_speaks_to_patient():
    """The override proposes the recovery to the patient, in the second
    person, for every action the dictionary could suggest."""
    from nodes.next_steps_node import _PATIENT_PHRASES, _recovery_message
    from planning.canonical_actions import CANONICAL_ACTIONS

    assert set(CANONICAL_ACTIONS) <= set(_PATIENT_PHRASES)
    message = _recovery_message([{
        "activity_name": "Take Ibuprofen",
        "canonical_action": "take_medication",
        "recovery_actions": [{"canonical_action": "wake_patient",
                              "description": "Wake up the patient"}],
    }])
    assert message == ("I'm sorry, we can't do that just yet. First you "
                       "need to wake up; then you can take your medication.")
    assert "the patient" not in message
    print("  Passed!")


def test_validator_disabled_arm():
    """With VALIDATOR_DISABLED everything is executable, and the decision the
    validator would have taken is still logged."""
    import json
    from utils.structured_logs import _ensure_log_file

    antibiotic = _activity("Take Antibiotic", "take_antibiotic",
                           critical=True)
    os.environ["VALIDATOR_DISABLED"] = "true"
    try:
        state = _state([antibiotic], DAY.replace(hour=10))
        log_path = _ensure_log_file()
        start = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        state = plan_validator_node(state)
    finally:
        os.environ.pop("VALIDATOR_DISABLED")
    activity = _by_name(state, "Take Antibiotic")
    assert activity["executable"] is True
    assert activity["arbitrated_status"] == "EXECUTABLE"
    with open(log_path, encoding="utf-8") as f:
        f.seek(start)
        entries = [json.loads(line) for line in f if line.strip()]
    disabled = [e for e in entries
                if e.get("decision_type") == "validator_disabled"]
    assert disabled and disabled[-1]["would_be_status"] == "BLOCKED_CRITICAL"
    assert "missing_precondition: after_meal" in (
        disabled[-1]["would_be_reasons"])

    # Without the switch the same activity is blocked.
    state = plan_validator_node(_state([antibiotic], DAY.replace(hour=10)))
    assert _by_name(state, "Take Antibiotic")["executable"] is False
    print("  Passed!")


if __name__ == "__main__":
    tests = [
        test_tiredness_lapses,
        test_meal_facts_lapse,
        test_second_activity_renews_fact,
        test_patient_report,
        test_rest_needs_no_tiredness,
        test_duplicate_critical_is_non_critical,
        test_expired_critical_gets_no_recovery,
        test_critical_unknown_stays_non_critical,
        test_dated_times,
        test_plain_language_view,
        test_goal_planner_leaves_care_plan_alone,
        test_task_progress_rejects_truncated_task,
        test_recovery_message_speaks_to_patient,
        test_validator_disabled_arm,
    ]
    for test in tests:
        print(f"\n=== {test.__name__} ===")
        test()
    print("\n=== ALL TESTS PASSED! ===")
