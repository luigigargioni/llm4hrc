"""Failure mode scenarios for robustness analysis (Fase B.4).

Each scenario deliberately injects an error and checks whether the
symbolic layer (plan_validator + world_state_updater) detects it.

Categories:
  A. Wrong canonicalization (LLM maps to wrong canonical action)
  B. Corrupted world state (inconsistent facts)
  C. Conflicting effects (done activity applies contradictory effects)
  D. Missing canonical action (activity without canonical_action field)
  E. Stale world state (effects not applied after activity done)
"""

from copy import deepcopy
from evaluation.scenarios import BASE_WORLD_STATE, _ws, _act


def _act_no_ca(name, critical=False, priority=3, time=None, without_time=True, duration=30, done=False):
    """Create an activity WITHOUT canonical_action field."""
    a = _act(name, "", critical, priority, time, without_time, duration, done)
    del a["canonical_action"]
    return a


FAILURE_MODE_SCENARIOS = [

    # ================================================================
    # A. WRONG CANONICALIZATION (5 scenarios)
    # LLM maps activity to wrong canonical action.
    # The plan_validator should still check preconditions of the
    # (wrong) canonical action — the error is not detected at the
    # validator level, but the validator prevents execution if the
    # wrong action's preconditions happen to be unmet.
    # ================================================================

    {
        "id": "F01",
        "description": "Wrong canonicalization: 'Take Antibiotic' mapped to take_medication instead of take_antibiotic",
        "fault_type": "wrong_canonicalization",
        "injected_error": "canonical_action='take_medication' instead of 'take_antibiotic'",
        "world_state": _ws(after_meal=False),
        "activities": [
            _act("Take Antibiotic", "take_medication", critical=True, priority=1, time="18:00", without_time=False),
        ],
        "fake_datetime": "18:00",
        "expected_detection": True,
        "expected_reason": "take_medication preconditions are met, so the wrong mapping is NOT caught by precondition check. BUT the consistency check detects no keyword overlap between 'Take Antibiotic' and 'take_medication', flagging it as suspicious_mapping.",
        "expected": {
            "executable": ["Take Antibiotic"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F02",
        "description": "Wrong canonicalization detected: 'Take Antibiotic' mapped to take_medication, but medication_available=False blocks it",
        "fault_type": "wrong_canonicalization",
        "injected_error": "canonical_action='take_medication' instead of 'take_antibiotic'",
        "world_state": _ws(medication_available=False, after_meal=True),
        "activities": [
            _act("Take Antibiotic", "take_medication", critical=True, priority=1, time="18:00", without_time=False),
        ],
        "fake_datetime": "18:00",
        "expected_detection": True,
        "expected_reason": "take_medication requires medication_available which is False, so the activity is blocked. The wrong mapping is accidentally caught because the wrong action also fails.",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Antibiotic"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F03",
        "description": "Wrong canonicalization: 'Read a book' mapped to exercise instead of reading_activity",
        "fault_type": "wrong_canonicalization",
        "injected_error": "canonical_action='exercise' instead of 'reading_activity'",
        "world_state": _ws(patient_not_tired=False, patient_tired=True),
        "activities": [
            _act("Read a book", "exercise", critical=False, priority=3),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "exercise requires patient_not_tired which is False, so the wrong mapping is accidentally caught.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Read a book"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F04",
        "description": "Wrong canonicalization NOT detected: 'Call relative' mapped to social_interaction (both have similar preconditions met)",
        "fault_type": "wrong_canonicalization",
        "injected_error": "canonical_action='social_interaction' instead of 'call_relative'",
        "world_state": _ws(patient_emotionally_stable=True),
        "activities": [
            _act("Call my daughter", "social_interaction", critical=False, priority=2),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "social_interaction preconditions are met, so the wrong mapping is NOT caught by precondition check. BUT the consistency check detects no keyword overlap between 'Call my daughter' and 'social_interaction', flagging it as suspicious_mapping.",
        "expected": {
            "executable": ["Call my daughter"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F05",
        "description": "Wrong canonicalization: 'Walk in park' mapped to exercise (loses good_weather check)",
        "fault_type": "wrong_canonicalization",
        "injected_error": "canonical_action='exercise' instead of 'walk_in_park'",
        "world_state": _ws(good_weather=False, patient_not_tired=True),
        "activities": [
            _act("Walk in the park", "exercise", critical=False, priority=3),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "exercise does not check good_weather, so the wrong mapping is NOT caught by precondition check. BUT the consistency check detects no keyword overlap between 'Walk in the park' and 'exercise', flagging it as suspicious_mapping.",
        "expected": {
            "executable": ["Walk in the park"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # B. CORRUPTED WORLD STATE (4 scenarios)
    # World state has inconsistent facts.
    # The validator does not cross-check fact consistency — it only
    # checks if required preconditions are True.
    # ================================================================

    {
        "id": "F06",
        "description": "Corrupted world state: patient_awake=False AND patient_sleeping=False (impossible state)",
        "fault_type": "corrupted_world_state",
        "injected_error": "patient_awake=False, patient_sleeping=False (both false is inconsistent)",
        "world_state": _ws(patient_awake=False, patient_sleeping=False),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "08:00",
        "expected_detection": True,
        "expected_reason": "take_medication requires patient_awake which is False, so the activity is blocked. The corruption is partially detected (activity blocked) but the inconsistency itself is not flagged.",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Meds"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F07",
        "description": "Corrupted world state: patient_tired=True AND patient_not_tired=True (contradictory)",
        "fault_type": "corrupted_world_state",
        "injected_error": "patient_tired=True, patient_not_tired=True (both true is contradictory)",
        "world_state": _ws(patient_tired=True, patient_not_tired=True),
        "activities": [
            _act("Exercise", "exercise", critical=True, priority=2),
            _act("Rest", "rest", critical=False, priority=3),
        ],
        "fake_datetime": "10:00",
        "expected_detection": True,
        "expected_reason": "The world state invariant validator detects that both patient_tired and patient_not_tired are True (contradictory) and logs an invariant_violation.",
        "expected": {
            "executable": ["Exercise", "Rest"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F08",
        "description": "Corrupted world state: patient_fed=True but lunch_completed=False and after_meal=False (inconsistent)",
        "fault_type": "corrupted_world_state",
        "injected_error": "patient_fed=True but lunch_completed=False, after_meal=False (should be consistent)",
        "world_state": _ws(patient_fed=True, lunch_completed=False, after_meal=False),
        "activities": [
            _act("Take Antibiotic", "take_antibiotic", critical=True, priority=1, time="18:00", without_time=False),
        ],
        "fake_datetime": "18:00",
        "expected_detection": True,
        "expected_reason": "take_antibiotic requires after_meal which is False, so the activity is blocked despite patient_fed being True. The inconsistency is partially exposed.",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Antibiotic"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Take Antibiotic": ["eat_meal"],
            },
        },
    },

    {
        "id": "F09",
        "description": "Corrupted world state: all facts False (total corruption)",
        "fault_type": "corrupted_world_state",
        "injected_error": "All world state facts set to False",
        "world_state": {k: False for k in BASE_WORLD_STATE},
        "activities": [
            _act("Drink Water", "drink_water", critical=False, priority=3),
        ],
        "fake_datetime": "10:00",
        "expected_detection": True,
        "expected_reason": "All preconditions fail, activity is blocked. The total corruption is detected as blocked activity.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Drink Water"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # C. CONFLICTING EFFECTS (3 scenarios)
    # Done activities apply effects that conflict with current world state.
    # ================================================================

    {
        "id": "F10",
        "description": "Conflicting effects: done activity sets patient_tired=True but world_state has patient_not_tired=True",
        "fault_type": "conflicting_effects",
        "injected_error": "exercise done=True applies patient_tired=True, but world_state still has patient_not_tired=True (not updated)",
        "world_state": _ws(patient_not_tired=True, patient_tired=False),
        "activities": [
            _act("Exercise", "exercise", critical=True, priority=2, done=True),
            _act("Rest", "rest", critical=False, priority=3),
        ],
        "fake_datetime": "11:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies exercise effects (patient_tired=True) and auto-sets patient_not_tired=False via mutual exclusivity. The mutual_exclusivity_correction log entry now makes this auto-correction auditable, so the fault is detected + corrected.",
        "expected": {
            "executable": ["Exercise", "Rest"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F11",
        "description": "Conflicting effects: done stretching sets stretching_completed=True, but full_rehab still blocked by patient_fed",
        "fault_type": "conflicting_effects",
        "injected_error": "stretching done=True but patient_fed still False (inconsistent: stretching requires patient_fed)",
        "world_state": _ws(patient_fed=False, stretching_completed=False),
        "activities": [
            _act("Stretching", "stretching_session", critical=False, priority=2, done=True),
            _act("Full Rehab", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies stretching effects (stretching_completed=True). But full_rehab still needs patient_fed=False -> blocked. The inconsistency (stretching done without patient_fed) is NOT detected, but full_rehab is still correctly blocked.",
        "expected": {
            "executable": ["Stretching"],
            "blocked_critical": ["Full Rehab"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab": ["eat_meal", "have_lunch"],
            },
        },
    },

    {
        "id": "F12",
        "description": "Conflicting effects: done activity sets medication_taken=True but world_state already had it True (idempotent, no conflict)",
        "fault_type": "conflicting_effects",
        "injected_error": "take_medication done=True but medication_taken already True (double application)",
        "world_state": _ws(medication_taken=True, patient_stabilized=True),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, done=True),
            _act("Check Vitals", "check_vitals", critical=False, priority=3),
        ],
        "fake_datetime": "09:00",
        "expected_detection": True,
        "expected_reason": "Idempotent application: effects already True. The idempotent_reapplication log entry now flags this, making the double application auditable.",
        "expected": {
            "executable": ["Take Meds", "Check Vitals"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # D. MISSING CANONICAL ACTION (3 scenarios)
    # Activity without canonical_action field (canonicalizer failed silently).
    # ================================================================

    {
        "id": "F13",
        "description": "Missing canonical_action: activity has no canonical_action field",
        "fault_type": "missing_canonical_action",
        "injected_error": "canonical_action field is missing from activity dict",
        "world_state": _ws(),
        "activities": [
            _act_no_ca("Do something", critical=True, priority=1),
        ],
        "fake_datetime": "10:00",
        "expected_detection": True,
        "expected_reason": "canonical_action is None, not in CANONICAL_ACTIONS, not 'unknown'. The validator falls through to the precondition check but canonical_action is None so it's not 'unknown' and not in CANONICAL_ACTIONS. Activity should be blocked.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Do something"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F14",
        "description": "Missing canonical_action on non-critical activity",
        "fault_type": "missing_canonical_action",
        "injected_error": "canonical_action field is missing from activity dict",
        "world_state": _ws(),
        "activities": [
            _act_no_ca("Relax a bit", critical=False, priority=3),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "Same as F13 but non-critical. Should be blocked.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Relax a bit"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F15",
        "description": "Missing canonical_action among valid activities (mixed)",
        "fault_type": "missing_canonical_action",
        "injected_error": "One activity has no canonical_action, one has valid one",
        "world_state": _ws(),
        "activities": [
            _act_no_ca("Mystery activity", critical=False, priority=3),
            _act("Read a book", "reading_activity", critical=False, priority=3),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "Mystery activity should be blocked, Read a book should be executable.",
        "expected": {
            "executable": ["Read a book"],
            "blocked_critical": [],
            "blocked_non_critical": ["Mystery activity"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # E. STALE WORLD STATE (3 scenarios)
    # Effects not applied: activity is done but world_state was not updated.
    # ================================================================

    {
        "id": "F16",
        "description": "Stale world state: have_lunch done=True but patient_fed still False (effects not applied)",
        "fault_type": "stale_world_state",
        "injected_error": "have_lunch done=True but patient_fed=False (world_state_updater should fix this)",
        "world_state": _ws(patient_fed=False, lunch_completed=False),
        "activities": [
            _act("Have Lunch", "have_lunch", critical=False, priority=2, done=True),
            _act("Full Rehab", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies have_lunch effects (patient_fed=True, lunch_completed=True). Then plan_validator sees patient_fed=True and stretching_completed=False -> full_rehab blocked by stretching. The stale state is AUTO-CORRECTED by the updater.",
        "expected": {
            "executable": ["Have Lunch"],
            "blocked_critical": ["Full Rehab"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab": ["stretching_exercises", "stretching_session"],
            },
        },
    },

    {
        "id": "F17",
        "description": "Stale world state: stretching done=True but stretching_completed still False (updater corrects it)",
        "fault_type": "stale_world_state",
        "injected_error": "stretching done=True but stretching_completed=False (world_state_updater should fix this)",
        "world_state": _ws(patient_fed=True, stretching_completed=False),
        "activities": [
            _act("Stretching", "stretching_session", critical=False, priority=2, done=True),
            _act("Full Rehab", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies stretching effects (stretching_completed=True). Then plan_validator sees both preconditions met -> full_rehab executable. The stale state is AUTO-CORRECTED.",
        "expected": {
            "executable": ["Stretching", "Full Rehab"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F18",
        "description": "Stale world state NOT corrected: activity done=True but canonical_action is wrong (effects don't match)",
        "fault_type": "stale_world_state",
        "injected_error": "have_lunch done=True but canonical_action='exercise' (wrong mapping, wrong effects applied)",
        "world_state": _ws(patient_fed=False, patient_not_tired=True, patient_tired=False),
        "activities": [
            _act("Have Lunch", "exercise", critical=False, priority=2, done=True),
            _act("Full Rehab", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies exercise effects (patient_tired=True) instead of have_lunch effects (patient_fed=True). patient_fed stays False -> full_rehab blocked. The wrong effects are applied, but the validator still catches the missing patient_fed.",
        "expected": {
            "executable": ["Have Lunch"],
            "blocked_critical": ["Full Rehab"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab": ["eat_meal", "have_lunch"],
            },
        },
    },

    # ================================================================
    # A2. WRONG CANONICALIZATION — additional scenarios (F19-F21)
    # ================================================================

    {
        "id": "F19",
        "description": "Wrong canonicalization: 'Memory game' mapped to reading_activity instead of memory_training",
        "fault_type": "wrong_canonicalization",
        "injected_error": "canonical_action='reading_activity' instead of 'memory_training'",
        "world_state": _ws(book_available=True, patient_focused=False),
        "activities": [
            _act("Memory game", "reading_activity", critical=False, priority=3),
        ],
        "fake_datetime": "15:00",
        "expected_detection": True,
        "expected_reason": "reading_activity requires book_available (True) and patient_awake (True), so preconditions are met. BUT consistency check flags no keyword overlap between 'Memory game' and 'reading_activity' -> suspicious_mapping.",
        "expected": {
            "executable": ["Memory game"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F20",
        "description": "Wrong canonicalization: 'Play some music' mapped to emotional_support (preconditions differ)",
        "fault_type": "wrong_canonicalization",
        "injected_error": "canonical_action='emotional_support' instead of 'play_music'",
        "world_state": _ws(patient_sad=False, patient_emotionally_stable=True, music_device_available=True),
        "activities": [
            _act("Play some music", "emotional_support", critical=False, priority=3),
        ],
        "fake_datetime": "16:00",
        "expected_detection": True,
        "expected_reason": "emotional_support requires patient_sad which is False, so the activity is blocked by precondition check. The wrong mapping is accidentally caught.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Play some music"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F21",
        "description": "Wrong canonicalization: 'Drink water' mapped to take_medication (lexically similar 'take')",
        "fault_type": "wrong_canonicalization",
        "injected_error": "canonical_action='take_medication' instead of 'drink_water'",
        "world_state": _ws(medication_available=True, correct_time=True, water_available=True),
        "activities": [
            _act("Drink water", "take_medication", critical=False, priority=3),
        ],
        "fake_datetime": "10:00",
        "expected_detection": True,
        "expected_reason": "take_medication preconditions are met (medication_available, correct_time, patient_awake), so the wrong mapping is NOT caught by precondition check. BUT consistency check flags no keyword overlap between 'Drink water' and 'take_medication' (after stopword removal 'take' is removed) -> suspicious_mapping.",
        "expected": {
            "executable": ["Drink water"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # B2. CORRUPTED WORLD STATE — additional scenarios (F22-F24)
    # ================================================================

    {
        "id": "F22",
        "description": "Corrupted world state: patient_sad=True AND patient_emotionally_stable=True (contradictory)",
        "fault_type": "corrupted_world_state",
        "injected_error": "patient_sad=True, patient_emotionally_stable=True (both true is contradictory)",
        "world_state": _ws(patient_sad=True, patient_emotionally_stable=True),
        "activities": [
            _act("Emotional support", "emotional_support", critical=False, priority=3),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "The world state invariant validator detects that both patient_sad and patient_emotionally_stable are True (contradictory) and logs an invariant_violation.",
        "expected": {
            "executable": ["Emotional support"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F23",
        "description": "Corrupted world state: patient_awake=True AND patient_sleeping=True (contradictory)",
        "fault_type": "corrupted_world_state",
        "injected_error": "patient_awake=True, patient_sleeping=True (both true is contradictory)",
        "world_state": _ws(patient_awake=True, patient_sleeping=True),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "08:00",
        "expected_detection": True,
        "expected_reason": "The world state invariant validator detects that both patient_awake and patient_sleeping are True (contradictory) and logs an invariant_violation. take_medication preconditions are met so it is executable, but the corruption is flagged.",
        "expected": {
            "executable": ["Take Meds"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F24",
        "description": "Corrupted world state: medication_taken=True but medication_available=False (inconsistent)",
        "fault_type": "corrupted_world_state",
        "injected_error": "medication_taken=True but medication_available=False (should not be possible)",
        "world_state": _ws(medication_taken=True, medication_available=False),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "08:00",
        "expected_detection": True,
        "expected_reason": "take_medication requires medication_available which is False, so the activity is blocked despite medication_taken already being True. The inconsistency is partially exposed via the block.",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Meds"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # C2. CONFLICTING EFFECTS — additional scenarios (F25-F27)
    # ================================================================

    {
        "id": "F25",
        "description": "Conflicting effects: done emotional_support sets patient_emotionally_stable=True but world_state has patient_sad=True",
        "fault_type": "conflicting_effects",
        "injected_error": "emotional_support done=True applies patient_emotionally_stable=True, but world_state has patient_sad=True (not updated)",
        "world_state": _ws(patient_sad=True, patient_emotionally_stable=False),
        "activities": [
            _act("Emotional support", "emotional_support", critical=False, priority=3, done=True),
            _act("Call relative", "call_relative", critical=False, priority=2),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies emotional_support effects (patient_emotionally_stable=True) and auto-sets patient_sad=False via mutual exclusivity. The mutual_exclusivity_correction log makes this auditable.",
        "expected": {
            "executable": ["Emotional support", "Call relative"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F26",
        "description": "Conflicting effects: done wake_patient sets patient_awake=True but world_state has patient_sleeping=True",
        "fault_type": "conflicting_effects",
        "injected_error": "wake_patient done=True applies patient_awake=True, but world_state has patient_sleeping=True (not updated)",
        "world_state": _ws(patient_awake=False, patient_sleeping=True),
        "activities": [
            _act("Wake patient", "wake_patient", critical=True, priority=1, done=True),
            _act("Take Meds", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "08:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies wake_patient effects (patient_awake=True) and auto-sets patient_sleeping=False via mutual exclusivity. The mutual_exclusivity_correction log makes this auditable.",
        "expected": {
            "executable": ["Wake patient", "Take Meds"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F27",
        "description": "Conflicting effects: done rest sets patient_not_tired=True but world_state has patient_tired=True (idempotent + correction)",
        "fault_type": "conflicting_effects",
        "injected_error": "rest done=True applies patient_not_tired=True, but world_state has patient_tired=True (not updated)",
        "world_state": _ws(patient_tired=True, patient_not_tired=False, patient_rested=False),
        "activities": [
            _act("Rest", "rest", critical=False, priority=3, done=True),
            _act("Exercise", "exercise", critical=False, priority=2),
        ],
        "fake_datetime": "15:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies rest effects (patient_not_tired=True, patient_rested=True) and auto-sets patient_tired=False via mutual exclusivity. The mutual_exclusivity_correction log makes this auditable.",
        "expected": {
            "executable": ["Rest", "Exercise"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # D2. MISSING CANONICAL ACTION — additional scenarios (F28-F30)
    # ================================================================

    {
        "id": "F28",
        "description": "Missing canonical_action: activity has canonical_action='unknown' (canonicalizer fail-closed)",
        "fault_type": "missing_canonical_action",
        "injected_error": "canonical_action='unknown' (canonicalizer returned unrecognized name)",
        "world_state": _ws(),
        "activities": [
            _act("Do something weird", "unknown", critical=True, priority=1),
        ],
        "fake_datetime": "10:00",
        "expected_detection": True,
        "expected_reason": "canonical_action is 'unknown', which the validator treats as non-executable (fail-closed). Activity is blocked.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Do something weird"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F29",
        "description": "Missing canonical_action: activity has canonical_action='nonexistent_action' (not in dictionary)",
        "fault_type": "missing_canonical_action",
        "injected_error": "canonical_action='nonexistent_action' (not in CANONICAL_ACTIONS)",
        "world_state": _ws(),
        "activities": [
            _act("Mystery task", "nonexistent_action", critical=False, priority=3),
        ],
        "fake_datetime": "11:00",
        "expected_detection": True,
        "expected_reason": "canonical_action is 'nonexistent_action', not in CANONICAL_ACTIONS and not 'unknown'. The validator falls through and treats it as non-executable.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Mystery task"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F30",
        "description": "Missing canonical_action: multiple activities, one missing, rest valid",
        "fault_type": "missing_canonical_action",
        "injected_error": "One of three activities has canonical_action='unknown'",
        "world_state": _ws(),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
            _act("Unknown task", "unknown", critical=False, priority=3),
            _act("Read a book", "reading_activity", critical=False, priority=3),
        ],
        "fake_datetime": "08:00",
        "expected_detection": True,
        "expected_reason": "Take Meds and Read a book are executable; Unknown task is blocked (fail-closed). The missing canonical action is detected.",
        "expected": {
            "executable": ["Take Meds", "Read a book"],
            "blocked_critical": [],
            "blocked_non_critical": ["Unknown task"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # E2. STALE WORLD STATE — additional scenarios (F31-F33)
    # ================================================================

    {
        "id": "F31",
        "description": "Stale world state: take_medication done=True but medication_taken still False (updater corrects)",
        "fault_type": "stale_world_state",
        "injected_error": "take_medication done=True but medication_taken=False (world_state_updater should fix this)",
        "world_state": _ws(medication_taken=False, patient_stabilized=False),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, done=True),
            _act("Check Vitals", "check_vitals", critical=False, priority=3),
        ],
        "fake_datetime": "09:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies take_medication effects (medication_taken=True, patient_stabilized=True). The stale state is AUTO-CORRECTED by the updater.",
        "expected": {
            "executable": ["Take Meds", "Check Vitals"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F32",
        "description": "Stale world state: drink_water done=True but patient_hydrated still False (updater corrects)",
        "fault_type": "stale_world_state",
        "injected_error": "drink_water done=True but patient_hydrated=False (world_state_updater should fix this)",
        "world_state": _ws(patient_hydrated=False, water_available=True),
        "activities": [
            _act("Drink Water", "drink_water", critical=False, priority=3, done=True),
        ],
        "fake_datetime": "10:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies drink_water effects (patient_hydrated=True). The stale state is AUTO-CORRECTED.",
        "expected": {
            "executable": ["Drink Water"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F33",
        "description": "Stale world state: eat_meal done=True but patient_fed and after_meal still False (updater corrects)",
        "fault_type": "stale_world_state",
        "injected_error": "eat_meal done=True but patient_fed=False, after_meal=False (world_state_updater should fix this)",
        "world_state": _ws(patient_fed=False, after_meal=False, food_available=True),
        "activities": [
            _act("Eat Meal", "eat_meal", critical=False, priority=2, done=True),
            _act("Take Antibiotic", "take_antibiotic", critical=True, priority=1, time="18:00", without_time=False),
        ],
        "fake_datetime": "18:00",
        "expected_detection": True,
        "expected_reason": "world_state_updater applies eat_meal effects (patient_fed=True, after_meal=True). Then take_antibiotic preconditions are met -> executable. The stale state is AUTO-CORRECTED.",
        "expected": {
            "executable": ["Eat Meal", "Take Antibiotic"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # F. WRONG TEMPORAL WINDOW (6 scenarios) — NEW CATEGORY
    # Activity has a malformed or impossible scheduled time.
    # The validator should catch the ValueError and block the activity.
    # ================================================================

    {
        "id": "F34",
        "description": "Wrong temporal window: activity_time='25:99' (impossible hour/minute)",
        "fault_type": "wrong_temporal_window",
        "injected_error": "activity_time='25:99' is not a valid HH:MM time",
        "world_state": _ws(),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="25:99", without_time=False),
        ],
        "fake_datetime": "08:00",
        "expected_detection": True,
        "expected_reason": "The validator tries to parse '25:99' with strptime, raises ValueError, and marks the activity non-executable with reason 'invalid_activity_time'. Since the activity is critical, arbitration assigns BLOCKED_CRITICAL.",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Meds"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F35",
        "description": "Wrong temporal window: activity_time='not-a-time' (non-numeric)",
        "fault_type": "wrong_temporal_window",
        "injected_error": "activity_time='not-a-time' is not a valid time format",
        "world_state": _ws(),
        "activities": [
            _act("Exercise", "exercise", critical=False, priority=3, time="not-a-time", without_time=False),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "The validator tries to parse 'not-a-time', raises ValueError, marks non-executable with 'invalid_activity_time'.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Exercise"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F36",
        "description": "Wrong temporal window: activity_time='' (empty string)",
        "fault_type": "wrong_temporal_window",
        "injected_error": "activity_time='' (empty string, not None)",
        "world_state": _ws(),
        "activities": [
            _act("Read a book", "reading_activity", critical=False, priority=3, time="", without_time=False),
        ],
        "fake_datetime": "15:00",
        "expected_detection": True,
        "expected_reason": "The validator now explicitly checks for an empty-string activity_time and blocks the activity with a malformed_activity_time:empty_string reason, logging a malformed_temporal_input entry. The malformed time IS caught.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Read a book"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F37",
        "description": "Wrong temporal window: activity_time='12:00' but without_time=True (contradictory flags)",
        "fault_type": "wrong_temporal_window",
        "injected_error": "activity_time='12:00' but without_time=True (contradictory: time set but flagged as no-time)",
        "world_state": _ws(),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="12:00", without_time=True),
        ],
        "fake_datetime": "08:00",
        "expected_detection": True,
        "expected_reason": "The validator now explicitly checks for contradictory temporal flags (activity_time set while without_time=True) and blocks the activity with a contradictory_time_flags reason, logging a malformed_temporal_input entry. The contradiction IS caught.",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Meds"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F38",
        "description": "Wrong temporal window: activity_time='12:00' with current time far outside window",
        "fault_type": "wrong_temporal_window",
        "injected_error": "activity scheduled at 12:00 but current time is 03:00 (far outside ±30min window)",
        "world_state": _ws(),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="12:00", without_time=False),
        ],
        "fake_datetime": "03:00",
        "expected_detection": True,
        "expected_reason": "Current time 03:00 is before the earliest window (11:30), so the activity is marked DEFERRED (early, not expired). The temporal fault is detected.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": ["Take Meds"],
            "recovery_actions": {},
        },
    },

    {
        "id": "F39",
        "description": "Wrong temporal window: activity_time='08:00' with current time far past window",
        "fault_type": "wrong_temporal_window",
        "injected_error": "activity scheduled at 08:00 but current time is 23:00 (far past ±30min window)",
        "world_state": _ws(),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "23:00",
        "expected_detection": True,
        "expected_reason": "Current time 23:00 is past the latest window (08:30), so the activity is marked BLOCKED_NON_CRITICAL (expired). The temporal fault is detected.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Take Meds"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # G. ORPHAN PRECONDITION (6 scenarios) — NEW CATEGORY
    # A precondition that is not an effect of any canonical action,
    # meaning it can never be achieved by any action chain (only by
    # initial world state). If it is False, the dependent activity is
    # permanently blocked with no recovery action available.
    # ================================================================

    {
        "id": "F40",
        "description": "Orphan precondition: take_medication requires 'correct_time' (not an effect of any action)",
        "fault_type": "orphan_precondition",
        "injected_error": "correct_time=False (orphan: no action sets correct_time=True, so it is permanently unachievable)",
        "world_state": _ws(correct_time=False),
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "08:00",
        "expected_detection": True,
        "expected_reason": "take_medication requires correct_time which is False. No canonical action has correct_time in its effects, so no recovery action is available. The activity is BLOCKED_CRITICAL with no recovery — the orphan precondition is indirectly detected via the missing precondition + empty recovery.",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Meds"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F41",
        "description": "Orphan precondition: call_relative requires 'phone_available' (not an effect of any action)",
        "fault_type": "orphan_precondition",
        "injected_error": "phone_available=False (orphan: no action sets phone_available=True)",
        "world_state": _ws(phone_available=False),
        "activities": [
            _act("Call relative", "call_relative", critical=False, priority=2),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "call_relative requires phone_available which is False. No canonical action has phone_available in its effects, so no recovery action is available. The activity is BLOCKED_NON_CRITICAL — the orphan precondition is indirectly detected via the missing precondition.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Call relative"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F42",
        "description": "Orphan precondition: walk_in_park requires 'good_weather' (not an effect of any action)",
        "fault_type": "orphan_precondition",
        "injected_error": "good_weather=False (orphan: no action sets good_weather=True)",
        "world_state": _ws(good_weather=False),
        "activities": [
            _act("Walk in park", "walk_in_park", critical=False, priority=3),
        ],
        "fake_datetime": "14:00",
        "expected_detection": True,
        "expected_reason": "walk_in_park requires good_weather which is False. No canonical action has good_weather in its effects. The activity is BLOCKED_NON_CRITICAL — the orphan precondition is indirectly detected.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Walk in park"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F43",
        "description": "Orphan precondition: reading_activity requires 'book_available' (not an effect of any action)",
        "fault_type": "orphan_precondition",
        "injected_error": "book_available=False (orphan: no action sets book_available=True)",
        "world_state": _ws(book_available=False),
        "activities": [
            _act("Read a book", "reading_activity", critical=False, priority=3),
        ],
        "fake_datetime": "15:00",
        "expected_detection": True,
        "expected_reason": "reading_activity requires book_available which is False. No canonical action has book_available in its effects. The activity is BLOCKED_NON_CRITICAL.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Read a book"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F44",
        "description": "Orphan precondition: speak_with_nurse requires 'nurse_available' (not an effect of any action)",
        "fault_type": "orphan_precondition",
        "injected_error": "nurse_available=False (orphan: no action sets nurse_available=True)",
        "world_state": _ws(nurse_available=False),
        "activities": [
            _act("Speak with nurse", "speak_with_nurse", critical=False, priority=3),
        ],
        "fake_datetime": "11:00",
        "expected_detection": True,
        "expected_reason": "speak_with_nurse requires nurse_available which is False. No canonical action has nurse_available in its effects. The activity is BLOCKED_NON_CRITICAL.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Speak with nurse"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "F45",
        "description": "Orphan precondition: video_call_relative requires 'internet_available' (not an effect of any action)",
        "fault_type": "orphan_precondition",
        "injected_error": "internet_available=False (orphan: no action sets internet_available=True)",
        "world_state": _ws(internet_available=False),
        "activities": [
            _act("Video call relative", "video_call_relative", critical=False, priority=3),
        ],
        "fake_datetime": "16:00",
        "expected_detection": True,
        "expected_reason": "video_call_relative requires internet_available which is False. No canonical action has internet_available in its effects. The activity is BLOCKED_NON_CRITICAL.",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Video call relative"],
            "deferred": [],
            "recovery_actions": {},
        },
    },
]
