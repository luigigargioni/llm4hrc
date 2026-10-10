"""Scenario definitions for the symbolic layer evaluation suite.

Each scenario is a dict with:
  - id: unique identifier
  - description: human-readable description
  - world_state: dict of facts (str -> bool)
  - activities: list of activity dicts (already canonicalized)
  - fake_datetime: str "HH:MM" for simulated time
  - patient_request: str (simulated patient request)
  - expected: dict with expected outcomes
    - executable: list of activity_names that should be executable
    - blocked_critical: list of activity_names
    - blocked_non_critical: list of activity_names
    - deferred: list of activity_names
    - recovery_actions: dict {activity_name: list of canonical_action names}
"""

from copy import deepcopy

# Base world state. It matches INITIAL_GRAPH_STATE from interfaces.py except
# after_meal, which is True here and False in a live session: scenarios that
# test the meal precondition set it explicitly.
BASE_WORLD_STATE = {
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


def _ws(**overrides):
    """Create a world state from BASE_WORLD_STATE with overrides."""
    ws = deepcopy(BASE_WORLD_STATE)
    ws.update(overrides)
    return ws


def _act(name, ca, critical=False, priority=3, time=None, without_time=True, duration=30, done=False):
    """Create an activity dict."""
    return {
        "activity_name": name,
        "canonical_action": ca,
        "critical": critical,
        "priority": priority,
        "activity_time": time,
        "without_time": without_time,
        "activity_duration": duration,
        "done": done,
        "skipped": False,
        "activity_details": "",
        "execution_notes": None,
    }


SCENARIOS = [

    # ================================================================
    # A. PRECONDITIONS BASE (5 scenarios)
    # ================================================================

    {
        "id": "S01",
        "description": "All preconditions satisfied for critical medication",
        "world_state": _ws(),
        "activities": [
            _act("Take Paracetamol", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "08:00",
        "patient_request": "I want to take my medication",
        "expected": {
            "executable": ["Take Paracetamol"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S02",
        "description": "Critical full rehabilitation blocked by missing patient_fed, recovery available",
        "world_state": _ws(patient_fed=False, stretching_completed=True),
        "activities": [
            _act("Full Rehab Session", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "10:00",
        "patient_request": "I want to do my full rehabilitation",
        "expected": {
            "executable": [],
            "blocked_critical": ["Full Rehab Session"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab Session": ["eat_meal", "have_lunch"],
            },
        },
    },

    {
        "id": "S03",
        "description": "Non-critical stretching blocked by missing patient_fed",
        "world_state": _ws(patient_fed=False),
        "activities": [
            _act("Stretching Session", "stretching_session", critical=False, priority=3),
        ],
        "fake_datetime": "10:00",
        "patient_request": "I want to do some stretching",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Stretching Session"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S04",
        "description": "Unknown canonical action triggers fail-closed",
        "world_state": _ws(),
        "activities": [
            _act("Do some yoga", "unknown", critical=False, priority=3),
        ],
        "fake_datetime": "10:00",
        "patient_request": "I want to do yoga",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Do some yoga"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S05",
        "description": "Critical full rehab blocked by multiple missing preconditions, partial recovery",
        "world_state": _ws(patient_fed=False, stretching_completed=False),
        "activities": [
            _act("Full Rehab Session", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "10:00",
        "patient_request": "I want to do rehabilitation",
        "expected": {
            "executable": [],
            "blocked_critical": ["Full Rehab Session"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab Session": ["eat_meal", "have_lunch"],
            },
        },
    },

    # ================================================================
    # B. TEMPORAL WINDOW (4 scenarios)
    # ================================================================

    {
        "id": "S06",
        "description": "Activity within time window -> EXECUTABLE",
        "world_state": _ws(),
        "activities": [
            _act("Take Paracetamol", "take_medication", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "08:15",
        "patient_request": "Time for my medication?",
        "expected": {
            "executable": ["Take Paracetamol"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S07",
        "description": "Activity too early -> DEFERRED",
        "world_state": _ws(),
        "activities": [
            _act("Take Antibiotic", "take_antibiotic", critical=True, priority=1, time="18:00", without_time=False),
        ],
        "fake_datetime": "10:00",
        "patient_request": "Can I take my antibiotic now?",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": ["Take Antibiotic"],
            "recovery_actions": {},
        },
    },

    {
        "id": "S08",
        "description": "Activity expired (past time window + tolerance) -> BLOCKED_NON_CRITICAL",
        "world_state": _ws(),
        "activities": [
            _act("Take Vitamin D", "take_medication", critical=False, priority=1, time="07:30", without_time=False),
        ],
        "fake_datetime": "12:00",
        "patient_request": "Can I take my vitamin now?",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Take Vitamin D"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S09",
        "description": "Activity without_time -> EXECUTABLE if preconditions OK",
        "world_state": _ws(),
        "activities": [
            _act("Read a book", "reading_activity", critical=False, priority=3, without_time=True),
        ],
        "fake_datetime": "14:00",
        "patient_request": "I want to read something",
        "expected": {
            "executable": ["Read a book"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # C. DEPENDENCY CHAINS (4 scenarios)
    # ================================================================

    {
        "id": "S10",
        "description": "Chain: have_lunch (done) -> stretching (done) -> full_rehab (executable)",
        "world_state": _ws(patient_fed=True, stretching_completed=True),
        "activities": [
            _act("Full Rehab Session", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "14:00",
        "patient_request": "I'm ready for my full rehabilitation",
        "expected": {
            "executable": ["Full Rehab Session"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S11",
        "description": "Chain: full_rehab requested but patient_fed=False -> recovery suggests have_lunch",
        "world_state": _ws(patient_fed=False, stretching_completed=True),
        "activities": [
            _act("Full Rehab Session", "full_rehabilitation_session", critical=True, priority=1),
            _act("Have Lunch", "have_lunch", critical=False, priority=2),
        ],
        "fake_datetime": "12:00",
        "patient_request": "I want to do my rehabilitation now",
        "expected": {
            "executable": ["Have Lunch"],
            "blocked_critical": ["Full Rehab Session"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab Session": ["eat_meal", "have_lunch"],
            },
        },
    },

    {
        "id": "S12",
        "description": "Chain with fatigue: exercise executable, rest allowed although not tired yet",
        "world_state": _ws(patient_not_tired=True, patient_tired=False),
        "activities": [
            _act("Legs Exercise", "exercise", critical=True, priority=2),
            _act("Rest", "rest", critical=False, priority=3, done=False),
        ],
        "fake_datetime": "10:00",
        "patient_request": "I want to exercise",
        "expected": {
            "executable": ["Legs Exercise", "Rest"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S13",
        "description": "Goal decomposition: backward chaining for rehabilitation_completed",
        "world_state": _ws(patient_fed=False, stretching_completed=False, patient_not_tired=True),
        "activities": [
            _act("Full Rehab Session", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "11:00",
        "patient_request": "I want to complete my rehabilitation",
        "expected": {
            "executable": [],
            "blocked_critical": ["Full Rehab Session"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab Session": ["eat_meal", "have_lunch"],
            },
        },
    },

    # ================================================================
    # D. CRITICALITY AND PRIORITY (3 scenarios)
    # ================================================================

    {
        "id": "S14",
        "description": "Critical activity blocked with recovery available",
        "world_state": _ws(patient_fed=False, stretching_completed=True),
        "activities": [
            _act("Full Rehab", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "13:00",
        "patient_request": "Let's do the rehab",
        "expected": {
            "executable": [],
            "blocked_critical": ["Full Rehab"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab": ["eat_meal", "have_lunch"],
            },
        },
    },

    {
        "id": "S15",
        "description": "Critical activity blocked with NO recovery (patient asleep)",
        "world_state": _ws(patient_awake=False, patient_sleeping=True, patient_fed=False, stretching_completed=True),
        "activities": [
            _act("Full Rehab", "full_rehabilitation_session", critical=True, priority=1),
        ],
        "fake_datetime": "13:00",
        "patient_request": "Let's do the rehab",
        "expected": {
            "executable": [],
            "blocked_critical": ["Full Rehab"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S16",
        "description": "Critical vs non-critical both blocked, different statuses",
        "world_state": _ws(patient_fed=False, stretching_completed=False),
        "activities": [
            _act("Full Rehab", "full_rehabilitation_session", critical=True, priority=1),
            _act("Stretching", "stretching_session", critical=False, priority=3),
        ],
        "fake_datetime": "13:00",
        "patient_request": "I want to do rehab and stretching",
        "expected": {
            "executable": [],
            "blocked_critical": ["Full Rehab"],
            "blocked_non_critical": ["Stretching"],
            "deferred": [],
            "recovery_actions": {
                "Full Rehab": ["eat_meal", "have_lunch"],
            },
        },
    },

    # ================================================================
    # E. EDGE CASES AND ROBUSTNESS (4 scenarios)
    # ================================================================

    {
        "id": "S17",
        "description": "Empty world state -> all activities with preconditions fail",
        "world_state": {},
        "activities": [
            _act("Take Meds", "take_medication", critical=True, priority=1),
            _act("Drink Water", "drink_water", critical=False, priority=3),
        ],
        "fake_datetime": "10:00",
        "patient_request": "I want my meds and water",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Meds"],
            "blocked_non_critical": ["Drink Water"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S18",
        "description": "Activity already done -> world_state_updater applies effects, validator skips it",
        "world_state": _ws(medication_taken=False, patient_stabilized=False),
        "activities": [
            _act("Take Paracetamol", "take_medication", critical=True, priority=1, done=True),
            _act("Check Vitals", "check_vitals", critical=False, priority=3),
        ],
        "fake_datetime": "09:00",
        "patient_request": "What's next?",
        "expected": {
            "executable": ["Take Paracetamol", "Check Vitals"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S19",
        "description": "Malformed activity_time -> invalid_activity_time",
        "world_state": _ws(),
        "activities": [
            _act("Take Meds", "take_medication", critical=False, priority=1, time="25:99", without_time=False),
        ],
        "fake_datetime": "10:00",
        "patient_request": "Can I take my meds?",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Take Meds"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S20",
        "description": "All activities executable -> no blocks, no overrides",
        "world_state": _ws(),
        "activities": [
            _act("Drink Water", "drink_water", critical=False, priority=3),
            _act("Read a Book", "reading_activity", critical=False, priority=3),
            _act("Speak with Nurse", "speak_with_nurse", critical=False, priority=2),
        ],
        "fake_datetime": "14:00",
        "patient_request": "What can I do now?",
        "expected": {
            "executable": ["Drink Water", "Read a Book", "Speak with Nurse"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    # ================================================================
    # F. REAL-WORLD CONVERSATION SCENARIOS (8 scenarios)
    # Derived from actual user-study conversations (14 participants,
    # base paper gargioni2026base). Each scenario is grounded in a
    # specific interaction pattern observed during the user study.
    # ================================================================

    {
        "id": "S21",
        "description": "Real: patient sleeping at medication time (User 2, 07:38)",
        "world_state": _ws(patient_awake=False, patient_sleeping=True),
        "activities": [
            _act("Take Ibuprofen", "take_medication", critical=True, priority=1, time="07:30", without_time=False),
        ],
        "fake_datetime": "07:30",
        "patient_request": "(patient sleeping, no request)",
        "expected": {
            "executable": [],
            "blocked_critical": ["Take Ibuprofen"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Take Ibuprofen": ["wake_patient"],
            },
        },
    },

    {
        "id": "S22",
        "description": "Real: medication taken late, window expired (User 9, 08:05 vs 07:30)",
        "world_state": _ws(),
        "activities": [
            _act("Take Ibuprofen", "take_medication", critical=True, priority=1, time="07:30", without_time=False),
        ],
        "fake_datetime": "08:05",
        "patient_request": "Can I still take the pill? Because it's the 8:00",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Take Ibuprofen"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S23",
        "description": "Real: skipped breakfast, antibiotic blocked (User 1, no colazione)",
        "world_state": _ws(after_meal=False, patient_fed=False, breakfast_completed=False),
        "activities": [
            _act("Take Antibiotic", "take_antibiotic", critical=True, priority=1, time="12:00", without_time=False),
        ],
        "fake_datetime": "12:00",
        "patient_request": "non ho ancora fatto colazione",
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
        "id": "S24",
        "description": "Real: patient wants walk before scheduled time (User 2, 09:00 vs 10:00)",
        "world_state": _ws(),
        "activities": [
            _act("Walk in Park", "walk_in_park", critical=False, priority=2, time="10:00", without_time=False),
        ],
        "fake_datetime": "09:00",
        "patient_request": "can i do it now?",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": ["Walk in Park"],
            "recovery_actions": {},
        },
    },

    {
        "id": "S25",
        "description": "Real: tired after walk, rehab blocked (User 5, post-passeggiata)",
        "world_state": _ws(patient_tired=True, patient_not_tired=False, mobility_improved=True, patient_relaxed=True),
        "activities": [
            _act("Physical Rehabilitation", "physical_rehabilitation", critical=True, priority=1, time="14:00", without_time=False),
        ],
        "fake_datetime": "14:00",
        "patient_request": "La passeggiata è stata piacevole, ma ora sono stanca",
        "expected": {
            "executable": [],
            "blocked_critical": ["Physical Rehabilitation"],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {
                "Physical Rehabilitation": ["rest"],
            },
        },
    },

    {
        "id": "S26",
        "description": "Real: correct breakfast-then-antibiotic sequence (User 11, 07:50)",
        "world_state": _ws(after_meal=True, patient_fed=True, breakfast_completed=True),
        "activities": [
            _act("Take Antibiotic", "take_antibiotic", critical=True, priority=1, time="08:00", without_time=False),
        ],
        "fake_datetime": "08:00",
        "patient_request": "Nice breakfast, I'm taking the pill now",
        "expected": {
            "executable": ["Take Antibiotic"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S27",
        "description": "Real: patient wants to read but no book available (User 11+14)",
        "world_state": _ws(book_available=False),
        "activities": [
            _act("Read a Book", "reading_activity", critical=False, priority=3),
        ],
        "fake_datetime": "08:00",
        "patient_request": "I'd like to read for some time",
        "expected": {
            "executable": [],
            "blocked_critical": [],
            "blocked_non_critical": ["Read a Book"],
            "deferred": [],
            "recovery_actions": {},
        },
    },

    {
        "id": "S28",
        "description": "Real: patient wants to rest although not tired, and may (User 14, rest-all-day)",
        "world_state": _ws(patient_tired=False, patient_not_tired=True),
        "activities": [
            _act("Rest", "rest", critical=False, priority=3),
        ],
        "fake_datetime": "10:00",
        "patient_request": "Today i want to rest all the day",
        "expected": {
            "executable": ["Rest"],
            "blocked_critical": [],
            "blocked_non_critical": [],
            "deferred": [],
            "recovery_actions": {},
        },
    },
]
