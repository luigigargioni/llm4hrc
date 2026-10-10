CANONICAL_ACTIONS = {

    # ---------------------------------------------------
    # BASIC HEALTHCARE
    # ---------------------------------------------------

    "take_medication": {

        "description": "Take prescribed medication",

        "aliases": [
            "pill",
            "pills",
            "tablet",
            "capsule",
            "medicine",
            "drug",
            "paracetamol",
            "ibuprofen",
            "vitamin",
            "supplement",
            "dose",
        ],

        "preconditions": [
            "patient_awake",
            "medication_available",
            "correct_time"
        ],

        "effects": [
            "medication_taken",
            "patient_stabilized"
        ]
    },

    # A medication the care plan says to take after a meal ("Take with water
    # after breakfast"). take_medication carries no meal precondition, so the
    # dependency the clinician wrote was invisible to the validator.
    "take_medication_after_meal": {

        "description": "Take prescribed medication that must follow a meal",

        "aliases": [
            "after breakfast",
            "after lunch",
            "after a meal",
            "after eating",
            "with food",
        ],

        "preconditions": [
            "patient_awake",
            "medication_available",
            "correct_time",
            "after_meal"
        ],

        "effects": [
            "medication_taken",
            "patient_stabilized"
        ]
    },

    "take_antibiotic": {

        "description": "Take antibiotic medication",

        "aliases": [
            "antibiotic",
            "amoxicillin",
            "pill",
            "tablet",
            "medicine",
            "infection",
        ],

        "preconditions": [
            "patient_awake",
            "medication_available",
            "after_meal"
        ],

        "effects": [
            "infection_treated",
            "medication_taken"
        ]
    },

    "drink_water": {

        "description": "Drink water for hydration",

        "aliases": [
            "water",
            "drink",
            "hydrate",
            "glass",
        ],

        "preconditions": [
            "patient_awake",
            "water_available"
        ],

        "effects": [
            "patient_hydrated"
        ]
    },

    "eat_meal": {

        "description": "Consume a meal",

        "aliases": [
            "meal",
            "eat",
            "food",
            "breakfast",
            "dinner",
            "snack",
        ],

        "preconditions": [
            "food_available",
            "patient_awake"
        ],

        "effects": [
            "patient_fed",
            "after_meal"
        ]
    },

    "have_lunch": {

        "description": "Have lunch, the main meal of the day",

        "aliases": [
            "lunch",
            "meal",
            "eat",
            "food",
        ],

        "preconditions": [
            "patient_awake",
            "patient_available"
        ],

        # after_meal as well: a medication due after lunch must be unblocked
        # by lunch, not only by eat_meal.
        "effects": [
            "patient_fed",
            "lunch_completed",
            "after_meal"
        ]
    },

    # ---------------------------------------------------
    # PHYSICAL REHABILITATION
    # ---------------------------------------------------

    "physical_rehabilitation": {

        "description": "Perform physical rehabilitation exercises",

        "aliases": [
            "rehab",
            "rehabilitation",
            "physio",
            "therapy",
            "exercise",
        ],

        "preconditions": [
            "patient_awake",
            "patient_available",
            "patient_not_tired"
        ],

        "effects": [
            "rehabilitation_completed",
            "patient_tired"
        ]
    },

    "stretching_exercises": {

        "description": "Perform stretching exercises",

        "aliases": [
            "stretch",
            "stretching",
            "exercise",
            "legs",
        ],

        "preconditions": [
            "patient_awake",
            "patient_available",
            "patient_fed",
            "patient_not_tired"
        ],

        "effects": [
            "muscles_relaxed",
            "patient_tired",
            "stretching_completed"
        ]
    },

    "stretching_session": {

        "description": "Perform a stretching session for rehabilitation",

        "aliases": [
            "stretch",
            "stretching",
            "session",
        ],

        # Same requirements as stretching_exercises: the decomposer picks
        # this action for rehabilitation_completed, and without them it
        # would plan a stretching session for a sleeping patient.
        "preconditions": [
            "patient_awake",
            "patient_available",
            "patient_fed",
            "patient_not_tired"
        ],

        "effects": [
            "stretching_completed"
        ]
    },

    "exercise": {

        "description": "Perform general physical exercise",

        "aliases": [
            "exercise",
            "workout",
            "training",
            "gym",
            "legs",
            "physical",
        ],

        "preconditions": [
            "patient_awake",
            "patient_available",
            "patient_not_tired"
        ],

        "effects": [
            "patient_tired"
        ]
    },

    "walk_in_park": {

        "description": "Walk outside for mobility and relaxation",

        "aliases": [
            "walk",
            "stroll",
            "park",
            "outside",
            "outdoor",
        ],

        "preconditions": [
            "patient_awake",
            "good_weather",
            "patient_available",
            "patient_not_tired"
        ],

        "effects": [
            "patient_relaxed",
            "patient_tired",
            "mobility_improved"
        ]
    },

    "rest": {

        "description": "Patient rests to recover energy",

        "aliases": [
            "rest",
            "nap",
            "sleep",
            "lie",
            "relax",
            "break",
            "postpone",
        ],

        # A patient may rest whenever they ask to; requiring patient_tired
        # refused a request to rest from a patient who was not tired (S28).
        "preconditions": [
            "patient_available"
        ],

        "effects": [
            "patient_not_tired",
            "patient_rested"
        ]
    },

    # ---------------------------------------------------
    # SOCIAL ACTIVITIES
    # ---------------------------------------------------

    "call_relative": {

        "description": "Call a family member or relative",

        "aliases": [
            "call",
            "phone",
            "relative",
            "family",
            "wife",
            "husband",
            "daughter",
            "son",
        ],

        "preconditions": [
            "patient_awake",
            "phone_available",
            "patient_emotionally_stable"
        ],

        "effects": [
            "relative_contacted",
            "patient_reassured",
            "patient_socialized"
        ]
    },

    "video_call_relative": {

        "description": "Perform a video call with a relative",

        "aliases": [
            "video",
            "call",
            "relative",
            "family",
        ],

        "preconditions": [
            "patient_awake",
            "internet_available",
            "device_available"
        ],

        "effects": [
            "relative_contacted",
            "patient_happier"
        ]
    },

    "social_interaction": {

        "description": "Social interaction with staff or relatives",

        "aliases": [
            "chat",
            "talk",
            "socialise",
            "socialize",
            "company",
            "conversation",
            "visit",
        ],

        "preconditions": [
            "patient_available",
            "patient_awake"
        ],

        "effects": [
            "patient_socialized",
            "patient_happier"
        ]
    },

    "speak_with_nurse": {

        "description": "Speak with the nurse",

        "aliases": [
            "nurse",
            "staff",
            "speak",
            "talk",
        ],

        "preconditions": [
            "nurse_available",
            "patient_awake"
        ],

        "effects": [
            "patient_reassured",
            "patient_socialized"
        ]
    },

    # ---------------------------------------------------
    # COGNITIVE ACTIVITIES
    # ---------------------------------------------------

    "reading_activity": {

        "description": "Read a book or magazine",

        "aliases": [
            "read",
            "reading",
            "book",
            "magazine",
            "journal",
            "newspaper",
            "novel",
        ],

        "preconditions": [
            "patient_awake",
            "book_available"
        ],

        "effects": [
            "cognitive_training_completed",
            "patient_relaxed"
        ]
    },

    "memory_training": {

        "description": "Perform cognitive memory exercises",

        "aliases": [
            "memory",
            "cognitive",
            "puzzle",
            "game",
            "quiz",
        ],

        "preconditions": [
            "patient_awake",
            "patient_focused"
        ],

        "effects": [
            "cognitive_training_completed",
            "patient_tired"
        ]
    },

    # ---------------------------------------------------
    # EMOTIONAL SUPPORT
    # ---------------------------------------------------

    "emotional_support": {

        "description": "Provide emotional support to patient",

        "aliases": [
            "comfort",
            "support",
            "sad",
            "reassure",
            "upset",
        ],

        "preconditions": [
            "patient_sad"
        ],

        "effects": [
            "patient_reassured",
            "patient_emotionally_stable"
        ]
    },

    "play_music": {

        "description": "Play relaxing music for the patient",

        "aliases": [
            "music",
            "song",
            "listen",
            "radio",
        ],

        "preconditions": [
            "music_device_available",
            "patient_awake"
        ],

        "effects": [
            "patient_relaxed",
            "patient_happier"
        ]
    },

    # ---------------------------------------------------
    # DAILY ROUTINE
    # ---------------------------------------------------

    "wake_patient": {

        "description": "Wake up the patient",

        "aliases": [
            "wake",
            "awake",
            "get up",
            "morning",
        ],

        "preconditions": [
            "patient_sleeping"
        ],

        "effects": [
            "patient_awake"
        ]
    },

    "prepare_medication": {

        "description": "Prepare medication before administration",

        "aliases": [
            "prepare",
            "medication",
            "pill",
            "dose",
        ],

        "preconditions": [
            "medication_available"
        ],

        "effects": [
            "medication_prepared"
        ]
    },

    "do_canteen_volunteering": {

        "description": "Prepare breakfast for the patient",

        "aliases": [
            "canteen",
            "volunteer",
            "kitchen",
        ],

        "preconditions": [],

        "effects": [
            "food_available"
        ]
    },

    "check_vitals": {

        "description": "Check patient vital signs",

        "aliases": [
            "vitals",
            "blood",
            "pressure",
            "temperature",
            "glucose",
            "monitor",
        ],

        "preconditions": [
            "patient_awake"
        ],

        "effects": [
            "vitals_checked",
            "patient_monitored"
        ]
    },

    # ---------------------------------------------------
    # HIERARCHICAL / COMPLEX GOALS
    # ---------------------------------------------------

    "morning_routine": {

        "description": "Complete the full morning routine",

        "aliases": [
            "morning",
            "routine",
        ],

        "preconditions": [
            "patient_awake"
        ],

        "effects": [
            "breakfast_completed",
            "medication_taken",
            "patient_prepared_for_day"
        ]
    },

    "prepare_for_sleep": {

        "description": "Prepare patient for sleep",

        "aliases": [
            "sleep",
            "bed",
            "night",
            "pyjamas",
            "evening",
        ],

        "preconditions": [
            "patient_tired"
        ],

        "effects": [
            "patient_sleeping"
        ]
    },

    "full_rehabilitation_session": {

        "description": "Complete full rehabilitation session",

        "aliases": [
            "rehab",
            "rehabilitation",
            "full",
            "session",
            "physio",
        ],

        "preconditions": [
            "stretching_completed",
            "patient_fed"
        ],

        "effects": [
            "rehabilitation_completed",
            "mobility_improved",
            "patient_tired"
        ]
    }
}