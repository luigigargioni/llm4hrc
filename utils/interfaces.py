from datetime import datetime
import os
from typing import List, Optional
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from typing_extensions import TypedDict
from enum import Enum
from langchain_chroma import Chroma
from langchain_core.runnables.config import RunnableConfig


class Activity(TypedDict):
    activity_name: str
    activity_time: Optional[str]
    without_time: bool
    activity_duration: int
    activity_details: str
    critical: bool
    priority: int
    execution_notes: Optional[str]
    done: bool
    skipped: bool


class Task(TypedDict):
    patient_name: str
    patient_surname: str
    patient_information: str
    datetime: str
    activities: list[Activity]


class RobotBehaviours(str, Enum):
    PROACTIVE = "proactive"
    REACTIVE = "reactive"


class ProgressCheckingResponseTypes(str, Enum):
    OK = "OK"
    KO = "KO"


class ProgressCheckingResponses(str, Enum):
    result: ProgressCheckingResponseTypes
    reason: str


class GraphState(TypedDict):
    task: Task | None
    progress_checking_output: bool | None
    conversation_history: List[SystemMessage | AIMessage | HumanMessage] | None
    updated_task: Task | None
    robot_action: str | None
    vocal_answer: str | None
    patient_request: str | None
    situation_perception: str | None
    robot_behaviour: RobotBehaviours
    doctor_specs: str | None
    vector_store: Chroma | None
    end: bool
    fake_datetime: datetime | None
    patient_id: str | None
    doctor_specifications: str | None
    scheduled_activities: list[Activity] | None
    next_activities: list[Activity] | None
    progress_checking_similarity_search: str | None
    world_state: dict  # world state, to be further defined in the future, may include information about environment, objects, people, etc.
    # Bookkeeping of the World State Updater: which completed activities have
    # already had their effects applied, and since when each lapsing fact
    # has been true (see world_state_updater_node).
    world_state_meta: dict


INITIAL_GRAPH_STATE: GraphState = {
    "task": None,
    "progress_checking_output": None,
    "conversation_history": None,
    "updated_task": None,
    "robot_action": None,
    "vocal_answer": None,
    "patient_request": None,
    "situation_perception": None,
    "robot_behaviour": RobotBehaviours(
        os.getenv("ROBOT_BEHAVIOUR", "proactive").lower()
    ),
    "doctor_specs": None,
    "vector_store": None,
    "end": False,
    "fake_datetime": None,
    "patient_id": None,
    "doctor_specifications": None,
    "scheduled_activities": None,
    "next_activities": None,
    "progress_checking_similarity_search": None,
    "world_state_meta": {"applied": [], "true_since": {}},

    # simplified world model defined here
    "world_state": {

    # -------------------------
    # PATIENT PHYSICAL STATE
    # -------------------------
    "patient_awake": True,
    "patient_sleeping": False,
    "patient_available": True,
    "patient_not_tired": True,
    "patient_tired": False,
    "patient_rested": False,
    "patient_in_room": True,
    "patient_fed": False,
    "patient_hydrated": True,
    "patient_in_pain": False,
    "patient_stressed": False,
    "patient_lonely": True,
    "patient_bored": True,
    "patient_motivated": True,
    "patient_relaxed": False,
    "patient_happier": False,
    "patient_sad": False,
    "patient_emotionally_stable": True,
    "patient_focused": True,
    "patient_stabilized": False,
    "patient_reassured": False,
    "patient_socialized": False,
    "patient_prepared_for_day": False,

    # -------------------------
    # MEDICATION / HEALTH
    # -------------------------
    "medication_available": True,
    "medication_prepared": False,
    "correct_time": True,
    "medication_taken": False,
    # False until a meal is recorded: starting it True meant the meal
    # precondition of take_antibiotic was never checked in a live session.
    "after_meal": False,
    "infection_treated": False,
    "meal_completed": True,
    "blood_pressure_checked": False,
    "blood_glucose_checked": False,
    "vitals_checked": False,
    "patient_monitored": False,

    # -------------------------
    # SOCIAL / COMMUNICATION
    # -------------------------
    "phone_available": True,
    "relative_contacted": False,
    "nurse_available": True,
    "internet_available": True,
    "device_available": True,

    # -------------------------
    # REHABILITATION
    # -------------------------
    "rehabilitation_needed": True,
    "rehabilitation_completed": False,
    "stretching_completed": False,
    "mobility_checked": False,
    "mobility_improved": False,
    "muscles_relaxed": False,

    # -------------------------
    # COGNITIVE ACTIVITIES
    # -------------------------
    "book_available": True,
    "reading_completed": False,
    "cognitive_training_completed": False,

    # -------------------------
    # ENVIRONMENT
    # -------------------------
    "quiet_environment": True,
    "tv_on": False,
    "lights_on": True,
    "food_available": True,
    "water_available": True,
    "good_weather": True,
    "music_device_available": True,

    # -------------------------
    # DAILY ROUTINE
    # -------------------------
    "breakfast_completed": True,
    "lunch_completed": False,
    "dinner_completed": False,
    "morning_routine_completed": True,
    "evening_routine_completed": False,
    }
}


class GraphNodes(str, Enum):
    ROBOT_EFFECTOR = "robot_effector_node"
    SITUATION_ASSESSMENT = "situation_assessment_node"
    KNOWLEDGE_MANAGER = "knowledge_manager_node"
    PROGRESS_CHECKING = "progress_checking_node"
    ROBOT_PERCEPTION = "robot_perception_node"
    TASK_SYNTHESIZER = "task_synthesizer_node"
    TASK_PROGRESS = "task_progress_node"
    NEXT_STEPS = "next_steps_node"
    # added new node
    GOAL_PLANNER = "goal_planner_node"
    ACTION_CANONICALIZER = "action_canonicalizer_node"
    PLAN_VALIDATOR = "plan_validator_node"
    WORLD_STATE_UPDATER = "world_state_updater_node"


GRAPH_CONFIG: RunnableConfig = {
    "recursion_limit": 10000,
    "configurable": {
        "thread_id": "1",
    },
}

PATIENTS_ID_LIST: list[str] = ["123", "456", "789", "0789"]
