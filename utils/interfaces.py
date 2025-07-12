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


GRAPH_CONFIG: RunnableConfig = {
    "recursion_limit": 10000,
    "configurable": {
        "thread_id": "1",
    },
}

PATIENTS_ID_LIST: list[str] = ["123", "456", "789", "0789"]
