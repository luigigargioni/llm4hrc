import json
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from graph.graph_init import GraphState

from deterministic.activity_planner import (
    schedule_activities,
    TOLERANCE_START_TIME_ACTIVITY,
)

from utils.models import define_model
from utils.logs import log_graph_state, log_message

INITIAL_PROMPT = """
# CONTEXT #
You are a robot assistant in a care house. You must help the patient to complete his/her activities.
You must be kind and calm with the patient. You can sometimes refer to him/her only with his/her name to be more personal, but not always in a robotic way.
Use your knowledge and common sense to help the patient to complete the activities correctly and manage the situation. Apply Theory of Mind to understand the patient's needs and feelings.

# INPUT #
You will receive these information:
- The actual state of the activities of the day.
- The patient information.
- The next activities to do.
- What the patient just said.
- What the patient is doing and what is happening around the patient (the situation perception).
- The actual datetime.

# OUTPUT #
You have to reply with a JSON object with the following fields:

- "robot_action": The physical action that the robot must perform. For example, to move a specific medication next to the patient, or speak to the patient.
You must call the doctor if the patient does an activity in the wrong way, for example if the patient takes a wrong critical medication, or a wrong dosage or at the wrong time.
Express the action in uppercase and snake_case for the action. For example, "MOVE_MEDICATION", "REMOVE_MEDICATION", "CALL_DOCTOR", "NO_ACTION", "SPEAKING", "WAKE_UP_PATIENT", "WAIT_FOR_PATIENT", etc.

- "vocal_answer": The vocal answer for the patient. For example, "Please take this medication, etc..." or other explanations about the activty.
When you reply suggesting to do an activity, you should also provide the detail of the activity. Stay adherent to what it is reported in the prescription. For example, "Please, take the medicine X, it is a pill and it helps for ...".
You must consider also the information about the patient to provide the best help to the patient based on his/her impairment.
You must consider the list of the next activities to be done while providing the vocal answer.
You can also say nothing (reply with "...") if there are no activities to do and the patient doesn't ask anything or if the situation perception doesn't require a vocal answer (e.g. the patient is sleeping and he/she must not do any activity).
If there is an activity to do, you must provide the vocal answer to help the patient to do the activity correctly (e.g. wake up the patient to take the medication).
Try to convince the patient to do the activity correctly and why it is important before calling the doctor. After some attempts, if the acitivity is critical and the patient still refuses to do the activity correctly, you must call the doctor. If the activity is not critical, you can just wait for the patient to do it correctly or skip it if the patient doesn't want to do it.

# TASK #
Provide the robot_action and vocal_answer based on the patient request and the situation perception.
You must rely only on the next activities to do. The whole list of the scheduled activities for the day is not relevant for the next steps, but only if patient asks about the scheduled activities.
If the next activities to do are more than one, you must ask the patient which activity he/she wants to do.
If the next activity is without_time, you must refer to this activity as a self-schedulable activity that can be done at any time during the day.
If the patient for example is sleeping, and there is still time for the activity, wait as much as possible without exceeding the tolerance time waking up the patient to do the activity. Try to accommodate the patient's needs as much as possible.
The tolerance for each activity is {tolerance_start_time_activity} minutes before and after the scheduled time.

# DATETIME #
Take into account the actual datetime that will be provided in the each new request.

# LANGUAGE #
Always reply in {SYSTEM_LANGUAGE} language, both for the robot_action and vocal_answer.
"""


NEW_REQUEST_PROMPT = """
The actual state of the task is:
{updated_task}.

Patient information:
{patient_information}.

The next activities to be done are:
{next_activities}.

The scheduled activities for today are:
{scheduled_activities}.

The patient request is:
{patient_request}

The situation perception is:
{situation_perception}

The actual datetime is {datetime}.
"""


@tool
def response_template(robot_action: str, vocal_answer: str) -> bool:
    """The JSON format for the output of the next steps node.

    Args:
        robot_action: str
        vocal_answer: str

    """
    return True


def next_steps_node(state: GraphState) -> GraphState:
    log_graph_state(state, "next_steps_node")

    if state["updated_task"] is None:
        raise ValueError("next_steps_node - updated_task is None")

    # formatted_datetime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    if state["fake_datetime"] is None:
        raise ValueError("next_steps_node - fake_datetime is None")
    formatted_datetime = state["fake_datetime"].strftime("%d-%m-%Y %H:%M:%S")

    initial_prompt_template = PromptTemplate.from_template(INITIAL_PROMPT)
    initial_prompt = initial_prompt_template.invoke(
        {
            "tolerance_start_time_activity": TOLERANCE_START_TIME_ACTIVITY,
            "SYSTEM_LANGUAGE": os.getenv("SYSTEM_LANGUAGE"),
        }
    )

    scheduled_activities, next_activities = schedule_activities(
        state["updated_task"]["activities"], state["fake_datetime"]
    )

    # Ablation test: Skip scheduler if the environment variable is set
    if os.getenv("ABLATION_MODE", "false").lower() == "true":
        log_message("Skipping scheduler as per environment variable ABLATION_MODE")
        scheduled_activities = state["updated_task"]["activities"]
        next_activities = state["updated_task"]["activities"]

    log_message("next_steps_node - scheduled_activities")
    log_message(json.dumps(scheduled_activities))
    log_message("next_steps_node - next_activities")
    log_message(json.dumps(next_activities))

    next_activities_prompt = (
        next_activities
        if len(next_activities) > 0
        else "No activities to do at the moment."
    )

    new_request_prompt_template = PromptTemplate.from_template(NEW_REQUEST_PROMPT)
    new_request_prompt = new_request_prompt_template.invoke(
        {
            "datetime": formatted_datetime,
            "updated_task": state["updated_task"],
            "next_activities": next_activities_prompt,
            "scheduled_activities": scheduled_activities,
            "patient_request": state["patient_request"] or "",
            "situation_perception": state["situation_perception"] or "",
            "patient_information": state["updated_task"]["patient_information"] or "",
        }
    )

    model = define_model(response_template)

    messages = state["conversation_history"] or []
    if len(messages) == 0:
        messages.append(SystemMessage(content=initial_prompt.to_string()))

    messages.append(HumanMessage(content=new_request_prompt.to_string()))

    result = model.invoke(
        messages,
    ).to_json()

    log_message("\nNEXT_STEPS_NODE - result")
    if result["kwargs"]["tool_calls"] is None:
        raise ValueError("nNEXT_STEPS_NODE - tool_calls is None")

    log_message(
        "robot_action - " + result["kwargs"]["tool_calls"][0]["args"]["robot_action"]
    )
    log_message(
        "vocal_answer - " + result["kwargs"]["tool_calls"][0]["args"]["vocal_answer"]
    )

    task_progress_history = {
        "situation_perception": state["situation_perception"],
        "patient_request": state["patient_request"],
        "robot_action": result["kwargs"]["tool_calls"][0]["args"]["robot_action"],
        "vocal_answer": result["kwargs"]["tool_calls"][0]["args"]["vocal_answer"],
    }
    messages.append(AIMessage(content=json.dumps(task_progress_history)))

    return {
        **state,
        "robot_action": result["kwargs"]["tool_calls"][0]["args"]["robot_action"],
        "vocal_answer": result["kwargs"]["tool_calls"][0]["args"]["vocal_answer"],
        "conversation_history": messages,
        "scheduled_activities": scheduled_activities,
        "next_activities": next_activities,
    }
