import json
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import copy

from graph.graph_init import GraphState
from deterministic.activity_planner import set_skipped_activities
from utils.interfaces import Task
from utils.models import define_model
from utils.logs import log_graph_state, log_message


INITIAL_PROMPT = """
# CONTEXT #
You are a robot assistant in a care home. You must help the patient to do his/her activities.

# INPUT #
You will receive this information as input:
- The previous state of the activities of the day (previous state of the task).
- What the patient is doing and what is happening around the patient (the situation perception).
- The patient request.
- The list of the next scheduled activities.

# OUTPUT #
You must reply with a JSON object with the following field:
- "updated_task": The updated state of the task with 'execution_notes' and 'done' activities fields updated based on the situation perception.

# INSTRUCTIONS #
You must update the "Done" fields of the activities that the patient has done, and update the "execution_notes" field with the information about the execution of the activity.
Do not update the "Skipped" field of the activities, as it is not your responsibility to decide if an activity is skipped or not.
The priority of the activities is from 1 to 3, where 1 is the highest priority and 3 is the lowest priority.

- "Done" field:
You must update the "Done" field of the activities to True only if you understand from the situation perception that the patient has really done the activity.
Update the "Done" field based only on the situation perception, not on the patient request. The patient might say that he/she has done an activity, but you must update the "Done" field only if you understand from the situation perception that the patient has really done the activity.

- "Activity_Time" field:
If the patient decides to schedule an activity that has "without_time" to true specifying a time, you must fill the "Activity Time" field with that time in format HH:mm.
If the patient doesn't want to schedule it (e.g., saying "do it later"), you must leave the "Activity Time" field blank.
You can update the "Skipped" field only for this type of activities, if the patient doesn't want to do it in the whole day.
"""

NEW_REQUEST_PROMPT = """
The previous state of the task is:
{task}.

The situation perception is:
{situation_perception}

The patient request is:
{patient_request}

The previous robot answer was:
{vocal_answer}

The robot action was:
{robot_action}
"""
# The next scheduled activities are:
# {next_activities}

NEW_REQUEST_PROMPT_ADDITIONAL = """
The previous state of the task is:
{task}.

The situation perception is:
{situation_perception}

The patient request is:
{patient_request}

Additional information:
{additional_info}
"""


@tool
def response_template(updated_task: Task) -> bool:
    """The JSON format for the output of the task progress node.

    Args:
        updated_task: Task | None
    """

    return True


def task_progress_node(state: GraphState) -> GraphState:
    log_graph_state(state, "task_progress_node")

    if state["fake_datetime"] is None:
        raise ValueError("next_steps_node - fake_datetime is None")

    if state["task"] is None:
        raise ValueError("task_progress_node - task is None")
    task_with_skipped_activities = copy.deepcopy(state["task"])
    set_skipped_activities(task_with_skipped_activities, state["fake_datetime"])

    log_message("\nTASK_PROGRESS_NODE - task_with_skipped_activities")
    log_message(json.dumps(task_with_skipped_activities))

    if os.getenv("ABLATION_MODE", "false").lower() == "true":
        log_message(
            "Skipping skipped_activities as per environment variable ABLATION_MODE"
        )
        task_with_skipped_activities = state["task"]

    initial_prompt_template = PromptTemplate.from_template(INITIAL_PROMPT)
    initial_prompt = initial_prompt_template.invoke({})

    new_request_prompt_template = PromptTemplate.from_template(NEW_REQUEST_PROMPT)
    new_request_prompt = new_request_prompt_template.invoke(
        {
            "task": task_with_skipped_activities,
            "situation_perception": state["situation_perception"],
            "patient_request": state["patient_request"],
            # "next_activities": state["next_activities"],
            "vocal_answer": state["vocal_answer"],
            "robot_action": state["robot_action"],
        }
    )

    # If the progress checking output is True, we need to go to the robot effector node
    # or to the task progress node if the output is False to propose a different step of the task
    if state["progress_checking_output"] is False and state["updated_task"] is not None:
        # Query the model with different prompt
        additional_info = f"""Your previous response was not adherent to the prescription. Please provide the correct information.
        These are the not adherent information you provided:
        - Updated Task: {state["updated_task"]}
        - Robot action: {state["robot_action"]}
        - Vocal answer: {state["vocal_answer"]}
        """
        new_request_prompt_template_additional = PromptTemplate.from_template(
            NEW_REQUEST_PROMPT_ADDITIONAL
        )
        new_request_prompt = new_request_prompt_template_additional.invoke(
            {
                "task": task_with_skipped_activities,
                "situation_perception": state["situation_perception"],
                "patient_request": state["patient_request"],
                "additional_info": additional_info,
            }
        )

    model = define_model(response_template)

    messages = []
    messages.append(SystemMessage(content=initial_prompt.to_string()))
    messages.append(HumanMessage(content=new_request_prompt.to_string()))

    result = model.invoke(
        messages,
    ).to_json()

    log_message("\nTASK_PROGRESS_NODE - result")
    if result["kwargs"]["tool_calls"] is None:
        return {
            **state,
            "updated_task": None,
        }

    updated_task = None
    # TODO: Sometimes the updated_task is a dictionary with keys "type" and "value"
    if "type" in result["kwargs"]["tool_calls"][0]["args"]["updated_task"]:
        updated_task = result["kwargs"]["tool_calls"][0]["args"]["updated_task"][
            "value"
        ]
    else:
        updated_task = result["kwargs"]["tool_calls"][0]["args"]["updated_task"]

    return {
        **state,
        "updated_task": updated_task,
    }
