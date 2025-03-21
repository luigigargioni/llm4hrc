import json
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import copy

from graph.graph_init import GraphState
from utils.interfaces import Task
from nodes.task_synthesizer_node import set_skipped_activities
from utils.generics import define_model
from utils.logs import log_graph_state, log_message


INITIAL_PROMPT = """
# CONTEXT #
You are a robot assistant in a care home. You must help the patient to do his/her activities.

# INPUT #
You will receive these information as input:
- The previous state of the activities of the day (previous state of the task).
- What the patient is doing and what is happening around the patient (the situation perception).
- The patient request.
- The list of the next scheduled activities.

# OUTPUT #
You have to reply with a JSON object with the following fields:
- "updated_task": The updated state of the task with execution_notes, and skipped and done activities fields updated based on the situation perception. 

# INSTRUCTIONS #
You have to update the "Done" and "Skipped" fields of the activities that the patient has done, and update the "execution_notes" field with the information about the execution of the activity (for example "The patient took the medicine correctly" or "The patient did only half of the exercises prescribed").
You must be sure that the "Done" and "Skipped" fields are updated correctly only based on the situation perception.
If there are more scheduled activities to do next, means that the patient has to decide which activity he/she wants to do next. So, if the patient chooses an activity, you must update the "Skipped" field of the other activities to True.

- "Done" field:
You must update the "Done" field of the activities to True only if you understand from the situation perception that the patient has really done the activity.

- "Skipped" field:
You must update the "Skipped" field of the activities to True only if you understand from the situation perception that those activities have been skipped by the patient (e.g. the doctor says that the patient can skip the activity).

- "Activity_Time" field:
If the patient decides to schedule an activity that doesn't have an "Activity Time" field, you must fill the "Activity Time" field with the time that the patient asks you to schedule it.
If the patient specifies a time, you must fill the "Activity Time" field with that time in format HH:mm. If the patient doesn't want to schedule it (e.g., "do it later"), you must fill the "Activity Time" field with None.
"""
# IMPORTANT: The JSON structure of the task and the other information should be kept as the previous state task. Don't remove or add activities to the task, but only update the fields of the activities.

NEW_REQUEST_PROMPT = """
The previous state of the task is:
{task}.

The situation perception is:
{situation_perception}

The patient request is:
{patient_request}
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

    initial_prompt_template = PromptTemplate.from_template(INITIAL_PROMPT)
    initial_prompt = initial_prompt_template.invoke({})

    new_request_prompt_template = PromptTemplate.from_template(NEW_REQUEST_PROMPT)
    new_request_prompt = new_request_prompt_template.invoke(
        {
            "task": task_with_skipped_activities,
            "situation_perception": state["situation_perception"],
            "patient_request": state["patient_request"],
            # "next_activities": state["next_activities"],
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

    log_message(result["kwargs"]["tool_calls"][0]["args"]["updated_task"])

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
