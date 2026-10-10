
#Questo nodo aggiorna lo stato del task. Ad esempio, se il paziente ha fatto un'attività, il nodo aggiorna il campo "Done" dell'attività
#  a True, e aggiunge eventuali note di esecuzione. Se il paziente ha deciso di non fare un'attività, il nodo lascia il campo "Done" 
# a False, ma non aggiorna il campo "Skipped", in quanto non è responsabilità del robot decidere se un'attività è saltata o meno.

#In pratica, è il nodo che mantiene sincronizzato il “mondo reale” con la rappresentazione interna delle attività.

# è il ponte tra:

# percezione,
# task representation.

# Ed è qui che il sistema:

# “capisce” cosa è successo,
# aggiorna la memoria operativa,
# e prepara il terreno per il reasoning successivo.

import json
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import copy
from colorama import Fore, Style

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
It's mandatory that you must reply with a JSON object with EXACTLY the following field:
- "updated_task": The updated state of the task with 'execution_notes' and 'done' activities fields updated based on the situation perception.

# INSTRUCTIONS #
You must update the "done" fields of the activities that the patient has done, and update the "execution_notes" field with the information about the execution of the activity.
Do not update the "Skipped" field of the activities, as it is not your responsibility to decide if an activity is skipped or not.
The priority of the activities is from 1 to 3, where 1 is the highest priority and 3 is the lowest priority.

- "done" field:
You must update the "done" field of the activities to True only if you understand from the situation perception that the patient has really done the activity.
Update the "done" field based only on the situation perception, not on the patient request. The patient might say that he/she has done an activity, but you must update the "done" field only if you understand from the situation perception that the patient has really done the activity.

- "Activity_Time" field:
If the patient decides to schedule an activity that has "without_time" to true specifying a time, you must fill the "Activity Time" field with that time in format HH:mm.
If the patient doesn't want to schedule it (e.g., saying "do it later"), you must leave the "Activity Time" field blank.
You can update the "Skipped" field only for this type of activities, if the patient doesn't want to do it in the whole day.

Return ONLY a valid JSON. Do not include any explanation or text outside JSON.
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
    print(Fore.GREEN + "[position: TASK PROGRESS NODE]" + Style.RESET_ALL)

    log_graph_state(state, "task_progress_node")

    #tempo simulato e task devono esistere.
    if state["fake_datetime"] is None:
        raise ValueError("next_steps_node - fake_datetime is None")

    if state["task"] is None:
        raise ValueError("task_progress_node - task is None")
    
    # Creo una copia del task in cui aggiorno le attività saltate in base al tempo simulato, così da poterlo passare al modello senza modificare
    #  lo stato originale del task
    task_with_skipped_activities = copy.deepcopy(state["task"])

    #verifica skippate in base a orario
    set_skipped_activities(task_with_skipped_activities, state["fake_datetime"])

    log_message("\nTASK_PROGRESS_NODE - task_with_skipped_activities")
    log_message(json.dumps(task_with_skipped_activities))

    #salta gli skipped in caso ablation mode sia true (per studi scentifici)
    if os.getenv("ABLATION_MODE", "false").lower() == "true":
        log_message(
            "Skipping skipped_activities as per environment variable ABLATION_MODE"
        )
        task_with_skipped_activities = state["task"]

    #creo prompt
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

    #(sovrascrivo new_request_prompt con un prompt che dice al modello "la tua precedente risposta non era aderente alla prescrizione, 
    # riformula la risposta correttamente", e gli passo l'invoke del new_request_prompt_template_additional invece che del 
    # new_request_prompt_template)
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

    #print("sto creando la task in task_progress_node")
    model = define_model(response_template)

    messages = []
    messages.append(SystemMessage(content=initial_prompt.to_string()))
    messages.append(HumanMessage(content=new_request_prompt.to_string()))
    """
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
    """
    result = model.invoke(messages).to_json()

    log_message("\nTASK_PROGRESS_NODE - result")
    log_message(str(result))

    #ottengo l'updated_task in due modi: se il modello ha chiamato lo strumento response_template, lo prendo da lì, altrimenti faccio un
    #  parsing del JSON raw di output (fallback in caso Ollama non chiami correttamente lo strumento)

    #il risultato che prendo con tool calls da result mi aggiorna il mio updated_task

    if result["kwargs"].get("tool_calls"):
        updated_task = result["kwargs"]["tool_calls"][0]["args"]["updated_task"]

        # gestione caso {"type": ..., "value": ...}
        if isinstance(updated_task, dict) and "type" in updated_task:
            updated_task = updated_task["value"]

    else:
        print("⚠️ Ollama fallback: parsing raw JSON")

        raw_output = result["kwargs"].get("content", "")
        updated_task = None

        try:
            parsed = json.loads(raw_output)
            updated_task = parsed.get("updated_task", None)
        except Exception:
            # The model sometimes wraps the object in prose, or spends its
            # whole output budget on reasoning and returns nothing usable.
            import re

            embedded = re.search(r"\{.*\}", str(raw_output), re.S)
            if embedded:
                try:
                    parsed = json.loads(embedded.group(0))
                    updated_task = parsed.get("updated_task", None)
                except Exception:
                    updated_task = None

    # A reply can also parse and still be unusable: the model has returned a
    # task cut short, with activities dropped and others missing their
    # done/skipped flags ("... (truncated)"), which later nodes index
    # directly. Such a task is treated like one that could not be read.
    # Updating a task never removes activities (a postponed or skipped one
    # stays, flagged), so fewer activities than the current task means some
    # were lost: in the replay a one-activity task made the whole day look
    # completed and ended the session. The count is compared rather than
    # the names, so that a renamed activity is still accepted.
    def _complete(task) -> bool:
        activities = task.get("activities")
        current = (state["task"] or {}).get("activities") or []
        return (
            isinstance(activities, list)
            and len(activities) >= len(current)
            and all(
                isinstance(a, dict)
                and all(k in a for k in ("activity_name", "done", "skipped"))
                for a in activities
            )
        )

    if not updated_task or not isinstance(updated_task, dict) \
            or "activities" not in updated_task \
            or not _complete(updated_task):
        # Failing to produce an updated task is a generation failure, not a
        # reason to abandon the session: carrying the current task forward
        # leaves the plan and the world state untouched, and the next cycle
        # gets another chance. Aborting here used to discard a whole run
        # over one unusable reply.
        print("⚠️ TASK_PROGRESS: unusable task, keeping the current one")
        log_message(
            "TASK_PROGRESS_NODE - unusable model output, "
            "carrying the existing task forward"
        )
        updated_task = state["task"]

    for activity in updated_task["activities"]:
        if activity.get("done", False):
            activity["executable"] = True
    
    from typing import cast

    #debug
    print("stato eseguite:")
    for activity in updated_task["activities"]:
        print(f"Activity: {activity['activity_name']}, Done: {activity.get('done', False)}")
        
    #lo ritorno
    return cast(GraphState, {
        **state,
        "updated_task": updated_task,
    })
