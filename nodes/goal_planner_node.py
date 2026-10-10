# nodes/goal_planner_node.py

import json
import os
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from colorama import Fore, Style

from graph.graph_init import GraphState
from utils.logs import log_graph_state, log_message
from utils.structured_logs import (
    log_goal_predicate_extraction,
    log_goal_decomposition,
)
from utils.models import define_model
from deterministic.goal_decomposer import GoalDecomposer
from planning.canonical_actions import CANONICAL_ACTIONS


INITIAL_PROMPT = """
# CONTEXT #
You are a goal-oriented planning module for a care-home robot.

Your task is to determine whether one or more new future activities should be added to the patient's schedule in order to achieve the patient's goal and intentions.

You must ONLY generate new activities if:
- the patient explicitly asks for a reminder,
- the patient asks to postpone something,
- the patient expresses a future intention,
- the patient requests a future action.

# GOAL DECOMPOSITION #
If the patient's request maps to one of the following canonical action effects (goals),
set "goal_predicate" to that effect name. The system will use backward chaining
to automatically decompose the goal into the full chain of required activities.

Available goal predicates (effects of canonical actions):
{available_effects}

If the patient's request does NOT map to any of these goals (e.g., simple reminder,
postponement, or a specific one-off activity), set "goal_predicate" to null and
use "new_activities" to generate activities directly.

# OUTPUT FORMAT #

Return ONLY valid JSON.

If no new activity is needed:

{{
  "generate_new_activities": false,
  "goal_predicate": null
}}

If a goal predicate is identified (backward chaining will decompose it):

{{
  "generate_new_activities": true,
  "goal_predicate": "EFFECT_NAME",
  "new_activities": []
}}

If no goal predicate matches (generate activities directly):

{{
  "generate_new_activities": true,
  "goal_predicate": null,
  "new_activities": [
    {{
      "activity_name": "STRING",
      "activity_time": "HH:MM",
      "without_time": false,
      "activity_duration": 10,
      "activity_details": "STRING",
      "execution_notes": "",
      "done": false,
      "skipped": false,
      "critical": false,
      "priority": 3
    }}
  ]
}}
"""


NEW_REQUEST_PROMPT = """
Patient request:
{patient_request}

Situation perception:
{situation_perception}

Current datetime:
{datetime}

Current task:
{task}

Current world state:
{world_state}
"""


def goal_planner_node(state: GraphState) -> GraphState:
    print(Fore.GREEN + "[position: GOAL PLANNER NODE]" + Style.RESET_ALL)

    log_graph_state(state, "goal_planner_node")

    if state["task"] is None:
        raise ValueError("goal_planner_node - task is None")

    if state["fake_datetime"] is None:
        raise ValueError("goal_planner_node - fake_datetime is None")

    formatted_datetime = state["fake_datetime"].strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    # Build list of available effects for the prompt
    all_effects = set()
    for action_def in CANONICAL_ACTIONS.values():
        for effect in action_def.get("effects", []):
            all_effects.add(effect)
    available_effects = "\n".join(sorted(all_effects))

    initial_prompt = PromptTemplate.from_template(
        INITIAL_PROMPT
    ).invoke({"available_effects": available_effects})

    request_prompt = PromptTemplate.from_template(
        NEW_REQUEST_PROMPT
    ).invoke(
        {
            "patient_request": state["patient_request"],
            "situation_perception": state["situation_perception"],
            "datetime": formatted_datetime,
            "task": json.dumps(state["task"]),
            "world_state": json.dumps(state.get("world_state", {})),
        }
    )

    model = define_model(None)

    messages = [
        SystemMessage(content=initial_prompt.to_string()),
        HumanMessage(content=request_prompt.to_string()),
    ]

    raw_response = model.invoke(messages)

    raw_content = raw_response.content

    if not isinstance(raw_content, str):
        raw_content = str(raw_content)

    raw_content = raw_content.strip()

    if raw_content.startswith("```json"):
        raw_content = raw_content.removeprefix("```json").strip()

    if raw_content.startswith("```"):
        raw_content = raw_content.removeprefix("```").strip()

    if raw_content.endswith("```"):
        raw_content = raw_content.removesuffix("```").strip()

    try:
        parsed_response = json.loads(raw_content)

    except Exception:

        print("❌ GOAL PLANNER JSON ERROR")
        print(raw_content)

        return state

    generate_new_activity = parsed_response.get(
        "generate_new_activities",
        []
    )

    if not (parsed_response.get("generate_new_activities", [])):

        log_message(
            "GOAL_PLANNER_NODE - no new activities generated"
        )
        log_goal_predicate_extraction(
            patient_request=state.get("patient_request", ""),
            goal_predicate=parsed_response.get("goal_predicate"),
            generate_new_activities=False,
        )
        print("no new action to generate!")
        return state

    # ---------------------------------------------------
    # GOAL DECOMPOSITION via backward chaining
    # ---------------------------------------------------
    goal_predicate = parsed_response.get("goal_predicate")
    new_activities = []

    log_goal_predicate_extraction(
        patient_request=state.get("patient_request", ""),
        goal_predicate=goal_predicate,
        generate_new_activities=True,
    )

    if goal_predicate:
        log_message(
            f"GOAL_PLANNER_NODE - goal predicate identified: "
            f"{goal_predicate}. Running backward chaining decomposition..."
        )
        print(
            Fore.CYAN + f"[BACKWARD CHAINING] Decomposing goal: "
            f"{goal_predicate}" + Style.RESET_ALL
        )

        decomposer = GoalDecomposer()
        world_state = state.get("world_state", {})
        existing_activities = state["task"].get("activities", [])

        action_chain = decomposer.decompose_goal(
            goal_predicate,
            world_state,
            existing_activities,
        )

        if action_chain:
            new_activities = decomposer.actions_to_activities(action_chain)
            print(
                Fore.CYAN + f"[BACKWARD CHAINING] Decomposed into "
                f"{len(new_activities)} actions:" + Style.RESET_ALL
            )
            for act in new_activities:
                print(f"  - {act['activity_name']} "
                      f"({act['canonical_action']})")
            log_goal_decomposition(
                goal_predicate=goal_predicate,
                decomposition_success=True,
                action_chain=action_chain,
                fallback_used=False,
                num_activities_generated=len(new_activities),
            )
        else:
            log_message(
                "GOAL_PLANNER_NODE - backward chaining produced "
                "no actions, falling back to LLM-generated activities"
            )
            # Fall back to LLM-generated activities
            new_activities = parsed_response.get("new_activities", [])
            log_goal_decomposition(
                goal_predicate=goal_predicate,
                decomposition_success=False,
                action_chain=None,
                fallback_used=True,
                num_activities_generated=len(new_activities),
            )

    else:
        # No goal predicate: use LLM-generated activities directly
        new_activities = parsed_response.get("new_activities", [])
        log_goal_decomposition(
            goal_predicate=None,
            decomposition_success=False,
            action_chain=None,
            fallback_used=True,
            num_activities_generated=len(new_activities),
        )

    if not new_activities:

        log_message(
            "GOAL_PLANNER_NODE - invalid activities"
        )
        print("invalid activities")
        return state

    log_message(
        "GOAL_PLANNER_NODE - generated activities"
    )
    log_message(json.dumps(new_activities))

    updated_task = state["task"]

    updated_task["activities"].extend(new_activities)

    # The new activities belong to this session's task only. They used to be
    # appended to the patient's file in mock_data, which is the care plan
    # the Task Synthesizer builds every later session from: a replayed
    # session then inherited the activities generated in earlier ones, and
    # the file grew from five activities to thirty. The care plan is the
    # clinician's, and an emergent request does not amend it.

    # ---------------------------------------------------
    # REBUILD VECTOR STORE / UPDATE EMBEDDINGS - removed, delegating control to knowledge_manager
    # ---------------------------------------------------
    """
    try:

        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        patient_file = f"mock_data/{patient_id}.txt"

        loader = TextLoader(patient_file, encoding="utf-8")

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        split_docs = splitter.split_documents(documents)

        # Remove old patient documents
        # (depends on the vector db, this is the most common form)

        if state["vector_store"] is not None and hasattr(state["vector_store"], "delete"):  # pylint doesn't allow delete

            try:
                state["vector_store"].delete(
                    filter={"patient_id": patient_id}
                )

            except Exception as e:
                print("Vector delete warning:", e)

        # Re-add updated chunks

        for doc in split_docs:

            doc.metadata["patient_id"] = patient_id

        if state["vector_store"] is not None and hasattr(state["vector_store"], "add_documents"):  # pylint doesn't allow delete
            state["vector_store"].add_documents(split_docs)

        log_message(
            "GOAL_PLANNER_NODE - vector store updated"
        )

    except Exception as e:

        print("VECTOR STORE UPDATE ERROR")
        print(e)

    """
    print("goal decomposed into:")
    for activity in new_activities:
        print(f"{activity['activity_name']} at {activity.get('activity_time', 'unspecified time')}")

    return {
        **state,
        "task": updated_task,
    }