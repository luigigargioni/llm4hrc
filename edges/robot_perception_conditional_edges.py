from typing import Literal
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from graph.graph_init import GraphState
from utils.interfaces import GraphNodes
from utils.models import define_model
from utils.logs import log_message


ROUTER_PROMPT = """
You are a routing system for a care-home robot.

Your job is to decide whether the patient request or situation
requires creating a NEW FUTURE ACTIVITY (goal) that is NOT already in the schedule.

Return ONLY valid JSON:

{{
  "need_goal_planning": true or false,
  "reason": "short explanation"
}}

Rules:
- Return true ONLY if the patient expresses:
  - desire to schedule something later
  - request for reminder
  - postponement ("later", "tomorrow", "after")
  - new future intention
- Return false if:
  - it's an immediate action
  - it's already part of scheduled activities
  - it's casual conversation
  - it's already handled by current task execution
"""



def _scheduled_summary(task) -> str:
    """List what is already scheduled, as names and times only.

    The router answers one boolean: does this request introduce something
    that is not on the schedule yet. To decide that it needs to know which
    activities exist, not their details, preconditions, blocking reasons or
    recovery actions. Sending the whole task object cost around 2,300
    tokens per turn — a fifth of a session's input — for an answer of about
    eighty.
    """
    if not task:
        return "(no task yet)"
    activities = task.get("activities") or []
    if not activities:
        return "(no activities)"
    lines = []
    for activity in activities:
        name = activity.get("activity_name", "?")
        when = activity.get("activity_time") or "no fixed time"
        state = " [done]" if activity.get("done") else ""
        lines.append(f"- {name} ({when}){state}")
    return "\n".join(lines)


def robot_perception_conditional_edges(
    state: GraphState,
) -> Literal[
    GraphNodes.GOAL_PLANNER.value,
    GraphNodes.SITUATION_ASSESSMENT.value,
]:
    
    patient_request = state["patient_request"] or ""
    situation_perception = state["situation_perception"] or ""

    prompt_template = PromptTemplate.from_template(ROUTER_PROMPT)

    prompt = prompt_template.invoke({})

    model = define_model(None)

    messages = [
        SystemMessage(content=prompt.to_string()),
        HumanMessage(
            content=f"""
        Patient request:
        {patient_request}

        Situation perception:
        {situation_perception}

        Already scheduled activities:
        {_scheduled_summary(state.get("task"))}
        """
                ),
            ]
    
    print("reasoning about whether a new activity is needed...")
    raw = model.invoke(messages)

    content = raw.content
    if not isinstance(content, str):
        content = str(content)

    #print(f"raw model output for routing decision: {content}")

    content = content.strip()

    # rimuove markdown fences
    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")

    content = content.strip()

    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ Robot perception edge: JSON parsing failed, defaulting to SITUATION_ASSESSMENT")
        log_message("ROBOT_PERCEPTION -> JSON parse error, defaulting to SITUATION_ASSESSMENT")
        log_message(f"Raw content: {content}")
        return GraphNodes.SITUATION_ASSESSMENT.value

    need_goal = decision.get("need_goal_planning", False)

    if need_goal:
        print("new activity detected from perception! Routing to GOAL PLANNER")
        log_message("ROBOT_PERCEPTION -> GOAL_PLANNER")
        return GraphNodes.GOAL_PLANNER.value

    log_message("ROBOT_PERCEPTION -> SITUATION_ASSESSMENT")
    print(f"no new activity detected from perception. Routing to SITUATION_ASSESSMENT")
    return GraphNodes.SITUATION_ASSESSMENT.value