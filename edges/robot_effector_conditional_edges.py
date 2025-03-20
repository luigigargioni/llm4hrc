from typing import Literal

from utils.interfaces import GraphNodes, GraphState
from utils.logs import log_message


def robot_effector_conditional_edges(
    state: GraphState,
) -> Literal[
    GraphNodes.ROBOT_PERCEPTION.value,
    GraphNodes.KNOWLEDGE_MANAGER.value,
]:
    if state["task"] is None:
        raise ValueError("robot_effector_conditional_edges - task is None")

    # If all activities are completed, we need to go to the Knowledge Manager node to save the conversation history and end the task
    all_activities_completed = all(
        activity["done"] is True or activity["skipped"] is True
        for activity in state["task"]["activities"]
    )
    if all_activities_completed:
        log_message("\nROBOT_EFFECTOR_CONDITIONAL_EDGES - KNOWLEDGE_MANAGER")
        return GraphNodes.KNOWLEDGE_MANAGER.value

    return GraphNodes.ROBOT_PERCEPTION.value
