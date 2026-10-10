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

    # Se tutte le activities del task sono state completate o saltate, vai alla Knowledge Manager dire che è tutto finito.
    # altrimenti torna alla Robot Perception per continuare a monitorare la situazione e ricevere eventuali nuove richieste del paziente.

    all_activities_completed = all(
        activity.get("done") is True or activity.get("skipped") is True
        for activity in state["task"]["activities"]
    )
    if all_activities_completed:
        log_message("\nROBOT_EFFECTOR_CONDITIONAL_EDGES - KNOWLEDGE_MANAGER")
        return GraphNodes.KNOWLEDGE_MANAGER.value
    #(else)
    return GraphNodes.ROBOT_PERCEPTION.value
