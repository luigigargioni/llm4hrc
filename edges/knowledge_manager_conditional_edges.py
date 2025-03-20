from typing import Literal
from langgraph.graph import END

from utils.interfaces import GraphNodes, GraphState
from utils.logs import log_message


def knowledge_manager_conditional_edges(
    state: GraphState,
) -> Literal[
    GraphNodes.SITUATION_ASSESSMENT.value,
    END,
]:
    # If the end flag is True, set by the Knowledge Manager node, go to the End node
    if state["end"] is True:
        log_message("\nKNOWLEDGE_MANAGER_CONDITIONAL_EDGES - END")
        return END

    # If the end flag is not True, go to the Situation Assessment node to start the iteration of the graph
    if state["end"] is not True:
        log_message("\nKNOWLEDGE_MANAGER_CONDITIONAL_EDGES - SITUATION_ASSESSMENT")
        return GraphNodes.SITUATION_ASSESSMENT.value
