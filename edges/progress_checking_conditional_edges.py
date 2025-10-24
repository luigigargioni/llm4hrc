from typing import Literal

from utils.interfaces import GraphNodes, GraphState
from utils.logs import log_message


def progress_checking_conditional_edges(
    state: GraphState,
) -> Literal[
    GraphNodes.ROBOT_EFFECTOR.value,
    GraphNodes.SITUATION_ASSESSMENT.value,
]:
    # If the progress checking output is True, go to the Robot Effector node to execute the robot_action and the vocal_answer
    if state["progress_checking_output"] is True:
        log_message("\nPROGRESS_CHECKING_CONDITIONAL_EDGES - ROBOT_EFFECTOR")
        return GraphNodes.ROBOT_EFFECTOR.value

    # If the progress checking output is False, go to the Situation Assessment node to decide if generate a new updated_task, robot_action and vocal_answer
    if state["progress_checking_output"] is False:
        log_message("\nPROGRESS_CHECKING_CONDITIONAL_EDGES - TASK_PROGRESS")
        return GraphNodes.SITUATION_ASSESSMENT.value
