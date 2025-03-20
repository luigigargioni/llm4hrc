from typing import Literal
from langgraph.graph import END

from graph.graph_init import GraphState
from utils.interfaces import GraphNodes, RobotBehaviours
from utils.logs import log_message


def situation_assessment_conditional_edges(
    state: GraphState,
) -> Literal[
    GraphNodes.TASK_SYNTHESIZER.value,
    GraphNodes.TASK_PROGRESS.value,
    GraphNodes.ROBOT_EFFECTOR.value,
]:
    # If the task is None, it is the first iteration of the graph. Go to the Task Synthesizer node to define the task
    if state["task"] is None:
        log_message("\nSITUATION_ASSESSMENT_CONDITIONAL_EDGES - TASK_SYNTHESIZER")
        return GraphNodes.TASK_SYNTHESIZER.value

    # If the progress checking output is False, go to the Task Progress node to generate a new updated_task, robot_action and vocal_answer
    if state["progress_checking_output"] is False:
        # TODO: If there will not be any other condition to check, it is possible to move the decision directly in the Progress Checking node
        log_message(
            "\nSITUATION_ASSESSMENT_CONDITIONAL_EDGES - TASK_PROGRESS after PROGRESS_CHECKING"
        )
        return GraphNodes.TASK_PROGRESS.value

    # If the patient_request, situation_perception are None and the robot is reactive, go to the Robot Effector node and wait for a patient request or a situation perception.
    # This can happen during the first iteration of the graph, because it is supposed that the situation_perception is not None in the other cases.
    if (
        state["patient_request"] is None
        and state["situation_perception"] is None
        and state["robot_behaviour"] == RobotBehaviours.REACTIVE
    ):
        log_message("\nSITUATION_ASSESSMENT_CONDITIONAL_EDGES - ROBOT_EFFECTOR")
        return GraphNodes.ROBOT_EFFECTOR.value

    # If the task progress output is None and situation perception is not None, we need to go to the task progress node
    # If there is a situation perception, it means that it is possible to interpret the situation and progress in the task
    if (
        state["updated_task"] is None
        and state["situation_perception"] is not None
        and state["end"] is False
    ):
        log_message(
            "\nSITUATION_ASSESSMENT_CONDITIONAL_EDGES - TASK_PROGRESS after ROBOT_PERCEPTION"
        )
        return GraphNodes.TASK_PROGRESS.value

    log_message("\nSITUATION_ASSESSMENT_CONDITIONAL_EDGES - no edge to follow")
    return END
