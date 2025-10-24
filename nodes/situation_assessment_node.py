from graph.graph_init import GraphState
from utils.interfaces import RobotBehaviours

from utils.logs import log_graph_state


def situation_assessment_node(state: GraphState) -> GraphState:
    log_graph_state(state, "situation_assessment_node")

    # TODO: Add signals (booleans) from robot's sensors or medical devices to manage emergency situations

    # If the updated_task, the patient_request, situation_perception are None and the robot is reactive, go to the Robot Effector node and wait for a patient request or a situation perception.
    # This can happen during the first iteration of the graph, because it is supposed that the situation_perception is not None in the other cases.
    if (
        state["updated_task"] is None
        and state["patient_request"] is None
        and state["situation_perception"] is None
        and state["robot_behaviour"] == RobotBehaviours.REACTIVE
    ):
        return {
            **state,
            "robot_action": "WAITING",
            "vocal_answer": "How can I help you?",
        }

    # If the updated_task, the patient_request, situation_perception are None and the robot is proactive, go to the Task Progress node and help the patient to progress in the task.
    # This can happen during the first iteration of the graph, because it is supposed that the situation_perception is not None in the other cases.
    if (
        state["updated_task"] is None
        and state["patient_request"] is None
        and state["situation_perception"] is None
        and state["robot_behaviour"] == RobotBehaviours.PROACTIVE
    ):
        return {
            **state,
            "situation_perception": "The patient is waiting for your help for his/her activities.",
        }

    return state
