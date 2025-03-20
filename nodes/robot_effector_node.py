from colorama import Fore, Back, Style
from graph.graph_init import GraphState
from utils.logs import conversation_message, log_graph_state


def robot_effector_node(state: GraphState) -> GraphState:
    log_graph_state(state, "robot_effector_node")

    if state["fake_datetime"] is None:
        raise ValueError("robot_effector_node - fake_datetime is None")

    # TODO: Add information about task and domain (or pre-defined knowledge)

    print(
        Fore.RED + "\nROBOT_EFFECTOR_NODE - Play vocal response: " + Style.RESET_ALL,
        state["vocal_answer"],
    )
    # TODO: Play vocal response on the real robot through the speakers

    print(
        Fore.GREEN + "\nROBOT_EFFECTOR_NODE - Execute robot action: " + Style.RESET_ALL,
        state["robot_action"],
    )
    # TODO: Execute robot action on the real robot through motion planning

    conversation_message(
        state["fake_datetime"].strftime("%H:%M"),
        "Robot (speaking)",
        state["vocal_answer"],
    )
    conversation_message(
        state["fake_datetime"].strftime("%H:%M"),
        "Robot (acting)",
        state["robot_action"],
    )

    return {
        **state,
        "task": (
            state["updated_task"]
            if state["updated_task"] is not None
            else state["task"]
        ),
        "updated_task": None,
        "robot_action": None,
        "vocal_answer": None,
        "patient_request": None,
        "situation_perception": None,
        "progress_checking_output": None,
    }
