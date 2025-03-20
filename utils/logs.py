from datetime import datetime
import json
from utils.interfaces import GraphState


LOG_FILENAME = f"log/{datetime.now().strftime('%d%m%Y_%H%M%S')}.txt"
CONVERSATION_FILENAME = (
    f"conversation/{datetime.now().strftime('%d%m%Y_%H%M%S')}_conversation.txt"
)


def create_log_file() -> None:
    with open(LOG_FILENAME, "w") as file:
        file.write("")


def log_graph_state(state: GraphState, node: str) -> None:
    state_without_vector_store = {
        "task": state["task"],
        "progress_checking_output": state["progress_checking_output"],
        # "conversation_history": state["conversation_history"],
        "updated_task": state["updated_task"],
        "robot_action": state["robot_action"],
        "vocal_answer": state["vocal_answer"],
        "patient_request": state["patient_request"],
        "situation_perception": state["situation_perception"],
        "robot_behaviour": state["robot_behaviour"],
        "doctor_specs": state["doctor_specs"],
        "end": state["end"],
        "patient_id": state["patient_id"],
        "doctor_specifications": state["doctor_specifications"],
        "fake_datetime": (
            state["fake_datetime"].strftime("%d/%m/%Y, %H:%M:%S")
            if state["fake_datetime"] is not None
            else None
        ),
    }

    formatted_datetime = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    separator = "-" * 30
    with open(LOG_FILENAME, "a") as file:
        file.write(
            f"Datetime: {formatted_datetime}\nNode: {node}\n\n{json.dumps(state_without_vector_store, indent=4)}\n\n{separator}\n\n"
        )


def log_message(message: str) -> None:
    separator = "-" * 30
    with open(LOG_FILENAME, "a") as file:
        file.write(f"{message}\n\n{separator}\n\n")


def conversation_header(
    message: str,
) -> None:
    with open(CONVERSATION_FILENAME, "a") as file:
        file.write(message)


def conversation_message(
    datetime: str,
    actor: str,
    message: str | None = None,
) -> None:
    with open(CONVERSATION_FILENAME, "a") as file:
        file.write(f"{datetime} - {actor}: {message}\n\n")
