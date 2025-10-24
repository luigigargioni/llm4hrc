from datetime import datetime
from graph.graph_init import GraphState
from colorama import Fore, Back, Style

from utils.logs import conversation_message, log_graph_state


def robot_perception_node(state: GraphState) -> GraphState:
    log_graph_state(state, "robot_perception_node")

    # TODO: Add information about task and domain (or pre-defined knowledge)

    # TODO: Datetime set by the user to simulate a specific time
    while True:
        if state["fake_datetime"] is not None:
            fake_datetime = input(
                Fore.YELLOW
                + f"""\nEnter the datetime (HH:MM) (actual: {state['fake_datetime'].strftime("%H:%M")}): """
                + Style.RESET_ALL
            ).strip()
        else:
            fake_datetime = input(
                Fore.YELLOW + "\nEnter the datetime (HH:MM): " + Style.RESET_ALL
            ).strip()

        try:
            new_datetime = datetime.strptime(
                datetime.now().strftime("%d-%m-%Y") + " " + fake_datetime + ":00",
                "%d-%m-%Y %H:%M:%S",
            )

            if (
                state["fake_datetime"] is not None
                and new_datetime < state["fake_datetime"]
            ):
                print(
                    Fore.YELLOW
                    + "The entered datetime cannot be earlier than the current datetime."
                    + Style.RESET_ALL
                )
                continue
            else:
                fake_datetime = new_datetime
                break
        except ValueError:
            print(
                Fore.YELLOW
                + "Invalid format. Please enter the datetime in HH:MM format."
                + Style.RESET_ALL
            )
            continue

    # TODO: Record the human request from the robot's microphone
    patient_request = input(
        Fore.CYAN + "\nEnter the request of the patient: " + Style.RESET_ALL
    ).strip()
    if patient_request == "":
        patient_request = None

    # TODO: Record the situation perception from the robot's camera and interpret it using a VLM
    while True:
        situation_perception = input(
            Fore.MAGENTA
            + "\nEnter the situation perception (default: Patient is speaking): "
            + Style.RESET_ALL
        ).strip()
        if situation_perception != "":
            break
        else:
            situation_perception = "Patient is speaking"
            break
            # print(
            #     Fore.MAGENTA
            #     + "Situation perception cannot be empty. Please enter a valid perception."
            #     + Style.RESET_ALL
            # )

    conversation_message(
        fake_datetime.strftime("%H:%M"),
        "Patient",
        patient_request,
    )
    conversation_message(
        fake_datetime.strftime("%H:%M"),
        "Perception",
        situation_perception,
    )

    return {
        **state,
        "patient_request": patient_request,
        "situation_perception": situation_perception,
        "fake_datetime": fake_datetime,
    }
