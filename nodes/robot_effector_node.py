import os
from colorama import Fore, Style
from graph.graph_init import GraphState
from utils.logs import conversation_message, log_graph_state

import asyncio
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

openai = AsyncOpenAI()


async def speak_openai(text: str) -> None:
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="echo",
        input=text,
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)


def robot_effector_node(state: GraphState) -> GraphState:
    log_graph_state(state, "robot_effector_node")

    if state["fake_datetime"] is None:
        raise ValueError("robot_effector_node - fake_datetime is None")

    # TODO: Add information about task and domain (or pre-defined knowledge)

    print(
        Fore.RED + "\nROBOT - Play vocal response: " + Style.RESET_ALL,
        state["vocal_answer"],
    )
    if (
        state["vocal_answer"] is not None
        and os.getenv("AUDIO_ENABLED", "false").lower() == "true"
    ):
        asyncio.run(speak_openai(state["vocal_answer"]))

    # TODO: Play vocal response on the real robot through the speakers

    print(
        Fore.GREEN + "\nROBOT - Execute robot action: " + Style.RESET_ALL,
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
