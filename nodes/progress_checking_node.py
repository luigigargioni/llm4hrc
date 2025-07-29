from datetime import datetime
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from utils.interfaces import ProgressCheckingResponseTypes, ProgressCheckingResponses
import json

from graph.graph_init import GraphState
from utils.models import TEMPERATURE, OllamaModel, define_model
from utils.logs import log_graph_state, log_message


INITIAL_PROMPT = """
# CONTEXT #
You are a robot assistant in a care home. You must help the patient to do his/her activities.
As input you have your action and the vocal answer for the next step of the task.
Check if they are compliant with the prescribed activities, the patient information, and the additional instructions reported below.

# OUTPUT #
The output is a JSON object with this structure:
- "result": "OK" if the robot action and the vocal answer are compliant, "KO" otherwise.
- "reason": A string with the reason for the result.
Reply only with "OK" if the robot action and the vocal answer are compliant with the prescribed activities, the patient information, and the additional instructions reported below, or "KO" otherwise.

# INPUT #
- Prescribed activities and patient information:
{docs_content}

- Robot action:
{robot_action}

- Vocal answer:
{vocal_answer}

- Situation perception:
{situation_perception}

- Patient request:
{patient_request}

# INSTRUCTIONS #
1. Politeness and Helpfulness
   - You must always be polite, kind, and helpful when interacting with patients.
   - Use clear, professional language, avoiding offensive or inappropriate expressions.

2. Ethics and Confidentiality
   - You must respect ethical principles and the privacy of patients in all interactions.
   - Do not disclose or share sensitive information about patients or healthcare professionals.

3. Limitations on Providing Medical Advice
   - You must not provide diagnoses, prescriptions, or medical advice beyond the prescribed activities.
   - Always encourage users to consult a qualified healthcare professional for medical concerns.
   - Do not interpret medical tests or suggest treatments without explicit instructions from qualified professionals.

4. Accuracy and Truthfulness
   - You must provide only verified and accurate information.
   - Do not generate or spread false, misleading, or unverified information.
   - If unsure about an answer, you must clearly state it and refer the user to reliable sources or competent professionals.

5. Handling Patient Information
   - You must not alter, manipulate, or fabricate patient-related information.
   - You must not disclose a patient’s personal information, even to the patient themselves, unless explicitly authorized.

6. Neutrality and Non-Judgmental Approach
   - Maintain a neutral and unbiased approach in responses.
   - Do not discriminate based on ethnicity, gender, religion, sexual orientation, age, or any other personal characteristic.

7. Limitations on Actions and Autonomy
   - You must not make autonomous decisions that could directly impact a patient's health or well-being.

8. Prevention of Abuse and Improper Use
   - Do not engage in inappropriate, threatening, or illegal conversations.
   - Avoid any form of manipulation or influence that could lead users to harmful decisions.

9. Clarity About Capabilities and Limitations
   - Explain your limitations and the need for human intervention when necessary.
"""


@tool
def response_template(progress_checking: ProgressCheckingResponses) -> bool:
    """The JSON format for the output of the next steps node.

    Args:
        progress_checking: ProgressCheckingResponses

    """
    return True


def progress_checking_node(state: GraphState) -> GraphState:
    log_graph_state(state, "progress_checking_node")

    # REMOVE
    return {**state, "progress_checking_output": True}

    if state["vector_store"] is None:
        raise ValueError("PROGRESS_CHECKING_NODE - vector_store is None")

    if state["patient_id"] is None:
        raise ValueError("PROGRESS_CHECKING_NODE - patient_id is None")

    docs_content = state["progress_checking_similarity_search"]
    if docs_content is None:
        # TODO: Query must be dynamic based on the robot_action and vocal_answer
        retrieved_vectors = state["vector_store"].similarity_search(
            "Retrieve information about the prescribed activities and the patient for the patient with Record Number: "
            + state["patient_id"],
            k=1,
            filter={"patient_id": state["patient_id"]},
        )
        docs_content = "\n\n".join(vector.page_content for vector in retrieved_vectors)

    prompt_template = PromptTemplate.from_template(INITIAL_PROMPT)

    if state["fake_datetime"] is None:
        raise ValueError("next_steps_node - fake_datetime is None")
    formatted_datetime = state["fake_datetime"].strftime("%d-%m-%Y %H:%M:%S")

    prompt = prompt_template.invoke(
        {
            "situation_perception": state["situation_perception"],
            "patient_request": state["patient_request"],
            "robot_action": state["robot_action"],
            "vocal_answer": state["vocal_answer"],
            "docs_content": docs_content,
            # "datetime": formatted_datetime,
        }
    )

    checking_model = os.getenv("PROGRESS_CHECKING_MODEL") or OllamaModel.LLAMA_3_1.value
    model = ChatOllama(
        model=checking_model, temperature=TEMPERATURE, format="json"
    ).bind_tools([response_template])

    result = model.invoke(prompt).to_json()
    progress_checking_result = result["kwargs"]["tool_calls"][0]["args"][
        "progress_checking"
    ]

    log_message(f"PROGRESS_CHECKING_NODE - result: {progress_checking_result}")
    print(f"PROGRESS_CHECKING_NODE - result: {progress_checking_result}")

    progress_checking_output = (
        True
        if progress_checking_result["result"] == ProgressCheckingResponseTypes.OK.value
        else False
    )

    return {**state, "progress_checking_output": progress_checking_output}
